pub mod attention;
pub mod config;
pub mod error;
pub mod experts;
/// Heterogeneous Attention-based Sparse MoE Feed-Forward Network (HAS-MoE-FFN)
///
/// Advanced sparse mixture of experts implementation for transformer models
pub mod layers;
pub mod routing;
pub mod types;

// Re-export main components
pub use attention::*;
pub use config::*;
pub use error::*;
pub use experts::*;
pub use layers::*;
pub use routing::*;
pub use types::*;

use serde::{Deserialize, Serialize};

/// Main HAS-MoE-FFN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasMoeFFNConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub use_dropout: bool,
    pub dropout_rate: f32,
}

impl Default for HasMoeFFNConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            hidden_size: 768,
            intermediate_size: 3072,
            use_dropout: true,
            dropout_rate: 0.1,
        }
    }
}

/// Main HAS-MoE-FFN implementation
pub struct HasMoeFFN {
    config: HasMoeFFNConfig,
    experts: Vec<crate::experts::Expert>,
    router: crate::routing::Router,
}

impl HasMoeFFN {
    /// Create new HAS-MoE-FFN
    pub fn new(config: HasMoeFFNConfig) -> Self {
        let mut experts = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts {
            experts.push(crate::experts::Expert::new(
                config.hidden_size,
                config.intermediate_size,
                config.use_dropout,
                config.dropout_rate,
            ));
        }

        let router =
            crate::routing::Router::new(config.hidden_size, config.num_experts, config.top_k);

        Self {
            config,
            experts,
            router,
        }
    }

    /// Forward pass through HAS-MoE-FFN
    /// GPU-accelerated: groups tokens by expert, processes batches on GPU.
    pub fn forward(&self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        let (batch_size, hidden_size) = input.dim();

        // Route to experts
        let routing_weights = self.router.forward(input);

        // Try CUDA fused path: all experts on GPU, single readback
        #[cfg(feature = "cuda")]
        if let Some(result) = self.forward_cuda_fused(input, &routing_weights) {
            return result;
        }

        // Try wgpu fused path: all experts on GPU, single readback
        #[cfg(feature = "gpu")]
        if let Some(result) = self.forward_wgpu_fused(input, &routing_weights) {
            return result;
        }

        // Initialize output tensor
        let mut output = ndarray::Array2::zeros((batch_size, hidden_size));

        // Group tokens by expert for batched processing
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.config.num_experts];
        for i in 0..batch_size {
            let top_experts = self.get_top_experts(&routing_weights, i);
            for &expert_idx in &top_experts {
                let weight = routing_weights[[i, expert_idx]];
                expert_tokens[expert_idx].push((i, weight));
            }
        }

        // Process each expert's assigned tokens in batch
        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            let n = tokens.len();
            if n == 0 {
                continue;
            }

            // Gather token embeddings into batch matrix
            let mut batch_input = ndarray::Array2::zeros((n, hidden_size));
            for (k, &(token_idx, _)) in tokens.iter().enumerate() {
                let row_view = input.row(token_idx);
                batch_input
                    .row_mut(k)
                    .assign(&row_view);
            }

            // Batched expert forward (GPU if available, CPU fallback)
            let batch_output = self.experts[expert_idx].forward_batched(&batch_input);

            // Scatter weighted results back to output
            for (k, &(token_idx, weight)) in tokens.iter().enumerate() {
                let out_row = batch_output.row(k);
                for j in 0..hidden_size {
                    output[[token_idx, j]] += out_row[j] * weight;
                }
            }
        }

        output
    }

    /// wgpu fused forward: all expert computations on GPU before any readback.
    /// Batches all expert forwards, then reads back all results at once —
    /// avoids GPU idling between per-expert readback stalls.
    #[cfg(feature = "gpu")]
    fn forward_wgpu_fused(
        &self,
        input: &ndarray::Array2<f32>,
        routing_weights: &ndarray::Array2<f32>,
    ) -> Option<ndarray::Array2<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global().ok()?;
        let (batch_size, hidden_size) = input.dim();
        let num_experts = self.config.num_experts;

        // Group tokens by expert (CPU routing)
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for i in 0..batch_size {
            let top_experts = self.get_top_experts(routing_weights, i);
            for &expert_idx in &top_experts {
                let weight = routing_weights[[i, expert_idx]];
                expert_tokens[expert_idx].push((i, weight));
            }
        }

        ctx.begin_batch_mode();

        // Process each expert on GPU, store GPU tensor (no readback between experts)
        struct GpuExpertResult {
            tokens: Vec<(usize, f32)>,
            output_gpu: GpuTensor,
        }
        let mut gpu_results: Vec<GpuExpertResult> = Vec::new();

        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            let n = tokens.len();
            if n == 0 {
                continue;
            }

            // Gather token embeddings (CPU)
            let mut batch_input = ndarray::Array2::zeros((n, hidden_size));
            for (k, &(token_idx, _)) in tokens.iter().enumerate() {
                let row_view = input.row(token_idx);
                batch_input.row_mut(k).assign(&row_view);
            }

            // Expert forward on GPU (no readback)
            let expert_out = self.experts[expert_idx]
                .forward_batched_gpu_keep_gpu(&batch_input)?;

            gpu_results.push(GpuExpertResult {
                tokens: tokens.clone(),
                output_gpu: expert_out,
            });
        }

        ctx.end_batch_mode();

        // Read back all results and scatter-accumulate on CPU
        let mut output = ndarray::Array2::zeros((batch_size, hidden_size));
        for result in &gpu_results {
            let cpu_d = result.output_gpu.to_cpu().ok()?;
            let batch_output = cpu_d.into_dimensionality::<ndarray::Ix2>().ok()?;
            for (k, &(token_idx, weight)) in result.tokens.iter().enumerate() {
                let out_row = batch_output.row(k);
                for j in 0..hidden_size {
                    output[[token_idx, j]] += out_row[j] * weight;
                }
            }
        }

        Some(output)
    }

    /// CUDA fused forward: all expert computations on GPU with single readback.
    /// Processes each expert sequentially on GPU, scatters weighted results
    /// into an output buffer, then reads back once — avoids N-1 CPU readbacks.
    #[cfg(feature = "cuda")]
    fn forward_cuda_fused(
        &self,
        input: &ndarray::Array2<f32>,
        routing_weights: &ndarray::Array2<f32>,
    ) -> Option<ndarray::Array2<f32>> {
        use nexora_autograd::gpu::cuda::{CudaTensor, CudaSlice};
        use nexora_autograd::gpu::{GpuContext, GpuBackend};
        let ctx = GpuContext::global().ok()?;
        if ctx.backend() != GpuBackend::Cuda {
            return None;
        }
        let cuda = ctx.cuda_runtime()?;

        let (batch_size, hidden_size) = input.dim();
        let num_experts = self.config.num_experts;

        // Group tokens by expert (CPU routing)
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for i in 0..batch_size {
            let top_experts = self.get_top_experts(routing_weights, i);
            for &expert_idx in &top_experts {
                let weight = routing_weights[[i, expert_idx]];
                expert_tokens[expert_idx].push((i, weight));
            }
        }

        // Allocate output buffer on GPU
        let output_zeros = vec![0.0f32; batch_size * hidden_size];
        let mut output_gpu =
            CudaTensor::from_cpu(&cuda.device, vec![batch_size, hidden_size], &output_zeros).ok()?;

        // Process each expert on GPU, scatter results in-place
        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            let n = tokens.len();
            if n == 0 {
                continue;
            }

            // Gather token embeddings (CPU)
            let mut batch_input = ndarray::Array2::zeros((n, hidden_size));
            for (k, &(token_idx, _)) in tokens.iter().enumerate() {
                let row_view = input.row(token_idx);
                batch_input.row_mut(k).assign(&row_view);
            }

            // Upload to CUDA
            let batch_flat: Vec<f32> = batch_input.iter().copied().collect();
            let input_group =
                CudaTensor::from_cpu(&cuda.device, vec![n, hidden_size], &batch_flat).ok()?;

            // Expert computation (all on GPU, no CPU readback between ops)
            let (w1, b1, w2, b2) = self.experts[expert_idx].ensure_weights_cuda(cuda)?;
            let mut hidden = cuda.matmul(&input_group, w1).ok()?;
            hidden = cuda.add(&hidden, b1).ok()?;
            cuda.gelu_inplace(&mut hidden).ok()?;
            let mut expert_out = cuda.matmul(&hidden, w2).ok()?;
            expert_out = cuda.add(&expert_out, b2).ok()?;

            // Upload indices & weights for GPU scatter
            let indices: Vec<i32> = tokens.iter().map(|(idx, _)| *idx as i32).collect();
            let weights: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();
            let indices_gpu: CudaSlice<i32> = cuda.device.htod_sync_copy(&indices).ok()?;
            let weights_gpu: CudaSlice<f32> = cuda.device.htod_sync_copy(&weights).ok()?;

            cuda.scatter_add_weighted(&mut output_gpu, &expert_out, &indices_gpu, &weights_gpu)
                .ok()?;
        }

        // Single readback
        let out_vec = output_gpu.to_cpu_vec(&cuda.device).ok()?;
        ndarray::Array2::from_shape_vec((batch_size, hidden_size), out_vec).ok()
    }

    /// Get top-k experts for a specific token
    fn get_top_experts(
        &self,
        routing_weights: &ndarray::Array2<f32>,
        token_idx: usize,
    ) -> Vec<usize> {
        let mut expert_scores: Vec<(usize, f32)> = (0..self.config.num_experts)
            .map(|j| (j, routing_weights[[token_idx, j]]))
            .collect();

        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        expert_scores
            .iter()
            .take(k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect()
    }

    /// Initialize all sub-components with random weights
    pub fn init_random(&mut self) {
        self.router.init_random();
        for expert in &mut self.experts {
            expert.init_random();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &HasMoeFFNConfig {
        &self.config
    }
}

impl Default for HasMoeFFN {
    fn default() -> Self {
        Self::new(HasMoeFFNConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_moe() -> HasMoeFFN {
        let mut moe = HasMoeFFN::new(HasMoeFFNConfig {
            num_experts: 4,
            top_k: 2,
            hidden_size: 4,
            intermediate_size: 8,
            use_dropout: false,
            dropout_rate: 0.0,
        });
        moe.init_random();
        moe
    }

    #[test]
    fn test_new_creates_correct_number_of_experts() {
        let moe = small_moe();
        assert_eq!(moe.experts.len(), 4);
        assert_eq!(moe.config.num_experts, 4);
    }

    #[test]
    fn test_forward_output_shape() {
        let moe = small_moe();
        let input = ndarray::Array2::from_shape_fn((3, 4), |(_, _)| rand::random::<f32>());
        let output = moe.forward(&input);
        assert_eq!(output.dim(), (3, 4));
    }

    #[test]
    fn test_forward_preserves_batch_dim() {
        let moe = small_moe();
        let input = ndarray::Array2::from_shape_fn((5, 4), |(_, _)| 0.5);
        let output = moe.forward(&input);
        assert_eq!(output.dim(), (5, 4));
    }

    #[test]
    fn test_forward_deterministic() {
        let moe = small_moe();
        let input = ndarray::Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f32 / 8.0);
        let out1 = moe.forward(&input);
        let out2 = moe.forward(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_forward_zero_input() {
        let moe = small_moe();
        let input = ndarray::Array2::zeros((2, 4));
        let output = moe.forward(&input);
        assert_eq!(output.dim(), (2, 4));
    }

    #[test]
    fn test_get_top_experts_returns_correct_count() {
        let moe = small_moe();
        let weights = ndarray::Array2::from_shape_vec((1, 4), vec![0.1, 0.4, 0.3, 0.2]).unwrap();
        let top = moe.get_top_experts(&weights, 0);
        assert_eq!(top.len(), 2);
        assert!(top.contains(&1));
    }

    #[test]
    fn test_config_returns_config() {
        let moe = small_moe();
        let cfg = moe.config();
        assert_eq!(cfg.hidden_size, 4);
        assert_eq!(cfg.top_k, 2);
    }

    #[test]
    fn test_default_uses_8_experts() {
        let moe = HasMoeFFN::default();
        let cfg = moe.config();
        assert_eq!(cfg.num_experts, 8);
        assert_eq!(cfg.hidden_size, 768);
    }

    #[test]
    fn test_forward_batch_consistency() {
        let moe = small_moe();
        let input = ndarray::Array2::from_shape_fn((3, 4), |(_, j)| j as f32 / 4.0);
        let output = moe.forward(&input);
        for i in 0..3 {
            for j in 0..4 {
                assert!(output[[i, j]].is_finite());
            }
        }
    }
}
