//! Expert modules for HAS-MoE-FFN

use serde::{Deserialize, Serialize};

/// GELU activation function (standalone, for sharing across modules)
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let x_cubed = x * x * x;
    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
    x * 0.5 * (1.0 + tanh_arg.tanh())
}

/// Expert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub use_dropout: bool,
    pub dropout_rate: f32,
}

/// Individual expert in the MoE system
pub struct Expert {
    config: ExpertConfig,
    fc1_weights: Option<Vec<Vec<f32>>>,
    fc1_bias: Option<Vec<f32>>,
    fc2_weights: Option<Vec<Vec<f32>>>,
    fc2_bias: Option<Vec<f32>>,
    #[cfg(feature = "gpu")]
    fc1_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc1_bias_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc2_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc2_bias_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "cuda")]
    fc1_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
    #[cfg(feature = "cuda")]
    fc1_bias_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
    #[cfg(feature = "cuda")]
    fc2_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
    #[cfg(feature = "cuda")]
    fc2_bias_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
}

impl Expert {
    /// Create new expert
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        use_dropout: bool,
        dropout_rate: f32,
    ) -> Self {
        let config = ExpertConfig {
            hidden_size,
            intermediate_size,
            use_dropout,
            dropout_rate,
        };

        Self {
            config,
            fc1_weights: None,
            fc1_bias: None,
            fc2_weights: None,
            fc2_bias: None,
            #[cfg(feature = "gpu")]
            fc1_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc1_bias_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc2_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc2_bias_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc1_cuda: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc1_bias_cuda: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc2_cuda: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc2_bias_cuda: std::sync::OnceLock::new(),
        }
    }

    fn get_fc1_weights(&self) -> &Vec<Vec<f32>> {
        self.fc1_weights.as_ref().expect("fc1_weights not initialized — call init_random()")
    }

    fn get_fc1_bias(&self) -> &Vec<f32> {
        self.fc1_bias.as_ref().expect("fc1_bias not initialized — call init_random()")
    }

    fn get_fc2_weights(&self) -> &Vec<Vec<f32>> {
        self.fc2_weights.as_ref().expect("fc2_weights not initialized — call init_random()")
    }

    fn get_fc2_bias(&self) -> &Vec<f32> {
        self.fc2_bias.as_ref().expect("fc2_bias not initialized — call init_random()")
    }

    pub fn init_random(&mut self) {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        self.fc1_weights = Some(Self::init_weights(h, i));
        self.fc1_bias = Some(vec![0.0; i]);
        self.fc2_weights = Some(Self::init_weights(i, h));
        self.fc2_bias = Some(vec![0.0; h]);
    }

    pub fn drop_cpu_weights(&mut self) {
        self.fc1_weights = None;
        self.fc1_bias = None;
        self.fc2_weights = None;
        self.fc2_bias = None;
    }

    pub fn has_weights(&self) -> bool {
        self.fc1_weights.is_some()
    }

    /// Initialize weights with Xavier/Glorot initialization
    fn init_weights(input_size: usize, output_size: usize) -> Vec<Vec<f32>> {
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            let row: Vec<f32> = (0..input_size)
                .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            weights.push(row);
        }
        weights
    }

    /// Forward pass through expert
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        {
            let gpu_result = self.forward_gpu(input);
            if let Some(result) = gpu_result {
                return result;
            }
            tracing::warn!("MoE expert GPU forward failed, falling back to CPU");
        }

        // First linear layer + activation
        let hidden = self.fc1_forward(input);
        let activated = self.apply_gelu(&hidden);

        // Apply dropout if enabled
        let dropped = if self.config.use_dropout {
            self.apply_dropout(&activated)
        } else {
            activated
        };

        // Second linear layer
        let output = self.fc2_forward(&dropped);

        output
    }

    /// GPU-accelerated forward pass
    #[cfg(feature = "gpu")]
    fn forward_gpu(&self, input: &[f32]) -> Option<Vec<f32>> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        use tracing::warn;
        let ctx = match GpuContext::global() {
            Ok(ctx) => ctx,
            Err(e) => {
                warn!("GPU forward failed (GPU unavailable): {e}");
                return None;
            }
        };
        let dim = input.len();
        let fc1_w = self.get_fc1_weights();
        let fc1_b = self.get_fc1_bias();

        // Upload input as [1, hidden_size]
        let input_gpu =
            GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![1, dim], input.to_vec()).ok()?)
                .ok()?;

        // fc1 weights: [intermediate_size, hidden_size] → transpose → [hidden_size, intermediate_size]
        let fc1_flat: Vec<f32> = fc1_w.iter().flat_map(|r| r.iter()).copied().collect();
        let w1_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![fc1_w.len(), dim], fc1_flat).ok()?,
        )
        .ok()?;
        let w1t_gpu = ctx.transpose(&w1_gpu).ok()?;
        let b1_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, fc1_b.len()], fc1_b.clone()).ok()?,
        )
        .ok()?;

        let hidden_gpu = ctx.matmul(&input_gpu, &w1t_gpu).ok()?;
        let mut hidden_gpu = ctx.add(&hidden_gpu, &b1_gpu).ok()?;

        // GELU in-place on GPU (no CPU roundtrip)
        ctx.gelu_inplace(&mut hidden_gpu).ok()?;

        // fc2 weights: [hidden_size, intermediate_size] → transpose → [intermediate_size, hidden_size]
        let fc2_w = self.get_fc2_weights();
        let fc2_b = self.get_fc2_bias();
        let fc2_flat: Vec<f32> = fc2_w.iter().flat_map(|r| r.iter()).copied().collect();
        let inter_size = fc2_w.len();
        let w2_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![inter_size, dim], fc2_flat).ok()?,
        )
        .ok()?;
        let w2t_gpu = ctx.transpose(&w2_gpu).ok()?;
        let b2_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, fc2_b.len()], fc2_b.clone()).ok()?,
        )
        .ok()?;

        let act_gpu = if self.config.use_dropout {
            let hidden_cpu = hidden_gpu.to_cpu().ok()?;
            let hidden_slice: Vec<f32> = hidden_cpu.iter().copied().collect();
            let dropped = self.apply_dropout(&hidden_slice);
            GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![1, dropped.len()], dropped).ok()?,
            )
            .ok()?
        } else {
            hidden_gpu
        };

        let out_gpu = ctx.matmul(&act_gpu, &w2t_gpu).ok()?;
        let out_gpu = ctx.add(&out_gpu, &b2_gpu).ok()?;

        let out_cpu = out_gpu.to_cpu().ok()?;
        Some(out_cpu.iter().copied().collect())
    }

    /// Lazily upload and transpose expert weights to GPU — cached via OnceLock.
    /// Returns (fc1_w_t, fc1_bias, fc2_w_t, fc2_bias) as GpuTensors or None.
    #[cfg(feature = "gpu")]
    fn ensure_weights_gpu(&self) -> Option<(&nexora_autograd::gpu::GpuTensor, &nexora_autograd::gpu::GpuTensor, &nexora_autograd::gpu::GpuTensor, &nexora_autograd::gpu::GpuTensor)> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global().ok()?;
        let fc1_w = self.fc1_weights.as_ref()?;
        let fc1_b = self.fc1_bias.as_ref()?;
        let fc2_w = self.fc2_weights.as_ref()?;
        let fc2_b = self.fc2_bias.as_ref()?;
        let w1 = self.fc1_gpu.get_or_init(|| {
            let flat: Vec<f32> = fc1_w.iter().flatten().copied().collect();
            let cpu = ArrayD::from_shape_vec(vec![fc1_w.len(), self.config.hidden_size], flat).ok()?;
            let gpu = GpuTensor::from_cpu(&cpu).ok()?;
            ctx.transpose(&gpu).ok()
        }).as_ref()?;
        let b1 = self.fc1_bias_gpu.get_or_init(|| {
            GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![1, fc1_b.len()], fc1_b.clone()).ok()?).ok()
        }).as_ref()?;
        let w2 = self.fc2_gpu.get_or_init(|| {
            let flat: Vec<f32> = fc2_w.iter().flatten().copied().collect();
            let cpu = ArrayD::from_shape_vec(vec![fc2_w.len(), self.config.intermediate_size], flat).ok()?;
            let gpu = GpuTensor::from_cpu(&cpu).ok()?;
            ctx.transpose(&gpu).ok()
        }).as_ref()?;
        let b2 = self.fc2_bias_gpu.get_or_init(|| {
            GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![1, fc2_b.len()], fc2_b.clone()).ok()?).ok()
        }).as_ref()?;
        Some((w1, b1, w2, b2))
    }

    /// Batched GPU forward: processes N tokens in one GPU call.
    /// Returns None if GPU unavailable (CPU fallback handled by caller).
    #[cfg(feature = "gpu")]
    pub fn forward_batched_gpu(&self, inputs: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let (w1, b1, w2, b2) = self.ensure_weights_gpu()?;
        let ctx = GpuContext::global().ok()?;
        let input_gpu = GpuTensor::from_cpu(&inputs.clone().into_dyn()).ok()?;
        let mut hidden = ctx.matmul(&input_gpu, w1).ok()?;
        hidden = ctx.add(&hidden, b1).ok()?;
        ctx.gelu_inplace(&mut hidden).ok()?;
        let output = ctx.matmul(&hidden, w2).ok()?;
        let output = ctx.add(&output, b2).ok()?;
        let cpu_d = output.to_cpu().ok()?;
        cpu_d.into_dimensionality::<ndarray::Ix2>().ok()
    }

    /// Lazily upload expert weights to CUDA — cached via OnceLock.
    /// Returns (fc1, fc1_bias, fc2, fc2_bias) as CudaTensors or None.
    #[cfg(feature = "cuda")]
    pub(crate) fn ensure_weights_cuda(&self, cuda: &nexora_autograd::gpu::cuda::CudaRuntime) -> Option<(&nexora_autograd::gpu::cuda::CudaTensor, &nexora_autograd::gpu::cuda::CudaTensor, &nexora_autograd::gpu::cuda::CudaTensor, &nexora_autograd::gpu::cuda::CudaTensor)> {
        use nexora_autograd::gpu::cuda::CudaTensor;
        let fc1_w = self.fc1_weights.as_ref()?;
        let fc1_b = self.fc1_bias.as_ref()?;
        let fc2_w = self.fc2_weights.as_ref()?;
        let fc2_b = self.fc2_bias.as_ref()?;
        let w1 = self.fc1_cuda.get_or_init(|| {
            let flat: Vec<f32> = fc1_w.iter().flatten().copied().collect();
            // Store transposed: [hidden_size, intermediate_size]
            CudaTensor::from_cpu(&cuda.device, vec![fc1_w.len(), self.config.hidden_size], &flat).ok()
        }).as_ref()?;
        let b1 = self.fc1_bias_cuda.get_or_init(|| {
            CudaTensor::from_cpu(&cuda.device, vec![1, fc1_b.len()], fc1_b).ok()
        }).as_ref()?;
        let w2 = self.fc2_cuda.get_or_init(|| {
            let flat: Vec<f32> = fc2_w.iter().flatten().copied().collect();
            // Store transposed: [intermediate_size, hidden_size]
            CudaTensor::from_cpu(&cuda.device, vec![fc2_w.len(), self.config.intermediate_size], &flat).ok()
        }).as_ref()?;
        let b2 = self.fc2_bias_cuda.get_or_init(|| {
            CudaTensor::from_cpu(&cuda.device, vec![1, fc2_b.len()], fc2_b).ok()
        }).as_ref()?;
        Some((w1, b1, w2, b2))
    }

    /// Batched CUDA forward: processes N tokens via cuBLAS (no CPU round-trip for weights).
    /// Returns None if CUDA unavailable.
    #[cfg(feature = "cuda")]
    pub fn forward_batched_cuda(&self, inputs: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuBackend};
        let ctx = GpuContext::global().ok()?;
        if ctx.backend() != GpuBackend::Cuda {
            return None;
        }
        let cuda = ctx.cuda_runtime()?;
        let (w1, b1, w2, b2) = self.ensure_weights_cuda(cuda)?;

        let n = inputs.shape()[0];
        let dim = inputs.shape()[1];

        // Upload input to GPU
        let input_flat: Vec<f32> = inputs.iter().copied().collect();
        let input_gpu = nexora_autograd::gpu::cuda::CudaTensor::from_cpu(
            &cuda.device, vec![n, dim], &input_flat,
        ).ok()?;

        // fc1: [n, dim] @ [inter, dim] → [n, inter]
        // w1 stored as [inter, dim]; cuBLAS transposes both operands internally
        let mut hidden = cuda.matmul(&input_gpu, w1).ok()?;
        hidden = cuda.add(&hidden, b1).ok()?;
        cuda.gelu_inplace(&mut hidden).ok()?;

        // fc2: [n, inter] @ [dim, inter] → [n, dim]
        // w2 stored as [dim, inter]; cuBLAS handles transpose
        let output = cuda.matmul(&hidden, w2).ok()?;
        let output = cuda.add(&output, b2).ok()?;

        // Readback
        let out_cpu = output.to_cpu_vec(&cuda.device).ok()?;
        let out_shape = vec![n, self.config.hidden_size];
        let out_flat: Vec<f32> = out_cpu;
        ndarray::Array2::from_shape_vec(out_shape, out_flat).ok()
    }

    /// Batched forward: processes N tokens, tries CUDA → wgpu → CPU fallback.
    pub fn forward_batched(&self, inputs: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        #[cfg(feature = "cuda")]
        if let Some(result) = self.forward_batched_cuda(inputs) {
            return result;
        }
        #[cfg(feature = "gpu")]
        if let Some(result) = self.forward_batched_gpu(inputs) {
            return result;
        }
        let n = inputs.shape()[0];
        let h = self.config.hidden_size;
        let mut outputs = ndarray::Array2::zeros((n, h));
        for i in 0..n {
            let token = inputs.row(i);
            let result = self.forward(token.as_slice().unwrap_or(&[]));
            for (j, &v) in result.iter().enumerate() {
                outputs[[i, j]] = v;
            }
        }
        outputs
    }

    /// First linear layer forward pass
    fn fc1_forward(&self, input: &[f32]) -> Vec<f32> {
        let w = self.get_fc1_weights();
        let b = self.get_fc1_bias();
        w.iter()
            .zip(b.iter())
            .map(|(weights, bias)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    sum += w * x;
                }
                sum
            })
            .collect()
    }

    /// Second linear layer forward pass
    fn fc2_forward(&self, input: &[f32]) -> Vec<f32> {
        let w = self.get_fc2_weights();
        let b = self.get_fc2_bias();
        w.iter()
            .zip(b.iter())
            .map(|(weights, bias)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    sum += w * x;
                }
                sum
            })
            .collect()
    }

    /// GELU activation function
    fn apply_gelu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| gelu(x)).collect()
    }

    /// Apply dropout during training
    fn apply_dropout(&self, input: &[f32]) -> Vec<f32> {
        let rate = self.config.dropout_rate;
        let scale = 1.0 / (1.0 - rate);
        input
            .iter()
            .map(|&x| {
                if rand::random::<f32>() < rate {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect()
    }

    /// Get expert configuration
    pub fn config(&self) -> &ExpertConfig {
        &self.config
    }

    /// Get expert size information
    pub fn size_info(&self) -> (usize, usize) {
        (self.config.hidden_size, self.config.intermediate_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_expert() -> Expert {
        let mut e = Expert::new(4, 8, false, 0.0);
        e.init_random();
        e
    }

    #[test]
    fn test_gelu_shape() {
        let x = gelu(0.0);
        assert!((x - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu_positive() {
        let x = gelu(1.0);
        assert!(x > 0.8);
    }

    #[test]
    fn test_gelu_negative() {
        let x = gelu(-1.0);
        assert!(x < 0.0);
        assert!(x > -0.2);
    }

    #[test]
    fn test_new_expert_config() {
        let mut e = Expert::new(16, 32, true, 0.1);
        e.init_random();
        let (h, i) = e.size_info();
        assert_eq!(h, 16);
        assert_eq!(i, 32);
        assert_eq!(e.get_fc1_weights().len(), 32);
        assert_eq!(e.get_fc1_weights()[0].len(), 16);
        assert_eq!(e.get_fc2_weights().len(), 16);
        assert_eq!(e.get_fc2_weights()[0].len(), 32);
    }

    #[test]
    fn test_forward_output_dim() {
        let e = small_expert();
        let input = vec![0.5; 4];
        let output = e.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_forward_deterministic_without_dropout() {
        let e = small_expert();
        let input = vec![1.0, 0.5, 0.0, -0.5];
        let out1 = e.forward(&input);
        let out2 = e.forward(&input);
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fc1_forward_output_dim() {
        let e = small_expert();
        let input = vec![0.5; 4];
        let hidden = e.fc1_forward(&input);
        assert_eq!(hidden.len(), 8);
    }

    #[test]
    fn test_fc2_forward_output_dim() {
        let e = small_expert();
        let input = vec![0.5; 8];
        let output = e.fc2_forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_apply_gelu() {
        let e = small_expert();
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let activated = e.apply_gelu(&input);
        assert_eq!(activated.len(), 4);
        assert!(activated[1].abs() < 1e-5);
        assert!(activated[0] < activated[2]);
    }

    #[test]
    fn test_apply_dropout_with_zero_rate() {
        let e = Expert::new(4, 8, true, 0.0);
        let input = vec![1.0; 4];
        let dropped = e.apply_dropout(&input);
        assert_eq!(dropped.len(), 4);
        for v in &dropped {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_dropout_with_full_rate() {
        let e = Expert::new(4, 8, true, 1.0);
        let input = vec![1.0; 4];
        let dropped = e.apply_dropout(&input);
        for v in &dropped {
            assert!((v - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_forward_zero_input() {
        let e = small_expert();
        let input = vec![0.0; 4];
        let output = e.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_config() {
        let e = small_expert();
        let config = e.config();
        assert_eq!(config.hidden_size, 4);
        assert_eq!(config.intermediate_size, 8);
    }

    #[test]
    fn test_init_weights_dimensions() {
        let w = Expert::init_weights(10, 5);
        assert_eq!(w.len(), 5);
        assert_eq!(w[0].len(), 10);
    }
}
