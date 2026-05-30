//! ORACLE Backbone - Sparse MoE + MLA Implementation
//!
//! Fondasi arsitektur ORACLE yang menggabungkan Mixture of Experts (MoE)
//! dengan Multi-head Latent Attention (MLA) untuk efisiensi komputasi
//! dan context window yang besar.

use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, ArrayBase, Data};
use nexora_autograd::{ops, Tensor, TensorOps};
use serde::{Deserialize, Serialize};

/// Konfigurasi ORACLE Backbone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleBackboneConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of experts in MoE
    pub n_experts: usize,
    /// Top-k experts to activate
    pub top_k: usize,
    /// Latent dimension for MLA
    pub latent_dim: usize,
    /// Context window size
    pub context_size: usize,
    /// MLP hidden dimension
    pub mlp_hidden: usize,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for OracleBackboneConfig {
    fn default() -> Self {
        Self {
            d_model: 4096,
            n_heads: 32,
            n_experts: 8,
            top_k: 2,
            latent_dim: 512,
            context_size: 32768,
            mlp_hidden: 16384,
            dropout: 0.1,
        }
    }
}

/// Sparse Mixture of Experts Layer
pub struct SparseMoELayer {
    pub _config: OracleBackboneConfig,
    pub experts: Vec<MLPExpert>,
    pub gate: LinearLayer,
    pub router: Router,
}

impl SparseMoELayer {
    pub fn new(config: OracleBackboneConfig) -> Self {
        let mut experts = Vec::new();
        for _ in 0..config.n_experts {
            experts.push(MLPExpert::new(config.d_model, config.mlp_hidden));
        }

        Self {
            _config: config.clone(),
            experts,
            gate: LinearLayer::new(config.d_model, config.n_experts),
            router: Router::new(config.n_experts, config.top_k),
        }
    }

    pub fn forward<S>(&self, x: &ArrayBase<S, ndarray::Ix2>) -> Result<Array2<f32>>
    where
        S: Data<Elem = f32>,
    {
        let (batch_size, d_model) = (x.dim().0, x.dim().1);

        let gate_scores = self.gate.forward(x)?;
        let expert_indices = self.router.route(&gate_scores)?;

        let top_k = self.router.top_k;
        let mut output = Array2::zeros((batch_size, d_model));
        let mut weight_sum = vec![0.0f32; batch_size];

        for token_idx in 0..batch_size {
            for k in 0..top_k {
                let flat_idx = token_idx * top_k + k;
                if flat_idx >= expert_indices.len() {
                    break;
                }
                let expert_idx = expert_indices[flat_idx];
                let expert_input = x.slice(s![token_idx, ..]);
                let expert_output = self.experts[expert_idx].forward(&expert_input)?;

                let mut output_row = output.slice_mut(s![token_idx, ..]);
                output_row
                    .iter_mut()
                    .zip(&expert_output)
                    .for_each(|(o, e)| *o += e);
                weight_sum[token_idx] += 1.0;
            }
        }

        for token_idx in 0..batch_size {
            if weight_sum[token_idx] > 0.0 {
                let mut output_row = output.slice_mut(s![token_idx, ..]);
                output_row.mapv_inplace(|v| v / weight_sum[token_idx]);
            }
        }

        Ok(output)
    }

    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;

        let (batch_size, d_model) = (x.shape()[0], x.shape()[1]);
        let top_k = self.router.top_k;
        let n_experts = self.experts.len();

        // 1. Gate scores on GPU
        let gate_gpu = self.gate.forward_gpu(x)?;
        let gate_arr = gate_gpu.to_cpu()?;
        let gate_vec: Vec<f32> = gate_arr.iter().copied().collect();
        let gate_arr = ndarray::Array2::from_shape_vec((batch_size, n_experts), gate_vec)
            .map_err(|e| anyhow::anyhow!("Gate shape error: {}", e))?;

        // 2. Route on CPU (cheap)
        let expert_indices = self.router.route(&gate_arr)?;

        // 3. Group tokens by expert
        let mut expert_tokens: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); n_experts];
        let x_arr = x.to_cpu()?;
        let x_vec: Vec<f32> = x_arr.iter().copied().collect();
        for token_idx in 0..batch_size {
            for k in 0..top_k {
                let flat_idx = token_idx * top_k + k;
                if flat_idx >= expert_indices.len() {
                    break;
                }
                let expert_idx = expert_indices[flat_idx];
                let start = token_idx * d_model;
                expert_tokens[expert_idx].push((token_idx, x_vec[start..start + d_model].to_vec()));
            }
        }

        // 4. Run batched expert forward on GPU, scatter back
        if expert_tokens.iter().all(|e| e.is_empty()) {
            return GpuTensor::zeros(&[batch_size, d_model])
                .map_err(|e| anyhow::anyhow!("GPU zeros: {}", e));
        }

        let mut output_data: Vec<f32> = vec![0.0; batch_size * d_model];
        let mut weight_sum = vec![0.0f32; batch_size];

        for (expert_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let n_tokens = tokens.len();
            let mut batch_input = Vec::with_capacity(n_tokens * d_model);
            for (_, token_data) in tokens {
                batch_input.extend_from_slice(token_data);
            }
            let input_arr = ndarray::ArrayView2::from_shape((n_tokens, d_model), &batch_input)
                .map_err(|e| anyhow::anyhow!("Input shape: {}", e))?;
            let gpu_input = GpuTensor::from_slice(vec![n_tokens, d_model], &batch_input)
                .map_err(|e| anyhow::anyhow!("GPU upload: {}", e))?;
            let gpu_output = self.experts[expert_idx].forward_gpu(&gpu_input)?;
            let out_arr = gpu_output
                .to_cpu()
                .map_err(|e| anyhow::anyhow!("GPU readback: {}", e))?;
            let out_vec: Vec<f32> = out_arr.iter().copied().collect();

            for (j, (token_idx, _)) in tokens.iter().enumerate() {
                let out_start = j * d_model;
                let dst_start = token_idx * d_model;
                for k in 0..d_model {
                    output_data[dst_start + k] += out_vec[out_start + k];
                }
                weight_sum[*token_idx] += 1.0;
            }
        }

        for token_idx in 0..batch_size {
            if weight_sum[token_idx] > 0.0 {
                let inv = 1.0 / weight_sum[token_idx];
                let start = token_idx * d_model;
                for v in output_data[start..start + d_model].iter_mut() {
                    *v *= inv;
                }
            }
        }

        let out_arr = ndarray::ArrayView2::from_shape((batch_size, d_model), &output_data)
            .map_err(|e| anyhow::anyhow!("Output shape: {}", e))?;
        GpuTensor::from_slice(vec![batch_size, d_model], &output_data)
            .map_err(|e| anyhow::anyhow!("GPU upload: {}", e))
    }

    pub fn get_expert_usage<S>(&self, x: &ArrayBase<S, ndarray::Ix2>) -> Result<Vec<f32>>
    where
        S: Data<Elem = f32>,
    {
        let (_batch_size, _d_model) = (x.dim().0, x.dim().1);

        let gate_scores = self.gate.forward(x)?;
        self.router.get_usage_stats(&gate_scores)
    }
}

/// Individual Expert (MLP)
pub struct MLPExpert {
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub b1: Array1<f32>,
    pub b2: Array1<f32>,
}

fn xavier_uniform(in_dim: usize, out_dim: usize) -> f32 {
    let limit = (6.0 / (in_dim + out_dim) as f32).sqrt();
    rand::random::<f32>() * 2.0 * limit - limit
}

fn xavier_uniform_1d(dim: usize) -> f32 {
    let limit = (6.0 / dim as f32).sqrt();
    rand::random::<f32>() * 2.0 * limit - limit
}

impl MLPExpert {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w1: Array2::from_shape_fn((input_dim, hidden_dim), |_| {
                xavier_uniform(input_dim, hidden_dim)
            }),
            w2: Array2::from_shape_fn((hidden_dim, input_dim), |_| {
                xavier_uniform(hidden_dim, input_dim)
            }),
            b1: Array1::from_shape_fn(hidden_dim, |_| xavier_uniform_1d(hidden_dim)),
            b2: Array1::from_shape_fn(input_dim, |_| xavier_uniform_1d(input_dim)),
        }
    }

    pub fn forward<S>(&self, x: &ArrayBase<S, ndarray::Ix1>) -> Result<Array1<f32>>
    where
        S: Data<Elem = f32>,
    {
        let hidden = x.dot(&self.w1) + &self.b1;
        let activated = self.gelu(&hidden);
        let output = activated.dot(&self.w2) + &self.b2;
        Ok(output)
    }

    fn gelu(&self, x: &Array1<f32>) -> Array1<f32> {
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        x.mapv(|x| {
            let x3 = x * x * x;
            0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x3)).tanh())
        })
    }

    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;

        let w1_shape = vec![self.w1.shape()[0], self.w1.shape()[1]];
        let w1_flat: Vec<f32> = self.w1.iter().copied().collect();
        let w1_gpu = GpuTensor::from_slice(w1_shape, &w1_flat)?;

        let w2_shape = vec![self.w2.shape()[0], self.w2.shape()[1]];
        let w2_flat: Vec<f32> = self.w2.iter().copied().collect();
        let w2_gpu = GpuTensor::from_slice(w2_shape, &w2_flat)?;

        let b1_flat: Vec<f32> = self.b1.iter().copied().collect();
        let b1_gpu = GpuTensor::from_slice(vec![1, self.b1.len()], &b1_flat)?;

        let b2_flat: Vec<f32> = self.b2.iter().copied().collect();
        let b2_gpu = GpuTensor::from_slice(vec![1, self.b2.len()], &b2_flat)?;

        let hidden = ctx.matmul(x, &w1_gpu)?;
        let hidden_bias = ctx.add(&hidden, &b1_gpu)?;
        let activated = ctx.gelu(&hidden_bias)?;
        let output = ctx.matmul(&activated, &w2_gpu)?;
        Ok(ctx.add(&output, &b2_gpu)?)
    }
}

/// Router untuk expert selection
pub struct Router {
    n_experts: usize,
    pub top_k: usize,
}

impl Router {
    pub fn new(n_experts: usize, top_k: usize) -> Self {
        Self { n_experts, top_k }
    }

    pub fn route(&self, gate_scores: &Array2<f32>) -> Result<Vec<usize>> {
        let (batch_size, _) = gate_scores.dim();
        let mut selected_experts = Vec::new();

        for i in 0..batch_size {
            let row = gate_scores.slice(s![i, ..]);
            let mut indices: Vec<usize> = (0..self.n_experts).collect();
            indices.sort_by(|&a, &b| {
                row[a]
                    .partial_cmp(&row[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            selected_experts.extend(indices.iter().take(self.top_k));
        }

        Ok(selected_experts)
    }

    pub fn get_usage_stats(&self, gate_scores: &Array2<f32>) -> Result<Vec<f32>> {
        let mut usage = vec![0.0; self.n_experts];
        let (batch_size, _) = gate_scores.dim();

        for i in 0..batch_size {
            let row = gate_scores.slice(s![i, ..]);
            let mut indices: Vec<usize> = (0..self.n_experts).collect();
            indices.sort_by(|&a, &b| {
                row[a]
                    .partial_cmp(&row[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for &idx in indices.iter().take(self.top_k) {
                usage[idx] += 1.0;
            }
        }

        // Normalize
        let total: f32 = usage.iter().sum();
        if total > 0.0 {
            for usage_val in usage.iter_mut() {
                *usage_val /= total;
            }
        }

        Ok(usage)
    }
}

/// Multi-head Latent Attention (MLA)
pub struct MultiHeadLatentAttention {
    pub config: OracleBackboneConfig,
    pub latent_projection: LinearLayer,
    pub attention_heads: Vec<LatentAttentionHead>,
    pub output_projection: LinearLayer,
    pub latent_compression: LatentCompression,
}

impl MultiHeadLatentAttention {
    pub fn new(config: OracleBackboneConfig) -> Self {
        let mut attention_heads = Vec::new();
        let head_dim = config.d_model / config.n_heads;

        for _ in 0..config.n_heads {
            attention_heads.push(LatentAttentionHead::new(head_dim, config.latent_dim));
        }

        Self {
            config: config.clone(),
            latent_projection: LinearLayer::new(config.d_model, config.latent_dim),
            attention_heads,
            output_projection: LinearLayer::new(config.latent_dim, config.d_model),
            latent_compression: LatentCompression::new(config.latent_dim),
        }
    }

    pub fn forward(&self, x: &Array3<f32>, mask: Option<&Array2<f32>>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, d_model) = (x.dim().0, x.dim().1, x.dim().2);

        // Project to latent space
        let x_reshaped = x.view().into_shape((batch_size * seq_len, d_model))?;
        let latent = self.latent_projection.forward(&x_reshaped)?;
        let latent_3d = latent.into_shape((batch_size, seq_len, self.config.latent_dim))?;

        // Compress latent representation
        let compressed = self.latent_compression.compress(&latent_3d, None)?;

        // Multi-head attention in latent space
        let mut head_outputs = Vec::new();
        for head in &self.attention_heads {
            let head_output = head.forward(&compressed, mask)?;
            head_outputs.push(head_output);
        }

        // Concatenate heads
        let concatenated = self.concatenate_heads(&head_outputs)?;

        // Project back to original dimension
        let output_reshaped = concatenated
            .view()
            .into_shape((batch_size * seq_len, self.config.latent_dim))?;
        let output = self.output_projection.forward(&output_reshaped)?;

        Ok(output.into_shape((batch_size, seq_len, d_model))?)
    }

    fn concatenate_heads(&self, heads: &[Array3<f32>]) -> Result<Array3<f32>> {
        if heads.is_empty() {
            return Err(anyhow::anyhow!("No heads to concatenate"));
        }

        let (batch_size, seq_len, head_dim) = heads[0].dim();
        let mut concatenated = Array3::zeros((batch_size, seq_len, head_dim * heads.len()));

        for (i, head) in heads.iter().enumerate() {
            let mut slice = concatenated.slice_mut(s![.., .., i * head_dim..(i + 1) * head_dim]);
            slice.assign(head);
        }

        Ok(concatenated)
    }
}

/// Individual Latent Attention Head
pub struct LatentAttentionHead {
    pub head_dim: usize,
    pub latent_dim: usize,
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub out_proj: LinearLayer,
}

impl LatentAttentionHead {
    pub fn new(head_dim: usize, latent_dim: usize) -> Self {
        Self {
            head_dim,
            latent_dim,
            q_proj: LinearLayer::new(latent_dim, head_dim),
            k_proj: LinearLayer::new(latent_dim, head_dim),
            v_proj: LinearLayer::new(latent_dim, head_dim),
            out_proj: LinearLayer::new(head_dim, latent_dim),
        }
    }

    pub fn forward(&self, x: &Array3<f32>, mask: Option<&Array2<f32>>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, _) = x.dim();

        // Project to Q, K, V
        let x_reshaped = x
            .view()
            .into_shape((batch_size * seq_len, self.latent_dim))?;

        let q = self.q_proj.forward(&x_reshaped)?;
        let k = self.k_proj.forward(&x_reshaped)?;
        let v = self.v_proj.forward(&x_reshaped)?;

        let q_3d = q.into_shape((batch_size, seq_len, self.head_dim))?;
        let k_3d = k.into_shape((batch_size, seq_len, self.head_dim))?;
        let v_3d = v.into_shape((batch_size, seq_len, self.head_dim))?;

        // Scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q_3d, &k_3d, &v_3d, mask)?;

        // Project output
        let output_reshaped = attention_output.into_shape((batch_size * seq_len, self.head_dim))?;
        let output = self.out_proj.forward(&output_reshaped)?;

        Ok(output.into_shape((batch_size, seq_len, self.latent_dim))?)
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, head_dim) = q.dim();
        let scale = (head_dim as f32).sqrt();

        // Per-batch attention to avoid cross-batch contamination
        let mut output = Array3::zeros((batch_size, seq_len, head_dim));

        for b in 0..batch_size {
            // Q[b] · K[b]^T  — shape: (seq_len, seq_len)
            let q_b = q.index_axis(ndarray::Axis(0), b);
            let k_b = k.index_axis(ndarray::Axis(0), b);
            let mut scores = q_b.dot(&k_b.t()) / scale;

            // Apply causal mask — vectorized
            if let Some(mask) = mask {
                ndarray::Zip::from(&mut scores)
                    .and(mask)
                    .for_each(|s, &m| *s += m);
            } else {
                // Default causal mask: prevent attending to future tokens
                for i in 0..seq_len {
                    for j in i + 1..seq_len {
                        scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
            }

            // Row-wise softmax in-place (vectorized)
            for i in 0..seq_len {
                let mut row = scores.slice_mut(ndarray::s![i, ..]);
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                row.mapv_inplace(|x| (x - max_val).exp());
                let sum_exp: f32 = row.iter().sum();
                if sum_exp > 0.0 {
                    row.mapv_inplace(|x| x / sum_exp);
                }
            }

            // Attention: softmax(scores) @ V[b] — cache-optimized matmul
            let v_b = v.index_axis(ndarray::Axis(0), b);
            let attn_out = scores.dot(&v_b);
            output.slice_mut(ndarray::s![b, .., ..]).assign(&attn_out);
        }

        Ok(output)
    }
}

/// Latent Compression untuk mengurangi context storage
pub struct LatentCompression {
    compression_ratio: f32,
    latent_dim: usize,
}

impl LatentCompression {
    pub fn new(latent_dim: usize) -> Self {
        Self {
            compression_ratio: 0.25,
            latent_dim,
        }
    }

    pub fn compress(&self, x: &Array3<f32>, mask: Option<&Array3<f32>>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, latent_dim) = x.dim();
        let compressed_dim = (latent_dim as f32 * self.compression_ratio) as usize;

        // Apply mask if provided
        let x = if let Some(mask) = mask {
            let mask_expanded = match mask.broadcast((batch_size, seq_len, latent_dim)) {
                Some(broadcasted) => broadcasted,
                None => return Err(anyhow::anyhow!("Broadcast shape mismatch")),
            };
            x * &mask_expanded
        } else {
            x.to_owned()
        };

        // Simple compression: take top-k dimensions
        let mut compressed = Array3::zeros((batch_size, seq_len, compressed_dim));

        for b in 0..batch_size {
            for i in 0..seq_len {
                let row = x.slice(s![b, i, ..]);
                let mut indices: Vec<usize> = (0..latent_dim).collect();
                indices.sort_by(|&a, &b| {
                    row[a]
                        .partial_cmp(&row[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for (j, &idx) in indices.iter().take(compressed_dim).enumerate() {
                    compressed[[b, i, j]] = row[idx];
                }
            }
        }

        Ok(compressed)
    }

    pub fn decompress(&self, x: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, compressed_dim) = x.dim();
        let mut decompressed = Array3::zeros((batch_size, seq_len, self.latent_dim));

        for b in 0..batch_size {
            for i in 0..seq_len {
                let compressed_row = x.slice(s![b, i, ..]);

                // Simple decompression: pad with zeros
                for j in 0..compressed_dim {
                    decompressed[[b, i, j]] = compressed_row[j];
                }
            }
        }

        Ok(decompressed)
    }
}

/// Simple Linear Layer
pub struct LinearLayer {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weight: Array2::from_shape_fn((input_dim, output_dim), |_| {
                xavier_uniform(input_dim, output_dim)
            }),
            bias: Array1::from_shape_fn(output_dim, |_| xavier_uniform_1d(output_dim)),
        }
    }

    pub fn forward<S>(&self, x: &ArrayBase<S, ndarray::Ix2>) -> Result<Array2<f32>>
    where
        S: Data<Elem = f32>,
    {
        Ok(x.dot(&self.weight) + &self.bias)
    }

    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;

        let w_shape = vec![self.weight.shape()[0], self.weight.shape()[1]];
        let w_flat: Vec<f32> = self.weight.iter().copied().collect();
        let w_gpu = GpuTensor::from_slice(w_shape, &w_flat)?;

        let b_flat: Vec<f32> = self.bias.iter().copied().collect();
        let b_gpu = GpuTensor::from_slice(vec![1, self.bias.len()], &b_flat)?;

        let result = ctx.matmul(x, &w_gpu)?;
        Ok(ctx.add(&result, &b_gpu)?)
    }
}

/// ORACLE Backbone Model
pub struct OracleBackbone {
    config: OracleBackboneConfig,
    embedding: EmbeddingLayer,
    moe_layers: Vec<SparseMoELayer>,
    attention_layers: Vec<MultiHeadLatentAttention>,
    norm_layers: Vec<LayerNorm>,
    output_projection: LinearLayer,
}

impl OracleBackbone {
    pub fn new(config: OracleBackboneConfig, vocab_size: usize) -> Self {
        let n_layers = 12; // Default number of transformer layers

        Self {
            config: config.clone(),
            embedding: EmbeddingLayer::new(vocab_size, config.d_model),
            moe_layers: (0..n_layers)
                .map(|_| SparseMoELayer::new(config.clone()))
                .collect(),
            attention_layers: (0..n_layers)
                .map(|_| MultiHeadLatentAttention::new(config.clone()))
                .collect(),
            norm_layers: (0..n_layers + 1)
                .map(|_| LayerNorm::new(config.d_model))
                .collect(),
            output_projection: LinearLayer::new(config.d_model, vocab_size),
        }
    }

    pub fn forward(
        &self,
        input_ids: &Array2<i32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len) = input_ids.dim();

        // Embedding
        let mut hidden = self.embedding.forward(input_ids)?;

        // Transformer layers
        for i in 0..self.moe_layers.len() {
            // Pre-norm
            hidden = self.norm_layers[i].forward(&hidden)?;

            // MoE layer - reshape from 3D to 2D
            let (b, s, d) = hidden.dim();
            let hidden_2d = hidden.view().into_shape((b * s, d))?;
            let moe_output = self.moe_layers[i].forward(&hidden_2d)?;
            let moe_output_3d = moe_output.into_shape((b, s, d))?;
            hidden = hidden + moe_output_3d;

            // Pre-norm
            hidden = self.norm_layers[i + 1].forward(&hidden)?;

            // Attention layer
            let attn_output = self.attention_layers[i].forward(&hidden, mask)?;
            hidden = hidden + attn_output;
        }

        // Output projection
        let hidden_reshaped = hidden
            .view()
            .into_shape((batch_size * seq_len, self.config.d_model))?;
        let logits = self.output_projection.forward(&hidden_reshaped)?;

        Ok(logits.into_shape((batch_size, seq_len, self.output_projection.weight.dim().1))?)
    }

    #[cfg(feature = "gpu")]
    pub fn forward_gpu(&self, input_ids: &Array2<i32>) -> Result<Array3<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;

        let (batch_size, seq_len) = input_ids.dim();

        // Embedding on CPU (lookup — not worth GPU upload)
        let hidden_cpu = self.embedding.forward(input_ids)?;
        let d_model = hidden_cpu.dim().2;

        // Upload embedding to GPU as [batch*seq, d_model]
        let hidden_flat: Vec<f32> = hidden_cpu.iter().copied().collect();
        let mut hidden = GpuTensor::from_slice(vec![batch_size * seq_len, d_model], &hidden_flat)?;

        // Transformer layers
        for i in 0..self.moe_layers.len() {
            // CPU round-trip for LayerNorm (elementwise, not worth GPU)
            let hidden_arr = hidden.to_cpu()?;
            let hidden_3d = hidden_arr
                .into_shape((batch_size, seq_len, d_model))
                .map_err(|e| anyhow::anyhow!("Reshape: {}", e))?;
            let normed = self.norm_layers[i].forward(&hidden_3d)?;
            let normed_flat: Vec<f32> = normed.iter().copied().collect();
            let gpu_in = GpuTensor::from_slice(vec![batch_size * seq_len, d_model], &normed_flat)?;

            // MoE layer on GPU
            let moe_out = self.moe_layers[i].forward_gpu(&gpu_in)?;
            hidden = ctx.add(&gpu_in, &moe_out)?;

            // CPU round-trip for norm + attention
            let hidden_arr2 = hidden.to_cpu()?;
            let hidden_3d2 = hidden_arr2
                .into_shape((batch_size, seq_len, d_model))
                .map_err(|e| anyhow::anyhow!("Reshape: {}", e))?;
            let normed2 = self.norm_layers[i + 1].forward(&hidden_3d2)?;
            let attn_out = self.attention_layers[i].forward(&normed2, None)?;

            // Upload attention + residual add on GPU
            let attn_flat: Vec<f32> = attn_out.iter().copied().collect();
            let normed2_flat: Vec<f32> = normed2.iter().copied().collect();
            let attn_gpu = GpuTensor::from_slice(vec![batch_size * seq_len, d_model], &attn_flat)?;
            let normed2_gpu =
                GpuTensor::from_slice(vec![batch_size * seq_len, d_model], &normed2_flat)?;
            hidden = ctx.add(&normed2_gpu, &attn_gpu)?;
        }

        // Output projection on GPU
        let logits_gpu = self.output_projection.forward_gpu(&hidden)?;
        let logits_arr = logits_gpu.to_cpu()?;
        let vocab_size = self.output_projection.weight.dim().1;
        let logits_data: Vec<f32> = logits_arr.iter().copied().collect();
        let result = Array3::from_shape_vec((batch_size, seq_len, vocab_size), logits_data)
            .map_err(|e| anyhow::anyhow!("Logits shape error: {}", e))?;
        Ok(result)
    }

    pub fn get_expert_usage_stats(&self, input_ids: &Array2<i32>) -> Result<Vec<Vec<f32>>> {
        let hidden = self.embedding.forward(input_ids)?;
        let (batch_size, seq_len, d_model) = hidden.dim();
        let hidden_2d = hidden.view().into_shape((batch_size * seq_len, d_model))?;
        let mut usage_stats = Vec::new();

        for moe_layer in &self.moe_layers {
            let usage = moe_layer.get_expert_usage(&hidden_2d)?;
            usage_stats.push(usage);
        }

        Ok(usage_stats)
    }
}

/// Embedding Layer
pub struct EmbeddingLayer {
    pub embeddings: Array2<f32>,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            embeddings: Array2::from_shape_fn((vocab_size, d_model), |_| {
                xavier_uniform(vocab_size, d_model)
            }),
        }
    }

    pub fn forward(&self, input_ids: &Array2<i32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len) = input_ids.dim();
        let mut output = Array3::zeros((batch_size, seq_len, self.embeddings.dim().1));

        for b in 0..batch_size {
            for i in 0..seq_len {
                let token_id = input_ids[[b, i]] as usize;
                if token_id < self.embeddings.dim().0 {
                    let embedding = self.embeddings.slice(s![token_id, ..]);
                    let mut output_slice = output.slice_mut(s![b, i, ..]);
                    output_slice.assign(&embedding);
                }
            }
        }

        Ok(output)
    }
}

/// Layer Normalization
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(d_model: usize) -> Self {
        Self {
            weight: Array1::ones(d_model),
            bias: Array1::zeros(d_model),
            eps: 1e-6,
        }
    }

    pub fn forward(&self, x: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, _d_model) = x.dim();
        let mut output = Array3::zeros(x.dim());

        for b in 0..batch_size {
            for i in 0..seq_len {
                let row = x.slice(s![b, i, ..]);
                let mean = row
                    .mean()
                    .ok_or_else(|| anyhow::anyhow!("Failed to calculate mean"))?;
                let variance = row.var(0.0);
                let std = (variance + self.eps).sqrt();

                let normalized = row.mapv(|x| (x - mean) / std);
                let scaled = normalized * &self.weight + &self.bias;

                let mut output_row = output.slice_mut(s![b, i, ..]);
                output_row.assign(&scaled);
            }
        }

        Ok(output)
    }
}

// ─── Tensor-based training methods ──────────────────────────────────────────────

impl OracleBackbone {
    /// Create differentiable parameter wrappers for ALL backbone weights.
    /// Returns (tensors, a closure to sync back).
    /// Number of tensors is deterministic:
    ///   1 embedding + 1 output_w + 1 output_b
    ///   + layers × (2 norms + 2 norm_b + gate_w + gate_b
    ///                + n_experts × (w1 + b1 + w2 + b2)
    ///                + q_proj + k_proj + v_proj + attn_out_proj)
    pub fn collect_training_tensors(&self) -> Vec<Tensor> {
        let mut tensors = Vec::new();

        let mk_tensor = |data: &[f32], shape: &[usize]| -> Tensor {
            let t = Tensor::from_slice(data, shape);
            t.set_requires_grad(true);
            t
        };

        // 0: embedding
        {
            let d: Vec<f32> = self.embedding.embeddings.iter().copied().collect();
            let s = self.embedding.embeddings.shape().to_vec();
            tensors.push(mk_tensor(&d, &s));
        }

        // Per-layer parameters
        for i in 0..self.moe_layers.len() {
            // norm1_weight, norm1_bias
            {
                let d: Vec<f32> = self.norm_layers[i].weight.iter().copied().collect();
                let s = self.norm_layers[i].weight.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }
            {
                let d: Vec<f32> = self.norm_layers[i].bias.iter().copied().collect();
                let s = self.norm_layers[i].bias.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }

            // MoE gate weight + bias
            {
                let d: Vec<f32> = self.moe_layers[i].gate.weight.iter().copied().collect();
                let s = self.moe_layers[i].gate.weight.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }
            {
                let d: Vec<f32> = self.moe_layers[i].gate.bias.iter().copied().collect();
                let s = self.moe_layers[i].gate.bias.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }

            // Experts
            for e in 0..self.config.n_experts {
                let exp = &self.moe_layers[i].experts[e];
                {
                    let d: Vec<f32> = exp.w1.iter().copied().collect();
                    let s = exp.w1.shape().to_vec();
                    tensors.push(mk_tensor(&d, &s));
                }
                {
                    let d: Vec<f32> = exp.b1.iter().copied().collect();
                    let s = exp.b1.shape().to_vec();
                    tensors.push(mk_tensor(&d, &s));
                }
                {
                    let d: Vec<f32> = exp.w2.iter().copied().collect();
                    let s = exp.w2.shape().to_vec();
                    tensors.push(mk_tensor(&d, &s));
                }
                {
                    let d: Vec<f32> = exp.b2.iter().copied().collect();
                    let s = exp.b2.shape().to_vec();
                    tensors.push(mk_tensor(&d, &s));
                }
            }

            // norm2_weight, norm2_bias
            {
                let d: Vec<f32> = self.norm_layers[i + 1].weight.iter().copied().collect();
                let s = self.norm_layers[i + 1].weight.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }
            {
                let d: Vec<f32> = self.norm_layers[i + 1].bias.iter().copied().collect();
                let s = self.norm_layers[i + 1].bias.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }

            // Attention projections from first head (all heads use same projection in MLA)
            let head0 = &self.attention_layers[i].attention_heads[0];
            // q_proj: [latent_dim, head_dim]
            {
                let d: Vec<f32> = head0.q_proj.weight.iter().copied().collect();
                let s = head0.q_proj.weight.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }
            // k_proj
            {
                let d: Vec<f32> = head0.k_proj.weight.iter().copied().collect();
                let s = head0.k_proj.weight.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }
            // v_proj
            {
                let d: Vec<f32> = head0.v_proj.weight.iter().copied().collect();
                let s = head0.v_proj.weight.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }
            // out_proj (head output to latent)
            {
                let d: Vec<f32> = head0.out_proj.weight.iter().copied().collect();
                let s = head0.out_proj.weight.shape().to_vec();
                tensors.push(mk_tensor(&d, &s));
            }
        }

        // Output projection weight + bias
        {
            let d: Vec<f32> = self.output_projection.weight.iter().copied().collect();
            let s = self.output_projection.weight.shape().to_vec();
            tensors.push(mk_tensor(&d, &s));
        }
        {
            let d: Vec<f32> = self.output_projection.bias.iter().copied().collect();
            let s = self.output_projection.bias.shape().to_vec();
            tensors.push(mk_tensor(&d, &s));
        }

        tensors
    }

    /// Sync tensor values back to the backbone's ndarray parameters.
    /// `tensors` must be in the same order as `collect_training_tensors`.
    pub fn sync_from_tensors(&mut self, tensors: &[Tensor]) {
        let mut idx = 0usize;

        // Embedding
        copy_tensor_to_arr2(&tensors[idx].data(), &mut self.embedding.embeddings);
        idx += 1;

        for i in 0..self.moe_layers.len() {
            // norm1
            copy_tensor_to_arr1(&tensors[idx].data(), &mut self.norm_layers[i].weight);
            idx += 1;
            copy_tensor_to_arr1(&tensors[idx].data(), &mut self.norm_layers[i].bias);
            idx += 1;

            // gate
            copy_tensor_to_arr2(&tensors[idx].data(), &mut self.moe_layers[i].gate.weight);
            idx += 1;
            copy_tensor_to_arr1(&tensors[idx].data(), &mut self.moe_layers[i].gate.bias);
            idx += 1;

            // experts
            for e in 0..self.config.n_experts {
                let exp = &mut self.moe_layers[i].experts[e];
                copy_tensor_to_arr2(&tensors[idx].data(), &mut exp.w1);
                idx += 1;
                copy_tensor_to_arr1(&tensors[idx].data(), &mut exp.b1);
                idx += 1;
                copy_tensor_to_arr2(&tensors[idx].data(), &mut exp.w2);
                idx += 1;
                copy_tensor_to_arr1(&tensors[idx].data(), &mut exp.b2);
                idx += 1;
            }

            // norm2
            copy_tensor_to_arr1(&tensors[idx].data(), &mut self.norm_layers[i + 1].weight);
            idx += 1;
            copy_tensor_to_arr1(&tensors[idx].data(), &mut self.norm_layers[i + 1].bias);
            idx += 1;

            // attention projections (first head)
            let head0 = &mut self.attention_layers[i].attention_heads[0];
            copy_tensor_to_arr2(&tensors[idx].data(), &mut head0.q_proj.weight);
            idx += 1;
            copy_tensor_to_arr2(&tensors[idx].data(), &mut head0.k_proj.weight);
            idx += 1;
            copy_tensor_to_arr2(&tensors[idx].data(), &mut head0.v_proj.weight);
            idx += 1;
            copy_tensor_to_arr2(&tensors[idx].data(), &mut head0.out_proj.weight);
            idx += 1;
        }

        // Output projection
        copy_tensor_to_arr2(&tensors[idx].data(), &mut self.output_projection.weight);
        idx += 1;
        copy_tensor_to_arr1(&tensors[idx].data(), &mut self.output_projection.bias);
    }

    /// Differentiable forward pass using Tensor ops.
    /// Returns logits [batch, seq_len, vocab_size].
    /// Uses soft MoE (all experts weighted by gate scores) for differentiability.
    pub fn forward_tensor(
        &self,
        input_ids: &Array2<i32>,
        params: &[Tensor],
        _mask: Option<&Tensor>,
        d_model: usize,
        vocab_size: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dim();
        let n = batch_size * seq_len;
        let param = TensorParamIndexer::new(self.config.n_experts);

        // Flatten input for embedding
        let flat: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
        let ids = Tensor::from_slice(&flat, &[n]);

        // --- Embedding ---
        let mut h = ops::nn::embedding(&ids, &params[param.emb]);
        // h: [n, d_model]

        // --- Precompute block-diagonal causal mask (reused across all layers) ---
        // Each batch element is isolated; within each, future positions are masked.
        // Created once per forward call to avoid O(n²) per-layer recomputation.
        let causal_mask = {
            let mut mask_data = vec![f32::NEG_INFINITY; n * n];
            for bi in 0..batch_size {
                let base = bi * seq_len;
                for i in 0..seq_len {
                    let row_start = (base + i) * n + base;
                    // Fill lower triangle (i >= j) with 0.0
                    for j in 0..=i {
                        mask_data[row_start + j] = 0.0;
                    }
                }
            }
            Tensor::from_slice(&mask_data, &[n, n])
        };

        // --- Transformer layers ---
        for li in 0..self.moe_layers.len() {
            let lo = param.layer_offset(li);

            // Pre-norm (norm1)
            let norm_w = &params[lo.norm1_w];
            let norm_b = &params[lo.norm1_b];
            h = ops::nn::layer_norm_2d(&h, Some(norm_w), Some(norm_b), 1e-6);

            // --- MoE with soft gating (differentiable) ---
            let gate_scores =
                ops::nn::softmax(&h.matmul(&params[lo.gate_w]).add(&params[lo.gate_b]), 1);
            // gate_scores: [n, n_experts]

            // Soft MoE: weighted sum of all experts
            let n_experts = self.config.n_experts;
            let mut moe_out = Tensor::zeros(&[n, d_model], false);

            // Compute each expert's output and aggregate
            for e in 0..n_experts {
                let ew1 = &params[lo.expert_w1(e)];
                let eb1 = &params[lo.expert_b1(e)];
                let ew2 = &params[lo.expert_w2(e)];
                let eb2 = &params[lo.expert_b2(e)];

                // expert_out = gelu(x @ W1 + b1) @ W2 + b2
                let expert_h = h.matmul(ew1).add(eb1).gelu();
                let expert_out = expert_h.matmul(ew2).add(eb2);

                // Create gate score mask for this expert: select column e of gate_scores
                // Build a one-hot selector: [n, n_experts], column e = 1
                let mut sel = vec![0.0f32; n * n_experts];
                for i in 0..n {
                    sel[i * n_experts + e] = 1.0;
                }
                let selector = Tensor::from_slice(&sel, &[n, n_experts]);
                let score = gate_scores.mul(&selector);
                let ones = Tensor::from_slice(&vec![1.0f32; n_experts], &[n_experts, 1]);
                let score_2d = score.matmul(&ones);

                let weighted = expert_out.mul(&score_2d);
                moe_out = moe_out.add(&weighted);
            }

            h = h.add(&moe_out);

            // Post-norm (norm2)
            let norm2_w = &params[lo.norm2_w];
            let norm2_b = &params[lo.norm2_b];
            h = ops::nn::layer_norm_2d(&h, Some(norm2_w), Some(norm2_b), 1e-6);

            // --- Multi-head latent attention (flattened 2D matmul with block+causal mask) ---
            let head_dim = d_model / self.config.n_heads;
            let q = h.matmul(&params[lo.q_proj]);
            let k = h.matmul(&params[lo.k_proj]);
            let v = h.matmul(&params[lo.v_proj]);

            let k_t = k.transpose();
            let scale = (head_dim as f32).sqrt();
            let scale_t = Tensor::from_slice(&[1.0 / scale], &[1]);
            let mut scores = q.matmul(&k_t).mul(&scale_t);

            scores = scores.add(&causal_mask);

            let attn_weights = ops::nn::softmax(&scores, 1);
            let attn_out = attn_weights.matmul(&v);
            let attn_proj = attn_out.matmul(&params[lo.attn_out_proj]);

            h = h.add(&attn_proj);
        }

        // --- Output projection ---
        let out_w_idx = param.output_w(self.moe_layers.len());
        let out_b_idx = param.output_b(self.moe_layers.len());
        let logits = h.matmul(&params[out_w_idx]).add(&params[out_b_idx]);
        let logits_3d = logits.reshape(&[batch_size, seq_len, vocab_size]);

        Ok(logits_3d)
    }
}

/// Helper: copy tensor data into a 2D ndarray
fn copy_tensor_to_arr2(data: &ndarray::ArrayD<f32>, target: &mut Array2<f32>) {
    let flat: Vec<f32> = data.iter().copied().collect();
    let (rows, cols) = target.dim();
    if flat.len() == rows * cols {
        for i in 0..rows {
            for j in 0..cols {
                target[[i, j]] = flat[i * cols + j];
            }
        }
    }
}

/// Helper: copy tensor data into a 1D ndarray
fn copy_tensor_to_arr1(data: &ndarray::ArrayD<f32>, target: &mut Array1<f32>) {
    let flat: Vec<f32> = data.iter().copied().collect();
    if flat.len() == target.len() {
        for i in 0..target.len() {
            target[i] = flat[i];
        }
    }
}

/// Parameter index calculator for the flat tensor layout used in forward_tensor.
struct TensorParamIndexer {
    n_experts: usize,
    pub emb: usize,
    pub params_per_layer: usize,
}

impl TensorParamIndexer {
    fn new(n_experts: usize) -> Self {
        Self {
            emb: 0,
            params_per_layer: 2  // norms (w+b)
                + 2              // gate (w+b)
                + n_experts * 4  // experts (w1+b1+w2+b2)
                + 2              // norm2 (w+b)
                + 4, // q,k,v,out projections
            n_experts,
        }
    }

    fn layer_offset(&self, layer: usize) -> LayerTensorOffset {
        let base = self.emb + 1 + layer * self.params_per_layer;
        LayerTensorOffset {
            norm1_w: base,
            norm1_b: base + 1,
            gate_w: base + 2,
            gate_b: base + 3,
            expert_start: base + 4,
            norm2_w: base + 4 + self.n_experts * 4,
            norm2_b: base + 5 + self.n_experts * 4,
            q_proj: base + 6 + self.n_experts * 4,
            k_proj: base + 7 + self.n_experts * 4,
            v_proj: base + 8 + self.n_experts * 4,
            attn_out_proj: base + 9 + self.n_experts * 4,
        }
    }

    /// Index of output weight tensor (embedding + all layers first)
    fn output_w(&self, n_layers: usize) -> usize {
        self.emb + 1 + n_layers * self.params_per_layer
    }

    fn output_b(&self, n_layers: usize) -> usize {
        self.output_w(n_layers) + 1
    }
}

struct LayerTensorOffset {
    norm1_w: usize,
    norm1_b: usize,
    gate_w: usize,
    gate_b: usize,
    expert_start: usize,
    norm2_w: usize,
    norm2_b: usize,
    q_proj: usize,
    k_proj: usize,
    v_proj: usize,
    attn_out_proj: usize,
}

impl LayerTensorOffset {
    fn expert_w1(&self, e: usize) -> usize {
        self.expert_start + e * 4
    }
    fn expert_b1(&self, e: usize) -> usize {
        self.expert_start + e * 4 + 1
    }
    fn expert_w2(&self, e: usize) -> usize {
        self.expert_start + e * 4 + 2
    }
    fn expert_b2(&self, e: usize) -> usize {
        self.expert_start + e * 4 + 3
    }
}

/// Compute a differentiable cross-entropy loss from logits and labels.
/// Labels with value -100 are ignored (they contribute zero gradient).
pub fn compute_tensor_loss(logits: &Tensor, labels: &[i32], vocab_size: usize) -> (Tensor, f32) {
    let logits_data = logits.data();
    let shape = logits_data.shape();
    let n = shape[0]; // batch * seq_len flattened

    // Build one-hot targets: only positions with label >= 0 contribute
    let mut one_hot = vec![0.0f32; n * vocab_size];
    let mut valid = 0;

    for i in 0..n {
        if i < labels.len() && labels[i] >= 0 {
            let li = labels[i] as usize;
            if li < vocab_size {
                one_hot[i * vocab_size + li] = 1.0;
                valid += 1;
            }
        }
    }

    // Create differentiable loss:
    // loss = -sum(log_softmax(logits) * one_hot) / valid_count
    let log_probs = ops::nn::log_softmax(logits, 1);
    // neg_log_probs = -log_softmax
    let neg_log_probs = ops::math::neg(&log_probs);
    let weighted = neg_log_probs.mul(&Tensor::from_slice(&one_hot, &[n, vocab_size]));

    // weighted.sum() gives scalar, divide by valid count
    let total = weighted.sum();
    let scale = if valid > 0 { 1.0 / valid as f32 } else { 1.0 };
    let scale_t = Tensor::from_slice(&[scale], &[1]);

    let loss_t = total.mul(&scale_t);
    let loss_val = loss_t.data()[0];

    (loss_t, loss_val)
}

/// Utility functions
pub mod utils {
    use super::*;

    /// Calculate model parameters count
    pub fn count_parameters(model: &OracleBackbone) -> usize {
        // This is a simplified calculation
        // In practice, you'd sum all parameters from all layers
        model.config.d_model * model.config.d_model * 100 // Rough estimate
    }

    /// Calculate FLOPs for forward pass
    pub fn calculate_flops(
        config: &OracleBackboneConfig,
        batch_size: usize,
        seq_len: usize,
    ) -> u64 {
        let n_layers: u64 = 12;
        let batch_size: u64 = batch_size as u64;
        let moe_flops = (config.d_model * config.mlp_hidden * 2) as u64; // Forward + backward
        let attention_flops = (seq_len * seq_len * config.d_model * 4) as u64; // Q, K, V, O

        (moe_flops + attention_flops) * n_layers * batch_size
    }

    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(
        config: &OracleBackboneConfig,
        batch_size: usize,
        seq_len: usize,
    ) -> usize {
        let activation_memory = batch_size * seq_len * config.d_model * 4; // f32
        let kv_cache_memory =
            batch_size * config.n_heads * seq_len * (config.d_model / config.n_heads) * 4;
        let mla_compressed = batch_size * seq_len * config.latent_dim * 4;

        activation_memory + kv_cache_memory + mla_compressed
    }
}
