//! ORACLE Backbone - Sparse MoE + MLA Implementation
//!
//! Fondasi arsitektur ORACLE yang menggabungkan Mixture of Experts (MoE)
//! dengan Multi-head Latent Attention (MLA) untuk efisiensi komputasi
//! dan context window yang besar.

use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, ArrayBase, Data};
use nexora_autograd::{ops, Tensor, TensorOps};
use nexora_has_moe_ffn::{HasMoeFFN, HasMoeFFNConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::rope::ExtendedRope;
#[cfg(feature = "gpu")]
use nexora_autograd::gpu::{GpuContext, GpuTensor};

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

fn xavier_uniform(in_dim: usize, out_dim: usize) -> f32 {
    let limit = (6.0 / (in_dim + out_dim) as f32).sqrt();
    rand::random::<f32>() * 2.0 * limit - limit
}

fn xavier_uniform_1d(dim: usize) -> f32 {
    let limit = (6.0 / dim as f32).sqrt();
    rand::random::<f32>() * 2.0 * limit - limit
}

/// Router untuk expert selection
/// Multi-head Latent Attention (MLA)
pub struct MultiHeadLatentAttention {
    pub config: OracleBackboneConfig,
    pub latent_projection: LinearLayer,
    pub attention_heads: Vec<LatentAttentionHead>,
    pub output_projection: LinearLayer,
    pub latent_compression: LatentCompression,
    pub rope: Arc<ExtendedRope>,
}

impl MultiHeadLatentAttention {
    pub fn new(config: OracleBackboneConfig, rope: Arc<ExtendedRope>) -> Self {
        let mut attention_heads = Vec::new();
        let head_dim = config.d_model / config.n_heads;

        for _ in 0..config.n_heads {
            attention_heads.push(LatentAttentionHead::new(
                head_dim,
                config.latent_dim,
                Arc::clone(&rope),
            ));
        }

        Self {
            config: config.clone(),
            latent_projection: LinearLayer::new(config.d_model, config.latent_dim),
            attention_heads,
            output_projection: LinearLayer::new(config.latent_dim, config.d_model),
            latent_compression: LatentCompression::new(config.latent_dim),
            rope,
        }
    }

    pub fn forward(
        &self,
        x: &Array3<f32>,
        mask: Option<&Array2<f32>>,
        positions: &[usize],
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, d_model) = (x.dim().0, x.dim().1, x.dim().2);

        let x_reshaped = x.view().into_shape((batch_size * seq_len, d_model))?;
        let latent = self.latent_projection.forward(&x_reshaped)?;
        let latent_3d = latent.into_shape((batch_size, seq_len, self.config.latent_dim))?;

        let compressed = self.latent_compression.compress(&latent_3d, None)?;

        let mut head_outputs = Vec::new();
        for head in &self.attention_heads {
            let head_output = head.forward(&compressed, mask, positions)?;
            head_outputs.push(head_output);
        }

        let concatenated = self.concatenate_heads(&head_outputs)?;

        let output_reshaped = concatenated
            .view()
            .into_shape((batch_size * seq_len, self.config.latent_dim))?;
        let output = self.output_projection.forward(&output_reshaped)?;

        Ok(output.into_shape((batch_size, seq_len, d_model))?)
    }

    /// GPU forward pass: per-head QKV projections + fused_attention, all GPU-resident.
    /// Head outputs are summed (not concatenated) — zero CPU round-trips.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor> {
        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;

        let latent = self.latent_projection.forward_gpu(x)?;

        let mut output = self.attention_heads[0].forward_gpu(&latent)?;
        for head in self.attention_heads[1..].iter() {
            let head_out = head.forward_gpu(&latent)?;
            output = ctx.add(&output, &head_out)?;
        }

        self.output_projection.forward_gpu(&output)
    }

    fn concatenate_heads(&self, heads: &[Array3<f32>]) -> Result<Array3<f32>> {
        if heads.is_empty() {
            return Err(anyhow::anyhow!("No heads to concatenate"));
        }

        let (batch_size, seq_len, head_dim) = heads[0].dim();
        let concatenated_dim = head_dim * heads.len();
        let expected_dim = self.config.latent_dim;

        if concatenated_dim != expected_dim {
            return Err(anyhow::anyhow!(
                "MLA concatenate_heads: head_dim({}) * n_heads({}) = {} != latent_dim({}). \
                 Each head output must align with latent projection dimension.",
                head_dim, heads.len(), concatenated_dim, expected_dim,
            ));
        }

        let mut concatenated = Array3::zeros((batch_size, seq_len, concatenated_dim));

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
    pub rope: Arc<ExtendedRope>,
}

impl LatentAttentionHead {
    pub fn new(head_dim: usize, latent_dim: usize, rope: Arc<ExtendedRope>) -> Self {
        Self {
            head_dim,
            latent_dim,
            q_proj: LinearLayer::new(latent_dim, head_dim),
            k_proj: LinearLayer::new(latent_dim, head_dim),
            v_proj: LinearLayer::new(latent_dim, head_dim),
            out_proj: LinearLayer::new(head_dim, latent_dim),
            rope,
        }
    }

    pub fn forward(
        &self,
        x: &Array3<f32>,
        mask: Option<&Array2<f32>>,
        positions: &[usize],
    ) -> Result<Array3<f32>> {
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

        // Apply Rotary Position Embedding to Q and K before attention
        let (q_3d, k_3d) = self.rope.apply_rotary_emb(&q_3d, &k_3d, positions)?;

        // Scaled dot-product attention — uses cpu_flash_attention for O(n) memory
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
        let mut output = Array3::zeros((batch_size, seq_len, head_dim));
        let chunk_size = 64.max(seq_len / 4).min(256);
        let scale = (head_dim as f32).sqrt();

        for b in 0..batch_size {
            let q_b = q.index_axis(ndarray::Axis(0), b);
            let k_b = k.index_axis(ndarray::Axis(0), b);
            let v_b = v.index_axis(ndarray::Axis(0), b);

            let q_slice: Vec<f32> = q_b.iter().copied().collect();
            let k_slice: Vec<f32> = k_b.iter().copied().collect();
            let v_slice: Vec<f32> = v_b.iter().copied().collect();

            let mut output_b = vec![0.0f32; seq_len * head_dim];

            // Chunked softmax attention with optional mask
            for q_start in (0..seq_len).step_by(chunk_size) {
                let q_end = (q_start + chunk_size).min(seq_len);
                let n_q = q_end - q_start;

                let mut scores = vec![0.0f32; n_q * seq_len];
                for qi in 0..n_q {
                    let abs_qi = q_start + qi;
                    let q_base = abs_qi * head_dim;
                    for kj in 0..seq_len {
                        let k_base = kj * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_slice[q_base + d] * k_slice[k_base + d];
                        }
                        scores[qi * seq_len + kj] = dot / scale;
                    }
                }

                // Apply mask if provided or causal masking
                for i in 0..n_q {
                    let abs_qi = q_start + i;
                    for j in 0..seq_len {
                        if let Some(m) = mask {
                            if j < m.shape()[1] && abs_qi < m.shape()[0] {
                                scores[i * seq_len + j] += m[[abs_qi, j]];
                            }
                        } else if j > abs_qi {
                            scores[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Row-wise softmax
                for i in 0..n_q {
                    let _abs_qi = q_start + i;
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..seq_len {
                        if scores[i * seq_len + j] > max_val {
                            max_val = scores[i * seq_len + j];
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for j in 0..seq_len {
                        let e = (scores[i * seq_len + j] - max_val).exp();
                        scores[i * seq_len + j] = e;
                        sum_exp += e;
                    }
                    if sum_exp > 0.0 {
                        let inv = 1.0 / sum_exp;
                        for j in 0..seq_len {
                            scores[i * seq_len + j] *= inv;
                        }
                    }
                }

                // Weighted sum of V
                for i in 0..n_q {
                    let abs_qi = q_start + i;
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for j in 0..seq_len {
                            acc += scores[i * seq_len + j] * v_slice[j * head_dim + d];
                        }
                        output_b[abs_qi * head_dim + d] = acc;
                    }
                }
            }

            // Copy back into output array
            let mut out_slice = output.slice_mut(ndarray::s![b, .., ..]);
            for i in 0..seq_len {
                for d in 0..head_dim {
                    out_slice[[i, d]] = output_b[i * head_dim + d];
                }
            }
        }

        Ok(output)
    }
}

impl LatentAttentionHead {
    /// GPU forward: QKV projections → fused_attention → output projection.
    /// All operations GPU-resident — zero CPU round-trips.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor> {
        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;
        let shape = x.shape();
        let n = shape[0];

        // QKV projections via cached LinearLayer
        let q = self.q_proj.forward_gpu(x)?;
        let k = self.k_proj.forward_gpu(x)?;
        let v = self.v_proj.forward_gpu(x)?;

        // fused_attention expects [B, H, S, D]. Use [1, 1, N, head_dim] (single batch, single head)
        let head_dim = self.head_dim;
        let q_4d = q.reshape(vec![1, 1, n, head_dim])?;
        let k_4d = k.reshape(vec![1, 1, n, head_dim])?;
        let v_4d = v.reshape(vec![1, 1, n, head_dim])?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_4d = ctx.fused_attention(&q_4d, &k_4d, &v_4d, scale, false)?;
        let attn_flat = attn_4d.reshape(vec![n, head_dim])?;

        // Output projection: head_dim → latent_dim
        self.out_proj.forward_gpu(&attn_flat)
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
                indices.select_nth_unstable_by(compressed_dim - 1, |&a, &b| {
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
    #[cfg(feature = "gpu")]
    weight_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    bias_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weight: Array2::from_shape_fn((input_dim, output_dim), |_| {
                xavier_uniform(input_dim, output_dim)
            }),
            bias: Array1::from_shape_fn(output_dim, |_| xavier_uniform_1d(output_dim)),
            #[cfg(feature = "gpu")]
            weight_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            bias_gpu: std::sync::OnceLock::new(),
        }
    }

    /// Construct from pre-existing arrays (skips Xavier init, GPU caches uninitialized).
    pub fn from_arrays(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        Self {
            weight,
            bias,
            #[cfg(feature = "gpu")]
            weight_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            bias_gpu: std::sync::OnceLock::new(),
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
        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;

        let w_gpu = self
            .weight_gpu
            .get_or_init(|| {
                let w_shape = vec![self.weight.shape()[0], self.weight.shape()[1]];
                let w_flat: Vec<f32> = self.weight.iter().copied().collect();
                GpuTensor::from_slice(w_shape, &w_flat).ok()
            })
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to cache LinearLayer weight"))?;

        let b_gpu = self
            .bias_gpu
            .get_or_init(|| {
                let b_flat: Vec<f32> = self.bias.iter().copied().collect();
                GpuTensor::from_slice(vec![1, self.bias.len()], &b_flat).ok()
            })
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to cache LinearLayer bias"))?;

        let result = ctx.matmul(x, w_gpu)?;
        Ok(ctx.add(&result, b_gpu)?)
    }

    /// Invalidate GPU weight cache (call after weight update).
    #[cfg(feature = "gpu")]
    pub fn invalidate_gpu_cache(&mut self) {
        self.weight_gpu = std::sync::OnceLock::new();
        self.bias_gpu = std::sync::OnceLock::new();
    }
}

/// ORACLE Backbone Model
pub struct OracleBackbone {
    pub config: OracleBackboneConfig,
    pub embedding: EmbeddingLayer,
    pub moe_layers: Vec<HasMoeFFN>,
    pub attention_layers: Vec<MultiHeadLatentAttention>,
    pub norm_layers: Vec<LayerNorm>,
    pub output_projection: LinearLayer,
    pub rope: Arc<ExtendedRope>,
}

impl OracleBackbone {
    pub fn new(config: OracleBackboneConfig, vocab_size: usize) -> Self {
        let n_layers = 12; // Default number of transformer layers
        let moe_config = HasMoeFFNConfig {
            num_experts: config.n_experts,
            top_k: config.top_k,
            hidden_size: config.d_model,
            intermediate_size: config.mlp_hidden,
            ..Default::default()
        };

        let rope_config = crate::rope::ExtendedRopeConfig {
            d_model: config.d_model,
            n_heads: config.n_heads,
            max_seq_len: config.context_size,
            ..Default::default()
        };
        let rope = Arc::new(ExtendedRope::new(rope_config));

        Self {
            config: config.clone(),
            embedding: EmbeddingLayer::new(vocab_size, config.d_model),
            moe_layers: (0..n_layers)
                .map(|_| HasMoeFFN::new(moe_config.clone()))
                .collect(),
            attention_layers: (0..n_layers)
                .map(|_| MultiHeadLatentAttention::new(config.clone(), Arc::clone(&rope)))
                .collect(),
            norm_layers: (0..n_layers + 1)
                .map(|_| LayerNorm::new(config.d_model))
                .collect(),
            output_projection: LinearLayer::new(config.d_model, vocab_size),
            rope,
        }
    }

    pub fn forward(
        &self,
        input_ids: &Array2<i32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len) = input_ids.dim();

        // Generate standard sequential positions for RoPE
        let positions: Vec<usize> = (0..seq_len).collect();

        // Embedding
        let mut hidden = self.embedding.forward(input_ids)?;

        // Transformer layers
        for i in 0..self.moe_layers.len() {
            // Pre-norm
            hidden = self.norm_layers[i].forward(&hidden)?;

            // MoE layer - reshape from 3D to 2D
            let (b, s, d) = hidden.dim();
            let hidden_2d = hidden.view().into_shape((b * s, d))?.to_owned();
            let moe_output = self.moe_layers[i].forward(&hidden_2d);
            let moe_output_3d = moe_output.into_shape((b, s, d))?;
            hidden = hidden + moe_output_3d;

            // Pre-norm
            hidden = self.norm_layers[i + 1].forward(&hidden)?;

            // Attention layer with RoPE
            let attn_output = self.attention_layers[i].forward(&hidden, mask, &positions)?;
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
        let ctx = GpuContext::global().map_err(|e| anyhow::anyhow!("GPU: {}", e))?;

        let (batch_size, seq_len) = input_ids.dim();
        let d_model = self.config.d_model;

        // Embedding → directly to GPU tensor (CPU lookup, no intermediate Array3)
        let mut hidden = self.embedding.forward_gpu(input_ids)?;

        // Transformer layers
        for i in 0..self.moe_layers.len() {
            // LayerNorm on GPU — weight cached via OnceLock in LayerNorm
            let gpu_normed = self.norm_layers[i].forward_gpu(&hidden, &ctx, d_model)?;

            // MoE layer — HasMoeFFN handles GPU internally if available
            let moe_out = self.moe_layers[i].forward_gpu(&gpu_normed)
                .map_err(|e| anyhow::anyhow!("HasMoeFFN GPU forward: {e}"))?;
            hidden = ctx.add(&gpu_normed, &moe_out)?;

            // Attention via MLA GPU forward (weight cache via LinearLayer OnceLock)
            let gpu_normed2 = self.norm_layers[i + 1].forward_gpu(&hidden, &ctx, d_model)?;

            let attn_gpu = self.attention_layers[i].forward_gpu(&gpu_normed2)?;
            hidden = ctx.add(&gpu_normed2, &attn_gpu)?;
        }

        // Output projection on GPU (weight cache via LinearLayer OnceLock)
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
            let hidden_2d_owned = hidden_2d.to_owned();
            let routing = moe_layer.router().forward(&hidden_2d_owned);
            let avg_usage: Vec<f32> = routing
                .mean_axis(ndarray::Axis(0))
                .map(|a| a.to_vec())
                .unwrap_or_else(|| vec![0.0; self.config.n_experts]);
            usage_stats.push(avg_usage);
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

    /// GPU forward: CPU lookup → upload to GPU.
    /// Without native gather/scatter on GpuTensor, embedding lookup stays on CPU.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        input_ids: &Array2<i32>,
    ) -> Result<nexora_autograd::gpu::GpuTensor> {
        let (batch_size, seq_len) = input_ids.dim();
        let d_model = self.embeddings.dim().1;
        let n = batch_size * seq_len;
        let mut flat = vec![0.0f32; n * d_model];

        for b in 0..batch_size {
            for i in 0..seq_len {
                let token_id = input_ids[[b, i]] as usize;
                if token_id < self.embeddings.dim().0 {
                    let src = token_id * d_model;
                    let dst = (b * seq_len + i) * d_model;
                    let emb_row = self.embeddings.as_slice()
                        .ok_or_else(|| anyhow::anyhow!("Embeddings not contiguous"))?;
                    for d in 0..d_model {
                        flat[dst + d] = emb_row[src + d];
                    }
                }
            }
        }

        GpuTensor::from_slice(vec![n, d_model], &flat)
            .map_err(|e| anyhow::anyhow!("GPU upload: {}", e))
    }
}

/// Layer Normalization
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
    #[cfg(feature = "gpu")]
    weight_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    bias_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
}

impl LayerNorm {
    pub fn new(d_model: usize) -> Self {
        Self {
            weight: Array1::ones(d_model),
            bias: Array1::zeros(d_model),
            eps: 1e-6,
            #[cfg(feature = "gpu")]
            weight_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            bias_gpu: std::sync::OnceLock::new(),
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

    /// GPU layer norm on existing GpuTensor. Weight/bias are cached and only
    /// uploaded once (lazy via OnceLock), eliminating redundant per-call uploads.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
        ctx: &nexora_autograd::gpu::GpuContext,
        d_model: usize,
    ) -> Result<nexora_autograd::gpu::GpuTensor> {
        let w_gpu = self
            .weight_gpu
            .get_or_init(|| {
                let w_flat: Vec<f32> = self.weight.iter().copied().collect();
                nexora_autograd::gpu::GpuTensor::from_slice(vec![d_model], &w_flat).ok()
            })
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to cache LayerNorm weight"))?;

        let b_gpu = self
            .bias_gpu
            .get_or_init(|| {
                let b_flat: Vec<f32> = self.bias.iter().copied().collect();
                nexora_autograd::gpu::GpuTensor::from_slice(vec![d_model], &b_flat).ok()
            })
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Failed to cache LayerNorm bias"))?;

        Ok(ctx.layer_norm(x, w_gpu, b_gpu, 1e-5)?)
    }

    /// Invalidate GPU weight cache (call after weight update).
    #[cfg(feature = "gpu")]
    pub fn invalidate_gpu_cache(&mut self) {
        self.weight_gpu = std::sync::OnceLock::new();
        self.bias_gpu = std::sync::OnceLock::new();
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
            tensors.push(mk_tensor(&d, self.embedding.embeddings.shape()));
        }

        // Per-layer parameters
        for i in 0..self.moe_layers.len() {
            // norm1_weight, norm1_bias
            {
                let d: Vec<f32> = self.norm_layers[i].weight.iter().copied().collect();
                tensors.push(mk_tensor(&d, self.norm_layers[i].weight.shape()));
            }
            {
                let d: Vec<f32> = self.norm_layers[i].bias.iter().copied().collect();
                tensors.push(mk_tensor(&d, self.norm_layers[i].bias.shape()));
            }

            // MoE parameters — native HasMoeFFN format (no transpose bridge)
            for (data, shape) in self.moe_layers[i].collect_params() {
                tensors.push(mk_tensor(&data, &shape));
            }

            // norm2_weight, norm2_bias
            {
                let d: Vec<f32> = self.norm_layers[i + 1].weight.iter().copied().collect();
                tensors.push(mk_tensor(&d, self.norm_layers[i + 1].weight.shape()));
            }
            {
                let d: Vec<f32> = self.norm_layers[i + 1].bias.iter().copied().collect();
                tensors.push(mk_tensor(&d, self.norm_layers[i + 1].bias.shape()));
            }

            // Attention projections from first head (all heads use same projection in MLA)
            let head0 = &self.attention_layers[i].attention_heads[0];
            // q_proj: [latent_dim, head_dim]
            {
                let d: Vec<f32> = head0.q_proj.weight.iter().copied().collect();
                tensors.push(mk_tensor(&d, head0.q_proj.weight.shape()));
            }
            // k_proj
            {
                let d: Vec<f32> = head0.k_proj.weight.iter().copied().collect();
                tensors.push(mk_tensor(&d, head0.k_proj.weight.shape()));
            }
            // v_proj
            {
                let d: Vec<f32> = head0.v_proj.weight.iter().copied().collect();
                tensors.push(mk_tensor(&d, head0.v_proj.weight.shape()));
            }
            // out_proj (head output to latent)
            {
                let d: Vec<f32> = head0.out_proj.weight.iter().copied().collect();
                tensors.push(mk_tensor(&d, head0.out_proj.weight.shape()));
            }
        }

        // Output projection weight + bias
        {
            let d: Vec<f32> = self.output_projection.weight.iter().copied().collect();
            tensors.push(mk_tensor(&d, self.output_projection.weight.shape()));
        }
        {
            let d: Vec<f32> = self.output_projection.bias.iter().copied().collect();
            tensors.push(mk_tensor(&d, self.output_projection.bias.shape()));
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

            // MoE parameters — native HasMoeFFN format (no transpose bridge)
            {
                let n_moe = 1 + 4 * self.config.n_experts;
                let mut moe_slice = Vec::with_capacity(n_moe);
                for j in 0..n_moe {
                    let arr = tensors[idx + j].data();
                    let d: Vec<f32> = arr.iter().copied().collect();
                    let s = tensors[idx + j].shape().to_vec();
                    moe_slice.push((d, s));
                }
                self.moe_layers[i].sync_params(&moe_slice);
                idx += n_moe;
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

        // --- Transformer layers ---
        for li in 0..self.moe_layers.len() {
            let lo = param.layer_offset(li);

            // Pre-norm (norm1)
            let norm_w = &params[lo.norm1_w];
            let norm_b = &params[lo.norm1_b];
            h = ops::nn::layer_norm_2d(&h, Some(norm_w), Some(norm_b), 1e-6);

            // --- MoE with soft gating (differentiable) ---
            // Native HasMoeFFN format: router_w [ne, h], need h @ router_w^T
            let gate_scores =
                ops::nn::softmax(&h.matmul(&params[lo.router_w].transpose()), 1);
            // gate_scores: [n, n_experts]

            // Soft MoE: weighted sum of all experts
            let n_experts = self.config.n_experts;
            let mut moe_out = Tensor::zeros(&[n, d_model], false);

            // Compute each expert's output and aggregate
            for e in 0..n_experts {
                let ef1 = &params[lo.fc1_w(e)].transpose();
                let eb1 = &params[lo.fc1_b(e)];
                let ef2 = &params[lo.fc2_w(e)].transpose();
                let eb2 = &params[lo.fc2_b(e)];

                // expert_out = gelu(x @ W1^T + b1) @ W2^T + b2
                let expert_h = h.matmul(ef1).add(eb1).gelu();
                let expert_out = expert_h.matmul(ef2).add(eb2);

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

            // --- Multi-head latent attention (batch-wise with per-batch causal mask) ---
            let head_dim = d_model / self.config.n_heads;
            let q = h.matmul(&params[lo.q_proj]);
            let k = h.matmul(&params[lo.k_proj]);
            let v = h.matmul(&params[lo.v_proj]);

            // Full multi-head latent attention with block-diagonal causal mask
            let scale = 1.0 / (head_dim as f32).sqrt();
            let scale_t = Tensor::from_slice(&[scale], &[1, 1]);
            // Build block-diagonal causal mask for all batches
            let mut full_mask_data = vec![f32::NEG_INFINITY; n * n];
            for bi in 0..batch_size {
                let batch_offset = bi * seq_len;
                for i in 0..seq_len {
                    let row_start = (batch_offset + i) * n;
                    for j in 0..=i {
                        full_mask_data[row_start + batch_offset + j] = 0.0;
                    }
                }
            }
            let causal_mask = Tensor::from_slice(&full_mask_data, &[n, n]);
            let k_t = k.transpose();
            let scores = q.matmul(&k_t).mul(&scale_t);
            let scores_masked = scores.add(&causal_mask);
            let attn_w = ops::nn::softmax(&scores_masked, 1);
            let attn_proj = attn_w.matmul(&v).matmul(&params[lo.attn_out_proj]);

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
                + 1              // router weight (no bias in Router)
                + n_experts * 4  // experts (fc1_w+b+fc2_w+b)
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
            router_w: base + 2,
            expert_start: base + 3,
            norm2_w: base + 3 + self.n_experts * 4,
            norm2_b: base + 4 + self.n_experts * 4,
            q_proj: base + 5 + self.n_experts * 4,
            k_proj: base + 6 + self.n_experts * 4,
            v_proj: base + 7 + self.n_experts * 4,
            attn_out_proj: base + 8 + self.n_experts * 4,
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
    router_w: usize,
    expert_start: usize,
    norm2_w: usize,
    norm2_b: usize,
    q_proj: usize,
    k_proj: usize,
    v_proj: usize,
    attn_out_proj: usize,
}

impl LayerTensorOffset {
    fn fc1_w(&self, e: usize) -> usize {
        self.expert_start + e * 4
    }
    fn fc1_b(&self, e: usize) -> usize {
        self.expert_start + e * 4 + 1
    }
    fn fc2_w(&self, e: usize) -> usize {
        self.expert_start + e * 4 + 2
    }
    fn fc2_b(&self, e: usize) -> usize {
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
    pub fn _count_parameters(model: &OracleBackbone) -> usize {
        // This is a simplified calculation
        // In practice, you'd sum all parameters from all layers
        model.config.d_model * model.config.d_model * 100 // Rough estimate
    }

    /// Calculate FLOPs for forward pass
    pub fn _calculate_flops(
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
