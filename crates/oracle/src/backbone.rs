//! ORACLE Backbone - Sparse MoE + MLA Implementation
//! 
//! Fondasi arsitektur ORACLE yang menggabungkan Mixture of Experts (MoE)
//! dengan Multi-head Latent Attention (MLA) untuk efisiensi komputasi
//! dan context window yang besar.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, s};
use ndarray_rand::RandomExt;
use rand::{distributions::Standard, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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
    _config: OracleBackboneConfig,
    experts: Vec<MLPExpert>,
    gate: LinearLayer,
    router: Router,
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
    
    pub fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        let (batch_size, d_model) = (x.dim().0, x.dim().1);
        
        let x_reshaped = x.view().into_shape((batch_size, d_model))?.to_owned();
        
        let gate_scores = self.gate.forward(&x_reshaped)?;
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
                let expert_input = x_reshaped.slice(s![token_idx, ..]).to_owned();
                let expert_output = self.experts[expert_idx].forward(&expert_input)?;
                
                let mut output_row = output.slice_mut(s![token_idx, ..]);
                output_row.iter_mut().zip(&expert_output).for_each(|(o, e)| *o += e);
                weight_sum[token_idx] += 1.0;
            }
        }
        
        for token_idx in 0..batch_size {
            if weight_sum[token_idx] > 0.0 {
                let mut output_row = output.slice_mut(s![token_idx, ..]);
                output_row.mapv_inplace(|v| v / weight_sum[token_idx]);
            }
        }
        
        Ok(output.into_shape((batch_size, d_model))?.to_owned())
    }
    
    pub fn get_expert_usage(&self, x: &Array2<f32>) -> Result<Vec<f32>> {
        let (batch_size, d_model) = (x.dim().0, x.dim().1);
        let x_reshaped = x.view().into_shape((batch_size, d_model))?;
        
        let gate_scores = self.gate.forward(&x_reshaped.to_owned())?;
        self.router.get_usage_stats(&gate_scores)
    }
}

/// Individual Expert (MLP)
pub struct MLPExpert {
    w1: Array2<f32>,
    w2: Array2<f32>,
    b1: Array1<f32>,
    b2: Array1<f32>,
}

impl MLPExpert {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w1: Array2::from_shape_fn((input_dim, hidden_dim), |(_, _)| rand::random::<f32>()),
            w2: Array2::from_shape_fn((hidden_dim, input_dim), |(_, _)| rand::random::<f32>()),
            b1: Array1::from_shape_fn(hidden_dim, |_| rand::random::<f32>()),
            b2: Array1::from_shape_fn(input_dim, |_| rand::random::<f32>()),
        }
    }
    
    pub fn forward(&self, x: &Array1<f32>) -> Result<Array1<f32>> {
        // First linear layer + activation
        let hidden = x.dot(&self.w1) + &self.b1;
        let activated = self.gelu(&hidden);
        
        // Second linear layer
        let output = activated.dot(&self.w2) + &self.b2;
        Ok(output)
    }
    
    fn gelu(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|x| 0.5 * x * (1.0 + (1.41421356237 * x).tanh()))
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
            indices.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap_or(std::cmp::Ordering::Equal));
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
            indices.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap_or(std::cmp::Ordering::Equal));
            
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
    config: OracleBackboneConfig,
    latent_projection: LinearLayer,
    attention_heads: Vec<LatentAttentionHead>,
    output_projection: LinearLayer,
    latent_compression: LatentCompression,
}

impl MultiHeadLatentAttention {
    pub fn new(config: OracleBackboneConfig) -> Self {
        let mut attention_heads = Vec::new();
        let head_dim = config.d_model / config.n_heads;
        
        for _ in 0..config.n_heads {
            attention_heads.push(LatentAttentionHead::new(
                head_dim,
                config.latent_dim,
            ));
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
        let latent = self.latent_projection.forward(&x_reshaped.to_owned())?;
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
        let output_reshaped = concatenated.view().into_shape((batch_size * seq_len, self.config.latent_dim))?;
        let output = self.output_projection.forward(&output_reshaped.to_owned())?;
        
        Ok(output.into_shape((batch_size, seq_len, d_model))?.to_owned())
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
    head_dim: usize,
    latent_dim: usize,
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    out_proj: LinearLayer,
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
        let x_reshaped = x.view().into_shape((batch_size * seq_len, self.latent_dim))?.to_owned();
        
        let q = self.q_proj.forward(&x_reshaped.to_owned())?;
        let k = self.k_proj.forward(&x_reshaped.to_owned())?;
        let v = self.v_proj.forward(&x_reshaped.to_owned())?;
        
        let q_3d = q.into_shape((batch_size, seq_len, self.head_dim))?;
        let k_3d = k.into_shape((batch_size, seq_len, self.head_dim))?;
        let v_3d = v.into_shape((batch_size, seq_len, self.head_dim))?;
        
        // Scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q_3d, &k_3d, &v_3d, mask)?;
        
        // Project output
        let output_reshaped = attention_output.view().into_shape((batch_size * seq_len, self.head_dim))?.to_owned();
        let output = self.out_proj.forward(&output_reshaped)?;
        
        Ok(output.into_shape((batch_size, seq_len, self.latent_dim))?.to_owned())
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

            // Apply causal mask
            if let Some(mask) = mask {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        scores[[i, j]] += mask[[i, j]];
                    }
                }
            } else {
                // Default causal mask: prevent attending to future tokens
                for i in 0..seq_len {
                    for j in i + 1..seq_len {
                        scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
            }

            // Softmax over last dimension (per-row)
            for i in 0..seq_len {
                let row = scores.slice(ndarray::s![i, ..]);
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_row = row.mapv(|x| (x - max_val).exp());
                let sum_exp: f32 = exp_row.iter().sum();
                for j in 0..seq_len {
                    let attn = (scores[[i, j]] - max_val).exp() / sum_exp;
                    for d in 0..head_dim {
                        output[[b, i, d]] += attn * v[[b, j, d]];
                    }
                }
            }
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
            (x * &mask_expanded).to_owned()
        } else {
            x.to_owned()
        };
        
        // Simple compression: take top-k dimensions
        let mut compressed = Array3::zeros((batch_size, seq_len, compressed_dim));
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                let row = x.slice(s![b, i, ..]);
                let mut indices: Vec<usize> = (0..latent_dim).collect();
                indices.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap_or(std::cmp::Ordering::Equal));
                
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
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weight: Array2::from_shape_fn((input_dim, output_dim), |_| rand::random()),
            bias: Array1::from_shape_fn(output_dim, |_| rand::random()),
        }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        Ok(x.dot(&self.weight) + &self.bias)
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
            moe_layers: (0..n_layers).map(|_| SparseMoELayer::new(config.clone())).collect(),
            attention_layers: (0..n_layers).map(|_| MultiHeadLatentAttention::new(config.clone())).collect(),
            norm_layers: (0..n_layers + 1).map(|_| LayerNorm::new(config.d_model)).collect(),
            output_projection: LinearLayer::new(config.d_model, vocab_size),
        }
    }
    
    pub fn forward(&self, input_ids: &Array2<i32>, mask: Option<&Array2<f32>>) -> Result<Array3<f32>> {
        let (batch_size, seq_len) = input_ids.dim();
        
        // Embedding
        let mut hidden = self.embedding.forward(input_ids)?;
        
        // Transformer layers
        for i in 0..self.moe_layers.len() {
            // Pre-norm
            hidden = self.norm_layers[i].forward(&hidden)?;
            
            // MoE layer - reshape from 3D to 2D
            let (b, s, d) = hidden.dim();
            let hidden_2d = hidden.view().into_shape((b * s, d))?.to_owned();
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
        let hidden_reshaped = hidden.view().into_shape((batch_size * seq_len, self.config.d_model))?.to_owned();
        let logits = self.output_projection.forward(&hidden_reshaped)?;
        
        Ok(logits.into_shape((batch_size, seq_len, self.output_projection.weight.dim().1))?.to_owned())
    }
    
    pub fn get_expert_usage_stats(&self, input_ids: &Array2<i32>) -> Result<Vec<Vec<f32>>> {
        let hidden = self.embedding.forward(input_ids)?;
        let (batch_size, seq_len, d_model) = hidden.dim();
        let hidden_2d = hidden.view().into_shape((batch_size * seq_len, d_model))?.to_owned();
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
    embeddings: Array2<f32>,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            embeddings: Array2::from_shape_fn((vocab_size, d_model), |_| rand::random()),
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
    weight: Array1<f32>,
    bias: Array1<f32>,
    eps: f32,
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
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros(x.dim());
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                let row = x.slice(s![b, i, ..]);
                let mean = row.mean().ok_or_else(|| anyhow::anyhow!("Failed to calculate mean"))?;
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
    pub fn calculate_flops(config: &OracleBackboneConfig, batch_size: usize, seq_len: usize) -> u64 {
        let n_layers: u64 = 12;
        let batch_size: u64 = batch_size as u64;
        let moe_flops = (config.d_model * config.mlp_hidden * 2) as u64; // Forward + backward
        let attention_flops = (seq_len * seq_len * config.d_model * 4) as u64; // Q, K, V, O
        
        (moe_flops + attention_flops) * n_layers * batch_size
    }
    
    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(config: &OracleBackboneConfig, batch_size: usize, seq_len: usize) -> usize {
        let activation_memory = batch_size * seq_len * config.d_model * 4; // f32
        let kv_cache_memory = batch_size * config.n_heads * seq_len * (config.d_model / config.n_heads) * 4;
        let mla_compressed = batch_size * seq_len * config.latent_dim * 4;
        
        activation_memory + kv_cache_memory + mla_compressed
    }
}
