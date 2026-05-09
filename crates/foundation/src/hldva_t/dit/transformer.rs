//! Transformer Components for DiT
//!
//! Implementasi transformer blocks dan komponen terkait

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Transformer Block Configuration
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub attention_dropout: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_heads: 12,
            num_layers: 12,
            dropout: 0.1,
            attention_dropout: 0.1,
        }
    }
}

/// Residual Connection
pub struct ResidualConnection {
    dropout: f32,
}

impl ResidualConnection {
    pub fn new(dropout: f32) -> Self {
        Self { dropout }
    }
    
    pub fn forward(&self, x: &Tensor, sublayer_output: &Tensor) -> HLDVAResult<Tensor> {
        let x_data = x.data();
        let sub_data = sublayer_output.data();
        
        let mut output = Vec::with_capacity(x_data.len());
        
        for (i, &x_val) in x_data.iter().enumerate() {
            let sub_val = if i < sub_data.len() { sub_data[i] } else { 0.0 };
            
            // Residual connection + dropout
            let residual = x_val + sub_val;
            let final_val = if rand::random::<f32>() < self.dropout {
                0.0
            } else {
                residual
            };
            
            output.push(final_val);
        }
        
        Ok(Tensor::new(output, x.shape().to_vec()))
    }
}

/// Pre-Layer Normalization (style GPT)
pub struct PreLayerNorm {
    layer_norm: LayerNorm,
}

impl PreLayerNorm {
    pub fn new(hidden_dim: usize) -> HLDVAResult<Self> {
        let layer_norm = LayerNorm::new(hidden_dim)?;
        Ok(Self { layer_norm })
    }
    
    pub fn forward(&self, x: &Tensor) -> HLDVAResult<Tensor> {
        self.layer_norm.forward(x)
    }
}

/// Post-Layer Normalization (style original Transformer)
pub struct PostLayerNorm {
    layer_norm: LayerNorm,
}

impl PostLayerNorm {
    pub fn new(hidden_dim: usize) -> HLDVAResult<Self> {
        let layer_norm = LayerNorm::new(hidden_dim)?;
        Ok(Self { layer_norm })
    }
    
    pub fn forward(&self, x: &Tensor) -> HLDVAResult<Tensor> {
        self.layer_norm.forward(x)
    }
}

/// Transformer Layer dengan Pre-Norm
pub struct TransformerLayer {
    hidden_dim: usize,
    num_heads: usize,
    
    // Self-attention
    self_attention: MultiHeadAttention,
    attention_norm: PreLayerNorm,
    attention_dropout: ResidualConnection,
    
    // Cross-attention
    cross_attention: MultiHeadAttention,
    cross_attention_norm: PreLayerNorm,
    cross_attention_dropout: ResidualConnection,
    
    // Feed-forward
    feed_forward: FeedForward,
    ff_norm: PreLayerNorm,
    ff_dropout: ResidualConnection,
}

impl TransformerLayer {
    pub fn new(hidden_dim: usize, num_heads: usize) -> HLDVAResult<Self> {
        let self_attention = MultiHeadAttention::new(hidden_dim, num_heads)?;
        let cross_attention = MultiHeadAttention::new(hidden_dim, num_heads)?;
        let feed_forward = FeedForward::new(hidden_dim)?;
        
        Ok(Self {
            hidden_dim,
            num_heads,
            self_attention,
            attention_norm: PreLayerNorm::new(hidden_dim)?,
            attention_dropout: ResidualConnection::new(0.1),
            cross_attention,
            cross_attention_norm: PreLayerNorm::new(hidden_dim)?,
            cross_attention_dropout: ResidualConnection::new(0.1),
            feed_forward,
            ff_norm: PreLayerNorm::new(hidden_dim)?,
            ff_dropout: ResidualConnection::new(0.1),
        })
    }
    
    pub fn forward(
        &self,
        hidden: &Tensor,
        conditioning: &Tensor,
    ) -> HLDVAResult<Tensor> {
        // Self-attention block
        let norm_hidden = self.attention_norm.forward(hidden)?;
        let attn_output = self.self_attention.forward(&norm_hidden, &norm_hidden, &norm_hidden)?;
        let hidden_with_attn = self.attention_dropout.forward(hidden, &attn_output)?;
        
        // Cross-attention block
        let norm_hidden2 = self.cross_attention_norm.forward(&hidden_with_attn)?;
        let cross_attn_output = self.cross_attention.forward(&norm_hidden2, conditioning, conditioning)?;
        let hidden_with_cross = self.cross_attention_dropout.forward(&hidden_with_attn, &cross_attn_output)?;
        
        // Feed-forward block
        let norm_hidden3 = self.ff_norm.forward(&hidden_with_cross)?;
        let ff_output = self.feed_forward.forward(&norm_hidden3)?;
        let output = self.ff_dropout.forward(&hidden_with_cross, &ff_output)?;
        
        Ok(output)
    }
}

/// Transformer Encoder
pub struct TransformerEncoder {
    layers: Vec<TransformerLayer>,
    hidden_dim: usize,
    num_layers: usize,
}

impl TransformerEncoder {
    pub fn new(config: &TransformerConfig) -> HLDVAResult<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        
        for _ in 0..config.num_layers {
            let layer = TransformerLayer::new(config.hidden_dim, config.num_heads)?;
            layers.push(layer);
        }
        
        Ok(Self {
            layers,
            hidden_dim: config.hidden_dim,
            num_layers: config.num_layers,
        })
    }
    
    pub fn forward(
        &self,
        hidden: &Tensor,
        conditioning: &Tensor,
    ) -> HLDVAResult<Tensor> {
        let mut output = hidden.clone();
        
        for layer in &self.layers {
            output = layer.forward(&output, conditioning)?;
        }
        
        Ok(output)
    }
    
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// Adaptive Computation Time (ACT) untuk dynamic depth
pub struct AdaptiveComputationTime {
    hidden_dim: usize,
    
    // Halting probabilities
    halting_proj: Linear,
    
    // Threshold untuk stopping
    threshold: f32,
    
    // Maximum ponder time
    max_ponder: usize,
}

impl AdaptiveComputationTime {
    pub fn new(hidden_dim: usize) -> HLDVAResult<Self> {
        let halting_proj = Linear::new(hidden_dim, 1)?;
        
        Ok(Self {
            hidden_dim,
            halting_proj,
            threshold: 0.5,
            max_ponder: 10,
        })
    }
    
    pub fn forward(
        &self,
        hidden: &Tensor,
        layers: &[TransformerLayer],
        conditioning: &Tensor,
    ) -> HLDVAResult<Tensor> {
        let mut output = hidden.clone();
        let mut halting_prob = 0.0;
        let mut ponder_time = 0;
        
        for (i, layer) in layers.iter().enumerate() {
            if ponder_time >= self.max_ponder {
                break;
            }
            
            // Compute halting probability
            let halting_score = self.halting_proj.forward(&output)?;
            let halting_prob_step = 1.0 / (1.0 + (-halting_score.data()[0]).exp());
            
            // Update cumulative halting probability
            halting_prob += halting_prob_step;
            ponder_time += 1;
            
            // Apply layer
            output = layer.forward(&output, conditioning)?;
            
            // Weight output by halting probability
            let weighted_output = self.weight_output(&output, halting_prob_step)?;
            output = weighted_output;
            
            // Check if should halt
            if halting_prob >= self.threshold {
                break;
            }
        }
        
        Ok(output)
    }
    
    fn weight_output(&self, output: &Tensor, weight: f32) -> HLDVAResult<Tensor> {
        let data = output.data();
        let weighted: Vec<f32> = data.iter().map(|&x| x * weight).collect();
        Ok(Tensor::new(weighted, output.shape().to_vec()))
    }
}

/// Mixture of Experts (MoE) untuk conditional computation
pub struct MixtureOfExperts {
    hidden_dim: usize,
    num_experts: usize,
    top_k: usize,
    
    // Expert networks
    experts: Vec<FeedForward>,
    
    // Gating network
    gate: Linear,
    
    // Load balancing loss coefficient
    load_balance_coef: f32,
}

impl MixtureOfExperts {
    pub fn new(hidden_dim: usize, num_experts: usize, top_k: usize) -> HLDVAResult<Self> {
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            let expert = FeedForward::new(hidden_dim)?;
            experts.push(expert);
        }
        
        let gate = Linear::new(hidden_dim, num_experts)?;
        
        Ok(Self {
            hidden_dim,
            num_experts,
            top_k,
            experts,
            gate,
            load_balance_coef: 0.01,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<(Tensor, f32)> {
        // Compute gating scores
        let gate_scores = self.gate.forward(input)?;
        let gate_data = gate_scores.data();
        
        // Select top-k experts
        let (selected_experts, expert_weights) = self.select_top_experts(gate_data)?;
        
        // Apply selected experts
        let mut output = Tensor::new(vec![0.0; self.hidden_dim], vec![self.hidden_dim]);
        let mut load_balance_loss = 0.0;
        
        for (expert_idx, weight) in selected_experts.iter().zip(expert_weights.iter()) {
            if *expert_idx < self.experts.len() {
                let expert_output = self.experts[*expert_idx].forward(input)?;
                let weighted_output = self.weight_output(&expert_output, *weight)?;
                output = self.add_outputs(&output, &weighted_output)?;
                
                // Load balancing loss
                load_balance_loss += self.compute_load_balance_loss(gate_data, *expert_idx);
            }
        }
        
        Ok((output, load_balance_loss))
    }
    
    fn select_top_experts(&self, gate_scores: &[f32]) -> HLDVAResult<(Vec<usize>, Vec<f32>)> {
        let mut indexed_scores: Vec<(usize, f32)> = gate_scores.iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        // Sort by score (descending)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select top-k
        let selected_experts: Vec<usize> = indexed_scores.iter()
            .take(self.top_k)
            .map(|(idx, _)| *idx)
            .collect();
        
        let expert_weights: Vec<f32> = indexed_scores.iter()
            .take(self.top_k)
            .map(|(_, score)| *score)
            .collect();
        
        Ok((selected_experts, expert_weights))
    }
    
    fn weight_output(&self, output: &Tensor, weight: f32) -> HLDVAResult<Tensor> {
        let data = output.data();
        let weighted: Vec<f32> = data.iter().map(|&x| x * weight).collect();
        Ok(Tensor::new(weighted, output.shape().to_vec()))
    }
    
    fn add_outputs(&self, a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
        let a_data = a.data();
        let b_data = b.data();
        
        let mut sum = Vec::with_capacity(a_data.len());
        for i in 0..a_data.len() {
            let b_val = if i < b_data.len() { b_data[i] } else { 0.0 };
            sum.push(a_data[i] + b_val);
        }
        
        Ok(Tensor::new(sum, a.shape().to_vec()))
    }
    
    fn compute_load_balance_loss(&self, gate_scores: &[f32], expert_idx: usize) -> f32 {
        if expert_idx < gate_scores.len() {
            let gate_prob = 1.0 / (1.0 + (-gate_scores[expert_idx]).exp());
            let n = gate_scores.len() as f32;
            gate_prob * (gate_prob * n - 1.0).powi(2) * self.load_balance_coef
        } else {
            0.0
        }
    }
}

// Re-export dependencies
use super::attention::{
    MultiHeadAttention, FeedForward, Linear, LayerNorm
};
