//! Attention module for CLIP

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Multi-head attention for CLIP
pub struct ClipAttention {
    config: ClipAttentionConfig,
}

impl ClipAttention {
    /// Create new attention module
    pub fn new(config: ClipAttentionConfig) -> HLDVAResult<Self> {
        Ok(Self { config })
    }
    
    /// Apply attention to input
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> HLDVAResult<Tensor> {
        let query_data = query.data();
        let key_data = key.data();
        let value_data = value.data();
        
        let seq_len = query_data.len() / self.config.head_dim;
        let mut output_data = Vec::with_capacity(query_data.len());
        
        for head in 0..self.config.num_heads {
            let head_start = head * self.config.head_dim;
            let head_end = head_start + self.config.head_dim;
            
            for i in 0..seq_len {
                let mut attention_sum = 0.0;
                let mut attention_weight_sum = 0.0;
                
                for j in 0..seq_len {
                    let q_idx = head_start + i * self.config.head_dim;
                    let k_idx = head_start + j * self.config.head_dim;
                    let v_idx = head_start + j * self.config.head_dim;
                    
                    if q_idx < query_data.len() && k_idx < key_data.len() && v_idx < value_data.len() {
                        let attention_score = query_data[q_idx] * key_data[k_idx];
                        let attention_weight = attention_score.exp();
                        
                        attention_sum += attention_weight * value_data[v_idx];
                        attention_weight_sum += attention_weight;
                    }
                }
                
                if attention_weight_sum > 0.0 {
                    output_data.push(attention_sum / attention_weight_sum);
                } else {
                    output_data.push(0.0);
                }
            }
        }
        
        Ok(Tensor::new(output_data.clone(), vec![output_data.len()]))
    }
    
    /// Self-attention
    pub fn self_attention(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let seq_len = input_data.len() / self.config.embed_dim;
        
        // Create Q, K, V from input (simplified)
        let query = self.project_to_query(input)?;
        let key = self.project_to_key(input)?;
        let value = self.project_to_value(input)?;
        
        self.forward(&query, &key, &value)
    }
    
    fn project_to_query(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        self.linear_projection(input, self.config.embed_dim, self.config.embed_dim)
    }
    
    fn project_to_key(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        self.linear_projection(input, self.config.embed_dim, self.config.embed_dim)
    }
    
    fn project_to_value(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        self.linear_projection(input, self.config.embed_dim, self.config.embed_dim)
    }
    
    fn linear_projection(&self, input: &Tensor, in_dim: usize, out_dim: usize) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let mut output_data = Vec::with_capacity(out_dim);
        
        for i in 0..out_dim {
            let mut sum = 0.0;
            for j in 0..in_dim.min(input_data.len()) {
                let weight = ((i as f32 + 1.0) * (j as f32 + 1.0)).sin() / ((i as f32 + j as f32 + 1.0));
                sum += input_data[j] * weight;
            }
            output_data.push(sum);
        }
        
        Ok(Tensor::new(output_data, vec![out_dim]))
    }
}

/// CLIP Attention configuration
#[derive(Debug, Clone)]
pub struct ClipAttentionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout: f32,
}

impl ClipAttentionConfig {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            embed_dim,
            num_heads,
            head_dim,
            dropout: 0.1,
        }
    }
}

impl Default for ClipAttentionConfig {
    fn default() -> Self {
        Self::new(512, 8)
    }
}
