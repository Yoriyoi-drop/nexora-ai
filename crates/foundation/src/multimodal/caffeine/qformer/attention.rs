//! Attention mechanisms for Q-Former
//! 
//! Multi-head attention and query attention implementations

use crate::multimodal::caffeine::types::*;
use crate::multimodal::caffeine::error::Result;
use ndarray::ArrayD;

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    dropout_rate: f32,
    query_weights: Vec<Vec<f32>>,
    key_weights: Vec<Vec<f32>>,
    value_weights: Vec<Vec<f32>>,
    output_weights: Vec<f32>,
}

impl MultiHeadAttention {
    /// Create new multi-head attention
    pub fn new(hidden_dim: usize, num_heads: usize, dropout_rate: f32) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        
        // Initialize weights
        let mut query_weights = Vec::new();
        let mut key_weights = Vec::new();
        let mut value_weights = Vec::new();
        
        for _ in 0..num_heads {
            query_weights.push(Self::init_weights(hidden_dim, hidden_dim)?);
            key_weights.push(Self::init_weights(hidden_dim, hidden_dim)?);
            value_weights.push(Self::init_weights(hidden_dim, hidden_dim)?);
        }
        
        let output_weights = Self::init_weights(num_heads * head_dim, hidden_dim)?;
        
        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim,
            dropout_rate,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, query: &ArrayD<f32>, key: &ArrayD<f32>, value: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = query.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        // Project to Q, K, V for each head
        let mut head_outputs = Vec::new();
        
        for head in 0..self.num_heads {
            let q_head = self.linear_projection(query, &self.query_weights[head])?;
            let k_head = self.linear_projection(key, &self.key_weights[head])?;
            let v_head = self.linear_projection(value, &self.value_weights[head])?;
            
            // Compute attention for this head
            let attention_output = self.scaled_dot_product_attention(&q_head, &k_head, &v_head)?;
            head_outputs.push(attention_output);
        }
        
        // Concatenate heads and project output
        let concatenated = self.concatenate_heads(&head_outputs, batch_size, seq_len)?;
        let output = self.linear_projection(&concatenated, &self.output_weights)?;
        
        Ok(output)
    }
    
    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        query: &ArrayD<f32>,
        key: &ArrayD<f32>,
        value: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        let shape = query.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let head_dim = shape[2];
        
        // Compute attention scores
        let mut attention_scores = vec![0.0f32; batch_size * seq_len * seq_len];
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0f32;
                    
                    for d in 0..head_dim {
                        if let (Some(&q), Some(&k)) = (
                            query.get([b, i, d]),
                            key.get([b, j, d])
                        ) {
                            score += q * k;
                        }
                    }
                    
                    // Scale by sqrt(head_dim)
                    score = score / (head_dim as f32).sqrt();
                    attention_scores[b * seq_len * seq_len + i * seq_len + j] = score;
                }
            }
        }
        
        // Apply softmax
        let attention_weights = self.softmax_3d(&attention_scores, batch_size, seq_len)?;
        
        // Apply attention to values
        let mut output = vec![0.0f32; batch_size * seq_len * head_dim];
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut weighted_sum = 0.0f32;
                    
                    for j in 0..seq_len {
                        let weight_idx = b * seq_len * seq_len + i * seq_len + j;
                        if let (Some(&weight), Some(&v)) = (
                            attention_weights.get(weight_idx),
                            value.get([b, j, d])
                        ) {
                            weighted_sum += weight * v;
                        }
                    }
                    
                    let output_idx = b * seq_len * head_dim + i * head_dim + d;
                    output[output_idx] = weighted_sum;
                }
            }
        }
        
        let output_shape = vec![batch_size, seq_len, head_dim];
        Ok(ArrayD::from_shape_vec(output_shape, output)?)
    }
    
    /// Linear projection
    fn linear_projection(&self, input: &ArrayD<f32>, weights: &[f32]) -> Result<ArrayD<f32>> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let input_dim = input_shape[2];
        let output_dim = weights.len() / input_dim;
        
        let mut output = vec![0.0f32; batch_size * seq_len * output_dim];
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                for o in 0..output_dim {
                    let mut sum = 0.0f32;
                    
                    for d in 0..input_dim {
                        if let (Some(&input_val), Some(&weight)) = (
                            input.get([b, i, d]),
                            weights.get(d * output_dim + o)
                        ) {
                            sum += input_val * weight;
                        }
                    }
                    
                    let output_idx = b * seq_len * output_dim + i * output_dim + o;
                    output[output_idx] = sum;
                }
            }
        }
        
        let output_shape = vec![batch_size, seq_len, output_dim];
        Ok(ArrayD::from_shape_vec(output_shape, output)?)
    }
    
    /// Concatenate attention heads
    fn concatenate_heads(
        &self,
        heads: &[ArrayD<f32>],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<ArrayD<f32>> {
        let total_dim = self.num_heads * self.head_dim;
        let mut concatenated = vec![0.0f32; batch_size * seq_len * total_dim];
        
        for (head_idx, head) in heads.iter().enumerate() {
            for b in 0..batch_size {
                for i in 0..seq_len {
                    for d in 0..self.head_dim {
                        if let Some(&val) = head.get([b, i, d]) {
                            let output_idx = b * seq_len * total_dim + 
                                           i * total_dim + 
                                           head_idx * self.head_dim + d;
                            concatenated[output_idx] = val;
                        }
                    }
                }
            }
        }
        
        let output_shape = vec![batch_size, seq_len, total_dim];
        Ok(ArrayD::from_shape_vec(output_shape, concatenated)?)
    }
    
    /// Softmax for 3D tensor
    fn softmax_3d(&self, scores: &[f32], batch_size: usize, seq_len: usize) -> Result<Vec<f32>> {
        let mut softmax_scores = vec![0.0f32; scores.len()];
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                // Find max for numerical stability
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..seq_len {
                    let idx = b * seq_len * seq_len + i * seq_len + j;
                    max_score = max_score.max(scores[idx]);
                }
                
                // Compute exp and sum
                let mut sum = 0.0f32;
                for j in 0..seq_len {
                    let idx = b * seq_len * seq_len + i * seq_len + j;
                    let exp_val = (scores[idx] - max_score).exp();
                    softmax_scores[idx] = exp_val;
                    sum += exp_val;
                }
                
                // Normalize
                for j in 0..seq_len {
                    let idx = b * seq_len * seq_len + i * seq_len + j;
                    if sum > 0.0 {
                        softmax_scores[idx] /= sum;
                    }
                }
            }
        }
        
        Ok(softmax_scores)
    }
    
    /// Initialize weights
    fn init_weights(input_dim: usize, output_dim: usize) -> Result<Vec<f32>> {
        let mut weights = vec![0.0f32; input_dim * output_dim];
        
        for i in 0..input_dim * output_dim {
            // Xavier initialization
            weights[i] = ((i as f32 * 0.01).sin() * 2.0) / (input_dim as f32).sqrt();
        }
        
        Ok(weights)
    }
}

/// Query attention mechanism
pub struct QueryAttention {
    hidden_dim: usize,
    _num_queries: usize,
    attention_weights: Option<ArrayD<f32>>,
}

impl QueryAttention {
    /// Create new query attention
    pub fn new(hidden_dim: usize, _num_queries: usize) -> Self {
        Self {
            hidden_dim,
            _num_queries,
            attention_weights: None,
        }
    }
    
    /// Apply query attention
    pub fn forward(&mut self, queries: &ArrayD<f32>, context: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let query_shape = queries.shape();
        let context_shape = context.shape();
        
        let num_queries = query_shape[1];
        let context_len = context_shape[1];
        
        // Compute attention scores between queries and context
        let mut attention_scores = vec![0.0f32; num_queries * context_len];
        
        for i in 0..num_queries {
            for j in 0..context_len {
                let mut score = 0.0f32;
                
                for d in 0..self.hidden_dim {
                    if let (Some(&q), Some(&c)) = (
                        queries.get([0, i, d]),
                        context.get([0, j, d])
                    ) {
                        score += q * c;
                    }
                }
                
                attention_scores[i * context_len + j] = score;
            }
        }
        
        // Apply softmax
        let attention_weights = self.softmax_2d(&attention_scores, num_queries, context_len)?;
        
        // Store attention weights for visualization
        let weight_shape = vec![num_queries, context_len];
        self.attention_weights = Some(ArrayD::from_shape_vec(weight_shape, attention_weights.clone())?);
        
        // Apply attention to context
        let mut output = vec![0.0f32; num_queries * self.hidden_dim];
        
        for i in 0..num_queries {
            for d in 0..self.hidden_dim {
                let mut weighted_sum = 0.0f32;
                
                for j in 0..context_len {
                    let weight_idx = i * context_len + j;
                    if let (Some(&weight), Some(&c)) = (
                        attention_weights.get(weight_idx),
                        context.get([0, j, d])
                    ) {
                        weighted_sum += weight * c;
                    }
                }
                
                let output_idx = i * self.hidden_dim + d;
                output[output_idx] = weighted_sum;
            }
        }
        
        let output_shape = vec![1, num_queries, self.hidden_dim];
        Ok(ArrayD::from_shape_vec(output_shape, output)?)
    }
    
    /// Softmax for 2D tensor
    fn softmax_2d(&self, scores: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let mut softmax_scores = vec![0.0f32; scores.len()];
        
        for i in 0..rows {
            // Find max for numerical stability
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..cols {
                let idx = i * cols + j;
                max_score = max_score.max(scores[idx]);
            }
            
            // Compute exp and sum
            let mut sum = 0.0f32;
            for j in 0..cols {
                let idx = i * cols + j;
                let exp_val = (scores[idx] - max_score).exp();
                softmax_scores[idx] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for j in 0..cols {
                let idx = i * cols + j;
                if sum > 0.0 {
                    softmax_scores[idx] /= sum;
                }
            }
        }
        
        Ok(softmax_scores)
    }
    
    /// Get attention weights
    pub fn get_attention_weights(&self) -> Option<&ArrayD<f32>> {
        self.attention_weights.as_ref()
    }
}
