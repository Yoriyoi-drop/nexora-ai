//! Attention mechanisms for HAS-MoE-FFN

use serde::{Serialize, Deserialize};
use std::f32::consts::SQRT_2;

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub dropout_rate: f32,
}

/// Attention mechanism
pub struct Attention {
    config: AttentionConfig,
    head_dim: usize,
    // Query, Key, Value projections
    q_proj: Vec<Vec<f32>>,
    k_proj: Vec<Vec<f32>>,
    v_proj: Vec<Vec<f32>>,
    // Output projection
    o_proj: Vec<Vec<f32>>,
}

impl Attention {
    /// Create new attention mechanism
    pub fn new(hidden_size: usize, num_heads: usize, dropout_rate: f32) -> Self {
        let config = AttentionConfig {
            hidden_size,
            num_heads,
            dropout_rate,
        };
        
        let head_dim = hidden_size / num_heads;
        
        // Initialize projections with random weights
        let q_proj = Self::init_projection(hidden_size, hidden_size);
        let k_proj = Self::init_projection(hidden_size, hidden_size);
        let v_proj = Self::init_projection(hidden_size, hidden_size);
        let o_proj = Self::init_projection(hidden_size, hidden_size);
        
        Self { 
            config,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        }
    }
    
    /// Initialize projection weights
    fn init_projection(input_size: usize, output_size: usize) -> Vec<Vec<f32>> {
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            let row: Vec<f32> = (0..input_size)
                .map(|i| ((i as f32 * 0.01).sin() * 0.02))
                .collect();
            weights.push(row);
        }
        weights
    }
    
    /// Forward pass through attention
    pub fn forward(&self, query: &[f32], key: &[f32], value: &[f32]) -> Vec<f32> {
        // Project to Q, K, V
        let q = self.linear_forward(&self.q_proj, query);
        let k = self.linear_forward(&self.k_proj, key);
        let v = self.linear_forward(&self.v_proj, value);
        
        // Reshape for multi-head attention
        let q_heads = self.reshape_to_heads(&q);
        let k_heads = self.reshape_to_heads(&k);
        let v_heads = self.reshape_to_heads(&v);
        
        // Compute attention for each head
        let mut head_outputs = Vec::new();
        for i in 0..self.config.num_heads {
            let head_output = self.compute_scaled_dot_product_attention(
                &q_heads[i], 
                &k_heads[i], 
                &v_heads[i]
            );
            head_outputs.push(head_output);
        }
        
        // Concatenate heads and project output
        let concatenated = self.concatenate_heads(&head_outputs);
        let output = self.linear_forward(&self.o_proj, &concatenated);
        
        output
    }
    
    /// Linear projection
    fn linear_forward(&self, weights: &[Vec<f32>], input: &[f32]) -> Vec<f32> {
        weights.iter().map(|row| {
            row.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
        }).collect()
    }
    
    /// Reshape input to multi-head format
    fn reshape_to_heads(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut heads = Vec::with_capacity(self.config.num_heads);
        for i in 0..self.config.num_heads {
            let start = i * self.head_dim;
            let end = start + self.head_dim;
            let head = input[start..end].to_vec();
            heads.push(head);
        }
        heads
    }
    
    /// Concatenate heads back to single tensor
    fn concatenate_heads(&self, heads: &[Vec<f32>]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.config.hidden_size);
        for head in heads {
            result.extend(head);
        }
        result
    }
    
    /// Compute scaled dot-product attention
    fn compute_scaled_dot_product_attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Vec<f32> {
        // Compute attention scores: Q * K^T / sqrt(d_k)
        let mut scores = Vec::with_capacity(query.len());
        for i in 0..query.len() {
            let mut score = 0.0;
            for j in 0..key.len() {
                score += query[i] * key[j];
            }
            scores.push(score / (self.head_dim as f32).sqrt());
        }
        
        // Apply softmax
        let softmax_scores = self.softmax(&scores);
        
        // Apply attention to values
        let mut output = Vec::with_capacity(value.len());
        for i in 0..value.len() {
            let mut sum = 0.0;
            for j in 0..softmax_scores.len() {
                sum += softmax_scores[j] * value[j];
            }
            output.push(sum);
        }
        
        output
    }
    
    /// Softmax function
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp and sum
        let exp_vals: Vec<f32> = input.iter()
            .map(|x| (x - max_val).exp())
            .collect();
        let sum: f32 = exp_vals.iter().sum();
        
        // Normalize
        exp_vals.iter().map(|x| x / sum).collect()
    }
    
    /// Get configuration
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }
    
    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}
