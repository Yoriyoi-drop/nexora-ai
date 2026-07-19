//! Attention mechanisms for HAS-MoE-FFN

use crate::error::HasMoeFfnError;
use serde::{Deserialize, Serialize};

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub dropout_rate: f32,
}

/// Attention mechanism
#[derive(Debug)]
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
    pub fn new(hidden_size: usize, num_heads: usize, dropout_rate: f32) -> Result<Self, HasMoeFfnError> {
        if num_heads == 0 {
            return Err(HasMoeFfnError::config("num_heads must be positive"));
        }
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

        Ok(Self {
            config,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Initialize projection weights
    fn init_projection(input_size: usize, output_size: usize) -> Vec<Vec<f32>> {
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
            let head_output =
                self.compute_scaled_dot_product_attention(&q_heads[i], &k_heads[i], &v_heads[i]);
            head_outputs.push(head_output);
        }

        // Concatenate heads and project output
        let concatenated = self.concatenate_heads(&head_outputs);
        let output = self.linear_forward(&self.o_proj, &concatenated);

        output
    }

    /// Linear projection
    fn linear_forward(&self, weights: &[Vec<f32>], input: &[f32]) -> Vec<f32> {
        weights
            .iter()
            .map(|row| row.iter().zip(input.iter()).map(|(w, x)| w * x).sum())
            .collect()
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

    /// Compute scaled dot-product attention — streaming per-query to avoid O(S²) memory
    /// FIX 1+2+5: No full score matrix, compute-use-discard per query position
    fn compute_scaled_dot_product_attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Vec<f32> {
        let seq_len = query.len() / self.head_dim;
        let scale = (self.head_dim as f32).sqrt().recip();

        let mut output = vec![0.0; seq_len * self.head_dim];

        // Process each query position independently — O(S) memory
        for qi in 0..seq_len {
            let q_start = qi * self.head_dim;

            // Compute scores and softmax in one pass
            let mut max_val = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; seq_len];
            for kj in 0..seq_len {
                let k_start = kj * self.head_dim;
                let dot: f32 = (0..self.head_dim)
                    .map(|d| query[q_start + d] * key[k_start + d])
                    .sum();
                scores[kj] = dot * scale;
                if scores[kj] > max_val {
                    max_val = scores[kj];
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for kj in 0..seq_len {
                scores[kj] = (scores[kj] - max_val).exp();
                sum_exp += scores[kj];
            }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for kj in 0..seq_len {
                    scores[kj] *= inv;
                }
            }

            // Weighted sum of values
            for d in 0..self.head_dim {
                let mut sum = 0.0;
                for kj in 0..seq_len {
                    sum += scores[kj] * value[kj * self.head_dim + d];
                }
                output[qi * self.head_dim + d] = sum;
            }

            // FIX 6: Early free — scores dropped at end of scope
        }

        output
    }

    /// Softmax function
    #[allow(dead_code)]
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let exp_vals: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        // Normalize (guard NaN when sum == 0)
        let safe_sum = if sum == 0.0 { 1.0 } else { sum };
        exp_vals.iter().map(|x| x / safe_sum).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn small_attention() -> Attention {
        Attention::new(8, 2, 0.0).unwrap()
    }

    #[test]
    fn test_new_attention_config() {
        let a = Attention::new(12, 4, 0.1).unwrap();
        assert_eq!(a.config.hidden_size, 12);
        assert_eq!(a.config.num_heads, 4);
        assert_eq!(a.head_dim, 3);
    }

    #[test]
    fn test_forward_output_dim() {
        let a = small_attention();
        let input = vec![0.5; 8];
        let output = a.forward(&input, &input, &input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_forward_different_qkv_dims() {
        let a = Attention::new(6, 3, 0.0).unwrap();
        let q = vec![0.5; 6];
        let k = vec![0.3; 6];
        let v = vec![0.7; 6];
        let output = a.forward(&q, &k, &v);
        assert_eq!(output.len(), 6);
    }

    #[test]
    fn test_forward_deterministic() {
        let a = small_attention();
        let input = vec![1.0, 0.5, 0.0, -0.5, 0.2, -0.2, 0.8, -0.8];
        let out1 = a.forward(&input, &input, &input);
        let out2 = a.forward(&input, &input, &input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_linear_forward() {
        let a = small_attention();
        let weights = vec![vec![1.0, 0.5], vec![0.0, -1.0]];
        let input = vec![2.0, 4.0];
        let output = a.linear_forward(&weights, &input);
        assert_eq!(output.len(), 2);
        assert!((output[0] - 4.0).abs() < 1e-5);
        assert!((output[1] - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn test_reshape_to_heads() {
        let a = Attention::new(8, 4, 0.0).unwrap();
        let input = (0..8).map(|i| i as f32).collect::<Vec<_>>();
        let heads = a.reshape_to_heads(&input);
        assert_eq!(heads.len(), 4);
        assert_eq!(heads[0].len(), 2);
        assert_eq!(heads[0], vec![0.0, 1.0]);
        assert_eq!(heads[1], vec![2.0, 3.0]);
    }

    #[test]
    fn test_concatenate_heads() {
        let a = Attention::new(8, 4, 0.0).unwrap();
        let heads = vec![
            vec![0.0, 1.0],
            vec![2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0],
        ];
        let result = a.concatenate_heads(&heads);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_zero_heads_returns_error() {
        let result = Attention::new(8, 0, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("num_heads must be positive"));
    }

    #[test]
    fn test_softmax() {
        let a = small_attention();
        let result = a.softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_config() {
        let a = small_attention();
        let cfg = a.config();
        assert_eq!(cfg.hidden_size, 8);
        assert_eq!(cfg.num_heads, 2);
    }

    #[test]
    fn test_head_dim() {
        let a = Attention::new(16, 4, 0.1).unwrap();
        assert_eq!(a.head_dim(), 4);
    }

    #[test]
    fn test_compute_scaled_dot_product_attention() {
        let a = Attention::new(4, 2, 0.0).unwrap();
        let q = vec![0.5, 0.5, 0.5, 0.5];
        let k = vec![0.3, 0.3, 0.3, 0.3];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let output = a.compute_scaled_dot_product_attention(&q, &k, &v);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_q_proj_has_correct_shape() {
        let a = small_attention();
        assert_eq!(a.q_proj.len(), 8);
        assert_eq!(a.q_proj[0].len(), 8);
    }
}
