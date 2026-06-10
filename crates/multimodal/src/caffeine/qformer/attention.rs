//! Attention mechanisms for Q-Former
//!
//! Multi-head attention and query attention implementations

use crate::caffeine::error::Result;
use ndarray::ArrayD;

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    _hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    _dropout_rate: f32,
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
            _hidden_dim: hidden_dim,
            num_heads,
            head_dim,
            _dropout_rate: dropout_rate,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
        })
    }

    /// Forward pass — GPU-accelerated when available
    pub fn forward(
        &self,
        query: &ArrayD<f32>,
        key: &ArrayD<f32>,
        value: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        let shape = query.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d = shape[2];

        // Try GPU-accelerated path first
        if let Some(result) = self.forward_gpu(query, key, value, batch_size, seq_len, d) {
            return Ok(result);
        }

        // CPU fallback: project Q, K, V for each head
        let mut head_outputs = Vec::new();

        for head in 0..self.num_heads {
            let q_head = self.linear_projection(query, &self.query_weights[head])?;
            let k_head = self.linear_projection(key, &self.key_weights[head])?;
            let v_head = self.linear_projection(value, &self.value_weights[head])?;

            let attention_output = self.scaled_dot_product_attention(&q_head, &k_head, &v_head)?;
            head_outputs.push(attention_output);
        }

        let concatenated = self.concatenate_heads(&head_outputs, batch_size, seq_len)?;
        let output = self.linear_projection(&concatenated, &self.output_weights)?;

        Ok(output)
    }

    /// GPU-accelerated forward path — batched QKV + fused attention
    #[cfg(feature = "gpu")]
    fn forward_gpu(
        &self,
        query: &ArrayD<f32>,
        key: &ArrayD<f32>,
        value: &ArrayD<f32>,
        batch_size: usize,
        seq_len: usize,
        _hidden_dim: usize,
    ) -> Option<ArrayD<f32>> {
        use crate::caffeine::gpu_compute;

        let q_slice = query.as_slice()?;
        let k_slice = key.as_slice()?;
        let v_slice = value.as_slice()?;
        let nh = self.num_heads;
        let dh = self.head_dim;
        let n = batch_size * seq_len;

        // Batch all heads: project Q/K/V per head
        // layout per head: [batch*seq, hidden_dim] @ [hidden_dim, head_dim] = [batch*seq, head_dim]
        let mut q_all = Vec::with_capacity(n * nh * dh);
        let mut k_all = Vec::with_capacity(n * nh * dh);
        let mut v_all = Vec::with_capacity(n * nh * dh);

        for h in 0..nh {
            q_all.extend(gpu_compute::try_gpu_matmul(q_slice, &self.query_weights[h],
                batch_size, seq_len, self._hidden_dim, dh)?);
            k_all.extend(gpu_compute::try_gpu_matmul(k_slice, &self.key_weights[h],
                batch_size, seq_len, self._hidden_dim, dh)?);
            v_all.extend(gpu_compute::try_gpu_matmul(v_slice, &self.value_weights[h],
                batch_size, seq_len, self._hidden_dim, dh)?);
        }

        // Fused attention: [B, H, S, D] layout
        let attended = gpu_compute::try_gpu_attention(
            &q_all, &k_all, &v_all,
            batch_size, seq_len, nh, dh, false,
        )?;

        // attended layout: [B, H, S, D] -> concat heads: [B, S, H*D]
        let mut concat = vec![0.0f32; batch_size * seq_len * nh * dh];
        for b in 0..batch_size {
            for h in 0..nh {
                for s in 0..seq_len {
                    for d in 0..dh {
                        let src_idx = b * nh * seq_len * dh + h * seq_len * dh + s * dh + d;
                        let dst_idx = b * seq_len * nh * dh + s * nh * dh + h * dh + d;
                        concat[dst_idx] = attended[src_idx];
                    }
                }
            }
        }

        // Output projection
        let output = gpu_compute::try_gpu_matmul(
            &concat, &self.output_weights,
            batch_size, seq_len, nh * dh, self._hidden_dim,
        )?;

        let shape = vec![batch_size, seq_len, self._hidden_dim];
        ArrayD::from_shape_vec(shape, output).ok()
    }

    #[cfg(not(feature = "gpu"))]
    fn forward_gpu(
        &self, _query: &ArrayD<f32>, _key: &ArrayD<f32>, _value: &ArrayD<f32>,
        _batch_size: usize, _seq_len: usize, _hidden_dim: usize,
    ) -> Option<ArrayD<f32>> { None }

    /// Scaled dot-product attention — streaming per-query to avoid O(S²) memory
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
        let scale = (head_dim as f32).sqrt().recip();

        let mut output = vec![0.0f32; batch_size * seq_len * head_dim];

        // Process each query position independently — O(S) memory instead of O(S²)
        // FIX 5: Score Matrix Streaming — compute, use, discard
        for b in 0..batch_size {
            for i in 0..seq_len {
                // Compute scores for Q[i] against all K[j]
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];

                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        if let (Some(&q), Some(&k)) = (query.get([b, i, d]), key.get([b, j, d])) {
                            dot += q * k;
                        }
                    }
                    scores[j] = dot * scale;
                    if scores[j] > max_val {
                        max_val = scores[j];
                    }
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for j in 0..seq_len {
                    scores[j] = (scores[j] - max_val).exp();
                    sum_exp += scores[j];
                }
                if sum_exp > 0.0 {
                    let inv = 1.0 / sum_exp;
                    for j in 0..seq_len {
                        scores[j] *= inv;
                    }
                }

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..seq_len {
                        if let Some(&v) = value.get([b, j, d]) {
                            acc += scores[j] * v;
                        }
                    }
                    output[b * seq_len * head_dim + i * head_dim + d] = acc;
                }

                // FIX 6: Early free — scores dropped at end of scope
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
                        if let (Some(&input_val), Some(&weight)) =
                            (input.get([b, i, d]), weights.get(d * output_dim + o))
                        {
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
                            let output_idx = b * seq_len * total_dim
                                + i * total_dim
                                + head_idx * self.head_dim
                                + d;
                            concatenated[output_idx] = val;
                        }
                    }
                }
            }
        }

        let output_shape = vec![batch_size, seq_len, total_dim];
        Ok(ArrayD::from_shape_vec(output_shape, concatenated)?)
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
    _hidden_dim: usize,
    _num_queries: usize,
    attention_weights: Option<ArrayD<f32>>,
}

impl QueryAttention {
    /// Create new query attention
    pub fn new(hidden_dim: usize, _num_queries: usize) -> Self {
        Self {
            _hidden_dim: hidden_dim,
            _num_queries,
            attention_weights: None,
        }
    }

    /// Apply query attention — streaming per-query to avoid O(Nq × Ctx) full storage
    pub fn forward(&mut self, queries: &ArrayD<f32>, context: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let query_shape = queries.shape();
        let context_shape = context.shape();

        let num_queries = query_shape[1];
        let context_len = context_shape[1];

        // Allocate attention weights for visualization (small, Nq × Ctx)
        let mut attention_weights = vec![0.0f32; num_queries * context_len];

        // Streaming output: compute per-query, never store full score matrix
        let mut output = vec![0.0f32; num_queries * self._hidden_dim];

        for i in 0..num_queries {
            // Compute scores: Q[i] @ context[j] for all j
            let mut max_val = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; context_len];

            for j in 0..context_len {
                let mut dot = 0.0f32;
                for d in 0..self._hidden_dim {
                    if let (Some(&q), Some(&c)) = (queries.get([0, i, d]), context.get([0, j, d])) {
                        dot += q * c;
                    }
                }
                scores[j] = dot;
                if scores[j] > max_val {
                    max_val = scores[j];
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for j in 0..context_len {
                scores[j] = (scores[j] - max_val).exp();
                sum_exp += scores[j];
            }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for j in 0..context_len {
                    scores[j] *= inv;
                }
            }

            // Save for visualization
            for j in 0..context_len {
                attention_weights[i * context_len + j] = scores[j];
            }

            // Weighted sum of context
            for d in 0..self._hidden_dim {
                let mut acc = 0.0f32;
                for j in 0..context_len {
                    if let Some(&c) = context.get([0, j, d]) {
                        acc += scores[j] * c;
                    }
                }
                output[i * self._hidden_dim + d] = acc;
            }

            // FIX 6: Early free — scores dropped here
        }

        // Store attention weights for visualization
        let weight_shape = vec![num_queries, context_len];
        self.attention_weights = Some(ArrayD::from_shape_vec(
            weight_shape,
            attention_weights,
        )?);

        let output_shape = vec![1, num_queries, self._hidden_dim];
        Ok(ArrayD::from_shape_vec(output_shape, output)?)
    }

    /// Get attention weights
    pub fn get_attention_weights(&self) -> Option<&ArrayD<f32>> {
        self.attention_weights.as_ref()
    }
}
