//! Attention Mechanisms for DiT
//!
//! GPU-accelerated multi-head attention, cross-attention,
//! linear projections, layer norm, and activations.

use crate::gpu_ops;
use crate::types::*;
use nexora_atqs::Tensor;

/// Multi-Head Attention
pub struct MultiHeadAttention {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,

    q_projection: Linear,
    k_projection: Linear,
    v_projection: Linear,
    out_projection: Linear,

    _dropout: f32,
}

impl MultiHeadAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> HLDVAResult<Self> {
        let head_dim = hidden_dim / num_heads;

        let q_projection = Linear::new(hidden_dim, hidden_dim)?;
        let k_projection = Linear::new(hidden_dim, hidden_dim)?;
        let v_projection = Linear::new(hidden_dim, hidden_dim)?;
        let out_projection = Linear::new(hidden_dim, hidden_dim)?;

        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim,
            q_projection,
            k_projection,
            v_projection,
            out_projection,
            _dropout: 0.1,
        })
    }

    pub fn q_projection(&self) -> &Linear { &self.q_projection }
    pub fn k_projection(&self) -> &Linear { &self.k_projection }
    pub fn v_projection(&self) -> &Linear { &self.v_projection }
    pub fn out_projection(&self) -> &Linear { &self.out_projection }

    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> HLDVAResult<Tensor> {
        let q = self.q_projection.forward(query)?;
        let k = self.k_projection.forward(key)?;
        let v = self.v_projection.forward(value)?;

        let q_heads = self.reshape_to_heads(&q)?;
        let k_heads = self.reshape_to_heads(&k)?;
        let v_heads = self.reshape_to_heads(&v)?;

        let attention_output = self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads)?;

        let concatenated = self.reshape_from_heads(&attention_output)?;
        let output = self.out_projection.forward(&concatenated)?;

        Ok(output)
    }

    /// GPU-accelerated scaled dot-product attention via fused_attention.
    /// Falls back to CPU if GPU unavailable.
    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return self.scaled_dot_product_attention_gpu(q, k, v);
        }
        self.scaled_dot_product_attention_cpu(q, k, v)
    }

    /// GPU fused attention — all heads batched in one call
    fn scaled_dot_product_attention_gpu(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> HLDVAResult<Tensor> {
        let q_data = q.data();
        let seq_len = q_data.len() / (self.num_heads * self.head_dim);
        let scale = (self.head_dim as f32).sqrt();

        // Check seq_len — fused_attention on GPU needs at least 1 token and handles it
        if seq_len == 0 {
            return Ok(q.clone());
        }

        // Reshape to [B=1, H, S, D]
        let q_4d = Tensor::new(q_data.to_vec(), vec![1, self.num_heads, seq_len, self.head_dim]);
        let k_4d = Tensor::new(k.data().to_vec(), vec![1, self.num_heads, seq_len, self.head_dim]);
        let v_4d = Tensor::new(v.data().to_vec(), vec![1, self.num_heads, seq_len, self.head_dim]);

        let out = gpu_ops::gpu_fused_attention(&q_4d, &k_4d, &v_4d, 1.0 / scale, false)?;

        // Reshape back to [num_heads, seq_len, head_dim]
        let out_data = out.data();
        Ok(Tensor::new(out_data.to_vec(), q.shape().to_vec()))
    }

    /// CPU fallback attention (per-head dot-product)
    fn scaled_dot_product_attention_cpu(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> HLDVAResult<Tensor> {
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let seq_len = q_data.len() / (self.num_heads * self.head_dim);
        let scale = (self.head_dim as f32).sqrt();

        let mut output = Vec::with_capacity(q_data.len());

        for head in 0..self.num_heads {
            let head_start = head * self.head_dim * seq_len;

            for i in 0..seq_len {
                let q_start = head_start + i * self.head_dim;
                let q_end = q_start + self.head_dim;

                let mut attention_scores = Vec::with_capacity(seq_len);
                for j in 0..seq_len {
                    let k_start = head_start + j * self.head_dim;
                    let k_end = k_start + self.head_dim;

                    let mut score = 0.0;
                    for (q_idx, k_idx) in (q_start..q_end).zip(k_start..k_end) {
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    attention_scores.push(score / scale);
                }

                let softmax_scores = self.softmax_cpu(&attention_scores);

                for dim in 0..self.head_dim {
                    let mut weighted_sum = 0.0;
                    for (j, &score) in softmax_scores.iter().enumerate() {
                        let v_idx = head_start + j * self.head_dim + dim;
                        weighted_sum += score * v_data[v_idx];
                    }
                    output.push(weighted_sum);
                }
            }
        }

        Ok(Tensor::new(output, q.shape().to_vec()))
    }

    fn softmax_cpu(&self, scores: &[f32]) -> Vec<f32> {
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = scores.iter().map(|&x| (x - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        exp_scores.iter().map(|&x| x / sum_exp).collect()
    }

    fn reshape_to_heads(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        let data = tensor.data();
        let seq_len = data.len() / self.hidden_dim;

        let mut reshaped = Vec::with_capacity(data.len());
        for head in 0..self.num_heads {
            for pos in 0..seq_len {
                for dim in 0..self.head_dim {
                    let idx = pos * self.hidden_dim + head * self.head_dim + dim;
                    reshaped.push(if idx < data.len() { data[idx] } else { 0.0 });
                }
            }
        }
        Ok(Tensor::new(reshaped, tensor.shape().to_vec()))
    }

    fn reshape_from_heads(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        let data = tensor.data();
        let seq_len = data.len() / (self.num_heads * self.head_dim);

        let mut reshaped = Vec::with_capacity(data.len());
        for pos in 0..seq_len {
            for head in 0..self.num_heads {
                for dim in 0..self.head_dim {
                    let idx = head * seq_len * self.head_dim + pos * self.head_dim + dim;
                    reshaped.push(if idx < data.len() { data[idx] } else { 0.0 });
                }
            }
        }
        Ok(Tensor::new(reshaped, tensor.shape().to_vec()))
    }
}

/// Cross-Attention untuk CLIP conditioning
pub struct CrossAttention {
    hidden_dim: usize,
    num_heads: usize,
    q_projection: Linear,
    k_projection: Linear,
    v_projection: Linear,
    out_projection: Linear,
    conditioning_projection: Linear,
}

impl CrossAttention {
    pub fn new(hidden_dim: usize, num_heads: usize, conditioning_dim: usize) -> HLDVAResult<Self> {
        let q_projection = Linear::new(hidden_dim, hidden_dim)?;
        let k_projection = Linear::new(conditioning_dim, hidden_dim)?;
        let v_projection = Linear::new(conditioning_dim, hidden_dim)?;
        let out_projection = Linear::new(hidden_dim, hidden_dim)?;
        let conditioning_projection = Linear::new(conditioning_dim, hidden_dim)?;

        Ok(Self {
            hidden_dim,
            num_heads,
            q_projection,
            k_projection,
            v_projection,
            out_projection,
            conditioning_projection,
        })
    }

    pub fn forward(&self, hidden: &Tensor, conditioning: &Tensor) -> HLDVAResult<Tensor> {
        let processed_conditioning = self.conditioning_projection.forward(conditioning)?;
        let attention = MultiHeadAttention::new(self.hidden_dim, self.num_heads)?;
        attention.forward(hidden, &processed_conditioning, &processed_conditioning)
    }
}

/// GPU-accelerated Linear Layer
pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> HLDVAResult<Self> {
        let weight_data = Self::xavier_init(in_features, out_features);
        let weight = Tensor::new(weight_data, vec![out_features, in_features]);
        let bias = Some(Tensor::new(vec![0.0; out_features], vec![out_features]));

        Ok(Self { in_features, out_features, weight, bias })
    }

    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn weight_mut(&mut self) -> &mut Tensor { &mut self.weight }
    pub fn bias(&self) -> Option<&Tensor> { self.bias.as_ref() }
    pub fn bias_mut(&mut self) -> Option<&mut Tensor> { self.bias.as_mut() }

    /// Forward pass: y = x @ W^T + b
    /// Uses GPU matmul when available, CPU fallback otherwise.
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return self.forward_gpu(input);
        }
        self.forward_cpu(input)
    }

    /// GPU forward: upload input → matmul on GPU → add bias on GPU → readback
    fn forward_gpu(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let input_2d = if input_data.len() == self.in_features {
            Tensor::new(input_data.to_vec(), vec![1, self.in_features])
        } else {
            input.clone()
        };

        // W is [out, in], we need input @ W^T → result is [batch, out]
        // Use GPU matmul: input [1, in] × W^T [in, out] → [1, out]
        // For higher dims, reshape first
        let w_data = self.weight.data();
        let w_t_data: Vec<f32> = if self.weight.shape() == vec![self.out_features, self.in_features] {
            // Transpose on CPU then upload
            let mut w_t = Vec::with_capacity(self.in_features * self.out_features);
            for i in 0..self.in_features {
                for o in 0..self.out_features {
                    w_t.push(w_data[o * self.in_features + i]);
                }
            }
            w_t
        } else {
            w_data.to_vec()
        };

        let w_t = Tensor::new(w_t_data, vec![self.in_features, self.out_features]);
        let result = gpu_ops::gpu_matmul(&input_2d, &w_t)?;

        if let Some(ref bias) = self.bias {
            let mut result_data = result.data().to_vec();
            let bias_data = bias.data();
            for i in 0..result_data.len() {
                let b_idx = i % self.out_features;
                if b_idx < bias_data.len() {
                    result_data[i] += bias_data[b_idx];
                }
            }
            Ok(Tensor::new(result_data, vec![self.out_features]))
        } else {
            Ok(result)
        }
    }

    /// CPU fallback forward
    fn forward_cpu(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let weight_data = self.weight.data();

        let mut output = Vec::with_capacity(self.out_features);

        for out_idx in 0..self.out_features {
            let mut sum = 0.0;
            for in_idx in 0..self.in_features {
                if in_idx < input_data.len() {
                    let weight_idx = out_idx * self.in_features + in_idx;
                    if weight_idx < weight_data.len() {
                        sum += input_data[in_idx] * weight_data[weight_idx];
                    }
                }
            }
            if let Some(ref bias) = self.bias {
                let bias_data = bias.data();
                if out_idx < bias_data.len() {
                    sum += bias_data[out_idx];
                }
            }
            output.push(sum);
        }

        Ok(Tensor::new(output, vec![self.out_features]))
    }

    fn xavier_init(in_features: usize, out_features: usize) -> Vec<f32> {
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        (0..in_features * out_features)
            .map(|_| rand::random::<f32>() * 2.0 * limit - limit)
            .collect()
    }
}

/// GPU-accelerated Layer Normalization
pub struct LayerNorm {
    _hidden_dim: usize,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(hidden_dim: usize) -> HLDVAResult<Self> {
        let weight = Tensor::new(vec![1.0; hidden_dim], vec![hidden_dim]);
        let bias = Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]);
        Ok(Self { _hidden_dim: hidden_dim, weight, bias, eps: 1e-6 })
    }

    /// GPU layer_norm when available, CPU fallback
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_layer_norm(input, &self.weight, &self.bias, self.eps);
        }
        self.forward_cpu(input)
    }

    fn forward_cpu(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();

        let mean: f32 = input_data.iter().sum::<f32>() / input_data.len() as f32;
        let variance: f32 = input_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input_data.len() as f32;

        let mut output = Vec::with_capacity(input_data.len());
        for (i, &x) in input_data.iter().enumerate() {
            let normalized = (x - mean) / (variance + self.eps).sqrt();
            let w = weight_data[i % weight_data.len()];
            let b = bias_data[i % bias_data.len()];
            output.push(normalized * w + b);
        }

        Ok(Tensor::new(output, input.shape().to_vec()))
    }
}

/// GPU-accelerated Feed-Forward Network
pub struct FeedForward {
    hidden_dim: usize,
    intermediate_dim: usize,
    linear1: Linear,
    linear2: Linear,
    activation: GELU,
}

impl FeedForward {
    pub fn new(hidden_dim: usize) -> HLDVAResult<Self> {
        let intermediate_dim = hidden_dim * 4;
        let linear1 = Linear::new(hidden_dim, intermediate_dim)?;
        let linear2 = Linear::new(intermediate_dim, hidden_dim)?;
        Ok(Self { hidden_dim, intermediate_dim, linear1, linear2, activation: GELU })
    }

    pub fn linear1(&self) -> &Linear { &self.linear1 }
    pub fn linear2(&self) -> &Linear { &self.linear2 }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let hidden = self.linear1.forward(input)?;
        let activated = self.activation.forward(&hidden)?;
        self.linear2.forward(&activated)
    }
}

/// GELU Activation — GPU accelerated
pub struct GELU;

impl GELU {
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_gelu(input);
        }
        self.forward_cpu(input)
    }

    fn forward_cpu(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let mut output = Vec::with_capacity(input_data.len());
        for &x in input_data {
            let x3 = x * x * x;
            let tanh_arg = 0.7978845608 * (x + 0.044715 * x3);
            let gelu_val = 0.5 * x * (1.0 + tanh_arg.tanh());
            output.push(gelu_val);
        }
        Ok(Tensor::new(output, input.shape().to_vec()))
    }
}
