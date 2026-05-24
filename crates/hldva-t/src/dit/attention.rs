//! Attention Mechanisms for DiT
//!
//! Implementasi multi-head attention dan cross-attention untuk DiT

use crate::types::*;
use nexora_atqs::Tensor;

/// Multi-Head Attention
pub struct MultiHeadAttention {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,

    // Linear projections
    q_projection: Linear,
    k_projection: Linear,
    v_projection: Linear,
    out_projection: Linear,

    // Dropout
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

    /// Forward pass untuk multi-head attention
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> HLDVAResult<Tensor> {
        // Step 1: Project to Q, K, V
        let q = self.q_projection.forward(query)?;
        let k = self.k_projection.forward(key)?;
        let v = self.v_projection.forward(value)?;

        // Step 2: Reshape untuk multi-head
        let q_heads = self.reshape_to_heads(&q)?;
        let k_heads = self.reshape_to_heads(&k)?;
        let v_heads = self.reshape_to_heads(&v)?;

        // Step 3: Scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads)?;

        // Step 4: Reshape back dan final projection
        let concatenated = self.reshape_from_heads(&attention_output)?;
        let output = self.out_projection.forward(&concatenated)?;

        Ok(output)
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> HLDVAResult<Tensor> {
        #[cfg(feature = "gpu")]
        if let Some(result) = self.scaled_dot_product_attention_gpu(q, k, v) {
            return Ok(result);
        }

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let seq_len = q_data.len() / (self.num_heads * self.head_dim);
        let scale = (self.head_dim as f32).sqrt();

        let mut output = Vec::with_capacity(q_data.len());

        // Process each head
        for head in 0..self.num_heads {
            let head_start = head * self.head_dim * seq_len;
            let _head_end = (head + 1) * self.head_dim * seq_len;

            // Calculate attention scores
            for i in 0..seq_len {
                let q_start = head_start + i * self.head_dim;
                let q_end = q_start + self.head_dim;

                let mut attention_scores = Vec::with_capacity(seq_len);

                // Score dengan semua keys
                for j in 0..seq_len {
                    let k_start = head_start + j * self.head_dim;
                    let k_end = k_start + self.head_dim;

                    let mut score = 0.0;
                    for (q_idx, k_idx) in (q_start..q_end).zip(k_start..k_end) {
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    attention_scores.push(score / scale);
                }

                // Softmax
                let softmax_scores = self.softmax(&attention_scores);

                // Weighted sum dari values
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

    /// Softmax function
    fn softmax(&self, scores: &[f32]) -> Vec<f32> {
        // Find max untuk numerical stability
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Exponential dan sum
        let exp_scores: Vec<f32> = scores.iter().map(|&x| (x - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        // Normalize
        exp_scores.iter().map(|&x| x / sum_exp).collect()
    }

    /// GPU-accelerated scaled dot-product attention
    #[cfg(feature = "gpu")]
    fn scaled_dot_product_attention_gpu(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Option<Tensor> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        use ndarray::ArrayD;

        let ctx = GpuContext::global().ok()?;
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let seq_len = q_data.len() / (self.num_heads * self.head_dim);
        let scale = (self.head_dim as f32).sqrt();

        let mut output = Vec::with_capacity(q_data.len());

        for head in 0..self.num_heads {
            let offset = head * seq_len * self.head_dim;

            // Extract Q_h, K_h, V_h as [seq_len, head_dim]
            let q_slice: Vec<f32> = q_data[offset..offset + seq_len * self.head_dim].to_vec();
            let k_slice: Vec<f32> = k_data[offset..offset + seq_len * self.head_dim].to_vec();
            let v_slice: Vec<f32> = v_data[offset..offset + seq_len * self.head_dim].to_vec();

            let q_gpu = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![seq_len, self.head_dim], q_slice).ok()?
            ).ok()?;
            let k_gpu = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![seq_len, self.head_dim], k_slice).ok()?
            ).ok()?;
            let v_gpu = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![seq_len, self.head_dim], v_slice).ok()?
            ).ok()?;

            // scores = Q @ K^T / scale: [seq_len, seq_len]
            let kt_gpu = ctx.transpose(&k_gpu).ok()?;
            let mut scores_gpu = ctx.matmul(&q_gpu, &kt_gpu).ok()?;

            // Scale scores
            let scale_factor = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![1], vec![1.0 / scale]).ok()?
            ).ok()?;
            scores_gpu = ctx.mul(&scores_gpu, &scale_factor).ok()?;

            // Softmax on CPU
            let scores_cpu = scores_gpu.to_cpu().ok()?;
            let scores_vec: Vec<f32> = scores_cpu.iter().copied().collect();
            let softmax_vec = self.softmax(&scores_vec);

            // attention = softmax(scores) @ V: [seq_len, head_dim]
            let softmax_gpu = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![seq_len, seq_len], softmax_vec).ok()?
            ).ok()?;
            let attn_gpu = ctx.matmul(&softmax_gpu, &v_gpu).ok()?;

            let attn_cpu = attn_gpu.to_cpu().ok()?;
            output.extend(attn_cpu.iter().copied());
        }

        Some(Tensor::new(output, q.shape().to_vec()))
    }

    /// Reshape tensor ke multi-head format
    fn reshape_to_heads(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        let data = tensor.data();
        let seq_len = data.len() / self.hidden_dim;

        let mut reshaped = Vec::with_capacity(data.len());

        for head in 0..self.num_heads {
            for pos in 0..seq_len {
                for dim in 0..self.head_dim {
                    let idx = pos * self.hidden_dim + head * self.head_dim + dim;
                    if idx < data.len() {
                        reshaped.push(data[idx]);
                    }
                }
            }
        }

        Ok(Tensor::new(reshaped, tensor.shape().to_vec()))
    }

    /// Reshape dari multi-head format
    fn reshape_from_heads(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        let data = tensor.data();
        let seq_len = data.len() / (self.num_heads * self.head_dim);

        let mut reshaped = Vec::with_capacity(data.len());

        for pos in 0..seq_len {
            for head in 0..self.num_heads {
                for dim in 0..self.head_dim {
                    let idx = head * seq_len * self.head_dim + pos * self.head_dim + dim;
                    if idx < data.len() {
                        reshaped.push(data[idx]);
                    }
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

    // Projections
    q_projection: Linear,
    k_projection: Linear,
    v_projection: Linear,
    out_projection: Linear,

    // Untuk CLIP conditioning
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

    /// Forward pass untuk cross-attention
    pub fn forward(&self, hidden: &Tensor, conditioning: &Tensor) -> HLDVAResult<Tensor> {
        // Project conditioning untuk cross-attention
        let processed_conditioning = self.conditioning_projection.forward(conditioning)?;

        // Standard multi-head attention dengan conditioning sebagai K dan V
        let attention = MultiHeadAttention::new(self.hidden_dim, self.num_heads)?;
        attention.forward(hidden, &processed_conditioning, &processed_conditioning)
    }
}

/// Linear Layer
pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> HLDVAResult<Self> {
        // Initialize weight dengan Xavier initialization
        let weight_data = Self::xavier_init(in_features, out_features);
        let weight = Tensor::new(weight_data, vec![out_features, in_features]);

        let bias_data = vec![0.0; out_features];
        let bias = Some(Tensor::new(bias_data, vec![out_features]));

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        #[cfg(feature = "gpu")]
        if let Some(result) = self.forward_gpu(input) {
            return Ok(result);
        }

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

            // Add bias jika ada
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

    /// GPU-accelerated forward pass
    #[cfg(feature = "gpu")]
    fn forward_gpu(&self, input: &Tensor) -> Option<Tensor> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        use ndarray::ArrayD;

        let ctx = GpuContext::global().ok()?;
        let input_data = input.data();

        let input_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, self.in_features], input_data.to_vec()).ok()?
        ).ok()?;

        let w_data = self.weight.data();
        let w_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![self.out_features, self.in_features], w_data.to_vec()).ok()?
        ).ok()?;
        let wt_gpu = ctx.transpose(&w_gpu).ok()?;

        let mut out_gpu = ctx.matmul(&input_gpu, &wt_gpu).ok()?;

        if let Some(ref bias) = self.bias {
            let bias_gpu = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![1, self.out_features], bias.data().to_vec()).ok()?
            ).ok()?;
            out_gpu = ctx.add(&out_gpu, &bias_gpu).ok()?;
        }

        let out_cpu = out_gpu.to_cpu().ok()?;
        let result: Vec<f32> = out_cpu.iter().copied().collect();
        Some(Tensor::new(result, vec![self.out_features]))
    }

    /// Xavier initialization
    fn xavier_init(in_features: usize, out_features: usize) -> Vec<f32> {
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut weights = Vec::with_capacity(in_features * out_features);

        for _ in 0..(in_features * out_features) {
            let weight = rand::random::<f32>() * 2.0 * limit - limit;
            weights.push(weight);
        }

        weights
    }
}

/// Layer Normalization
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

        Ok(Self {
            _hidden_dim: hidden_dim,
            weight,
            bias,
            eps: 1e-6,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();

        // Calculate mean
        let mean: f32 = input_data.iter().sum::<f32>() / input_data.len() as f32;

        // Calculate variance
        let variance: f32 =
            input_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input_data.len() as f32;

        // Normalize
        let mut output = Vec::with_capacity(input_data.len());
        for (i, &x) in input_data.iter().enumerate() {
            let normalized = (x - mean) / (variance + self.eps).sqrt();
            let weighted =
                normalized * weight_data[i % weight_data.len()] + bias_data[i % bias_data.len()];
            output.push(weighted);
        }

        Ok(Tensor::new(output, input.shape().to_vec()))
    }
}

/// Feed-Forward Network
pub struct FeedForward {
    hidden_dim: usize,
    intermediate_dim: usize,

    linear1: Linear,
    linear2: Linear,
    activation: GELU,
}

impl FeedForward {
    pub fn new(hidden_dim: usize) -> HLDVAResult<Self> {
        let intermediate_dim = hidden_dim * 4; // Standard 4x expansion

        let linear1 = Linear::new(hidden_dim, intermediate_dim)?;
        let linear2 = Linear::new(intermediate_dim, hidden_dim)?;

        Ok(Self {
            hidden_dim,
            intermediate_dim,
            linear1,
            linear2,
            activation: GELU,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let hidden = self.linear1.forward(input)?;
        let activated = self.activation.forward(&hidden)?;
        let output = self.linear2.forward(&activated)?;

        Ok(output)
    }
}

/// GELU Activation
pub struct GELU;

impl GELU {
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let mut output = Vec::with_capacity(input_data.len());

        for &x in input_data {
            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let x3 = x * x * x;
            let tanh_arg = 0.7978845608 * (x + 0.044715 * x3); // sqrt(2/pi) ≈ 0.7978845608
            let tanh_val = tanh_arg.tanh();
            let gelu_val = 0.5 * x * (1.0 + tanh_val);
            output.push(gelu_val);
        }

        Ok(Tensor::new(output, input.shape().to_vec()))
    }
}
