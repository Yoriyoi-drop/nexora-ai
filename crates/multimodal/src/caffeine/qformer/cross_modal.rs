//! Cross-modal attention mechanisms for Q-Former
//!
//! Implements attention between different modalities

use crate::caffeine::error::Result;
use ndarray::ArrayD;
use std::collections::HashMap;

/// Cross-modal attention mechanism with proper Q/K/V projections
pub struct CrossModalAttention {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    _dropout_rate: f32,
    pub(crate) q_proj: Vec<f32>,
    pub(crate) k_proj: Vec<f32>,
    pub(crate) v_proj: Vec<f32>,
    pub(crate) o_proj: Vec<f32>,
    projection_weights: HashMap<String, Vec<f32>>,
}

impl CrossModalAttention {
    /// Create new cross-modal attention with proper Q/K/V projections
    pub fn new(hidden_dim: usize, num_heads: usize, dropout_rate: f32) -> Result<Self> {
        if hidden_dim == 0 || num_heads == 0 {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "Hidden dimension and number of heads must be greater than 0",
            ));
        }

        if hidden_dim % num_heads != 0 {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "Hidden dimension must be divisible by number of heads",
            ));
        }

        let head_dim = hidden_dim / num_heads;
        let scale = (2.0 / hidden_dim as f32).sqrt();

        let init_proj = |rows: usize, cols: usize| -> Vec<f32> {
            let size = rows * cols;
            (0..size).map(|_| rand::random::<f32>() * scale - scale / 2.0).collect()
        };

        let q_proj = init_proj(hidden_dim, hidden_dim);
        let k_proj = init_proj(hidden_dim, hidden_dim);
        let v_proj = init_proj(hidden_dim, hidden_dim);
        let o_proj = init_proj(hidden_dim, hidden_dim);

        // Initialize projection weights for different modality pairs
        let mut projection_weights = HashMap::new();
        let modality_pairs = vec![
            ("image_text", hidden_dim * hidden_dim),
            ("audio_text", hidden_dim * hidden_dim),
            ("video_text", hidden_dim * hidden_dim),
            ("image_audio", hidden_dim * hidden_dim),
            ("image_video", hidden_dim * hidden_dim),
            ("audio_video", hidden_dim * hidden_dim),
        ];

        for (pair_name, size) in modality_pairs {
            let weights = Self::xavier_weights(size, hidden_dim);
            projection_weights.insert(pair_name.to_string(), weights);
        }

        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim,
            _dropout_rate: dropout_rate,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            projection_weights,
        })
    }

    fn xavier_weights(size: usize, hidden_dim: usize) -> Vec<f32> {
        let scale = (2.0 / hidden_dim as f32).sqrt();
        (0..size).map(|_| rand::random::<f32>() * scale - scale / 2.0).collect()
    }

    /// Project input through weight matrix: output[b,i,o] = sum_d input[b,i,d] * W[d,o]
    /// Tries GPU first, falls back to CPU
    fn project(&self, input: &[f32], weight: &[f32], n: usize, d_in: usize, d_out: usize) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(result) = self.project_gpu(input, weight, n, d_in, d_out) {
            return result;
        }

        let mut out = vec![0.0f32; n * d_out];
        for i in 0..n {
            for o in 0..d_out {
                let mut s = 0.0f32;
                for d in 0..d_in {
                    s += input[i * d_in + d] * weight[d * d_out + o];
                }
                out[i * d_out + o] = s;
            }
        }
        out
    }

    /// GPU-accelerated projection
    #[cfg(feature = "gpu")]
    fn project_gpu(&self, input: &[f32], weight: &[f32], n: usize, _d_in: usize, d_out: usize) -> Option<Vec<f32>> {
        use crate::caffeine::gpu_compute;
        gpu_compute::try_gpu_matmul(input, weight, 1, n, _d_in, d_out)
    }

    /// Compute cross-modal self-attention — GPU-accelerated when available
    pub fn compute_cross_attention(
        &self,
        features: &[f32],
        num_queries: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        if features.len() != num_queries * hidden_dim {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "Feature dimensions don't match expected query dimensions",
            ));
        }

        let (n, d) = (num_queries, hidden_dim);
        let nh = self.num_heads;
        let dh = self.head_dim;

        // Try GPU-accelerated path first
        #[cfg(feature = "gpu")]
        if let Some(result) = self.compute_cross_attention_gpu(features, n, d, nh, dh) {
            return Ok(result);
        }

        // CPU fallback
        let q = self.project(features, &self.q_proj, n, d, d);
        let k = self.project(features, &self.k_proj, n, d, d);
        let v = self.project(features, &self.v_proj, n, d, d);

        let mut head_outputs = Vec::with_capacity(nh * n * dh);

        for h in 0..nh {
            let hd = h * dh;
            let qh = &q[hd * n..][..n * dh];
            let kh = &k[hd * n..][..n * dh];
            let vh = &v[hd * n..][..n * dh];

            let scale = (dh as f32).sqrt().recip();
            let mut attn = vec![0.0f32; n * n];

            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0f32;
                    for d in 0..dh {
                        dot += qh[i * dh + d] * kh[j * dh + d];
                    }
                    attn[i * n + j] = dot * scale;
                }
            }

            for i in 0..n {
                let row_start = i * n;
                let mut max_val = attn[row_start];
                for j in 1..n { max_val = max_val.max(attn[row_start + j]); }
                let mut sum_exp = 0.0f32;
                for j in 0..n {
                    let e = (attn[row_start + j] - max_val).exp();
                    attn[row_start + j] = e;
                    sum_exp += e;
                }
                if sum_exp > 0.0 {
                    let inv = 1.0 / sum_exp;
                    for j in 0..n { attn[row_start + j] *= inv; }
                }
            }

            for i in 0..n {
                for d in 0..dh {
                    let mut acc = 0.0f32;
                    for j in 0..n {
                        acc += attn[i * n + j] * vh[j * dh + d];
                    }
                    head_outputs.push(acc);
                }
            }
        }

        let mut concat = vec![0.0f32; n * d];
        for h in 0..nh {
            let h_offset = h * n * dh;
            for i in 0..n {
                for d in 0..dh {
                    concat[i * d + h * dh + d] = head_outputs[h_offset + i * dh + d];
                }
            }
        }

        let result = self.project(&concat, &self.o_proj, n, d, d);
        Ok(result)
    }

    /// GPU-accelerated cross-attention path
    #[cfg(feature = "gpu")]
    fn compute_cross_attention_gpu(
        &self,
        features: &[f32],
        n: usize,
        d: usize,
        nh: usize,
        dh: usize,
    ) -> Option<Vec<f32>> {
        use crate::caffeine::gpu_compute;

        // Project Q, K, V via GPU matmul — submit all 3 asynchronously so the GPU
        // can pipeline them without CPU blocking between submissions.
        let q_async = gpu_compute::try_gpu_matmul_async(features, &self.q_proj, 1, n, d, d)?;
        let k_async = gpu_compute::try_gpu_matmul_async(features, &self.k_proj, 1, n, d, d)?;
        let v_async = gpu_compute::try_gpu_matmul_async(features, &self.v_proj, 1, n, d, d)?;
        let q = q_async.recv().ok()?;
        let k = k_async.recv().ok()?;
        let v = v_async.recv().ok()?;

        // Reshape Q/K/V from [1, n, d] -> [1, nh, n, dh] per-head layout
        let mut q_4d = vec![0.0f32; nh * n * dh];
        let mut k_4d = vec![0.0f32; nh * n * dh];
        let mut v_4d = vec![0.0f32; nh * n * dh];
        for h in 0..nh {
            for i in 0..n {
                for dd in 0..dh {
                    let src = i * d + h * dh + dd;
                    let dst = h * n * dh + i * dh + dd;
                    q_4d[dst] = q[src];
                    k_4d[dst] = k[src];
                    v_4d[dst] = v[src];
                }
            }
        }

        // Fused attention (non-causal self-attention)
        let attended = gpu_compute::try_gpu_attention(&q_4d, &k_4d, &v_4d, 1, n, nh, dh, false)?;

        // attended layout: [1, nh, n, dh] -> concat heads: [n, d]
        let mut concat = vec![0.0f32; n * d];
        for h in 0..nh {
            for i in 0..n {
                for dd in 0..dh {
                    let src = h * n * dh + i * dh + dd;
                    let dst = i * d + h * dh + dd;
                    concat[dst] = attended[src];
                }
            }
        }

        // Output projection via GPU
        gpu_compute::try_gpu_matmul(&concat, &self.o_proj, 1, n, d, d)
    }

    /// Apply modality-specific projection with optimized computation
    pub fn apply_modality_projection(
        &self,
        features: &ArrayD<f32>,
        from_modality: &str,
        to_modality: &str,
    ) -> Result<ArrayD<f32>> {
        let projection_key = format!("{}_{}", from_modality, to_modality);

        if let Some(weights) = self.projection_weights.get(&projection_key) {
            let shape = features.shape();
            let batch_size = shape[0];
            let seq_len = shape[1];
            let input_dim = shape[2];
            let output_dim = self.hidden_dim;

            // Validate dimensions
            if input_dim * output_dim != weights.len() {
                return Err(crate::caffeine::error::CaffeineError::qformer(&format!(
                    "Weight dimensions {} don't match expected {}x{}",
                    weights.len(),
                    input_dim,
                    output_dim
                )));
            }

            // Optimized projection computation
            let projected = self.compute_projection_optimized(
                features, weights, batch_size, seq_len, input_dim, output_dim,
            )?;

            let output_shape = vec![batch_size, seq_len, output_dim];
            Ok(ArrayD::from_shape_vec(output_shape, projected)?)
        } else {
            Err(crate::caffeine::error::CaffeineError::qformer(&format!(
                "No projection weights found for {} -> {}",
                from_modality, to_modality
            )))
        }
    }

    /// Optimized projection computation — GPU-accelerated when available
    fn compute_projection_optimized(
        &self,
        features: &ArrayD<f32>,
        weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        if let Some(result) = self.compute_projection_gpu(
            features, weights, batch_size, seq_len, input_dim, output_dim,
        ) {
            return Ok(result);
        }

        let mut projected = vec![0.0f32; batch_size * seq_len * output_dim];

        let weight_matrix: Vec<&[f32]> = (0..output_dim)
            .map(|o| &weights[o * input_dim..(o + 1) * input_dim])
            .collect();

        for b in 0..batch_size {
            for i in 0..seq_len {
                for o in 0..output_dim {
                    let output_idx = b * seq_len * output_dim + i * output_dim + o;
                    let weight_row = weight_matrix[o];

                    let mut sum = 0.0f32;
                    for d in 0..input_dim {
                        if let Some(&input_val) = features.get([b, i, d]) {
                            sum += input_val * weight_row[d];
                        }
                    }

                    projected[output_idx] = sum;
                }
            }
        }

        Ok(projected)
    }

    /// GPU-accelerated projection
    #[cfg(feature = "gpu")]
    fn compute_projection_gpu(
        &self,
        features: &ArrayD<f32>,
        weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Option<Vec<f32>> {
        use crate::caffeine::gpu_compute;
        let input_slice = features.as_slice()?;
        gpu_compute::try_gpu_matmul(
            input_slice, weights,
            batch_size, seq_len, input_dim, output_dim,
        )
    }

    /// Compute modality fusion weights
    pub fn compute_fusion_weights(
        &self,
        modality_features: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Result<std::collections::HashMap<String, f32>> {
        let mut fusion_weights = std::collections::HashMap::new();

        // Compute importance scores for each modality
        for (modality, features) in modality_features {
            let importance_score = self.compute_modality_importance(features)?;
            fusion_weights.insert(modality.clone(), importance_score);
        }

        // Normalize weights
        let total_weight: f32 = fusion_weights.values().sum();
        if total_weight > 0.0 {
            for weight in fusion_weights.values_mut() {
                *weight /= total_weight;
            }
        }

        Ok(fusion_weights)
    }

    /// Compute importance score for a modality
    fn compute_modality_importance(&self, features: &ArrayD<f32>) -> Result<f32> {
        let shape = features.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];

        let mut total_activation = 0.0f32;
        let mut count = 0.0f32;

        for b in 0..batch_size {
            for i in 0..seq_len {
                for d in 0..embed_dim {
                    if let Some(&val) = features.get([b, i, d]) {
                        total_activation += val.abs();
                        count += 1.0;
                    }
                }
            }
        }

        Ok(if count > 0.0 {
            total_activation / count
        } else {
            0.0
        })
    }

    /// Apply gated fusion
    pub fn gated_fusion(
        &self,
        modality_features: &std::collections::HashMap<String, ArrayD<f32>>,
        fusion_weights: &std::collections::HashMap<String, f32>,
    ) -> Result<ArrayD<f32>> {
        if modality_features.is_empty() {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "No modality features provided for fusion",
            ));
        }

        // Get reference shape from first modality
        let first_features = modality_features.values().next().ok_or_else(|| {
            crate::caffeine::error::CaffeineError::input_validation(
                "No modality features available",
            )
        })?;
        let shape = first_features.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];

        let mut fused = vec![0.0f32; batch_size * seq_len * embed_dim];

        // Fuse modalities with learned weights
        for (modality, features) in modality_features {
            let weight = fusion_weights.get(modality).unwrap_or(&0.0);

            for b in 0..batch_size {
                for i in 0..seq_len {
                    for d in 0..embed_dim {
                        if let Some(&val) = features.get([b, i, d]) {
                            let idx = b * seq_len * embed_dim + i * embed_dim + d;
                            fused[idx] += val * weight;
                        }
                    }
                }
            }
        }

        let output_shape = vec![batch_size, seq_len, embed_dim];
        Ok(ArrayD::from_shape_vec(output_shape, fused)?)
    }
}

/// Modality alignment module
pub struct ModalityAlignment {
    alignment_methods: std::collections::HashMap<String, AlignmentMethod>,
}

impl ModalityAlignment {
    /// Create new modality alignment module
    pub fn new() -> Self {
        let mut alignment_methods = std::collections::HashMap::new();

        alignment_methods.insert("contrastive".to_string(), AlignmentMethod::Contrastive);
        alignment_methods.insert("attention".to_string(), AlignmentMethod::Attention);
        alignment_methods.insert("fusion".to_string(), AlignmentMethod::Fusion);

        Self { alignment_methods }
    }

    /// Align two modalities
    pub fn align_modalities(
        &self,
        modality_a: &ArrayD<f32>,
        modality_b: &ArrayD<f32>,
        method: &str,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        if let Some(alignment_method) = self.alignment_methods.get(method) {
            match alignment_method {
                AlignmentMethod::Contrastive => self.contrastive_alignment(modality_a, modality_b),
                AlignmentMethod::Attention => self.attention_alignment(modality_a, modality_b),
                AlignmentMethod::Fusion => self.fusion_alignment(modality_a, modality_b),
            }
        } else {
            Err(crate::caffeine::error::CaffeineError::qformer(&format!(
                "Unknown alignment method: {}",
                method
            )))
        }
    }

    /// Contrastive alignment
    fn contrastive_alignment(
        &self,
        modality_a: &ArrayD<f32>,
        modality_b: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        // Normalize features
        let normalized_a = self.l2_normalize(modality_a)?;
        let normalized_b = self.l2_normalize(modality_b)?;

        Ok((normalized_a, normalized_b))
    }

    /// Attention-based alignment
    fn attention_alignment(
        &self,
        modality_a: &ArrayD<f32>,
        modality_b: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        // Apply cross-attention between modalities
        let attended_a = self.cross_attention(modality_a, modality_b)?;
        let attended_b = self.cross_attention(modality_b, modality_a)?;

        Ok((attended_a, attended_b))
    }

    /// Fusion-based alignment
    fn fusion_alignment(
        &self,
        modality_a: &ArrayD<f32>,
        modality_b: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        // Simple averaging fusion
        let shape = modality_a.shape();
        let mut fused_a = vec![0.0f32; modality_a.len()];
        let mut fused_b = vec![0.0f32; modality_b.len()];

        for i in 0..modality_a.len() {
            if let Some(&val_a) = modality_a.get([i]) {
                if let Some(&val_b) = modality_b.get([i]) {
                    let fused_val = (val_a + val_b) / 2.0;
                    fused_a[i] = fused_val;
                    fused_b[i] = fused_val;
                }
            }
        }

        let output_shape = shape.to_vec();
        Ok((
            ArrayD::from_shape_vec(output_shape.clone(), fused_a)?,
            ArrayD::from_shape_vec(output_shape, fused_b)?,
        ))
    }

    /// L2 normalize tensor
    fn l2_normalize(&self, tensor: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let norm = tensor.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(crate::caffeine::error::CaffeineError::tensor_operation(
                "Cannot normalize zero tensor",
            ));
        }

        let normalized = tensor.mapv(|x| x / norm);
        Ok(normalized)
    }

    /// Cross-attention between modalities
    fn cross_attention(&self, query: &ArrayD<f32>, key_value: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = query.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];

        let mut attended = vec![0.0f32; query.len()];

        for b in 0..batch_size {
            for i in 0..seq_len {
                for d in 0..embed_dim {
                    let mut attention_sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for j in 0..seq_len {
                        if let (Some(&q_val), Some(&kv_val)) =
                            (query.get([b, i, d]), key_value.get([b, j, d]))
                        {
                            let attention_weight = q_val * kv_val;
                            attention_sum += attention_weight * kv_val;
                            weight_sum += attention_weight.abs();
                        }
                    }

                    let idx = b * seq_len * embed_dim + i * embed_dim + d;
                    attended[idx] = if weight_sum > 0.0 {
                        attention_sum / weight_sum
                    } else {
                        0.0
                    };
                }
            }
        }

        let output_shape = vec![batch_size, seq_len, embed_dim];
        Ok(ArrayD::from_shape_vec(output_shape, attended)?)
    }
}

/// Alignment methods
#[derive(Debug, Clone)]
enum AlignmentMethod {
    Contrastive,
    Attention,
    Fusion,
}
