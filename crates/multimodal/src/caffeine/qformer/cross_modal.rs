//! Cross-modal attention mechanisms for Q-Former
//! 
//! Implements attention between different modalities

use crate::caffeine::types::*;
use crate::caffeine::error::Result;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::sync::Arc;

/// Memory pool for efficient buffer reuse
#[derive(Debug, Clone)]
struct MemoryPool {
    buffers: Arc<std::sync::Mutex<Vec<Vec<f32>>>>,
    _max_pool_size: usize,
}

impl MemoryPool {
    fn new(max_pool_size: usize) -> Self {
        Self {
            buffers: Arc::new(std::sync::Mutex::new(Vec::with_capacity(max_pool_size))),
            _max_pool_size: max_pool_size,
        }
    }
    
    fn get_buffer(&self, size: usize) -> Vec<f32> {
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut buffer) = buffers.pop() {
            if buffer.capacity() >= size {
                buffer.clear();
                buffer.resize(size, 0.0);
                buffer
            } else {
                vec![0.0f32; size]
            }
        } else {
            vec![0.0f32; size]
        }
    }
    
    fn _return_buffer(&self, buffer: Vec<f32>) {
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        if buffers.len() < self._max_pool_size {
            buffers.push(buffer);
        }
    }
}

/// Cross-modal attention mechanism with memory optimization
pub struct CrossModalAttention {
    hidden_dim: usize,
    num_heads: usize,
    _dropout_rate: f32,
    projection_weights: HashMap<String, Vec<f32>>,
    memory_pool: MemoryPool,
}

impl CrossModalAttention {
    /// Create new cross-modal attention with memory optimization
    pub fn new(hidden_dim: usize, num_heads: usize, _dropout_rate: f32) -> Result<Self> {
        if hidden_dim == 0 || num_heads == 0 {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "Hidden dimension and number of heads must be greater than 0"
            ));
        }
        
        if hidden_dim % num_heads != 0 {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "Hidden dimension must be divisible by number of heads"
            ));
        }

        let mut projection_weights = HashMap::new();
        
        // Initialize projection weights for different modality pairs
        let modality_pairs = vec![
            ("image_text", hidden_dim * hidden_dim),
            ("audio_text", hidden_dim * hidden_dim),
            ("video_text", hidden_dim * hidden_dim),
            ("image_audio", hidden_dim * hidden_dim),
            ("image_video", hidden_dim * hidden_dim),
            ("audio_video", hidden_dim * hidden_dim),
        ];
        
        for (pair_name, size) in modality_pairs {
            let weights = Self::initialize_weights(size, hidden_dim)?;
            projection_weights.insert(pair_name.to_string(), weights);
        }
        
        // Initialize memory pool with reasonable size
        let memory_pool = MemoryPool::new(10);
        
        Ok(Self {
            hidden_dim,
            num_heads,
            _dropout_rate,
            projection_weights,
            memory_pool,
        })
    }
    
    /// Initialize weights using Xavier initialization
    fn initialize_weights(size: usize, hidden_dim: usize) -> Result<Vec<f32>> {
        let mut weights = vec![0.0f32; size];
        let scale = (2.0 / hidden_dim as f32).sqrt();
        
        for i in 0..size {
            weights[i] = rand::random::<f32>() * scale - scale / 2.0;
        }
        
        Ok(weights)
    }
    
    /// Compute cross-attention between modalities with memory optimization
    pub fn compute_cross_attention(
        &self,
        features: &[f32],
        num_queries: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        if features.len() != num_queries * hidden_dim {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "Feature dimensions don't match expected query dimensions"
            ));
        }
        
        // Use memory pool for output buffer
        let mut attended_features = self.memory_pool.get_buffer(features.len());
        
        // Process each head with optimized computation
        let head_dim = hidden_dim / self.num_heads;
        
        for head in 0..self.num_heads {
            let start_dim = head * head_dim;
            let end_dim = std::cmp::min((head + 1) * head_dim, hidden_dim);
            
            // Vectorized computation for this head
            self.compute_head_attention_optimized(
                features,
                &mut attended_features,
                start_dim,
                end_dim,
                num_queries,
                hidden_dim,
                head,
            )?;
        }
        
        // Return buffer to pool when done (in practice, we'd need RAII for this)
        // For now, we'll return the buffer directly
        Ok(attended_features)
    }
    
    /// Optimized head attention computation with SIMD-like operations
    fn compute_head_attention_optimized(
        &self,
        features: &[f32],
        attended_features: &mut [f32],
        start_dim: usize,
        end_dim: usize,
        num_queries: usize,
        hidden_dim: usize,
        head_idx: usize,
    ) -> Result<()> {
        let dim_count = end_dim - start_dim;
        
        // Process queries in batches for better cache performance
        const BATCH_SIZE: usize = 4;
        
        for i_batch in (0..num_queries).step_by(BATCH_SIZE) {
            let i_end = std::cmp::min(i_batch + BATCH_SIZE, num_queries);
            
            for i in i_batch..i_end {
                // Pre-compute attention scores for all dimensions at once
                let attention_scores = self.compute_attention_scores_vectorized(
                    features, i, start_dim, end_dim, num_queries, hidden_dim, head_idx
                )?;
                
                // Apply attention scores with vectorized operations
                for (d_idx, d) in (start_dim..end_dim).enumerate() {
                    let input_idx = i * hidden_dim + d;
                    let output_idx = input_idx;
                    
                    if input_idx < features.len() && d_idx < attention_scores.len() {
                        attended_features[output_idx] = features[input_idx] * attention_scores[d_idx];
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Vectorized attention score computation
    fn compute_attention_scores_vectorized(
        &self,
        features: &[f32],
        query_idx: usize,
        start_dim: usize,
        end_dim: usize,
        num_queries: usize,
        hidden_dim: usize,
        head_idx: usize,
    ) -> Result<Vec<f32>> {
        let dim_count = end_dim - start_dim;
        let mut scores = vec![0.0f32; dim_count];
        
        // Pre-compute head factor
        let head_factor = (head_idx as f32 + 1.0) * 0.1;
        let head_factor_sin = head_factor.sin();
        
        // Vectorized computation across dimensions
        for (d_idx, d) in (start_dim..end_dim).enumerate() {
            let query_val = features[query_idx * hidden_dim + d];
            
            // Compute dot product with all other queries
            let mut dot_product = 0.0f32;
            
            // Process in chunks for better cache performance
            const CHUNK_SIZE: usize = 8;
            for j_chunk in (0..num_queries).step_by(CHUNK_SIZE) {
                let j_end = std::cmp::min(j_chunk + CHUNK_SIZE, num_queries);
                
                for j in j_chunk..j_end {
                    let other_idx = j * hidden_dim + d;
                    if other_idx < features.len() {
                        let key_val = features[other_idx];
                        dot_product += query_val * key_val;
                    }
                }
            }
            
            scores[d_idx] = (dot_product * head_factor_sin) / num_queries as f32;
        }
        
        Ok(scores)
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
                return Err(crate::caffeine::error::CaffeineError::qformer(
                    &format!("Weight dimensions {} don't match expected {}x{}", 
                            weights.len(), input_dim, output_dim)
                ));
            }
            
            // Optimized projection computation
            let projected = self.compute_projection_optimized(
                features, weights, batch_size, seq_len, input_dim, output_dim
            )?;
            
            let output_shape = vec![batch_size, seq_len, output_dim];
            Ok(ArrayD::from_shape_vec(output_shape, projected)?)
        } else {
            Err(crate::caffeine::error::CaffeineError::qformer(
                &format!("No projection weights found for {} -> {}", from_modality, to_modality)
            ))
        }
    }
    
    /// Optimized projection computation with reduced nested loops
    fn compute_projection_optimized(
        &self,
        features: &ArrayD<f32>,
        weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut projected = vec![0.0f32; batch_size * seq_len * output_dim];
        
        // Pre-compute weight matrix for better cache locality
        let weight_matrix: Vec<&[f32]> = (0..output_dim)
            .map(|o| &weights[o * input_dim..(o + 1) * input_dim])
            .collect();
        
        // Optimized computation with better memory access patterns
        for b in 0..batch_size {
            for i in 0..seq_len {
                for o in 0..output_dim {
                    let output_idx = b * seq_len * output_dim + i * output_dim + o;
                    let weight_row = weight_matrix[o];
                    
                    // Vectorized dot product
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
        
        Ok(if count > 0.0 { total_activation / count } else { 0.0 })
    }
    
    /// Apply gated fusion
    pub fn gated_fusion(
        &self,
        modality_features: &std::collections::HashMap<String, ArrayD<f32>>,
        fusion_weights: &std::collections::HashMap<String, f32>,
    ) -> Result<ArrayD<f32>> {
        if modality_features.is_empty() {
            return Err(crate::caffeine::error::CaffeineError::qformer(
                "No modality features provided for fusion"
            ));
        }
        
        // Get reference shape from first modality
        let first_features = modality_features.values().next().ok_or_else(|| crate::caffeine::error::CaffeineError::input_validation("No modality features available"))?;
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
        
        Self {
            alignment_methods,
        }
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
                AlignmentMethod::Contrastive => {
                    self.contrastive_alignment(modality_a, modality_b)
                }
                AlignmentMethod::Attention => {
                    self.attention_alignment(modality_a, modality_b)
                }
                AlignmentMethod::Fusion => {
                    self.fusion_alignment(modality_a, modality_b)
                }
            }
        } else {
            Err(crate::caffeine::error::CaffeineError::qformer(
                &format!("Unknown alignment method: {}", method)
            ))
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
                "Cannot normalize zero tensor"
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
                        if let (Some(&q_val), Some(&kv_val)) = (
                            query.get([b, i, d]),
                            key_value.get([b, j, d])
                        ) {
                            let attention_weight = q_val * kv_val;
                            attention_sum += attention_weight * kv_val;
                            weight_sum += attention_weight.abs();
                        }
                    }
                    
                    let idx = b * seq_len * embed_dim + i * embed_dim + d;
                    attended[idx] = if weight_sum > 0.0 { attention_sum / weight_sum } else { 0.0 };
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
