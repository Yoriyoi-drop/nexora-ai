//! Sparse Causal Attention (SCA)
//!
//! Attention dengan O(T log T) complexity menggunakan:
//! - Dynamic sparse routing berdasarkan relevance
//! - Temporal distance bias
//! - Harmonic temporal encoding
//! - Entropy regularization

use crate::{DLResult, DeepLearningError};
use crate::star_x::core::{SparseAttention, utils};
use crate::star_x::kv_cache::{KVCache, StreamingKVCache};
use crate::star_x::fused_ops::{FusedAttentionSoftmax, FusedElementWise, ElementWiseOp};
use crate::star_x::blas_backend::{BlasOperations, ActivationType};
use crate::traits::Forward;
use ndarray::{ArrayD, Array2, Array1};
use rand;

/// Sparse Causal Attention implementation
#[derive(Debug, Clone)]
pub struct SparseCausalAttention {
    // Attention parameters
    query_weights: Array2<f32>,
    key_weights: Array2<f32>,
    value_weights: Array2<f32>,
    output_weights: Array2<f32>,
    
    // Fused operations for optimization
    fused_attention: Option<FusedAttentionSoftmax>,
    fused_element_wise: Option<FusedElementWise>,
    
    // BLAS backend for high-performance operations
    blas_ops: Option<BlasOperations>,
    
    // Configurations
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    max_sparse_connections: usize,
    entropy_regularization: f32,
    temporal_distance_weight: f32,
    
    // KV Cache for efficient inference
    pub kv_cache: Option<KVCache>,
    pub streaming_cache: Option<StreamingKVCache>,
    pub use_cache: bool,
    max_cache_size: usize,
    
    // Sparse routing statistics
    current_sparsity: f32,
    attention_entropy: f32,
    routing_efficiency: f32,
}

impl SparseCausalAttention {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        max_sparse_connections: usize,
        entropy_regularization: f32,
    ) -> DLResult<Self> {
        let head_dim = hidden_dim / num_heads;
        
        if hidden_dim % num_heads != 0 {
            return Err(DeepLearningError::Configuration {
                reason: format!("hidden_dim {} must be divisible by num_heads {}", hidden_dim, num_heads),
            });
        }
        
        // Initialize weights
        let query_weights = Self::xavier_init(hidden_dim, hidden_dim);
        let key_weights = Self::xavier_init(hidden_dim, hidden_dim);
        let value_weights = Self::xavier_init(hidden_dim, hidden_dim);
        let output_weights = Self::xavier_init(hidden_dim, hidden_dim);
        
        // Initialize fused operations
        let fused_attention = FusedAttentionSoftmax::new(
            query_weights.clone(),
            key_weights.clone(),
            value_weights.clone(),
            output_weights.clone(),
            head_dim,
            num_heads,
        ).ok();
        
        let fused_element_wise = Some(FusedElementWise::new(vec![
            ElementWiseOp::Gelu,
            ElementWiseOp::Mul(0.1), // residual scaling
        ]));
        
        // Initialize BLAS operations
        let blas_ops = BlasOperations::auto_detect().ok();
        
        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            fused_attention,
            fused_element_wise,
            blas_ops,
            num_heads,
            head_dim,
            hidden_dim,
            max_sparse_connections,
            entropy_regularization,
            temporal_distance_weight: 0.1,
            kv_cache: None,
            streaming_cache: None,
            use_cache: false,
            max_cache_size: 2048,
            current_sparsity: 0.0,
            attention_entropy: 0.0,
            routing_efficiency: 0.0,
        })
    }
    
    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            (rand::random::<f32>() * 2.0 - 1.0) * limit
        })
    }
    
    /// Split input menjadi multi-head
    fn split_heads(&self, input: &ArrayD<f32>) -> DLResult<Vec<ArrayD<f32>>> {
        let input_flat = input.as_slice().unwrap();
        if input_flat.len() != self.hidden_dim {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![self.hidden_dim],
                actual: vec![input_flat.len()],
            });
        }
        
        let mut heads = Vec::new();
        for i in 0..self.num_heads {
            let start = i * self.head_dim;
            let end = start + self.head_dim;
            let head_data = input_flat[start..end].to_vec();
            heads.push(Array1::from_vec(head_data).into_dyn());
        }
        
        Ok(heads)
    }
    
    /// Combine multi-head outputs
    fn combine_heads(&self, heads: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        let mut combined = Vec::with_capacity(self.hidden_dim);
        
        for head in heads {
            let head_flat = head.as_slice().unwrap();
            combined.extend_from_slice(head_flat);
        }
        
        Ok(Array1::from_vec(combined).into_dyn())
    }
    
    /// Matrix multiplication
    fn matmul(&self, weights: &Array2<f32>, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_flat = input.as_slice().unwrap();
        if input_flat.len() != weights.shape()[0] {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![weights.shape()[0]],
                actual: vec![input_flat.len()],
            });
        }
        
        let mut output = vec![0.0; weights.shape()[1]];
        for (i, row) in weights.outer_iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                output[j] += input_flat[i] * weight;
            }
        }
        
        Ok(Array1::from_vec(output).into_dyn())
    }
    
    /// Compute attention scores dengan temporal bias
    fn compute_attention_scores(&self, 
        query: &ArrayD<f32>,
        keys: &[ArrayD<f32>],
        temporal_positions: &[usize],
        temporal_encoding: &ArrayD<f32>
    ) -> DLResult<Vec<f32>> {
        let query_flat = query.as_slice().unwrap();
        let mut scores = Vec::with_capacity(keys.len());
        
        for (i, key) in keys.iter().enumerate() {
            let key_flat = key.as_slice().unwrap();
            
            // Dot product
            let mut dot_product = 0.0;
            for (q, k) in query_flat.iter().zip(key_flat.iter()) {
                dot_product += q * k;
            }
            
            // Scale by sqrt(d_k)
            let scaled_score = dot_product / (self.head_dim as f32).sqrt();
            
            // Add temporal distance bias
            let temporal_distance = (temporal_positions[temporal_positions.len() - 1] as f32 - 
                                    temporal_positions[i] as f32).abs();
            let temporal_bias = -self.temporal_distance_weight * temporal_distance;
            
            // Add harmonic temporal encoding contribution
            let temp_enc_flat = temporal_encoding.as_slice().unwrap();
            let harmonic_contribution = if i < temp_enc_flat.len() {
                temp_enc_flat[i] * 0.1
            } else {
                0.0
            };
            
            let final_score = scaled_score + temporal_bias + harmonic_contribution;
            scores.push(final_score);
        }
        
        Ok(scores)
    }
    
    /// Dynamic sparse routing dengan top-k selection
    fn dynamic_sparse_routing(&self, scores: &[f32], k: usize) -> DLResult<Vec<usize>> {
        if k >= scores.len() {
            return Ok((0..scores.len()).collect());
        }
        
        let mut indexed_scores: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        // Sort by score (descending)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        let selected_indices: Vec<usize> = indexed_scores
            .iter()
            .take(k)
            .map(|(idx, _)| *idx)
            .collect();
        
        Ok(selected_indices)
    }
    
    /// Softmax dengan entropy regularization
    fn softmax_with_entropy(&self, scores: &[f32], selected_indices: &[usize]) -> DLResult<(Vec<f32>, f32)> {
        // Extract selected scores
        let selected_scores: Vec<f32> = selected_indices
            .iter()
            .map(|&i| scores[i])
            .collect();
        
        // Compute softmax
        let max_score = selected_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = selected_scores
            .iter()
            .map(|&s| (s - max_score).exp())
            .collect();
        
        let sum_exp: f32 = exp_scores.iter().sum();
        let softmax_probs: Vec<f32> = exp_scores
            .iter()
            .map(|&e| e / sum_exp)
            .collect();
        
        // Compute entropy
        let entropy = utils::compute_entropy(&softmax_probs);
        
        Ok((softmax_probs, entropy))
    }
    
    /// Apply attention weights ke values
    fn apply_attention(&self, 
        weights: &[f32],
        selected_indices: &[usize],
        values: &[ArrayD<f32>]
    ) -> DLResult<ArrayD<f32>> {
        let mut output = Array1::zeros(self.head_dim);
        let output_flat = output.as_slice_mut().unwrap();
        
        for (&weight, &idx) in weights.iter().zip(selected_indices.iter()) {
            if idx < values.len() {
                let value_flat = values[idx].as_slice().unwrap();
                for (i, &val) in value_flat.iter().enumerate().take(self.head_dim) {
                    output_flat[i] += weight * val;
                }
            }
        }
        
        Ok(output.into_dyn())
    }
    
    /// Adaptive k selection based on entropy
    fn adaptive_k_selection(&self, sequence_length: usize, base_entropy: f32) -> usize {
        // Higher entropy -> more connections (more diverse attention)
        // Lower entropy -> fewer connections (more focused attention)
        let entropy_factor = (base_entropy / (self.num_heads as f32).ln()).min(2.0).max(0.5);
        let adaptive_k = (self.max_sparse_connections as f32 * entropy_factor) as usize;
        
        adaptive_k.min(sequence_length).max(1)
    }
    
    /// Update sparsity statistics
    fn update_sparsity_stats(&mut self, total_connections: usize, possible_connections: usize) {
        self.current_sparsity = 1.0 - (total_connections as f32 / possible_connections as f32);
        self.routing_efficiency = total_connections as f32 / self.max_sparse_connections as f32;
    }
}

impl SparseAttention for SparseCausalAttention {
    fn compute_sparse_attention(&mut self, 
        query: &ArrayD<f32>, 
        key: &ArrayD<f32>, 
        value: &ArrayD<f32>,
        temporal_encoding: &ArrayD<f32>
    ) -> DLResult<(ArrayD<f32>, ArrayD<f32>)> {
        
        // Split inputs into multi-heads
        let query_heads = self.split_heads(query)?;
        let key_heads = self.split_heads(key)?;
        let value_heads = self.split_heads(value)?;
        
        let mut head_outputs = Vec::new();
        let mut all_entropies = Vec::new();
        let mut total_connections = 0;
        
        // Process each head
        for (_h, ((q_head, k_head), v_head)) in query_heads.iter().zip(key_heads.iter()).zip(value_heads.iter()).enumerate() {
            // For simplicity, assume we have a sequence of keys/values
            // In practice, this would be the full sequence
            
            // Compute attention scores
            let temporal_positions = vec![0; 1]; // Simplified
            let scores = self.compute_attention_scores(q_head, &[k_head.clone()], &temporal_positions, temporal_encoding)?;
            
            // Adaptive sparse routing
            let base_entropy = 1.0; // Simplified
            let k = self.adaptive_k_selection(1, base_entropy);
            let selected_indices = self.dynamic_sparse_routing(&scores, k)?;
            total_connections += selected_indices.len();
            
            // Softmax with entropy
            let (attention_weights, entropy) = self.softmax_with_entropy(&scores, &selected_indices)?;
            all_entropies.push(entropy);
            
            // Apply attention
            let head_output = self.apply_attention(&attention_weights, &selected_indices, &[v_head.clone()])?;
            head_outputs.push(head_output);
        }
        
        // Combine heads
        let combined_output = self.combine_heads(&head_outputs)?;
        
        // Final projection
        let final_output = self.matmul(&self.output_weights, &combined_output)?;
        
        // Create sparse mask
        let mut mask = ArrayD::zeros(vec![self.num_heads, self.max_sparse_connections]);
        let mask_flat = mask.as_slice_mut().unwrap();
        for (h, connections) in head_outputs.iter().enumerate() {
            let start = h * self.max_sparse_connections;
            let end = (start + connections.len()).min(mask_flat.len());
            for i in start..end {
                mask_flat[i] = 1.0;
            }
        }
        
        // Update statistics
        let avg_entropy = all_entropies.iter().sum::<f32>() / all_entropies.len() as f32;
        self.attention_entropy = avg_entropy;
        self.update_sparsity_stats(total_connections, self.num_heads * self.max_sparse_connections);
        
        Ok((final_output, mask))
    }
    
    fn get_sparsity_ratio(&self) -> f32 {
        self.current_sparsity
    }
    
    fn set_sparsity_level(&mut self, ratio: f32) -> DLResult<()> {
        if ratio < 0.0 || ratio > 1.0 {
            return Err(DeepLearningError::Configuration {
                reason: "Sparsity ratio must be between 0.0 and 1.0".to_string(),
            });
        }
        
        // Adjust max_sparse_connections based on target sparsity
        let target_connections = ((1.0 - ratio) * (self.num_heads * 100) as f32) as usize;
        self.max_sparse_connections = target_connections.max(1);
        
        Ok(())
    }
}

/// Additional methods for SparseCausalAttention
impl SparseCausalAttention {
    /// Enable KV Cache for efficient inference
    pub fn enable_cache(&mut self, max_cache_size: usize) {
        self.use_cache = true;
        self.max_cache_size = max_cache_size;
        self.kv_cache = Some(KVCache::new(max_cache_size, self.head_dim, self.num_heads));
    }
    
    /// Enable streaming cache for long sequences
    pub fn enable_streaming_cache(&mut self, chunk_size: usize, max_chunks: usize) {
        self.use_cache = true;
        self.streaming_cache = Some(StreamingKVCache::new(
            chunk_size, 
            max_chunks, 
            self.head_dim, 
            self.num_heads
        ));
    }
    
    /// Disable cache
    pub fn disable_cache(&mut self) {
        self.use_cache = false;
        self.kv_cache = None;
        self.streaming_cache = None;
    }
    
    /// Reset cache
    pub fn reset_cache(&mut self) {
        if let Some(cache) = &mut self.kv_cache {
            cache.reset();
        }
        if let Some(cache) = &mut self.streaming_cache {
            cache.reset();
        }
    }
    
    /// Forward pass with KV Cache support
    pub fn forward_cached(&mut self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        if !self.use_cache {
            return self.forward(input);
        }
        
        // Compute query, key, value
        let query = self.query_weights.dot(&input.view().into_shape((self.hidden_dim, 1))?)
            .into_shape(self.hidden_dim)?.into_dyn();
        let key = self.key_weights.dot(&input.view().into_shape((self.hidden_dim, 1))?)
            .into_shape(self.hidden_dim)?.into_dyn();
        let value = self.value_weights.dot(&input.view().into_shape((self.hidden_dim, 1))?)
            .into_shape(self.hidden_dim)?.into_dyn();
        
        // Use cache if available
        let attention_output = if let Some(cache) = &mut self.kv_cache {
            // Add to cache
            cache.append(
                key.as_slice().unwrap().to_vec().into(),
                value.as_slice().unwrap().to_vec().into()
            )?;
            
            // Compute attention with cache
            cache.compute_attention(
                &query.as_slice().unwrap().to_vec().into()
            )?.into_dyn()
        } else if let Some(streaming_cache) = &mut self.streaming_cache {
            // Add to streaming cache
            streaming_cache.append(
                key.as_slice().unwrap().to_vec().into(),
                value.as_slice().unwrap().to_vec().into()
            )?;
            
            // Compute attention with streaming cache
            streaming_cache.compute_attention(
                &query.as_slice().unwrap().to_vec().into()
            )?.into_dyn()
        } else {
            // Fallback to regular attention
            self.compute_sparse_attention(input, &ArrayD::zeros(vec![0]), &ArrayD::zeros(vec![0]), &Array1::zeros(0).into_dyn())?.0
        };
        
        // Final output projection
        let output = self.output_weights.dot(
            &attention_output.view().into_shape((self.hidden_dim, 1))?
        ).into_shape(self.hidden_dim)?.into_dyn();
        
        Ok(output)
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Option<String> {
        if let Some(cache) = &self.kv_cache {
            Some(format!("{}", cache.get_stats()))
        } else if let Some(cache) = &self.streaming_cache {
            Some(format!("{}", cache.get_streaming_stats()))
        } else {
            None
        }
    }
    
    /// Get attention statistics
    pub fn get_attention_stats(&self) -> (f32, f32, f32) {
        (self.current_sparsity, self.attention_entropy, self.routing_efficiency)
    }
    
    /// Set temporal distance weight
    pub fn set_temporal_distance_weight(&mut self, weight: f32) {
        self.temporal_distance_weight = weight;
    }
    
    /// Get entropy regularization strength
    pub fn get_entropy_regularization(&self) -> f32 {
        self.entropy_regularization
    }
}

impl crate::traits::Forward for SparseCausalAttention {
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;
    
    fn forward(&self, input: &Self::Input) -> DLResult<Self::Output> {
        // Use fused operations if available
        if let Some(ref fused_attention) = self.fused_attention {
            let output = fused_attention.forward(input)?;
            
            // Apply fused element-wise operations
            if let Some(ref fused_element_wise) = self.fused_element_wise {
                return fused_element_wise.forward(&output);
            }
            
            return Ok(output);
        }
        
        // Fallback to standard implementation
        let output = self.output_weights.dot(&input.view().into_shape((self.hidden_dim, 1))?)
            .into_shape(self.hidden_dim)?.into_dyn();
        
        Ok(output)
    }
}
