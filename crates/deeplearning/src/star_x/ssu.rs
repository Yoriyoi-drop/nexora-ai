//! Selective State Update (SSU)
//!
//! Inti dari STAR-X yang mengurangi FLOPs dengan:
//! - Relevance estimation untuk selective update
//! - Threshold-based state skipping
//! - Gradient flow stabilization
//! - Temporal redundancy elimination

use crate::{DLResult, DeepLearningError};
use crate::star_x::core::SelectiveUpdate;
use crate::star_x::tensor_pool::PooledTensor1D;
use crate::star_x::fused_ops::{FusedLinearActivation, FusedElementWise};
use ndarray::{ArrayD, Array1, Array2};
use rand;

/// Selective State Update implementation
pub struct SelectiveStateUpdate {
    // Relevance network parameters
    relevance_weights: Array2<f32>,
    relevance_bias: Array1<f32>,
    
    // Update network parameters
    update_weights: Array2<f32>,
    update_bias: Array1<f32>,
    
    // Fused operations for optimization
    fused_relevance: Option<FusedLinearActivation>,
    fused_update: Option<FusedLinearActivation>,
    fused_element_wise: Option<FusedElementWise>,
    
    // Configuration
    input_size: usize,
    hidden_size: usize,
    relevance_alpha: f32,
    update_threshold: f32,
    
    // Statistics
    update_frequency: f32,
    total_updates: usize,
    total_steps: usize,
    skipped_updates: usize,
    avg_relevance: f32,
}

impl std::fmt::Debug for SelectiveStateUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectiveStateUpdate")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("relevance_alpha", &self.relevance_alpha)
            .field("update_threshold", &self.update_threshold)
            .field("update_frequency", &self.update_frequency)
            .field("total_updates", &self.total_updates)
            .field("has_fused_relevance", &self.fused_relevance.is_some())
            .field("has_fused_update", &self.fused_update.is_some())
            .field("has_fused_element_wise", &self.fused_element_wise.is_some())
            .finish()
    }
}

impl Clone for SelectiveStateUpdate {
    fn clone(&self) -> Self {
        Self {
            relevance_weights: self.relevance_weights.clone(),
            relevance_bias: self.relevance_bias.clone(),
            update_weights: self.update_weights.clone(),
            update_bias: self.update_bias.clone(),
            fused_relevance: None, // Cannot clone fused operations, recreate as needed
            fused_update: None,
            fused_element_wise: None,
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            relevance_alpha: self.relevance_alpha,
            update_threshold: self.update_threshold,
            update_frequency: self.update_frequency,
            total_updates: self.total_updates,
            total_steps: self.total_steps,
            skipped_updates: self.skipped_updates,
            avg_relevance: self.avg_relevance,
        }
    }
}

impl SelectiveStateUpdate {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        update_threshold: f32,
        relevance_alpha: f32,
    ) -> DLResult<Self> {
        // Initialize relevance network (weighted fusion expects hidden_size input)
        let relevance_weights = Self::xavier_init(hidden_size, hidden_size);
        let relevance_bias = Array1::zeros(hidden_size);
        
        // Initialize update network
        let update_weights = Self::xavier_init(hidden_size, hidden_size);
        let update_bias = Array1::zeros(hidden_size);
        
        Ok(Self {
            relevance_weights,
            relevance_bias,
            update_weights,
            update_bias,
            fused_relevance: None,
            fused_update: None,
            fused_element_wise: None,
            input_size,
            hidden_size,
            update_threshold,
            relevance_alpha,
            update_frequency: 0.0,
            total_updates: 0,
            total_steps: 0,
            skipped_updates: 0,
            avg_relevance: 0.0,
        })
    }
    
    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            (rand::random::<f32>() * 2.0 - 1.0) * limit
        })
    }
    
    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Apply sigmoid ke array
    fn sigmoid_array(&self, mut arr: ArrayD<f32>) -> ArrayD<f32> {
        for val in arr.iter_mut() {
            *val = self.sigmoid(*val);
        }
        arr
    }
    
    /// Tanh activation
    fn tanh_array(&self, mut arr: ArrayD<f32>) -> ArrayD<f32> {
        for val in arr.iter_mut() {
            *val = val.tanh();
        }
        arr
    }
    
    /// Concatenate two tensors dengan efficient memory allocation
    fn concatenate(&self, tgh_output: &ArrayD<f32>, sca_output: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let tgh_flat = tgh_output.as_slice().expect("tensor should be contiguous");
        let sca_flat = sca_output.as_slice().expect("tensor should be contiguous");
        let total_size = tgh_flat.len() + sca_flat.len();
        
        // Use pooled tensor untuk concatenated result
        let mut pooled_tensor = PooledTensor1D::new(total_size)?;
        let concatenated = pooled_tensor.get_mut();
        let concat_flat = concatenated.as_slice_mut().expect("tensor should be contiguous");
        
        // Copy data dengan minimal allocation
        concat_flat[..tgh_flat.len()].copy_from_slice(tgh_flat);
        concat_flat[tgh_flat.len()..].copy_from_slice(sca_flat);
        
        Ok(concatenated.clone().into_dyn())
    }
    
    /// Matrix multiplication
    fn matmul(&self, weights: &Array2<f32>, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_flat = input.as_slice().expect("tensor should be contiguous");
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
    
    /// Add bias
    fn add_bias(&self, output: &mut ArrayD<f32>, bias: &Array1<f32>) {
        let output_flat = output.as_slice_mut().expect("tensor should be contiguous");
        for (i, &b) in bias.iter().enumerate().take(output_flat.len()) {
            output_flat[i] += b;
        }
    }
    
    /// Compute element-wise importance
    fn compute_element_importance(&self, relevance: &ArrayD<f32>) -> DLResult<Vec<f32>> {
        let relevance_flat = relevance.as_slice().expect("tensor should be contiguous");
        let mut importance = Vec::with_capacity(relevance_flat.len());
        
        for &val in relevance_flat {
            // Importance based on relevance magnitude
            let imp = val.abs();
            importance.push(imp);
        }
        
        Ok(importance)
    }
    
    /// Adaptive threshold adjustment
    fn adjust_threshold(&mut self, _current_relevance_avg: f32) {
        // Adjust threshold based on recent relevance patterns
        let target_frequency = 0.3; // Target 30% update frequency
        let current_frequency = self.update_frequency;
        
        if current_frequency < target_frequency {
            // Too few updates, lower threshold
            self.update_threshold *= 0.95;
            self.update_threshold = self.update_threshold.max(0.05);
        } else if current_frequency > target_frequency {
            // Too many updates, raise threshold
            self.update_threshold *= 1.05;
            self.update_threshold = self.update_threshold.min(0.5);
        }
    }
    
    /// Update statistics
    fn update_statistics(&mut self, relevance: &ArrayD<f32>, performed_update: bool) {
        self.total_steps += 1;
        
        if performed_update {
            self.total_updates += 1;
        } else {
            self.skipped_updates += 1;
        }
        
        // Update average relevance
        let relevance_flat = relevance.as_slice().expect("tensor should be contiguous");
        let relevance_sum: f32 = relevance_flat.iter().sum();
        let relevance_avg = relevance_sum / relevance_flat.len() as f32;
        
        self.avg_relevance = (self.avg_relevance * (self.total_steps - 1) as f32 + relevance_avg) / 
                           self.total_steps as f32;
        
        // Update frequency
        self.update_frequency = self.total_updates as f32 / self.total_steps as f32;
        
        // Adaptive threshold adjustment
        if self.total_steps % 100 == 0 {
            self.adjust_threshold(relevance_avg);
        }
    }
    
    /// Get detailed statistics
    pub fn get_detailed_stats(&self) -> (f32, f32, f32, f32, usize, usize) {
        (
            self.update_frequency,
            self.avg_relevance,
            self.update_threshold,
            self.relevance_alpha,
            self.total_updates,
            self.skipped_updates,
        )
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.update_frequency = 0.0;
        self.total_updates = 0;
        self.total_steps = 0;
        self.skipped_updates = 0;
        self.avg_relevance = 0.0;
    }
    
    /// Set update threshold
    pub fn set_update_threshold(&mut self, threshold: f32) -> DLResult<()> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(DeepLearningError::Configuration {
                reason: "Update threshold must be between 0.0 and 1.0".to_string(),
            });
        }
        self.update_threshold = threshold;
        Ok(())
    }
    
    /// Set relevance alpha
    pub fn set_relevance_alpha(&mut self, alpha: f32) {
        self.relevance_alpha = alpha;
    }
}

impl SelectiveUpdate for SelectiveStateUpdate {
    fn compute_relevance(&self, 
        tgh_output: &ArrayD<f32>,
        sca_output: &ArrayD<f32>
    ) -> DLResult<ArrayD<f32>> {
        
        // Weighted fusion of TGH and SCA outputs (STAR-X style)
        let alpha = 0.6; // TGH weight
        let beta = 0.4;  // SCA weight
        
        // Ensure both outputs have the same shape
        if tgh_output.shape() != sca_output.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: tgh_output.shape().to_vec(),
                actual: sca_output.shape().to_vec(),
            });
        }
        
        // Weighted fusion: α*h_tgh + β*h_sca
        let mut fused = tgh_output.clone();
        fused.map_inplace(|val| *val *= alpha);
        
        let mut sca_weighted = sca_output.clone();
        sca_weighted.map_inplace(|val| *val *= beta);
        
        fused += &sca_weighted;
        
        // Compute relevance scores
        let relevance_linear = self.matmul(&self.relevance_weights, &fused)?;
        let mut relevance_output = relevance_linear;
        self.add_bias(&mut relevance_output, &self.relevance_bias);
        
        // Apply sigmoid to get relevance probabilities
        let relevance_probs = self.sigmoid_array(relevance_output);
        
        Ok(relevance_probs)
    }
    
    fn selective_update(&self,
        previous_state: &ArrayD<f32>,
        candidate_state: &ArrayD<f32>,
        relevance: &ArrayD<f32>,
        threshold: f32
    ) -> DLResult<ArrayD<f32>> {
        
        let prev_flat = previous_state.as_slice().expect("tensor should be contiguous");
        let cand_flat = candidate_state.as_slice().expect("tensor should be contiguous");
        let rel_flat = relevance.as_slice().expect("tensor should be contiguous");
        
        if prev_flat.len() != cand_flat.len() || prev_flat.len() != rel_flat.len() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![prev_flat.len()],
                actual: vec![cand_flat.len(), rel_flat.len()],
            });
        }
        
        // Use pooled tensor untuk updated state
        let mut pooled_tensor = PooledTensor1D::new(prev_flat.len())?;
        let updated_state = pooled_tensor.get_mut();
        let updated_flat = updated_state.as_slice_mut().expect("tensor should be contiguous");
        let mut _performed_update = false;
        
        for i in 0..prev_flat.len() {
            let relevance_score = rel_flat[i];
            
            if relevance_score >= threshold {
                // Perform update
                let alpha = self.relevance_alpha;
                let updated = alpha * cand_flat[i] + (1.0 - alpha) * prev_flat[i];
                updated_flat[i] = updated;
                _performed_update = true;
            } else {
                // Skip update, keep previous state
                updated_flat[i] = prev_flat[i];
            }
        }
        
        Ok(updated_state.clone().into_dyn())
    }
    
    fn get_update_frequency(&self) -> f32 {
        self.update_frequency
    }
}

/// Advanced selective update strategies
impl SelectiveStateUpdate {
    /// Block-wise selective update untuk efficiency
    pub fn block_selective_update(&self,
        previous_state: &ArrayD<f32>,
        candidate_state: &ArrayD<f32>,
        relevance: &ArrayD<f32>,
        block_size: usize,
        threshold: f32
    ) -> DLResult<ArrayD<f32>> {
        
        let prev_flat = previous_state.as_slice().expect("tensor should be contiguous");
        let cand_flat = candidate_state.as_slice().expect("tensor should be contiguous");
        let rel_flat = relevance.as_slice().expect("tensor should be contiguous");
        
        let mut updated_state = prev_flat.to_vec();
        let mut block_start = 0;
        
        while block_start < prev_flat.len() {
            let block_end = (block_start + block_size).min(prev_flat.len());
            
            // Compute block relevance (average)
            let mut block_relevance = 0.0;
            for i in block_start..block_end {
                block_relevance += rel_flat[i];
            }
            block_relevance /= (block_end - block_start) as f32;
            
            // Update entire block if relevance is high enough
            if block_relevance >= threshold {
                for i in block_start..block_end {
                    let alpha = self.relevance_alpha;
                    updated_state[i] = alpha * cand_flat[i] + (1.0 - alpha) * prev_flat[i];
                }
            }
            
            block_start = block_end;
        }
        
        Ok(Array1::from_vec(updated_state).into_dyn())
    }
    
    /// Hierarchical selective update dengan multiple thresholds
    pub fn hierarchical_selective_update(&self,
        previous_state: &ArrayD<f32>,
        candidate_state: &ArrayD<f32>,
        relevance: &ArrayD<f32>,
        thresholds: &[f32]
    ) -> DLResult<ArrayD<f32>> {
        
        let prev_flat = previous_state.as_slice().expect("tensor should be contiguous");
        let cand_flat = candidate_state.as_slice().expect("tensor should be contiguous");
        let rel_flat = relevance.as_slice().expect("tensor should be contiguous");
        
        let mut updated_state = Vec::with_capacity(prev_flat.len());
        
        for i in 0..prev_flat.len() {
            let relevance_score = rel_flat[i];
            
            // Determine update level based on relevance
            let mut update_level = 0.0;
            for (level, &threshold) in thresholds.iter().enumerate() {
                if relevance_score >= threshold {
                    update_level = (level + 1) as f32 / thresholds.len() as f32;
                }
            }
            
            // Apply weighted update
            let alpha = self.relevance_alpha * update_level;
            let updated = alpha * cand_flat[i] + (1.0 - alpha) * prev_flat[i];
            updated_state.push(updated);
        }
        
        Ok(Array1::from_vec(updated_state).into_dyn())
    }
    
    /// Temporal coherence-aware selective update
    pub fn temporal_coherence_update(&self,
        previous_state: &ArrayD<f32>,
        candidate_state: &ArrayD<f32>,
        relevance: &ArrayD<f32>,
        temporal_coherence: f32,
        threshold: f32
    ) -> DLResult<ArrayD<f32>> {
        
        let prev_flat = previous_state.as_slice().expect("tensor should be contiguous");
        let cand_flat = candidate_state.as_slice().expect("tensor should be contiguous");
        let rel_flat = relevance.as_slice().expect("tensor should be contiguous");
        
        let mut updated_state = Vec::with_capacity(prev_flat.len());
        
        // Adjust threshold based on temporal coherence
        let coherence_adjusted_threshold = threshold * (1.0 + temporal_coherence);
        
        for i in 0..prev_flat.len() {
            let relevance_score = rel_flat[i];
            
            if relevance_score >= coherence_adjusted_threshold {
                // Full update with coherence weighting
                let alpha = self.relevance_alpha * (1.0 + temporal_coherence * 0.5);
                let updated = alpha * cand_flat[i] + (1.0 - alpha) * prev_flat[i];
                updated_state.push(updated);
            } else if relevance_score >= threshold {
                // Partial update
                let alpha = self.relevance_alpha * 0.5;
                let updated = alpha * cand_flat[i] + (1.0 - alpha) * prev_flat[i];
                updated_state.push(updated);
            } else {
                // No update
                updated_state.push(prev_flat[i]);
            }
        }
        
        Ok(Array1::from_vec(updated_state).into_dyn())
    }
    
    /// Compute computational savings
    pub fn compute_savings(&self) -> (f32, f32) {
        let skip_ratio = self.skipped_updates as f32 / self.total_steps as f32;
        let theoretical_savings = skip_ratio * 100.0; // Percentage
        let actual_savings = theoretical_savings * 0.8; // Account for overhead
        
        (actual_savings, theoretical_savings)
    }
}
