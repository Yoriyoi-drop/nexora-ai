//! Adaptive Gradient Resonance (AGR)
//!
//! Stabilisasi gradient untuk mencegah:
//! - Exploding gradients
//! - Catastrophic forgetting
//! - Unstable gating
//! - Trajectory divergence

use crate::{DLResult, DeepLearningError};
use crate::star_x::core::GradientResonance;
use ndarray::{ArrayD, Array1};

/// Adaptive Gradient Resonance implementation
#[derive(Debug, Clone)]
pub struct AdaptiveGradientResonance {
    // Resonance parameters
    resonance_factor: f32,
    momentum_factor: f32,
    decay_rate: f32,
    
    // State tracking
    previous_norm: f32,
    resonance_history: Vec<f32>,
    history_capacity: usize,
    
    // Adaptive parameters
    stability_threshold: f32,
    adaptation_rate: f32,
    min_resonance: f32,
    max_resonance: f32,
    
    // Statistics
    total_resonance_applications: usize,
    stability_violations: usize,
    avg_resonance_factor: f32,
}

impl AdaptiveGradientResonance {
    pub fn new(
        initial_resonance: f32,
        momentum_factor: f32,
        decay_rate: f32,
    ) -> DLResult<Self> {
        if initial_resonance < 0.0 || initial_resonance > 1.0 {
            return Err(DeepLearningError::Configuration {
                reason: "Initial resonance must be between 0.0 and 1.0".to_string(),
            });
        }
        
        Ok(Self {
            resonance_factor: initial_resonance,
            momentum_factor,
            decay_rate,
            previous_norm: 0.0,
            resonance_history: Vec::new(),
            history_capacity: 100,
            stability_threshold: 0.5,
            adaptation_rate: 0.01,
            min_resonance: 0.01,
            max_resonance: 0.5,
            total_resonance_applications: 0,
            stability_violations: 0,
            avg_resonance_factor: initial_resonance,
        })
    }
    
    /// Compute L2 norm of tensor
    fn compute_norm(&self, tensor: &ArrayD<f32>) -> f32 {
        let mut sum_sq = 0.0;
        for &val in tensor.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }
    
    /// Compute cosine similarity between two tensors
    fn compute_cosine_similarity(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> DLResult<f32> {
        let a_flat = a.as_slice().unwrap();
        let b_flat = b.as_slice().unwrap();
        
        if a_flat.len() != b_flat.len() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![a_flat.len()],
                actual: vec![b_flat.len()],
            });
        }
        
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        
        for (a_val, b_val) in a_flat.iter().zip(b_flat.iter()) {
            dot_product += a_val * b_val;
            norm_a += a_val * a_val;
            norm_b += b_val * b_val;
        }
        
        if norm_a > 0.0 && norm_b > 0.0 {
            Ok(dot_product / (norm_a.sqrt() * norm_b.sqrt()))
        } else {
            Ok(0.0)
        }
    }
    
    /// Detect gradient explosion
    fn detect_explosion(&self, current_norm: f32, previous_norm: f32) -> bool {
        if previous_norm == 0.0 {
            return false;
        }
        
        let ratio = current_norm / previous_norm;
        ratio > 10.0 // Explosion threshold
    }
    
    /// Detect gradient vanishing
    fn detect_vanishing(&self, current_norm: f32, previous_norm: f32) -> bool {
        if previous_norm == 0.0 {
            return false;
        }
        
        let ratio = current_norm / previous_norm;
        ratio < 0.1 // Vanishing threshold
    }
    
    /// Adaptive resonance adjustment
    fn adjust_resonance(&mut self, current_norm: f32, previous_norm: f32) {
        if previous_norm == 0.0 {
            return;
        }
        
        let norm_ratio = current_norm / previous_norm;
        
        // Detect instability
        let is_exploding = norm_ratio > 5.0;
        let is_vanishing = norm_ratio < 0.2;
        
        if is_exploding {
            // Reduce resonance to dampen oscillations
            self.resonance_factor *= 1.0 - self.adaptation_rate;
            self.stability_violations += 1;
        } else if is_vanishing {
            // Increase resonance to maintain signal flow
            self.resonance_factor *= 1.0 + self.adaptation_rate;
            self.stability_violations += 1;
        } else {
            // Gradual decay towards baseline
            self.resonance_factor *= self.decay_rate;
        }
        
        // Clamp to valid range
        self.resonance_factor = self.resonance_factor.clamp(self.min_resonance, self.max_resonance);
    }
    
    /// Update resonance history
    fn update_history(&mut self, resonance: f32) {
        self.resonance_history.push(resonance);
        
        // Maintain fixed capacity
        if self.resonance_history.len() > self.history_capacity {
            self.resonance_history.remove(0);
        }
        
        // Update average
        if !self.resonance_history.is_empty() {
            self.avg_resonance_factor = self.resonance_history.iter().sum::<f32>() / 
                                      self.resonance_history.len() as f32;
        }
    }
    
    /// Compute resonance stability metric
    fn compute_stability_metric(&self) -> f32 {
        if self.resonance_history.len() < 2 {
            return 1.0;
        }
        
        let mut variance_sum = 0.0;
        let mean = self.avg_resonance_factor;
        
        for &val in &self.resonance_history {
            let diff = val - mean;
            variance_sum += diff * diff;
        }
        
        let variance = variance_sum / self.resonance_history.len() as f32;
        let std_dev = variance.sqrt();
        
        // Stability = 1 / (1 + std_dev)
        1.0 / (1.0 + std_dev)
    }
    
    /// Apply momentum smoothing
    fn apply_momentum_smoothing(&self, current_resonance: f32) -> f32 {
        if self.total_resonance_applications == 0 {
            return current_resonance;
        }
        
        // Exponential moving average
        self.momentum_factor * current_resonance + 
        (1.0 - self.momentum_factor) * self.avg_resonance_factor
    }
    
    /// Get resonance statistics
    pub fn get_resonance_stats(&self) -> (f32, f32, f32, usize, usize) {
        (
            self.resonance_factor,
            self.avg_resonance_factor,
            self.compute_stability_metric(),
            self.total_resonance_applications,
            self.stability_violations,
        )
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.resonance_history.clear();
        self.total_resonance_applications = 0;
        self.stability_violations = 0;
        self.avg_resonance_factor = self.resonance_factor;
    }
    
    /// Set adaptation parameters
    pub fn set_adaptation_params(&mut self, 
        stability_threshold: f32,
        adaptation_rate: f32,
        min_resonance: f32,
        max_resonance: f32
    ) {
        self.stability_threshold = stability_threshold;
        self.adaptation_rate = adaptation_rate;
        self.min_resonance = min_resonance;
        self.max_resonance = max_resonance;
    }
    
    /// Get stability score
    pub fn get_stability_score(&self) -> f32 {
        let stability_metric = self.compute_stability_metric();
        let violation_rate = if self.total_resonance_applications > 0 {
            self.stability_violations as f32 / self.total_resonance_applications as f32
        } else {
            0.0
        };
        
        // Combine stability and violation rate
        stability_metric * (1.0 - violation_rate)
    }
}

impl GradientResonance for AdaptiveGradientResonance {
    fn compute_resonance(&self, 
        current_state: &ArrayD<f32>,
        previous_state: &ArrayD<f32>
    ) -> DLResult<f32> {
        
        let current_norm = self.compute_norm(current_state);
        let previous_norm = self.compute_norm(previous_state);
        
        if previous_norm == 0.0 {
            return Ok(self.resonance_factor);
        }
        
        // Base resonance from norm ratio
        let norm_ratio = current_norm / previous_norm;
        let base_resonance = self.resonance_factor * norm_ratio.log10().abs();
        
        // Adjust by cosine similarity
        let similarity = self.compute_cosine_similarity(current_state, previous_state)?;
        let similarity_factor = 1.0 - similarity; // Lower similarity = higher resonance needed
        
        // Combined resonance
        let combined_resonance = base_resonance * (1.0 + similarity_factor);
        
        Ok(combined_resonance.clamp(0.0, 1.0))
    }
    
    fn apply_resonance(&self,
        candidate_state: &ArrayD<f32>,
        previous_state: &ArrayD<f32>,
        resonance_factor: f32
    ) -> DLResult<ArrayD<f32>> {
        
        let cand_flat = candidate_state.as_slice().unwrap();
        let prev_flat = previous_state.as_slice().unwrap();
        
        if cand_flat.len() != prev_flat.len() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![cand_flat.len()],
                actual: vec![prev_flat.len()],
            });
        }
        
        let mut resonated_state = Vec::with_capacity(cand_flat.len());
        
        for (cand_val, prev_val) in cand_flat.iter().zip(prev_flat.iter()) {
            // Apply resonance: h_tilde = h_candidate + resonance * h_previous
            let resonated = cand_val + resonance_factor * prev_val;
            resonated_state.push(resonated);
        }
        
        Ok(Array1::from_vec(resonated_state).into_dyn())
    }
    
    fn check_stability(&self, resonance_history: &[f32]) -> bool {
        if resonance_history.len() < 10 {
            return true; // Not enough data
        }
        
        // Compute variance of recent resonance values
        let recent = &resonance_history[resonance_history.len().saturating_sub(10)..];
        let mean = recent.iter().sum::<f32>() / recent.len() as f32;
        
        let mut variance = 0.0;
        for &val in recent {
            variance += (val - mean).powi(2);
        }
        variance /= recent.len() as f32;
        
        let std_dev = variance.sqrt();
        
        // Stable if standard deviation is below threshold
        std_dev < self.stability_threshold
    }
}

/// Advanced resonance strategies
impl AdaptiveGradientResonance {
    /// Frequency-aware resonance untuk oscillatory patterns
    pub fn frequency_aware_resonance(&mut self, 
        current_state: &ArrayD<f32>,
        previous_state: &ArrayD<f32>
    ) -> DLResult<ArrayD<f32>> {
        
        // Detect oscillation frequency
        let similarity = self.compute_cosine_similarity(current_state, previous_state)?;
        let oscillation_indicator = (1.0 - similarity).abs();
        
        // Adjust resonance based on oscillation frequency
        let frequency_adjusted_resonance = if oscillation_indicator > 0.5 {
            // High oscillation - increase resonance for damping
            self.resonance_factor * 1.5
        } else {
            // Low oscillation - use normal resonance
            self.resonance_factor
        };
        
        let clamped_resonance = frequency_adjusted_resonance.clamp(self.min_resonance, self.max_resonance);
        
        self.apply_resonance(current_state, previous_state, clamped_resonance)
    }
    
    /// Multi-scale resonance untuk different time scales
    pub fn multi_scale_resonance(&self,
        current_state: &ArrayD<f32>,
        previous_states: &[ArrayD<f32>]
    ) -> DLResult<ArrayD<f32>> {
        
        if previous_states.is_empty() {
            return Ok(current_state.clone());
        }
        
        let mut multi_scale_resonated = current_state.clone();
        let current_flat = multi_scale_resonated.as_slice_mut().unwrap();
        
        // Apply resonance at different time scales
        for (scale, prev_state) in previous_states.iter().enumerate() {
            let scale_factor = 1.0 / (scale + 1) as f32; // Decreasing influence for older states
            let scale_resonance = self.resonance_factor * scale_factor;
            
            let prev_flat = prev_state.as_slice().unwrap();
            for (i, &prev_val) in prev_flat.iter().enumerate().take(current_flat.len()) {
                current_flat[i] += scale_resonance * prev_val;
            }
        }
        
        Ok(multi_scale_resonated)
    }
    
    /// Adaptive resonance dengan gradient feedback
    pub fn gradient_feedback_resonance(&mut self,
        candidate_state: &ArrayD<f32>,
        previous_state: &ArrayD<f32>,
        gradient_norm: f32
    ) -> DLResult<ArrayD<f32>> {
        
        // Adjust resonance based on gradient magnitude
        let gradient_adjusted_resonance = if gradient_norm > 1.0 {
            // Large gradients - increase resonance for stability
            self.resonance_factor * (1.0 + gradient_norm.log10())
        } else if gradient_norm < 0.01 {
            // Small gradients - decrease resonance to avoid stagnation
            self.resonance_factor * gradient_norm
        } else {
            // Normal gradients - use standard resonance
            self.resonance_factor
        };
        
        let clamped_resonance = gradient_adjusted_resonance.clamp(self.min_resonance, self.max_resonance);
        
        // Update statistics
        self.total_resonance_applications += 1;
        self.update_history(clamped_resonance);
        
        self.apply_resonance(candidate_state, previous_state, clamped_resonance)
    }
    
    /// Predictive resonance untuk anticipating future states
    pub fn predictive_resonance(&self,
        current_state: &ArrayD<f32>,
        previous_state: &ArrayD<f32>,
        trend_direction: f32 // -1.0 to 1.0
    ) -> DLResult<ArrayD<f32>> {
        
        // Predictive resonance based on trend
        let predictive_factor = 1.0 + trend_direction * 0.5;
        let predictive_resonance = self.resonance_factor * predictive_factor;
        
        let clamped_resonance = predictive_resonance.clamp(self.min_resonance, self.max_resonance);
        
        self.apply_resonance(current_state, previous_state, clamped_resonance)
    }
}
