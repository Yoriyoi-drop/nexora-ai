//! Adaptive Smoothness Penalty untuk VOGP+
//!
//! Implementasi smoothness penalty yang adaptif dengan EMA stabilization
//! untuk membuat training stabil lintas arsitektur dan scale-invariant.

use ndarray::{ArrayD, ArrayViewD, s};
use serde::{Deserialize, Serialize};

/// Adaptive Smoothness Penalty Component
/// 
/// Formula: L_smooth = E_{x~∼D_aug} [max(0, ϵ+EMA(∥∇_x f_θ(x)∥²) - τ·(ϵ+μ_grad)/σ_eff)²]
#[derive(Debug, Clone)]
pub struct AdaptiveSmoothnessPenalty {
    /// EMA dari gradient norm historis
    pub ema_gradient_norm: f32,
    /// Adaptive threshold τ
    pub adaptive_threshold: f32,
    /// Konstanta stabilitas numerik ϵ
    pub epsilon: f32,
    /// Mean gradien batch μ_grad
    pub mean_gradient_batch: f32,
    /// Effective variance σ_eff
    pub effective_variance: f32,
    /// Konfigurasi smoothness
    config: SmoothnessConfig,
}

/// Konfigurasi untuk smoothness penalty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothnessConfig {
    /// Base threshold untuk smoothness penalty
    pub base_threshold: f32,
    /// Scale factor untuk adaptive threshold
    pub scale_factor: f32,
    /// Enable variance-aware scaling
    pub enable_variance_scaling: bool,
    /// Enable gradient normalization
    pub enable_gradient_normalization: bool,
    /// Maximum penalty value untuk clipping
    pub max_penalty: f32,
    /// Minimum threshold untuk prevent over-penalization
    pub min_threshold: f32,
}

impl Default for SmoothnessConfig {
    fn default() -> Self {
        Self {
            base_threshold: 1.0,
            scale_factor: 1.0,
            enable_variance_scaling: true,
            enable_gradient_normalization: true,
            max_penalty: 10.0,
            min_threshold: 0.1,
        }
    }
}

impl AdaptiveSmoothnessPenalty {
    /// Buat adaptive smoothness penalty baru
    pub fn new(config: SmoothnessConfig) -> Self {
        Self {
            ema_gradient_norm: 1.0,
            adaptive_threshold: config.base_threshold,
            epsilon: 1e-8,
            mean_gradient_batch: 0.0,
            effective_variance: 1.0,
            config,
        }
    }

    /// Hitung smoothness penalty untuk gradien saat ini
    pub fn compute_penalty(&mut self, current_gradient_norm: f32, effective_variance: f32) -> f32 {
        // Update effective variance
        self.effective_variance = effective_variance;
        
        // Hitung adaptive threshold dengan EMA dan variance
        self.compute_adaptive_threshold();
        
        // Hitung penalty dengan max(0, ...)^2
        let penalty = self.compute_smoothness_loss(current_gradient_norm);
        
        // Clip penalty untuk prevent explosion
        penalty.min(self.config.max_penalty)
    }

    /// Hitung adaptive threshold: τ·(ϵ+μ_grad)/σ_eff
    fn compute_adaptive_threshold(&mut self) {
        let variance_factor = if self.config.enable_variance_scaling {
            (self.epsilon + self.effective_variance.sqrt()).recip()
        } else {
            1.0
        };
        
        let gradient_factor = if self.config.enable_gradient_normalization {
            (self.epsilon + self.ema_gradient_norm) / (self.epsilon + self.mean_gradient_batch.abs())
        } else {
            1.0
        };
        
        self.adaptive_threshold = self.config.base_threshold 
            * self.config.scale_factor
            * variance_factor
            * gradient_factor;
        
        // Ensure minimum threshold
        self.adaptive_threshold = self.adaptive_threshold.max(self.config.min_threshold);
    }

    /// Hitung smoothness loss: max(0, current - threshold)^2
    fn compute_smoothness_loss(&self, current_gradient_norm: f32) -> f32 {
        if current_gradient_norm > self.adaptive_threshold {
            let diff = current_gradient_norm - self.adaptive_threshold;
            diff * diff
        } else {
            0.0
        }
    }

    /// Update EMA gradient norm
    pub fn update_ema(&mut self, new_gradient_norm: f32, ema_beta: f32) {
        self.ema_gradient_norm = ema_beta * self.ema_gradient_norm + (1.0 - ema_beta) * new_gradient_norm;
    }

    /// Update mean gradien batch
    pub fn update_batch_mean(&mut self, batch_gradients: &[f32]) {
        if !batch_gradients.is_empty() {
            self.mean_gradient_batch = batch_gradients.iter().sum::<f32>() / batch_gradients.len() as f32;
        }
    }

    /// Get current adaptive threshold
    pub fn get_adaptive_threshold(&self) -> f32 {
        self.adaptive_threshold
    }

    /// Get EMA gradient norm
    pub fn get_ema_gradient_norm(&self) -> f32 {
        self.ema_gradient_norm
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.ema_gradient_norm = 1.0;
        self.adaptive_threshold = self.config.base_threshold;
        self.mean_gradient_batch = 0.0;
        self.effective_variance = 1.0;
    }
}

/// Stochastic Gradient Approximation untuk smoothness penalty
/// 
/// Menggunakan stochastic approximation untuk menghitung norm Jacobian:
/// ∥∇_x f(x)∥² ≈ E_{v∼N(0,I)} [v^T ∇_x f(x)]
pub struct StochasticGradientApproximator {
    /// Jumlah sample untuk approximation
    num_samples: usize,
    /// Seed untuk reproducibility
    seed: u64,
    /// Enable variance reduction
    enable_variance_reduction: bool,
}

impl StochasticGradientApproximator {
    /// Buat stochastic gradient approximator baru
    pub fn new(num_samples: usize, seed: u64) -> Self {
        Self {
            num_samples,
            seed,
            enable_variance_reduction: true,
        }
    }

    /// Approximate gradient norm menggunakan stochastic sampling
    pub fn approximate_norm<F>(&self, gradient_fn: F, input_shape: &[usize]) -> f32 
    where 
        F: Fn(&ArrayD<f32>) -> ArrayD<f32>,
    {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut total_norm_sq = 0.0;
        let mut total_weight = 0.0;

        for _ in 0..self.num_samples {
            // Generate random noise vector v ~ N(0, I)
            let noise = ArrayD::from_shape_fn(input_shape, |_| {
                rng.sample::<f32, _>(StandardNormal)
            });
            
            // Compute gradient at noise point
            let gradient = gradient_fn(&noise);
            
            // Compute v^T ∇_x f(x)
            let dot_product = (&noise * &gradient).sum();
            
            // Accumulate squared norm
            let weight = if self.enable_variance_reduction { 1.0 } else { 1.0 };
            total_norm_sq += dot_product * dot_product * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            (total_norm_sq / total_weight).sqrt()
        } else {
            0.0
        }
    }

    /// Approximate gradient norm dengan control variate untuk variance reduction
    pub fn approximate_norm_with_control<F>(
        &self, 
        gradient_fn: F, 
        baseline_gradient: &ArrayD<f32>,
        input_shape: &[usize]
    ) -> f32 
    where 
        F: Fn(&ArrayD<f32>) -> ArrayD<f32>,
    {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut total_norm_sq = 0.0;

        for _ in 0..self.num_samples {
            // Generate random noise vector
            let noise = ArrayD::from_shape_fn(input_shape, |_| {
                rng.sample::<f32, _>(StandardNormal)
            });
            
            // Compute gradient at noise point
            let gradient = gradient_fn(&noise);
            
            // Control variate correction
            let corrected_gradient = &gradient - baseline_gradient;
            
            // Compute v^T ∇_x f(x) with correction
            let dot_product = (&noise * &corrected_gradient).sum();
            
            total_norm_sq += dot_product * dot_product;
        }
        
        (total_norm_sq / self.num_samples as f32).sqrt()
    }
}

/// Multi-scale Smoothness Penalty untuk berbagai level of detail
#[derive(Debug, Clone)]
pub struct MultiScaleSmoothnessPenalty {
    /// Smoothness penalties untuk berbagai scales
    scales: Vec<AdaptiveSmoothnessPenalty>,
    /// Weight untuk setiap scale
    scale_weights: Vec<f32>,
}

impl MultiScaleSmoothnessPenalty {
    /// Buat multi-scale smoothness penalty
    pub fn new(num_scales: usize, base_config: SmoothnessConfig) -> Self {
        let mut scales = Vec::with_capacity(num_scales);
        let mut scale_weights = Vec::with_capacity(num_scales);
        
        for i in 0..num_scales {
            // Adjust threshold untuk setiap scale
            let mut config = base_config.clone();
            config.base_threshold *= (i + 1) as f32; // Higher threshold for coarser scales
            
            scales.push(AdaptiveSmoothnessPenalty::new(config));
            
            // Weight: finer scales lebih penting
            let weight = 2.0_f32.powi(-(i as i32));
            scale_weights.push(weight);
        }
        
        // Normalize weights
        let total_weight: f32 = scale_weights.iter().sum();
        for weight in &mut scale_weights {
            *weight /= total_weight;
        }
        
        Self { scales, scale_weights }
    }

    /// Hitung multi-scale smoothness penalty
    pub fn compute_penalty(&mut self, gradient_norms: &[f32], effective_variance: f32) -> f32 {
        assert_eq!(gradient_norms.len(), self.scales.len(), "Scale mismatch");
        
        let mut total_penalty = 0.0;
        
        for (i, &norm) in gradient_norms.iter().enumerate() {
            let penalty = self.scales[i].compute_penalty(norm, effective_variance);
            total_penalty += self.scale_weights[i] * penalty;
        }
        
        total_penalty
    }

    /// Update EMA untuk semua scales
    pub fn update_ema(&mut self, new_gradient_norms: &[f32], ema_beta: f32) {
        assert_eq!(new_gradient_norms.len(), self.scales.len(), "Scale mismatch");
        
        for (i, &norm) in new_gradient_norms.iter().enumerate() {
            self.scales[i].update_ema(norm, ema_beta);
        }
    }

    /// Get adaptive thresholds untuk semua scales
    pub fn get_adaptive_thresholds(&self) -> Vec<f32> {
        self.scales.iter().map(|s| s.get_adaptive_threshold()).collect()
    }
}

/// Directional Smoothness Penalty untuk specific directions
#[derive(Debug, Clone)]
pub struct DirectionalSmoothnessPenalty {
    /// Smoothness penalties untuk berbagai directions
    directions: Vec<AdaptiveSmoothnessPenalty>,
    /// Direction vectors
    direction_vectors: Vec<ArrayD<f32>>,
}

impl DirectionalSmoothnessPenalty {
    /// Buat directional smoothness penalty
    pub fn new(num_directions: usize, feature_dim: usize, config: SmoothnessConfig) -> Self {
        let mut directions = Vec::with_capacity(num_directions);
        let mut direction_vectors = Vec::with_capacity(num_directions);
        
        for i in 0..num_directions {
            directions.push(AdaptiveSmoothnessPenalty::new(config.clone()));
            
            // Create unit vector untuk direction i
            let mut vec = ArrayD::zeros(vec![feature_dim]);
            vec[[i % feature_dim]] = 1.0; // Simple unit vectors
            direction_vectors.push(vec);
        }
        
        Self { directions, direction_vectors }
    }

    /// Hitung directional smoothness penalty
    pub fn compute_penalty(&mut self, gradient: &ArrayD<f32>, effective_variance: f32) -> f32 {
        let mut total_penalty = 0.0;
        
        for (i, direction_vec) in self.direction_vectors.iter().enumerate() {
            // Project gradient onto direction
            let directional_norm = (gradient * direction_vec).sum().abs();
            
            let penalty = self.directions[i].compute_penalty(directional_norm, effective_variance);
            total_penalty += penalty;
        }
        
        total_penalty / self.directions.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_adaptive_smoothness_penalty() {
        let config = SmoothnessConfig::default();
        let mut penalty = AdaptiveSmoothnessPenalty::new(config);
        
        let result = penalty.compute_penalty(2.0, 1.0);
        assert!(result >= 0.0);
        
        // Test dengan gradient norm lebih kecil dari threshold
        let result = penalty.compute_penalty(0.5, 1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_ema_update() {
        let config = SmoothnessConfig::default();
        let mut penalty = AdaptiveSmoothnessPenalty::new(config);
        
        let initial_ema = penalty.get_ema_gradient_norm();
        penalty.update_ema(2.0, 0.9);
        let updated_ema = penalty.get_ema_gradient_norm();
        
        assert!(updated_ema != initial_ema);
    }

    #[test]
    fn test_stochastic_approximation() {
        let approximator = StochasticGradientApproximator::new(10, 42);
        
        // Dummy gradient function
        let gradient_fn = |noise: &ArrayD<f32>| noise.clone();
        
        let norm = approximator.approximate_norm(gradient_fn, &[10]);
        assert!(norm >= 0.0);
    }

    #[test]
    fn test_multi_scale_penalty() {
        let config = SmoothnessConfig::default();
        let mut multi_penalty = MultiScaleSmoothnessPenalty::new(3, config);
        
        let gradient_norms = vec![1.0, 2.0, 3.0];
        let penalty = multi_penalty.compute_penalty(&gradient_norms, 1.0);
        
        assert!(penalty >= 0.0);
    }
}
