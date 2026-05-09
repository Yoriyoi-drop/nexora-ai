//! Variance-Optimized Gradient Penalty Plus (VOGP+)
//!
//! Framework Training AI Hemat Dataset Berbasis Adaptive Smoothness, 
//! Consistency Learning, dan Variance Stabilization
//!
//! ## Komponen Utama:
//! - Adaptive Smoothness Penalty dengan EMA stabilization
//! - Consistency Learning untuk virtual dataset expansion
//! - Variance Stabilization dengan uncertainty awareness
//! - Scale-invariant gradient normalization
//!
//! ## Formula Final:
//! L_VOGP+(θ) = (1/|B|) Σ ℓ(y_i, f_θ(x_i)) + λL_smooth + γL_cons
//!
//! dengan:
//! - L_smooth: Adaptive smoothness dengan EMA gradien historis
//! - L_cons: Consistency learning antara sampel asli dan augmentasi
//! - σ_eff: Variansi efektif = α·Var(f) + (1-α)·H(f)

use std::collections::HashMap;
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, IxDyn, s};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

pub mod config;
pub mod ema;
pub mod smoothness;
pub mod consistency;
pub mod variance;
pub mod utils;

// Re-export main components
pub use config::*;
pub use ema::*;
pub use smoothness::*;
pub use consistency::*;
pub use variance::*;
pub use utils::*;

/// Struktur utama VOGP+ untuk training dengan dataset kecil
#[derive(Debug, Clone)]
pub struct VOGPPlus {
    /// Konfigurasi hyperparameter VOGP+
    config: VOGPConfig,
    /// EMA stabilizer untuk gradien historis
    ema_stabilizer: EMAStabilizer,
    /// Statistik batch untuk adaptive threshold
    batch_stats: BatchStatistics,
    /// Counter untuk update iterasi
    step_counter: usize,
}

impl VOGPPlus {
    /// Buat instance VOGP+ baru dengan konfigurasi default
    pub fn new() -> Self {
        Self::with_config(VOGPConfig::default())
    }

    /// Buat instance VOGP+ dengan konfigurasi kustom
    pub fn with_config(config: VOGPConfig) -> Self {
        info!("Inisialisasi VOGP+ dengan config: {:?}", config);
        
        Self {
            ema_stabilizer: EMAStabilizer::new(config.ema_beta),
            batch_stats: BatchStatistics::new(),
            step_counter: 0,
            config,
        }
    }

    /// Hitung total loss VOGP+ untuk satu batch
    /// 
    /// # Arguments
    /// * `predictions` - Prediksi model untuk batch asli [batch_size, num_classes]
    /// * `targets` - Target labels [batch_size]
    /// * `augmented_predictions` - Prediksi untuk data augmentasi [batch_size, num_classes]
    /// * `gradients` - Gradien model (untuk smoothness penalty)
    /// 
    /// # Returns
    /// Tuple (total_loss, component_losses) di mana component_losses berisi detail masing-masing komponen
    pub fn compute_loss(
        &mut self,
        predictions: &Array2<f32>,
        targets: &Array1<usize>,
        augmented_predictions: &Array2<f32>,
        gradients: Option<&ArrayD<f32>>,
    ) -> (f32, VOGPLossComponents) {
        self.step_counter += 1;
        
        debug!("Step {}: Menghitung VOGP+ loss", self.step_counter);
        
        // 1. Hitung primary loss (cross-entropy/MSE)
        let primary_loss = self.compute_primary_loss(predictions, targets);
        
        // 2. Hitung adaptive smoothness penalty
        let smoothness_loss = if let Some(grad) = gradients {
            self.compute_adaptive_smoothness_loss(grad, predictions)
        } else {
            warn!("Gradients tidak disediakan, smoothness penalty diabaikan");
            0.0
        };
        
        // 3. Hitung consistency learning loss
        let consistency_loss = self.compute_consistency_loss(predictions, augmented_predictions);
        
        // 4. Update EMA dan batch statistics
        self.update_statistics(predictions, gradients);
        
        // 5. Hitung total loss dengan bobot
        let total_loss = primary_loss 
            + self.config.lambda_smooth * smoothness_loss 
            + self.config.gamma_consistency * consistency_loss;
        
        let components = VOGPLossComponents {
            primary_loss,
            smoothness_loss,
            consistency_loss,
            total_loss,
            effective_variance: self.get_effective_variance(predictions),
            ema_gradient_norm: self.ema_stabilizer.get_current_norm(),
        };
        
        debug!(
            "Loss components - Primary: {:.4}, Smoothness: {:.4}, Consistency: {:.4}, Total: {:.4}",
            primary_loss, smoothness_loss, consistency_loss, total_loss
        );
        
        (total_loss, components)
    }

    /// Hitung primary loss (cross-entropy untuk klasifikasi)
    fn compute_primary_loss(&self, predictions: &Array2<f32>, targets: &Array1<usize>) -> f32 {
        let batch_size = predictions.dim().0;
        let mut total_loss = 0.0;
        
        for i in 0..batch_size {
            let pred = predictions.row(i);
            let target = targets[i];
            
            // Cross-entropy loss: -log(p_target)
            let prob = pred[target].max(1e-8); // Prevent log(0)
            total_loss += -prob.ln();
        }
        
        total_loss / batch_size as f32
    }

    /// Hitung adaptive smoothness penalty dengan EMA stabilization
    fn compute_adaptive_smoothness_loss(&mut self, gradients: &ArrayD<f32>, predictions: &Array2<f32>) -> f32 {
        // Hitung norm gradien saat ini
        let current_grad_norm = self.compute_gradient_norm(gradients);
        
        // Update EMA dengan gradien saat ini
        self.ema_stabilizer.update(current_grad_norm);
        
        // Hitung effective variance
        let effective_variance = self.get_effective_variance(predictions);
        
        // Adaptive threshold dengan EMA dan variance
        let adaptive_threshold = self.config.adaptive_threshold 
            * (self.config.epsilon + self.ema_stabilizer.get_current_norm())
            / (self.config.epsilon + effective_variance.sqrt());
        
        // Smoothness penalty dengan max(0, ...)^2
        let penalty = if current_grad_norm > adaptive_threshold {
            let diff = current_grad_norm - adaptive_threshold;
            diff * diff
        } else {
            0.0
        };
        
        debug!(
            "Smoothness - Grad norm: {:.4}, EMA: {:.4}, Threshold: {:.4}, Penalty: {:.4}",
            current_grad_norm, self.ema_stabilizer.get_current_norm(), adaptive_threshold, penalty
        );
        
        penalty
    }

    /// Hitung consistency learning loss
    fn compute_consistency_loss(&self, predictions: &Array2<f32>, augmented_predictions: &Array2<f32>) -> f32 {
        let batch_size = predictions.dim().0;
        let mut total_loss = 0.0;
        
        for i in 0..batch_size {
            let pred = predictions.row(i);
            let aug_pred = augmented_predictions.row(i);
            
            // L2 consistency loss: ||f(x) - f(x~)||^2
            let diff = &pred - &aug_pred;
            let l2_norm = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
            total_loss += l2_norm;
        }
        
        total_loss / batch_size as f32
    }

    /// Hitung effective variance: σ_eff = α·Var(f) + (1-α)·H(f)
    fn get_effective_variance(&self, predictions: &Array2<f32>) -> f32 {
        let variance = self.compute_prediction_variance(predictions);
        let entropy = self.compute_prediction_entropy(predictions);
        
        self.config.alpha_variance * variance + (1.0 - self.config.alpha_variance) * entropy
    }

    /// Hitung variance dari prediksi
    fn compute_prediction_variance(&self, predictions: &Array2<f32>) -> f32 {
        let batch_size = predictions.dim().0;
        let num_classes = predictions.dim().1;
        
        // Hitung mean per class
        let mut class_means = Array1::zeros(num_classes);
        for i in 0..batch_size {
            class_means += &predictions.row(i);
        }
        class_means /= batch_size as f32;
        
        // Hitung variance
        let mut total_variance = 0.0;
        for i in 0..batch_size {
            let diff = &predictions.row(i) - &class_means;
            total_variance += diff.iter().map(|x| x * x).sum::<f32>();
        }
        
        total_variance / (batch_size * num_classes) as f32
    }

    /// Hitung entropy prediksi (rata-rata Shannon entropy)
    fn compute_prediction_entropy(&self, predictions: &Array2<f32>) -> f32 {
        let batch_size = predictions.dim().0;
        let mut total_entropy = 0.0;
        
        for i in 0..batch_size {
            let pred = predictions.row(i);
            // Normalize to probabilities
            let sum: f32 = pred.iter().sum();
            if sum > 0.0 {
                let mut entropy = 0.0;
                for &p in pred.iter() {
                    if p > 0.0 {
                        let prob = p / sum;
                        entropy -= prob * prob.ln();
                    }
                }
                total_entropy += entropy;
            }
        }
        
        total_entropy / batch_size as f32
    }

    /// Hitung norm gradien (L2 norm)
    fn compute_gradient_norm(&self, gradients: &ArrayD<f32>) -> f32 {
        gradients.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Update statistik batch untuk adaptive learning
    fn update_statistics(&mut self, predictions: &Array2<f32>, gradients: Option<&ArrayD<f32>>) {
        self.batch_stats.update(predictions, gradients);
    }

    /// Reset semua statistik (berguna untuk epoch baru)
    pub fn reset_statistics(&mut self) {
        self.batch_stats.reset();
        self.ema_stabilizer.reset();
        self.step_counter = 0;
        info!("VOGP+ statistics di-reset");
    }

    /// Get current configuration
    pub fn get_config(&self) -> &VOGPConfig {
        &self.config
    }

    /// Update configuration runtime
    pub fn update_config(&mut self, config: VOGPConfig) {
        info!("Update VOGP+ config");
        let ema_beta = config.ema_beta;
        self.config = config;
        self.ema_stabilizer.update_beta(ema_beta);
    }
}

/// Komponen-komponen loss untuk debugging dan monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOGPLossComponents {
    pub primary_loss: f32,
    pub smoothness_loss: f32,
    pub consistency_loss: f32,
    pub total_loss: f32,
    pub effective_variance: f32,
    pub ema_gradient_norm: f32,
}

/// Statistik batch untuk adaptive threshold
#[derive(Debug, Clone)]
struct BatchStatistics {
    mean_gradient_norm: f32,
    variance_gradient_norm: f32,
    batch_count: usize,
}

impl BatchStatistics {
    fn new() -> Self {
        Self {
            mean_gradient_norm: 0.0,
            variance_gradient_norm: 0.0,
            batch_count: 0,
        }
    }

    fn update(&mut self, predictions: &Array2<f32>, gradients: Option<&ArrayD<f32>>) {
        self.batch_count += 1;
        
        if let Some(grad) = gradients {
            let grad_norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            // Update running mean dan variance
            let alpha = 0.1; // Learning rate untuk statistics
            self.mean_gradient_norm = (1.0 - alpha) * self.mean_gradient_norm + alpha * grad_norm;
            
            let diff = grad_norm - self.mean_gradient_norm;
            self.variance_gradient_norm = (1.0 - alpha) * self.variance_gradient_norm + alpha * diff * diff;
        }
    }

    fn reset(&mut self) {
        self.mean_gradient_norm = 0.0;
        self.variance_gradient_norm = 0.0;
        self.batch_count = 0;
    }
}

/// Utility functions untuk VOGP+
pub mod vogp_utils {
    use super::*;
    use ndarray_rand::RandomExt;

    /// Generate random noise untuk stochastic gradient approximation
    /// ∥∇_x f(x)∥² ≈ E_{v∼N(0,I)} [v^T ∇_x f(x)]
    pub fn generate_noise_vector(shape: &[usize]) -> ArrayD<f32> {
        ArrayD::zeros(IxDyn(shape)).mapv(|_: f32| -> f32 {
            // Sample dari N(0,1)
            let mut rng = thread_rng();
            rng.sample::<f32, _>(StandardNormal)
        })
    }

    /// Approximate gradient norm dengan stochastic sampling
    pub fn approximate_gradient_norm<F>(gradient_fn: F, input_shape: &[usize], num_samples: usize) -> f32 
    where 
        F: Fn(&ArrayD<f32>) -> ArrayD<f32>,
    {
        let mut total_norm_sq = 0.0;
        
        for _ in 0..num_samples {
            let noise = generate_noise_vector(input_shape);
            let gradient = gradient_fn(&noise);
            let dot_product = (&noise * &gradient).sum();
            total_norm_sq += dot_product * dot_product;
        }
        
        (total_norm_sq / num_samples as f32).sqrt()
    }

    /// Apply data augmentation untuk consistency learning
    pub fn apply_augmentation(data: &Array2<f32>, augmentation_type: AugmentationType) -> Array2<f32> {
        match augmentation_type {
            AugmentationType::GaussianNoise { std } => {
                let mut rng = thread_rng();
                data.mapv(|x| {
                    let noise: f32 = rng.sample::<f32, _>(StandardNormal) * std;
                    x + noise
                })
            }
            AugmentationType::Flip => {
                // Flip horizontal (sederhana)
                data.slice(s![.., ..;-1]).to_owned()
            }
            AugmentationType::Crop { ratio } => {
                let (batch_size, features) = data.dim();
                let crop_size = (features as f32 * ratio) as usize;
                data.slice(s![.., 0..crop_size]).to_owned()
            }
            AugmentationType::None => data.clone(),
        }
    }
}

/// Tipe augmentasi yang didukung
#[derive(Debug, Clone)]
pub enum AugmentationType {
    None,
    GaussianNoise { std: f32 },
    Flip,
    Crop { ratio: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_vogp_initialization() {
        let vogp = VOGPPlus::new();
        assert_eq!(vogp.step_counter, 0);
    }

    #[test]
    fn test_primary_loss_computation() {
        let vogp = VOGPPlus::new();
        let predictions = Array::from_elem((2, 3), 0.5);
        let targets = Array::from_vec(vec![0, 1]);
        
        let loss = vogp.compute_primary_loss(&predictions, &targets);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_consistency_loss() {
        let vogp = VOGPPlus::new();
        let predictions = Array::from_elem((2, 3), 0.5);
        let augmented = Array::from_elem((2, 3), 0.6);
        
        let loss = vogp.compute_consistency_loss(&predictions, &augmented);
        assert!(loss > 0.0);
    }
}
