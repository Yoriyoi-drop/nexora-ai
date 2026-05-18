//! Training Integration untuk VOGP+
//!
//! Integrasi VOGP+ dengan training framework yang ada di foundation crate.
//! Menyediakan wrapper dan adapter untuk seamless integration.

use nexora_hldva_t::training::{TrainingConfig, TrainingState};
use ndarray::{Array1, Array2, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// VOGP+ Training Wrapper
/// 
/// Mengintegrasikan VOGP+ dengan training framework HLDVA-T yang sudah ada.
/// Menyediakan interface yang konsisten untuk training dengan regularisasi VOGP+.
#[derive(Debug, Clone)]
pub struct VOGPTrainingWrapper {
    /// VOGP+ instance
    vogp_plus: super::VOGPPlus,
    /// Training configuration
    training_config: VOGPTrainingConfig,
    /// Training state
    training_state: VOGPTrainingState,
    /// Metrics collector
    metrics_collector: super::utils::MetricsCollector,
}

/// Konfigurasi training dengan VOGP+
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOGPTrainingConfig {
    /// Base training configuration dari HLDVA-T
    pub base_config: TrainingConfig,
    /// VOGP+ configuration
    pub vogp_config: super::VOGPConfig,
    /// Enable VOGP+ regularization
    pub enable_vogp: bool,
    /// Frequency untuk VOGP+ update (setiap N steps)
    pub vogp_update_frequency: usize,
    /// Enable early stopping berdasarkan VOGP+ metrics
    pub enable_early_stopping: bool,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Enable learning rate scheduling berdasarkan VOGP+ metrics
    pub enable_adaptive_lr: bool,
    /// Learning rate adaptation factor
    pub lr_adaptation_factor: f32,
}

impl Default for VOGPTrainingConfig {
    fn default() -> Self {
        Self {
            base_config: TrainingConfig::default(),
            vogp_config: super::VOGPConfig::default(),
            enable_vogp: true,
            vogp_update_frequency: 1,
            enable_early_stopping: false,
            early_stopping_patience: 10,
            enable_adaptive_lr: false,
            lr_adaptation_factor: 0.1,
        }
    }
}

/// Training state untuk VOGP+
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOGPTrainingState {
    /// Base training state
    pub base_state: TrainingState,
    /// Current step
    pub current_step: usize,
    /// Total VOGP+ loss
    pub total_vogp_loss: f32,
    /// Best validation loss untuk early stopping
    pub best_validation_loss: f32,
    /// Steps since best validation loss
    pub steps_since_best: usize,
    /// Current learning rate
    pub current_learning_rate: f32,
    /// Training statistics
    pub statistics: VOGPTrainingStatistics,
    /// Previous consistency similarity for temporal smoothing
    pub previous_consistency_similarity: Option<f32>,
}

/// Training statistics untuk monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOGPTrainingStatistics {
    /// Average primary loss
    pub avg_primary_loss: f32,
    /// Average smoothness loss
    pub avg_smoothness_loss: f32,
    /// Average consistency loss
    pub avg_consistency_loss: f32,
    /// Average effective variance
    pub avg_effective_variance: f32,
    /// Training stability score
    pub stability_score: f32,
    /// Convergence rate
    pub convergence_rate: f32,
}

impl VOGPTrainingWrapper {
    /// Buat VOGP+ training wrapper baru
    pub fn new(config: VOGPTrainingConfig) -> Self {
        info!("Inisialisasi VOGP+ Training Wrapper");
        
        Self {
            vogp_plus: super::VOGPPlus::with_config(config.vogp_config.clone()),
            training_config: config,
            training_state: VOGPTrainingState {
                base_state: TrainingState::new(),
                current_step: 0,
                total_vogp_loss: 0.0,
                best_validation_loss: f32::INFINITY,
                steps_since_best: 0,
                current_learning_rate: 0.001,
                statistics: VOGPTrainingStatistics {
                    avg_primary_loss: 0.0,
                    avg_smoothness_loss: 0.0,
                    avg_consistency_loss: 0.0,
                    avg_effective_variance: 0.0,
                    stability_score: 1.0,
                    convergence_rate: 0.0,
                },
                previous_consistency_similarity: None,
            },
            metrics_collector: super::utils::MonitoringUtils::create_metrics_collector(),
        }
    }

    /// Execute training step dengan VOGP+ regularization
    pub fn training_step(
        &mut self,
        batch_data: &Array2<f32>,
        batch_targets: &Array1<usize>,
        model_forward: impl Fn(&Array2<f32>) -> Array2<f32>,
        model_backward: impl Fn(&Array2<f32>, &Array1<usize>) -> Option<ArrayD<f32>>,
    ) -> VOGPTrainingResult {
        let start_time = std::time::Instant::now();
        self.training_state.current_step += 1;

        debug!("Executing VOGP+ training step {}", self.training_state.current_step);

        // 1. Forward pass
        let predictions = model_forward(batch_data);

        // 2. Generate augmentations untuk consistency learning
        let augmented_data = if self.training_config.enable_vogp {
            self.generate_augmentations(batch_data)
        } else {
            Vec::new()
        };

        // 3. Forward pass untuk augmentations
        let augmented_predictions: Vec<Array2<f32>> = if self.training_config.enable_vogp {
            augmented_data.iter()
                .map(|aug_data| model_forward(aug_data))
                .collect()
        } else {
            Vec::new()
        };

        // 4. Compute gradients
        let gradients = if self.training_config.enable_vogp {
            model_backward(&predictions, batch_targets)
        } else {
            None
        };

        // 5. Compute VOGP+ loss
        let (vogp_loss, components) = if self.training_config.enable_vogp 
            && self.training_state.current_step % self.training_config.vogp_update_frequency == 0 {
            // Combine all augmented predictions untuk consistency
            let combined_aug_predictions = if !augmented_predictions.is_empty() {
                let batch_size = predictions.dim().0;
                let num_classes = predictions.dim().1;
                let mut combined = Array2::zeros((batch_size, num_classes));
                
                for aug_pred in &augmented_predictions {
                    combined += aug_pred;
                }
                combined / augmented_predictions.len() as f32
            } else {
                // Use slice reference instead of clone when no augmentation
                predictions.to_owned()
            };

            self.vogp_plus.compute_loss(
                &predictions,
                batch_targets,
                &combined_aug_predictions,
                gradients.as_ref(),
            )
        } else {
            (0.0, super::VOGPLossComponents {
                primary_loss: 0.0,
                smoothness_loss: 0.0,
                consistency_loss: 0.0,
                total_loss: 0.0,
                effective_variance: 0.0,
                ema_gradient_norm: 0.0,
            })
        };

        // 6. Update training statistics
        self.update_training_statistics(&components);

        // 7. Check untuk early stopping
        let should_stop = if self.training_config.enable_early_stopping {
            self.check_early_stopping(components.total_loss)
        } else {
            false
        };

        // 8. Adaptive learning rate adjustment
        if self.training_config.enable_adaptive_lr {
            self.adjust_learning_rate(&components);
        }

        // 9. Log metrics
        let training_time_ms = start_time.elapsed().as_millis() as u64;
        self.log_training_metrics(&components, training_time_ms);

        VOGPTrainingResult {
            step: self.training_state.current_step,
            loss: components.total_loss,
            components,
            should_stop,
            learning_rate: self.training_state.current_learning_rate,
            training_time_ms,
        }
    }

    /// Generate augmentations untuk consistency learning
    fn generate_augmentations(&self, data: &Array2<f32>) -> Vec<Array2<f32>> {
        let num_augmentations = self.training_config.vogp_config.augmentation_config.num_augmentations;
        let augmentation_strength = self.training_config.vogp_config.augmentation_config.augmentation_strength;
        
        let mut augmentations = Vec::with_capacity(num_augmentations);
        let mut rng = rand::thread_rng();
        
        for _ in 0..num_augmentations {
            let aug_data = super::utils::AugmentationUtils::apply_mixed_augmentation(
                data,
                augmentation_strength,
                &mut rng,
            );
            augmentations.push(aug_data);
        }
        
        augmentations
    }

    /// Update training statistics
    fn update_training_statistics(&mut self, components: &super::VOGPLossComponents) {
        let alpha = 0.01; // Smoothing factor
        
        self.training_state.statistics.avg_primary_loss = 
            (1.0 - alpha) * self.training_state.statistics.avg_primary_loss + alpha * components.primary_loss;
        
        self.training_state.statistics.avg_smoothness_loss = 
            (1.0 - alpha) * self.training_state.statistics.avg_smoothness_loss + alpha * components.smoothness_loss;
        
        self.training_state.statistics.avg_consistency_loss = 
            (1.0 - alpha) * self.training_state.statistics.avg_consistency_loss + alpha * components.consistency_loss;
        
        self.training_state.statistics.avg_effective_variance = 
            (1.0 - alpha) * self.training_state.statistics.avg_effective_variance + alpha * components.effective_variance;
        
        // Update stability score
        self.training_state.statistics.stability_score = 
            self.compute_stability_score();
        
        // Update convergence rate
        self.training_state.statistics.convergence_rate = 
            self.metrics_collector.get_loss_trend(10);
    }

    /// Compute stability score berdasarkan recent loss variance
    fn compute_stability_score(&self) -> f32 {
        let recent_metrics = self.metrics_collector.get_recent_metrics(20);
        if recent_metrics.len() < 5 {
            return 1.0;
        }
        
        let losses: Vec<f32> = recent_metrics.iter().map(|m| m.total_loss).collect();
        let mean = losses.iter().sum::<f32>() / losses.len() as f32;
        let variance = losses.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / losses.len() as f32;
        
        // Lower variance = higher stability
        (1.0 / (1.0 + variance)).min(1.0)
    }

    /// Check early stopping conditions
    fn check_early_stopping(&mut self, current_loss: f32) -> bool {
        if current_loss < self.training_state.best_validation_loss {
            self.training_state.best_validation_loss = current_loss;
            self.training_state.steps_since_best = 0;
            false
        } else {
            self.training_state.steps_since_best += 1;
            self.training_state.steps_since_best >= self.training_config.early_stopping_patience
        }
    }

    /// Adjust learning rate berdasarkan VOGP+ metrics
    fn adjust_learning_rate(&mut self, components: &super::VOGPLossComponents) {
        let stability = self.training_state.statistics.stability_score;
        
        if stability < 0.5 {
            // Low stability - reduce learning rate
            self.training_state.current_learning_rate *= (1.0 - self.training_config.lr_adaptation_factor);
        } else if stability > 0.8 && components.effective_variance > 1.0 {
            // High stability but high variance - can increase learning rate slightly
            self.training_state.current_learning_rate *= (1.0 + self.training_config.lr_adaptation_factor * 0.5);
        }
        
        // Clamp learning rate
        self.training_state.current_learning_rate = 
            self.training_state.current_learning_rate.clamp(1e-6, 1e-1);
    }

    /// Log training metrics
    fn log_training_metrics(&mut self, components: &super::VOGPLossComponents, training_time_ms: u64) {
        let metrics = super::utils::VOGPMetrics {
            step: self.training_state.current_step,
            total_loss: components.total_loss,
            primary_loss: components.primary_loss,
            smoothness_loss: components.smoothness_loss,
            consistency_loss: components.consistency_loss,
            effective_variance: components.effective_variance,
            ema_gradient_norm: components.ema_gradient_norm,
            adaptive_threshold: self.vogp_plus.get_config().adaptive_threshold,
            consistency_similarity: self.compute_consistency_similarity(&components),
            training_time_ms,
        };
        
        self.metrics_collector.add_metrics(metrics.clone());
        super::utils::MonitoringUtils::log_training_step(&metrics);
    }
    
    /// Compute consistency similarity from VOGP components
    fn compute_consistency_similarity(&self, components: &super::VOGPLossComponents) -> f32 {
        // Compute similarity between current and previous consistency states
        // Use a combination of consistency loss and smoothness metrics
        
        let consistency_ratio = if components.consistency_loss > 0.0 {
            components.smoothness_loss / components.consistency_loss
        } else {
            1.0
        };
        
        // Map ratio to similarity score (higher ratio = higher similarity)
        let similarity = (consistency_ratio / (1.0 + consistency_ratio)).max(0.0).min(1.0);
        
        // Apply temporal smoothing if available
        let smoothed_similarity = if let Some(prev_similarity) = self.training_state.previous_consistency_similarity {
            0.7 * prev_similarity + 0.3 * similarity // Exponential moving average
        } else {
            similarity
        };
        
        // Update previous similarity for next iteration
        self.training_state.previous_consistency_similarity = Some(smoothed_similarity);
        
        smoothed_similarity
    }

    /// Validation step dengan VOGP+ metrics
    pub fn validation_step(
        &mut self,
        val_data: &Array2<f32>,
        val_targets: &Array1<usize>,
        model_forward: impl Fn(&Array2<f32>) -> Array2<f32>,
    ) -> VOGPValidationResult {
        let predictions = model_forward(val_data);
        
        // Compute validation loss (tanpa VOGP+ regularization)
        let val_loss = self.vogp_plus.compute_primary_loss(&predictions, val_targets);
        
        // Compute VOGP+ metrics untuk monitoring
        let effective_variance = self.vogp_plus.get_effective_variance(&predictions);
        
        VOGPValidationResult {
            validation_loss: val_loss,
            effective_variance,
            step: self.training_state.current_step,
        }
    }

    /// Get current training state
    pub fn get_training_state(&self) -> &VOGPTrainingState {
        &self.training_state
    }

    /// Get training statistics
    pub fn get_statistics(&self) -> &VOGPTrainingStatistics {
        &self.training_state.statistics
    }

    /// Reset training state
    pub fn reset(&mut self) {
        self.vogp_plus.reset_statistics();
        self.training_state = VOGPTrainingState {
            base_state: TrainingState::new(),
            current_step: 0,
            total_vogp_loss: 0.0,
            best_validation_loss: f32::INFINITY,
            steps_since_best: 0,
            current_learning_rate: 0.001,
            statistics: VOGPTrainingStatistics {
                avg_primary_loss: 0.0,
                avg_smoothness_loss: 0.0,
                avg_consistency_loss: 0.0,
                avg_effective_variance: 0.0,
                stability_score: 1.0,
                convergence_rate: 0.0,
            },
            previous_consistency_similarity: None,
        };
        self.metrics_collector.reset();
        
        info!("VOGP+ training state di-reset");
    }

    /// Export training metrics
    pub fn export_metrics(&self) -> &Vec<super::utils::VOGPMetrics> {
        self.metrics_collector.export_metrics()
    }
}

/// Result dari training step
#[derive(Debug, Clone)]
pub struct VOGPTrainingResult {
    pub step: usize,
    pub loss: f32,
    pub components: super::VOGPLossComponents,
    pub should_stop: bool,
    pub learning_rate: f32,
    pub training_time_ms: u64,
}

/// Result dari validation step
#[derive(Debug, Clone)]
pub struct VOGPValidationResult {
    pub validation_loss: f32,
    pub effective_variance: f32,
    pub step: usize,
}

/// Training loop utilities
pub struct VOGPTrainingLoop;

impl VOGPTrainingLoop {
    /// Run complete training loop dengan VOGP+
    pub fn train<F, G>(
        wrapper: &mut VOGPTrainingWrapper,
        train_data: &Array2<f32>,
        train_targets: &Array1<usize>,
        val_data: &Array2<f32>,
        val_targets: &Array1<usize>,
        model_forward: F,
        model_backward: G,
        max_epochs: usize,
        batch_size: usize,
    ) -> VOGPTrainingSummary
    where
        F: Fn(&Array2<f32>) -> Array2<f32> + Clone,
        G: Fn(&Array2<f32>, &Array1<usize>) -> Option<ArrayD<f32>> + Clone,
    {
        info!("Memulai VOGP+ training loop");
        
        let mut summary = VOGPTrainingSummary {
            total_steps: 0,
            final_loss: 0.0,
            best_validation_loss: f32::INFINITY,
            training_time_ms: 0,
            convergence_achieved: false,
        };
        
        let start_time = std::time::Instant::now();
        
        for epoch in 0..max_epochs {
            info!("Epoch {}/{}", epoch + 1, max_epochs);
            
            // Training loop
            for (batch_idx, batch_start) in (0..train_data.dim().0).step_by(batch_size).enumerate() {
                let batch_end = (batch_start + batch_size).min(train_data.dim().0);
                
                let batch_data = train_data.slice(s![batch_start..batch_end, ..]).to_owned();
                let batch_targets = train_targets.slice(s![batch_start..batch_end]).to_owned();
                
                let result = wrapper.training_step(
                    &batch_data,
                    &batch_targets,
                    model_forward.clone(),
                    model_backward.clone(),
                );
                
                summary.total_steps += 1;
                
                if result.should_stop {
                    info!("Early stopping triggered at step {}", result.step);
                    summary.convergence_achieved = true;
                    break;
                }
                
                if batch_idx % 100 == 0 {
                    debug!("Batch {}: Loss {:.4}", batch_idx, result.loss);
                }
            }
            
            // Validation
            let val_result = wrapper.validation_step(val_data, val_targets, model_forward.clone());
            summary.best_validation_loss = summary.best_validation_loss.min(val_result.validation_loss);
            
            if epoch % 10 == 0 {
                info!("Epoch {} - Val Loss: {:.4}", epoch + 1, val_result.validation_loss);
            }
            
            if wrapper.get_training_state().steps_since_best > wrapper.training_config.early_stopping_patience {
                info!("Early stopping due to no improvement");
                break;
            }
        }
        
        summary.training_time_ms = start_time.elapsed().as_millis() as u64;
        summary.final_loss = wrapper.get_statistics().avg_primary_loss;
        
        info!("VOGP+ training selesai - Total steps: {}, Final loss: {:.4}", 
              summary.total_steps, summary.final_loss);
        
        summary
    }
}

/// Training summary
#[derive(Debug, Clone)]
pub struct VOGPTrainingSummary {
    pub total_steps: usize,
    pub final_loss: f32,
    pub best_validation_loss: f32,
    pub training_time_ms: u64,
    pub convergence_achieved: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_vogp_training_wrapper() {
        let config = VOGPTrainingConfig::default();
        let mut wrapper = VOGPTrainingWrapper::new(config);
        
        // Dummy data
        let data = Array::from_elem((4, 10), 0.5);
        let targets = Array::from_vec(vec![0, 1, 0, 1]);
        
        // Dummy model
        let model_forward = |input: &Array2<f32>| input.clone();
        let model_backward = |_: &Array2<f32>, _: &Array1<usize>| None;
        
        let result = wrapper.training_step(&data, &targets, model_forward, model_backward);
        
        assert!(result.step > 0);
        assert!(result.loss >= 0.0);
    }

    #[test]
    fn test_vogp_validation() {
        let config = VOGPTrainingConfig::default();
        let mut wrapper = VOGPTrainingWrapper::new(config);
        
        let val_data = Array::from_elem((2, 10), 0.5);
        let val_targets = Array::from_vec(vec![0, 1]);
        
        let model_forward = |input: &Array2<f32>| input.clone();
        
        let result = wrapper.validation_step(&val_data, &val_targets, model_forward);
        
        assert!(result.validation_loss >= 0.0);
        assert!(result.effective_variance >= 0.0);
    }
}
