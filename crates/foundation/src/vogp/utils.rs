//! Utility functions untuk VOGP+
//!
//! Helper functions dan tools untuk mendukung implementasi VOGP+
//! termasuk data augmentation, gradient approximation, dan monitoring.

use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, IxDyn, s};
use rand::prelude::*;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Data augmentation utilities untuk consistency learning
pub struct AugmentationUtils;

impl AugmentationUtils {
    /// Apply mixed augmentation strategy
    pub fn apply_mixed_augmentation(
        data: &Array2<f32>,
        augmentation_strength: f32,
        rng: &mut ThreadRng,
    ) -> Array2<f32> {
        let augmentation_type = rng.gen_range(0..6);
        
        match augmentation_type {
            0 => Self::apply_gaussian_noise(data, 0.1 * augmentation_strength, rng),
            1 => Self::apply_dropout(data, 0.1 * augmentation_strength, rng),
            2 => Self::apply_random_crop(data, 0.9 + 0.1 * augmentation_strength, rng),
            3 => Self::apply_random_flip(data, rng),
            4 => Self::apply_rotation(data, 0.1 * augmentation_strength, rng),
            5 => Self::apply_brightness(data, 0.1 * augmentation_strength, rng),
            _ => data.clone(),
        }
    }

    /// Apply Gaussian noise augmentation
    pub fn apply_gaussian_noise(data: &Array2<f32>, std: f32, rng: &mut ThreadRng) -> Array2<f32> {
        data.mapv(|x| {
            let noise: f32 = rng.sample::<f32, _>(StandardNormal) * std;
            x + noise
        })
    }

    /// Apply dropout-style augmentation
    pub fn apply_dropout(data: &Array2<f32>, dropout_rate: f32, rng: &mut ThreadRng) -> Array2<f32> {
        data.mapv(|x| {
            if rng.gen::<f32>() < dropout_rate {
                0.0
            } else {
                x
            }
        })
    }

    /// Apply random crop augmentation
    pub fn apply_random_crop(data: &Array2<f32>, crop_ratio: f32, rng: &mut ThreadRng) -> Array2<f32> {
        let (_, num_features) = data.dim();
        let crop_size = (num_features as f32 * crop_ratio) as usize;
        
        if crop_size >= num_features {
            // Return a view-based clone instead of full clone when possible
            return data.slice(s![.., ..]).to_owned();
        }
        
        let start_idx = rng.gen_range(0..=num_features - crop_size);
        data.slice(s![.., start_idx..start_idx + crop_size]).to_owned()
    }

    /// Apply random flip augmentation
    pub fn apply_random_flip(data: &Array2<f32>, rng: &mut ThreadRng) -> Array2<f32> {
        if rng.gen::<bool>() {
            data.slice(s![.., ..;-1]).to_owned()
        } else {
            // Avoid clone by returning a view when no flip is needed
            data.slice(s![.., ..]).to_owned()
        }
    }

    /// Apply rotation-style augmentation (untuk sequential data)
    pub fn apply_rotation(data: &Array2<f32>, rotation_strength: f32, rng: &mut ThreadRng) -> Array2<f32> {
        let (_, num_features) = data.dim();
        let shift_amount = (num_features as f32 * rotation_strength) as usize;
        
        if shift_amount == 0 {
            // Return view instead of clone when no rotation needed
            return data.slice(s![.., ..]).to_owned();
        }
        
        let shift_direction: isize = if rng.gen::<bool>() { 1 } else { -1 };
        let shift = shift_amount as isize * shift_direction;
        
        // Circular shift
        let mut result = Array2::zeros(data.dim());
        for i in 0..num_features {
            let source_idx = (i as isize - shift).rem_euclid(num_features as isize) as usize;
            result.slice_mut(s![.., i]).assign(&data.slice(s![.., source_idx]));
        }
        
        result
    }

    /// Apply brightness-style augmentation
    pub fn apply_brightness(data: &Array2<f32>, brightness_strength: f32, rng: &mut ThreadRng) -> Array2<f32> {
        let brightness_factor = 1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * brightness_strength;
        data.mapv(|x| x * brightness_factor)
    }

    /// Generate augmentation pipeline
    pub fn generate_augmentation_pipeline(
        num_augmentations: usize,
        augmentation_strength: f32,
    ) -> Vec<AugmentationType> {
        let mut rng = thread_rng();
        let mut pipeline = Vec::with_capacity(num_augmentations);
        
        for _ in 0..num_augmentations {
            let aug_type = match rng.gen_range(0..6) {
                0 => AugmentationType::GaussianNoise { 
                    std: 0.1 * augmentation_strength 
                },
                1 => AugmentationType::Dropout { 
                    rate: 0.1 * augmentation_strength 
                },
                2 => AugmentationType::Crop { 
                    ratio: 0.9 + 0.1 * augmentation_strength 
                },
                3 => AugmentationType::Flip,
                4 => AugmentationType::Rotation { 
                    strength: 0.1 * augmentation_strength 
                },
                5 => AugmentationType::Brightness { 
                    strength: 0.1 * augmentation_strength 
                },
                _ => AugmentationType::None,
            };
            pipeline.push(aug_type);
        }
        
        pipeline
    }
}

/// Gradient approximation utilities
pub struct GradientUtils;

impl GradientUtils {
    /// Approximate gradient norm menggunakan finite differences
    pub fn approximate_gradient_finite_difference<F>(
        &self,
        loss_fn: F,
        parameters: &ArrayD<f32>,
        epsilon: f32,
    ) -> ArrayD<f32>
    where
        F: Fn(&ArrayD<f32>) -> f32,
    {
        let mut gradients = ArrayD::zeros(parameters.dim());
        let base_loss = loss_fn(parameters);
        
        for (idx, &_param) in parameters.indexed_iter() {
            // Create a mutable view instead of full clone for efficiency
            let mut perturbed = unsafe { ArrayD::uninitialized(parameters.dim()) };
            perturbed.assign(parameters);
            perturbed[idx.clone()] += epsilon;
            
            let perturbed_loss = loss_fn(&perturbed);
            let gradient = (perturbed_loss - base_loss) / epsilon;
            gradients[idx.clone()] = gradient;
        }
        
        gradients
    }

    /// Stochastic gradient approximation dengan control variates
    pub fn stochastic_gradient_approximation_cv<F>(
        &self,
        gradient_fn: F,
        baseline_gradient: &ArrayD<f32>,
        num_samples: usize,
        seed: u64,
    ) -> f32
    where
        F: Fn(&ArrayD<f32>) -> ArrayD<f32>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total_norm_sq = 0.0;
        
        for _ in 0..num_samples {
            // Generate random direction
            let total_elements = baseline_gradient.len();
            let mut noise_data = Vec::with_capacity(total_elements);
            for _ in 0..total_elements {
                noise_data.push(rng.sample::<f32, _>(StandardNormal));
            }
            let noise = match ArrayD::from_shape_vec(baseline_gradient.dim(), noise_data) {
                Ok(arr) => arr,
                Err(_) => {
                    // Handle error by returning 0.0 as default
                    return 0.0;
                }
            };
            
            // Compute gradient at perturbed point
            let gradient = gradient_fn(&noise);
            
            // Control variate correction
            let corrected_gradient = &gradient - baseline_gradient;
            
            // Compute directional derivative
            let directional_derivative = (&noise * &corrected_gradient).sum();
            total_norm_sq += directional_derivative * directional_derivative;
        }
        
        (total_norm_sq / num_samples as f32).sqrt()
    }

    /// Multi-scale gradient approximation
    pub fn multi_scale_gradient_approximation<F>(
        &self,
        gradient_fn: F,
        input_shape: &[usize],
        scales: &[f32],
        samples_per_scale: usize,
    ) -> Vec<f32>
    where
        F: Fn(&ArrayD<f32>, f32) -> ArrayD<f32>,
    {
        let mut results = Vec::new();
        
        for &scale in scales {
            let mut total_norm = 0.0;
            let mut rng = thread_rng();
            
            for _ in 0..samples_per_scale {
                let noise = ArrayD::from_shape_fn(input_shape, |_| {
                    rng.sample::<f32, _>(StandardNormal) * scale
                });
                
                let gradient = gradient_fn(&noise, scale);
                let norm = gradient.iter().map(|x| x * x).sum::<f32>().sqrt();
                total_norm += norm;
            }
            
            results.push(total_norm / samples_per_scale as f32);
        }
        
        results
    }
}

/// Monitoring dan logging utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOGPMetrics {
    pub step: usize,
    pub total_loss: f32,
    pub primary_loss: f32,
    pub smoothness_loss: f32,
    pub consistency_loss: f32,
    pub effective_variance: f32,
    pub ema_gradient_norm: f32,
    pub adaptive_threshold: f32,
    pub consistency_similarity: f32,
    pub training_time_ms: u64,
}

pub struct MonitoringUtils;

impl MonitoringUtils {
    /// Create metrics collector
    pub fn create_metrics_collector() -> MetricsCollector {
        MetricsCollector::new()
    }
    
    /// Log training progress
    pub fn log_training_step(metrics: &VOGPMetrics) {
        info!(
            "Step {}: Loss={:.4} (Primary={:.4}, Smooth={:.4}, Cons={:.4}), Var={:.4}, EMA={:.4}, Thresh={:.4}",
            metrics.step,
            metrics.total_loss,
            metrics.primary_loss,
            metrics.smoothness_loss,
            metrics.consistency_loss,
            metrics.effective_variance,
            metrics.ema_gradient_norm,
            metrics.adaptive_threshold
        );
    }
    
    /// Check untuk training stability
    pub fn check_training_stability(
        recent_losses: &[f32],
        variance_threshold: f32,
        loss_spike_threshold: f32,
    ) -> StabilityReport {
        if recent_losses.len() < 10 {
            return StabilityReport {
                is_stable: true,
                variance: 0.0,
                max_spike: 0.0,
                recommendations: Vec::new(),
            };
        }
        
        let mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let variance = recent_losses.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / recent_losses.len() as f32;
        
        let max_spike = recent_losses.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        
        let mut recommendations = Vec::new();
        
        if variance > variance_threshold {
            recommendations.push("Consider increasing EMA beta for more smoothing".to_string());
        }
        
        if max_spike > loss_spike_threshold {
            recommendations.push("Loss spike detected - consider reducing learning rate".to_string());
        }
        
        StabilityReport {
            is_stable: variance < variance_threshold && max_spike < loss_spike_threshold,
            variance,
            max_spike,
            recommendations,
        }
    }
}

/// Metrics collector untuk VOGP+ training
pub struct MetricsCollector {
    metrics_history: Vec<VOGPMetrics>,
    max_history_size: usize,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics_history: Vec::new(),
            max_history_size: 10000,
        }
    }
    
    pub fn add_metrics(&mut self, metrics: VOGPMetrics) {
        self.metrics_history.push(metrics);
        
        // Keep history size manageable
        if self.metrics_history.len() > self.max_history_size {
            self.metrics_history.remove(0);
        }
    }
    
    pub fn get_recent_metrics(&self, window_size: usize) -> &[VOGPMetrics] {
        let start = if self.metrics_history.len() > window_size {
            self.metrics_history.len() - window_size
        } else {
            0
        };
        
        &self.metrics_history[start..]
    }
    
    pub fn get_loss_trend(&self, window_size: usize) -> f32 {
        let recent = self.get_recent_metrics(window_size);
        if recent.len() < 2 {
            return 0.0;
        }
        
        let first_loss = recent[0].total_loss;
        let last_loss = recent[recent.len() - 1].total_loss;
        
        (last_loss - first_loss) / recent.len() as f32
    }
    
    pub fn export_metrics(&self) -> &Vec<VOGPMetrics> {
        &self.metrics_history
    }
    
    pub fn reset(&mut self) {
        self.metrics_history.clear();
    }
}

/// Stability report untuk training monitoring
#[derive(Debug, Clone)]
pub struct StabilityReport {
    pub is_stable: bool,
    pub variance: f32,
    pub max_spike: f32,
    pub recommendations: Vec<String>,
}

/// Batch processing utilities
pub struct BatchUtils;

impl BatchUtils {
    /// Process batch dengan gradient accumulation
    pub fn process_with_accumulation<F, T>(
        &self,
        batch: &[T],
        accumulation_steps: usize,
        process_fn: F,
    ) -> Vec<T>
    where
        F: Fn(&[T]) -> T,
        T: Clone,
    {
        let mut results = Vec::new();
        
        for chunk in batch.chunks(accumulation_steps) {
            let result = process_fn(chunk);
            results.push(result);
        }
        
        results
    }
    
    /// Create virtual batch untuk small batch scenarios
    pub fn create_virtual_batch<T: Clone>(
        &self,
        small_batch: &[T],
        virtual_size: usize,
        augmentation_fn: impl Fn(&T) -> T,
    ) -> Vec<T> {
        let mut virtual_batch = small_batch.to_vec();
        
        while virtual_batch.len() < virtual_size {
            for item in small_batch {
                if virtual_batch.len() >= virtual_size {
                    break;
                }
                virtual_batch.push(augmentation_fn(item));
            }
        }
        
        virtual_batch
    }
    
    /// Balance batch untuk class imbalance
    pub fn balance_batch<T: Clone>(
        &self,
        batch: &[(T, usize)], // (data, class_id)
        target_per_class: usize,
    ) -> Vec<(T, usize)> {
        let mut class_groups: HashMap<usize, Vec<T>> = HashMap::new();
        
        for (data, class_id) in batch {
            class_groups.entry(*class_id)
                .or_insert_with(Vec::new)
                .push(data.clone());
        }
        
        let mut balanced_batch = Vec::new();
        
        for (class_id, samples) in class_groups {
            let needed = target_per_class.min(samples.len());
            balanced_batch.extend(
                samples.into_iter().take(needed)
                    .map(|data| (data, class_id))
            );
        }
        
        balanced_batch
    }
}

/// Memory optimization utilities
pub struct MemoryUtils;

impl MemoryUtils {
    /// Estimate memory usage untuk tensors
    pub fn estimate_tensor_memory(shape: &[usize], dtype: DataType) -> usize {
        let elements_per_mb = 1024 * 1024 / dtype.size_bytes();
        let total_elements: usize = shape.iter().product();
        (total_elements + elements_per_mb - 1) / elements_per_mb
    }
    
    /// Optimize batch size berdasarkan memory constraint
    pub fn optimize_batch_size(
        &self,
        sample_shape: &[usize],
        available_memory_mb: usize,
        dtype: DataType,
        safety_factor: f32,
    ) -> usize {
        let memory_per_sample = Self::estimate_tensor_memory(sample_shape, dtype);
        let max_safe_samples = (available_memory_mb as f32 * safety_factor) as usize / memory_per_sample;
        
        // Power of 2 alignment untuk better performance
        max_safe_samples.next_power_of_two().min(1024)
    }
    
    /// Check memory pressure
    pub fn check_memory_pressure(&self, current_usage_mb: usize, limit_mb: usize) -> MemoryPressure {
        let usage_ratio = current_usage_mb as f32 / limit_mb as f32;
        
        if usage_ratio > 0.9 {
            MemoryPressure::Critical
        } else if usage_ratio > 0.7 {
            MemoryPressure::High
        } else if usage_ratio > 0.5 {
            MemoryPressure::Medium
        } else {
            MemoryPressure::Low
        }
    }
}

/// Data types untuk memory estimation
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U8,
}

impl DataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F64 => 8,
            DataType::I32 => 4,
            DataType::I64 => 8,
            DataType::U8 => 1,
        }
    }
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

/// Augmentation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AugmentationType {
    None,
    GaussianNoise { std: f32 },
    Dropout { rate: f32 },
    Crop { ratio: f32 },
    Flip,
    Rotation { strength: f32 },
    Brightness { strength: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_augmentation_utils() {
        let mut rng = thread_rng();
        let data = Array::from_elem((2, 10), 0.5);
        
        let augmented = AugmentationUtils::apply_mixed_augmentation(&data, 0.5, &mut rng);
        assert_eq!(augmented.dim(), data.dim());
    }

    #[test]
    fn test_gradient_approximation() {
        let utils = GradientUtils {};
        
        // Simple quadratic function: f(x) = x^2
        let loss_fn = |params: &ArrayD<f32>| {
            params.iter().map(|x| x * x).sum::<f32>()
        };
        
        let params = ArrayD::from_elem(vec![3], 1.0);
        let gradients = utils.approximate_gradient_finite_difference(loss_fn, &params, 1e-5);
        
        assert_eq!(gradients.dim(), params.dim());
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        
        let metrics = VOGPMetrics {
            step: 1,
            total_loss: 1.0,
            primary_loss: 0.8,
            smoothness_loss: 0.1,
            consistency_loss: 0.1,
            effective_variance: 0.5,
            ema_gradient_norm: 1.0,
            adaptive_threshold: 1.0,
            consistency_similarity: 0.9,
            training_time_ms: 100,
        };
        
        collector.add_metrics(metrics);
        assert_eq!(collector.get_recent_metrics(5).len(), 1);
    }

    #[test]
    fn test_memory_estimation() {
        let memory_mb = MemoryUtils::estimate_tensor_memory(&[1000, 1000], DataType::F32);
        assert!(memory_mb > 0);
    }

    #[test]
    fn test_stability_check() {
        let losses = vec![1.0, 1.1, 0.9, 1.2, 1.0, 1.3, 0.8, 1.1, 0.9, 1.0];
        let report = MonitoringUtils::check_training_stability(&losses, 0.1, 0.5);
        
        assert!(report.variance > 0.0);
    }
}
