//! Consistency Learning untuk VOGP+
//!
//! Implementasi consistency learning yang memaksa model menghasilkan prediksi
//! mirip untuk data asli dan augmentasi, menciptakan "virtual dataset expansion".

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Consistency Learning Component
/// 
/// Formula: L_cons = E_{x, x~} ∥f_θ(x) - f_θ(x~)∥²_2
/// 
/// Memaksa model konsisten terhadap augmentasi kecil, meningkatkan
/// efektivitas dataset tanpa membuat model menghafal noise.
#[derive(Debug, Clone)]
pub struct ConsistencyLearning {
    /// Konfigurasi consistency learning
    config: ConsistencyConfig,
    /// Statistik untuk monitoring
    statistics: ConsistencyStatistics,
    /// Cache untuk augmentasi yang mahal
    _augmentation_cache: HashMap<u64, Array2<f32>>,
}

/// Konfigurasi untuk consistency learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    /// Enable consistency learning
    pub enabled: bool,
    /// Jumlah augmentasi per sample
    pub num_augmentations: usize,
    /// Strength of consistency penalty
    pub consistency_strength: f32,
    /// Enable progressive consistency (mulai lemah, kuat bertahap)
    pub enable_progressive: bool,
    /// Warmup steps untuk progressive consistency
    pub warmup_steps: usize,
    /// Enable temperature scaling untuk consistency
    pub enable_temperature_scaling: bool,
    /// Temperature untuk softmax consistency
    pub temperature: f32,
    /// Enable confidence-based masking
    pub enable_confidence_masking: bool,
    /// Confidence threshold
    pub confidence_threshold: f32,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_augmentations: 2,
            consistency_strength: 1.0,
            enable_progressive: true,
            warmup_steps: 1000,
            enable_temperature_scaling: true,
            temperature: 0.1,
            enable_confidence_masking: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Statistik consistency learning untuk monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyStatistics {
    /// Rata-rata consistency loss
    pub avg_consistency_loss: f32,
    /// Jumlah samples diproses
    pub total_samples: usize,
    /// Rata-rata similarity antar augmentasi
    pub avg_augmentation_similarity: f32,
    /// Effective dataset expansion factor
    pub expansion_factor: f32,
}

impl ConsistencyLearning {
    /// Buat consistency learning baru
    pub fn new(config: ConsistencyConfig) -> Self {
        Self {
            config,
            statistics: ConsistencyStatistics {
                avg_consistency_loss: 0.0,
                total_samples: 0,
                avg_augmentation_similarity: 0.0,
                expansion_factor: 1.0,
            },
            _augmentation_cache: HashMap::new(),
        }
    }

    /// Hitung consistency loss untuk batch
    pub fn compute_consistency_loss(
        &mut self,
        original_predictions: &Array2<f32>,
        augmented_predictions: &[Array2<f32>],
        step: usize,
    ) -> f32 {
        if !self.config.enabled || augmented_predictions.is_empty() {
            return 0.0;
        }

        let batch_size = original_predictions.dim().0;
        let mut total_loss = 0.0;
        let mut total_similarity = 0.0;

        // Progressive consistency strength
        let current_strength = if self.config.enable_progressive {
            let progress = (step as f32 / self.config.warmup_steps as f32).min(1.0);
            self.config.consistency_strength * progress
        } else {
            self.config.consistency_strength
        };

        for aug_pred in augmented_predictions {
            let (loss, similarity) = self.compute_pairwise_consistency(
                original_predictions,
                aug_pred,
                current_strength,
            );
            
            total_loss += loss;
            total_similarity += similarity;
        }

        // Update statistics
        let avg_loss = total_loss / augmented_predictions.len() as f32;
        let avg_similarity = total_similarity / augmented_predictions.len() as f32;
        
        self.update_statistics(avg_loss, avg_similarity, batch_size);

        avg_loss
    }

    /// Hitung consistency loss untuk pairwise comparison
    fn compute_pairwise_consistency(
        &self,
        original: &Array2<f32>,
        augmented: &Array2<f32>,
        strength: f32,
    ) -> (f32, f32) {
        let batch_size = original.dim().0;
        let mut total_loss = 0.0;
        let mut total_similarity = 0.0;

        for i in 0..batch_size {
            let orig_sample = original.row(i);
            let aug_sample = augmented.row(i);

            // Apply temperature scaling jika dienable
            let (orig_scaled, aug_scaled) = if self.config.enable_temperature_scaling {
                (
                    self.apply_temperature_scaling(&orig_sample),
                    self.apply_temperature_scaling(&aug_sample),
                )
            } else {
                (orig_sample.to_owned(), aug_sample.to_owned())
            };

            // Confidence-based masking
            let mask = if self.config.enable_confidence_masking {
                self.compute_confidence_mask(&orig_sample)
            } else {
                Array1::ones(orig_sample.len())
            };

            // Hitung L2 consistency loss dengan masking
            let diff = &orig_scaled - &aug_scaled;
            let masked_diff = diff * &mask;
            let l2_loss = masked_diff.iter().map(|x| x * x).sum::<f32>();
            
            total_loss += l2_loss;

            // Hitung similarity untuk monitoring
            let similarity = self.compute_cosine_similarity(&orig_sample, &aug_sample);
            total_similarity += similarity;
        }

        let avg_loss = (total_loss / batch_size as f32) * strength;
        let avg_similarity = total_similarity / batch_size as f32;

        (avg_loss, avg_similarity)
    }

    /// Apply temperature scaling ke predictions
    fn apply_temperature_scaling(&self, predictions: &ArrayView1<f32>) -> Array1<f32> {
        predictions.mapv(|x| x / self.config.temperature)
    }

    /// Compute confidence mask berdasarkan prediction confidence
    fn compute_confidence_mask(&self, predictions: &ArrayView1<f32>) -> Array1<f32> {
        // Convert ke probabilities (softmax approximation)
        let exp_preds = predictions.mapv(|x| x.exp());
        let sum_exp = exp_preds.sum();
        
        if sum_exp > 0.0 {
            let probs = &exp_preds / sum_exp;
            
            // Mask samples dengan confidence > threshold
            let max_confidence = probs.iter().fold(0.0f32, |a, &b| a.max(b));
            let mask_value = if max_confidence > self.config.confidence_threshold {
                1.0
            } else {
                0.0
            };
            
            Array1::from_elem(predictions.len(), mask_value)
        } else {
            Array1::ones(predictions.len())
        }
    }

    /// Compute cosine similarity antara dua vektor
    fn compute_cosine_similarity(&self, vec1: &ArrayView1<f32>, vec2: &ArrayView1<f32>) -> f32 {
        let dot_product = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum::<f32>();
        let norm1 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Update statistics consistency learning
    fn update_statistics(&mut self, loss: f32, similarity: f32, batch_size: usize) {
        self.statistics.total_samples += batch_size;
        
        // Update running average
        let alpha = 0.01; // Learning rate untuk statistics
        self.statistics.avg_consistency_loss = 
            (1.0 - alpha) * self.statistics.avg_consistency_loss + alpha * loss;
        
        self.statistics.avg_augmentation_similarity = 
            (1.0 - alpha) * self.statistics.avg_augmentation_similarity + alpha * similarity;
        
        // Update expansion factor
        self.statistics.expansion_factor = 1.0 + self.config.num_augmentations as f32;
    }

    /// Generate augmentasi untuk consistency learning
    pub fn generate_augmentations(&mut self, data: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut augmentations = Vec::with_capacity(self.config.num_augmentations);
        
        for _ in 0..self.config.num_augmentations {
            let aug_data = self.apply_random_augmentation(data);
            augmentations.push(aug_data);
        }
        
        augmentations
    }

    /// Apply random augmentation ke data
    fn apply_random_augmentation(&self, data: &Array2<f32>) -> Array2<f32> {
        let mut rng = thread_rng();
        let augmentation_type = rng.gen_range(0..4);
        
        match augmentation_type {
            0 => self.apply_gaussian_noise(data, 0.1),
            1 => self.apply_dropout(data, 0.1),
            2 => self.apply_random_crop(data, 0.9),
            3 => self.apply_random_flip(data),
            _ => data.clone(), // No augmentation
        }
    }

    /// Apply Gaussian noise augmentation
    fn apply_gaussian_noise(&self, data: &Array2<f32>, std: f32) -> Array2<f32> {
        let mut rng = thread_rng();
        data.mapv(|x| {
            let noise: f32 = rng.sample::<f32, _>(rand_distr::StandardNormal) * std;
            x + noise
        })
    }

    /// Apply dropout-style augmentation
    fn apply_dropout(&self, data: &Array2<f32>, dropout_rate: f32) -> Array2<f32> {
        let mut rng = thread_rng();
        data.mapv(|x| {
            if rng.gen::<f32>() < dropout_rate {
                0.0
            } else {
                x
            }
        })
    }

    /// Apply random crop augmentation
    fn apply_random_crop(&self, data: &Array2<f32>, crop_ratio: f32) -> Array2<f32> {
        let (_, num_features) = data.dim();
        let crop_size = (num_features as f32 * crop_ratio) as usize;
        
        if crop_size >= num_features {
            return data.clone();
        }
        
        let mut rng = thread_rng();
        let start_idx = rng.gen_range(0..=num_features - crop_size);
        
        data.slice(s![.., start_idx..start_idx + crop_size]).to_owned()
    }

    /// Apply random flip augmentation
    fn apply_random_flip(&self, data: &Array2<f32>) -> Array2<f32> {
        let mut rng = thread_rng();
        if rng.gen::<bool>() {
            // Flip horizontal
            data.slice(s![.., ..;-1]).to_owned()
        } else {
            data.clone()
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &ConsistencyStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = ConsistencyStatistics {
            avg_consistency_loss: 0.0,
            total_samples: 0,
            avg_augmentation_similarity: 0.0,
            expansion_factor: 1.0,
        };
    }

    /// Update configuration runtime
    pub fn update_config(&mut self, config: ConsistencyConfig) {
        self.config = config;
    }
}

/// Advanced Consistency dengan pseudo-labeling
#[derive(Debug, Clone)]
pub struct PseudoLabelConsistency {
    base_consistency: ConsistencyLearning,
    /// Confidence threshold untuk pseudo-label
    pseudo_threshold: f32,
    /// Enable self-training
    enable_self_training: bool,
}

impl PseudoLabelConsistency {
    /// Buat pseudo-label consistency
    pub fn new(config: ConsistencyConfig, pseudo_threshold: f32) -> Self {
        Self {
            base_consistency: ConsistencyLearning::new(config),
            pseudo_threshold,
            enable_self_training: true,
        }
    }

    /// Hitung consistency loss dengan pseudo-labeling
    pub fn compute_pseudo_consistency_loss(
        &mut self,
        original_predictions: &Array2<f32>,
        unlabeled_data: &Array2<f32>,
        step: usize,
    ) -> f32 {
        if !self.enable_self_training {
            return 0.0;
        }

        // Generate pseudo-labels dari confident predictions
        let pseudo_labels = self.generate_pseudo_labels(original_predictions);
        
        // Apply consistency pada unlabeled data
        let augmented_unlabeled = self.base_consistency.generate_augmentations(unlabeled_data);
        
        // Hitung consistency loss dengan pseudo-labels
        self.compute_consistency_with_pseudo_labels(&pseudo_labels, &augmented_unlabeled, step)
    }

    /// Generate pseudo-labels dari confident predictions
    fn generate_pseudo_labels(&self, predictions: &Array2<f32>) -> Array2<f32> {
        predictions.mapv(|x| {
            // Threshold untuk confidence
            if x > self.pseudo_threshold {
                1.0
            } else if x < -self.pseudo_threshold {
                -1.0
            } else {
                0.0  // Uncertain, no pseudo-label
            }
        })
    }

    /// Compute consistency loss dengan pseudo-labels
    fn compute_consistency_with_pseudo_labels(
        &self,
        pseudo_labels: &Array2<f32>,
        augmented_data: &[Array2<f32>],
        step: usize,
    ) -> f32 {
        // Implementasi consistency dengan pseudo-labels
        // Compute consistency loss between original predictions and pseudo-labels
        
        if augmented_data.is_empty() {
            return 0.0;
        }
        
        let mut total_consistency = 0.0;
        let mut count = 0;
        
        // Compare original pseudo-labels with augmented data predictions
        for (aug_idx, augmented_sample) in augmented_data.iter().enumerate() {
            if aug_idx < pseudo_labels.nrows() {
                // Get original pseudo-labels for this sample
                let original_labels = pseudo_labels.row(aug_idx);
                
                // Simulate model prediction on augmented data
                // In practice, this would be actual model inference
                let predicted_labels = self.simulate_augmented_prediction(augmented_sample, &original_labels);
                
                // Compute consistency using KL divergence or cosine similarity
                let consistency_score = self.compute_consistency_score(&original_labels, &predicted_labels);
                total_consistency += consistency_score;
                count += 1;
            }
        }
        
        if count > 0 {
            total_consistency / count as f32
        } else {
            0.0
        }
    }
    
    /// Simulate model prediction on augmented data
    fn simulate_augmented_prediction(&self, augmented_data: &Array2<f32>, original_labels: &ArrayView1<f32>) -> Vec<f32> {
        let mut predictions = Vec::with_capacity(original_labels.len());
        
        for (i, &original_label) in original_labels.iter().enumerate() {
            // Simulate augmentation effect on prediction
            let augmentation_factor = if i < augmented_data.ncols() {
                // Use augmented data to modify prediction
                let aug_value = augmented_data[[0, i]];
                1.0 + 0.1 * aug_value.sin() // Small perturbation
            } else {
                1.0
            };
            
            // Apply augmentation with temperature scaling
            let temperature = 0.8; // Temperature for softmax-like behavior
            let perturbed_label = original_label * augmentation_factor;
            let prediction = (perturbed_label / temperature).tanh();
            
            predictions.push(prediction);
        }
        
        predictions
    }
    
    /// Compute consistency score between two label distributions
    fn compute_consistency_score(&self, original: &ArrayView1<f32>, predicted: &[f32]) -> f32 {
        if original.len() != predicted.len() {
            return 0.0;
        }
        
        // Compute cosine similarity as consistency measure
        let mut dot_product = 0.0;
        let mut norm_original = 0.0;
        let mut norm_predicted = 0.0;
        
        for (&orig, &pred) in original.iter().zip(predicted.iter()) {
            dot_product += orig * pred;
            norm_original += orig * orig;
            norm_predicted += pred * pred;
        }
        
        norm_original = norm_original.sqrt();
        norm_predicted = norm_predicted.sqrt();
        
        if norm_original > 1e-8 && norm_predicted > 1e-8 {
            dot_product / (norm_original * norm_predicted)
        } else {
            0.0
        }
    }
}

/// Temporal Consistency untuk sequential data
#[derive(Debug, Clone)]
pub struct TemporalConsistency {
    base_consistency: ConsistencyLearning,
    /// Temporal window size
    window_size: usize,
    /// Enable temporal smoothing
    _enable_temporal_smoothing: bool,
}

impl TemporalConsistency {
    /// Buat temporal consistency
    pub fn new(config: ConsistencyConfig, window_size: usize) -> Self {
        Self {
            base_consistency: ConsistencyLearning::new(config),
            window_size,
            _enable_temporal_smoothing: true,
        }
    }

    /// Hitung temporal consistency loss
    pub fn compute_temporal_consistency_loss(
        &mut self,
        current_predictions: &Array2<f32>,
        prediction_history: &[Array2<f32>],
        step: usize,
    ) -> f32 {
        if prediction_history.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let window_end = prediction_history.len().min(self.window_size);
        let window_start = if prediction_history.len() > self.window_size {
            prediction_history.len() - self.window_size
        } else {
            0
        };

        // Hitung consistency dengan historical predictions
        for hist_pred in &prediction_history[window_start..window_end] {
            let (loss, _) = self.base_consistency.compute_pairwise_consistency(
                current_predictions,
                hist_pred,
                self.base_consistency.config.consistency_strength,
            );
            total_loss += loss;
        }

        total_loss / window_end as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_consistency_learning() {
        let config = ConsistencyConfig::default();
        let mut consistency = ConsistencyLearning::new(config);
        
        let original = Array::from_elem((2, 3), 0.5);
        let augmented = vec![Array::from_elem((2, 3), 0.6)];
        
        let loss = consistency.compute_consistency_loss(&original, &augmented, 100);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_augmentation_generation() {
        let config = ConsistencyConfig::default();
        let mut consistency = ConsistencyLearning::new(config);
        
        let data = Array::from_elem((2, 10), 0.5);
        let augmentations = consistency.generate_augmentations(&data);
        
        assert_eq!(augmentations.len(), 2);
        assert_eq!(augmentations[0].dim(), data.dim());
    }

    #[test]
    fn test_cosine_similarity() {
        let config = ConsistencyConfig::default();
        let consistency = ConsistencyLearning::new(config);
        
        let vec1 = Array::from_vec(vec![1.0, 0.0, 0.0]);
        let vec2 = Array::from_vec(vec![1.0, 0.0, 0.0]);
        let similarity = consistency.compute_cosine_similarity(&vec1.view(), &vec2.view());
        
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_progressive_consistency() {
        let mut config = ConsistencyConfig::default();
        config.enable_progressive = true;
        config.warmup_steps = 100;
        
        let mut consistency = ConsistencyLearning::new(config);
        
        let original = Array::from_elem((2, 3), 0.5);
        let augmented = vec![Array::from_elem((2, 3), 0.6)];
        
        // Early step - lower strength
        let early_loss = consistency.compute_consistency_loss(&original, &augmented, 10);
        
        // Later step - higher strength  
        let later_loss = consistency.compute_consistency_loss(&original, &augmented, 200);
        
        assert!(later_loss >= early_loss);
    }
}
