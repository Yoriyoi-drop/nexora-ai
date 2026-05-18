//! Variance Stabilization untuk VOGP+
//!
//! Implementasi variance dan uncertainty awareness untuk menghindari
//! overconfident predictions dan meningkatkan generalisasi.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Variance Stabilization Component
/// 
/// Formula: σ_eff = α·Var(f_θ(x)) + (1-α)·H(f_θ(x))
/// 
/// Menggabungkan variance prediksi dengan entropy untuk
/// uncertainty-aware regularization.
#[derive(Debug, Clone)]
pub struct VarianceStabilizer {
    /// Konfigurasi variance stabilization
    config: VarianceConfig,
    /// Statistik variance historis
    variance_history: Vec<f32>,
    /// Statistik entropy historis
    entropy_history: Vec<f32>,
    /// Running average untuk effective variance
    avg_effective_variance: f32,
}

/// Konfigurasi untuk variance stabilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceConfig {
    /// Bobot untuk variance vs entropy (α)
    pub alpha_variance: f32,
    /// Enable variance clipping
    pub enable_variance_clipping: bool,
    /// Maximum variance value
    pub max_variance: f32,
    /// Minimum variance value
    pub min_variance: f32,
    /// Enable adaptive alpha
    pub enable_adaptive_alpha: bool,
    /// Alpha adaptation rate
    pub alpha_adaptation_rate: f32,
    /// Enable variance smoothing
    pub enable_variance_smoothing: bool,
    /// Variance smoothing window
    pub variance_smoothing_window: usize,
}

impl Default for VarianceConfig {
    fn default() -> Self {
        Self {
            alpha_variance: 0.7,
            enable_variance_clipping: true,
            max_variance: 10.0,
            min_variance: 1e-8,
            enable_adaptive_alpha: false,
            alpha_adaptation_rate: 0.01,
            enable_variance_smoothing: true,
            variance_smoothing_window: 10,
        }
    }
}

impl VarianceStabilizer {
    /// Buat variance stabilizer baru
    pub fn new(config: VarianceConfig) -> Self {
        Self {
            config,
            variance_history: Vec::new(),
            entropy_history: Vec::new(),
            avg_effective_variance: 1.0,
        }
    }

    /// Hitung effective variance untuk batch predictions
    pub fn compute_effective_variance(&mut self, predictions: &Array2<f32>) -> f32 {
        let variance = self.compute_prediction_variance(predictions);
        let entropy = self.compute_prediction_entropy(predictions);
        
        // Update histories
        self.update_histories(variance, entropy);
        
        // Adaptive alpha jika dienable
        let current_alpha = if self.config.enable_adaptive_alpha {
            self.compute_adaptive_alpha(variance, entropy)
        } else {
            self.config.alpha_variance
        };
        
        // Hitung effective variance
        let effective_variance = current_alpha * variance + (1.0 - current_alpha) * entropy;
        
        // Apply clipping jika dienable
        let clipped_variance = if self.config.enable_variance_clipping {
            effective_variance.clamp(self.config.min_variance, self.config.max_variance)
        } else {
            effective_variance
        };
        
        // Update running average
        self.update_running_average(clipped_variance);
        
        clipped_variance
    }

    /// Compute prediction variance
    fn compute_prediction_variance(&self, predictions: &Array2<f32>) -> f32 {
        let (batch_size, num_classes) = predictions.dim();
        
        // Hitung mean per class
        let mut class_means = Array1::zeros(num_classes);
        for i in 0..batch_size {
            class_means += &predictions.row(i);
        }
        class_means /= batch_size as f32;
        
        // Hitung variance per class lalu rata-rata
        let mut total_variance = 0.0;
        for i in 0..batch_size {
            let diff = &predictions.row(i) - &class_means;
            let class_variance = diff.iter().map(|x| x * x).sum::<f32>();
            total_variance += class_variance;
        }
        
        total_variance / (batch_size * num_classes) as f32
    }

    /// Compute Shannon entropy dari predictions
    fn compute_prediction_entropy(&self, predictions: &Array2<f32>) -> f32 {
        let batch_size = predictions.dim().0;
        let mut total_entropy = 0.0;
        
        for i in 0..batch_size {
            let pred = predictions.row(i);
            let entropy = self.compute_sample_entropy(&pred);
            total_entropy += entropy;
        }
        
        total_entropy / batch_size as f32
    }

    /// Compute entropy untuk single sample
    fn compute_sample_entropy(&self, predictions: &ArrayView1<f32>) -> f32 {
        // Normalize ke probabilities
        let exp_preds = predictions.mapv(|x| x.exp());
        let sum_exp = exp_preds.sum();
        
        if sum_exp > 0.0 {
            let probs = &exp_preds / sum_exp;
            
            let mut entropy = 0.0;
            for &p in probs.iter() {
                if p > 1e-12 {
                    entropy -= p * p.ln();
                }
            }
            entropy
        } else {
            0.0
        }
    }

    /// Update variance dan entropy histories
    fn update_histories(&mut self, variance: f32, entropy: f32) {
        self.variance_history.push(variance);
        self.entropy_history.push(entropy);
        
        // Keep history size manageable
        let max_history = 1000;
        if self.variance_history.len() > max_history {
            self.variance_history.remove(0);
        }
        if self.entropy_history.len() > max_history {
            self.entropy_history.remove(0);
        }
    }

    /// Compute adaptive alpha berdasarkan variance dan entropy patterns
    fn compute_adaptive_alpha(&self, variance: f32, entropy: f32) -> f32 {
        if self.variance_history.len() < 10 {
            return self.config.alpha_variance;
        }
        
        // Analyze recent trends
        let recent_variance_trend = self.compute_trend(&self.variance_history, 10);
        let recent_entropy_trend = self.compute_trend(&self.entropy_history, 10);
        
        // Adjust alpha based on trends
        let mut alpha = self.config.alpha_variance;
        
        // If variance increasing, increase alpha weight
        if recent_variance_trend > 0.1 {
            alpha += self.config.alpha_adaptation_rate;
        }
        
        // If entropy increasing, decrease alpha weight (more entropy focus)
        if recent_entropy_trend > 0.1 {
            alpha -= self.config.alpha_adaptation_rate;
        }
        
        alpha.clamp(0.0, 1.0)
    }

    /// Compute trend dari time series data
    fn compute_trend(&self, history: &[f32], window: usize) -> f32 {
        if history.len() < window + 1 {
            return 0.0;
        }
        
        let recent_sum: f32 = history.iter().rev().take(window).sum();
        let older_sum: f32 = history.iter().rev().skip(window).take(window).sum();
        
        (recent_sum - older_sum) / window as f32
    }

    /// Update running average effective variance
    fn update_running_average(&mut self, value: f32) {
        let alpha = 0.01; // Smoothing factor
        self.avg_effective_variance = (1.0 - alpha) * self.avg_effective_variance + alpha * value;
    }

    /// Get smoothed variance jika dienable
    pub fn get_smoothed_variance(&self) -> f32 {
        if self.config.enable_variance_smoothing && self.variance_history.len() >= self.config.variance_smoothing_window {
            let window = self.config.variance_smoothing_window;
            let recent_sum: f32 = self.variance_history.iter().rev().take(window).sum();
            recent_sum / window as f32
        } else {
            self.variance_history.last().copied().unwrap_or(1.0)
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> VarianceStatistics {
        VarianceStatistics {
            current_variance: self.variance_history.last().copied().unwrap_or(0.0),
            current_entropy: self.entropy_history.last().copied().unwrap_or(0.0),
            avg_effective_variance: self.avg_effective_variance,
            smoothed_variance: self.get_smoothed_variance(),
            variance_trend: self.compute_trend(&self.variance_history, 10),
            entropy_trend: self.compute_trend(&self.entropy_history, 10),
            history_length: self.variance_history.len(),
        }
    }

    /// Reset semua statistics
    pub fn reset(&mut self) {
        self.variance_history.clear();
        self.entropy_history.clear();
        self.avg_effective_variance = 1.0;
    }
}

/// Statistics dari variance stabilizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceStatistics {
    pub current_variance: f32,
    pub current_entropy: f32,
    pub avg_effective_variance: f32,
    pub smoothed_variance: f32,
    pub variance_trend: f32,
    pub entropy_trend: f32,
    pub history_length: usize,
}

/// Uncertainty Quantification untuk model predictions
#[derive(Debug, Clone)]
pub struct UncertaintyQuantifier {
    variance_stabilizer: VarianceStabilizer,
    /// Confidence threshold untuk uncertainty
    confidence_threshold: f32,
    /// Enable epistemic uncertainty estimation
    enable_epistemic: bool,
}

impl UncertaintyQuantifier {
    /// Buat uncertainty quantifier baru
    pub fn new(config: VarianceConfig, confidence_threshold: f32) -> Self {
        Self {
            variance_stabilizer: VarianceStabilizer::new(config),
            confidence_threshold,
            enable_epistemic: true,
        }
    }

    /// Quantify uncertainty untuk predictions
    pub fn quantify_uncertainty(&mut self, predictions: &Array2<f32>) -> UncertaintyMetrics {
        let effective_variance = self.variance_stabilizer.compute_effective_variance(predictions);
        
        // Compute aleatoric uncertainty (data uncertainty)
        let aleatoric_uncertainty = self.compute_aleatoric_uncertainty(predictions);
        
        // Compute epistemic uncertainty (model uncertainty) jika dienable
        let epistemic_uncertainty = if self.enable_epistemic {
            self.compute_epistemic_uncertainty(predictions)
        } else {
            0.0
        };
        
        // Total uncertainty
        let total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty;
        
        // Confidence score
        let confidence = self.compute_confidence_score(predictions, total_uncertainty);
        
        UncertaintyMetrics {
            effective_variance,
            aleatoric_uncertainty,
            epistemic_uncertainty,
            total_uncertainty,
            confidence_score: confidence,
            is_confident: confidence > self.confidence_threshold,
        }
    }

    /// Compute aleatoric uncertainty (intrinsic data noise)
    fn compute_aleatoric_uncertainty(&self, predictions: &Array2<f32>) -> f32 {
        // Aleatoric uncertainty = prediction variance
        self.variance_stabilizer.compute_prediction_variance(predictions)
    }

    /// Compute epistemic uncertainty (model uncertainty)
    fn compute_epistemic_uncertainty(&self, predictions: &Array2<f32>) -> f32 {
        // Epistemic uncertainty = disagreement between predictions
        // Simplified: use entropy as proxy
        self.variance_stabilizer.compute_prediction_entropy(predictions)
    }

    /// Compute confidence score
    fn compute_confidence_score(&self, predictions: &Array2<f32>, uncertainty: f32) -> f32 {
        // Higher uncertainty = lower confidence
        let base_confidence = self.compute_prediction_confidence(predictions);
        let uncertainty_penalty = uncertainty / (uncertainty + 1.0);
        
        base_confidence * (1.0 - uncertainty_penalty)
    }

    /// Compute prediction confidence
    fn compute_prediction_confidence(&self, predictions: &Array2<f32>) -> f32 {
        let batch_size = predictions.dim().0;
        let mut total_confidence = 0.0;
        
        for i in 0..batch_size {
            let pred = predictions.row(i);
            let max_prob = pred.iter().fold(0.0f32, |a, &b| a.max(b));
            total_confidence += max_prob;
        }
        
        total_confidence / batch_size as f32
    }

    /// Get uncertainty statistics
    pub fn get_statistics(&self) -> VarianceStatistics {
        let current_variance = self.variance_stabilizer.variance_history.last().copied().unwrap_or(0.0);
        let current_entropy = self.variance_stabilizer.entropy_history.last().copied().unwrap_or(0.0);
        
        // Calculate smoothed variance if enabled
        let smoothed_variance = if self.variance_stabilizer.config.enable_variance_smoothing && 
                                 self.variance_stabilizer.variance_history.len() >= self.variance_stabilizer.config.variance_smoothing_window {
            let window_size = self.variance_stabilizer.config.variance_smoothing_window;
            let start_idx = self.variance_stabilizer.variance_history.len().saturating_sub(window_size);
            let window_sum: f32 = self.variance_stabilizer.variance_history[start_idx..].iter().sum();
            window_sum / window_size as f32
        } else {
            current_variance
        };
        
        // Calculate trends (rate of change)
        let (variance_trend, entropy_trend) = if self.variance_stabilizer.variance_history.len() >= 2 {
            let recent_variance = self.variance_stabilizer.variance_history.last().copied().unwrap_or(0.0);
            let prev_variance = self.variance_stabilizer.variance_history[self.variance_stabilizer.variance_history.len() - 2];
            let variance_trend = recent_variance - prev_variance;
            
            let recent_entropy = self.variance_stabilizer.entropy_history.last().copied().unwrap_or(0.0);
            let prev_entropy = self.variance_stabilizer.entropy_history[self.variance_stabilizer.entropy_history.len() - 2];
            let entropy_trend = recent_entropy - prev_entropy;
            
            (variance_trend, entropy_trend)
        } else {
            (0.0, 0.0)
        };
        
        VarianceStatistics {
            current_variance,
            current_entropy,
            avg_effective_variance: self.variance_stabilizer.avg_effective_variance,
            smoothed_variance,
            variance_trend,
            entropy_trend,
            history_length: self.variance_stabilizer.variance_history.len(),
        }
    }
}

/// Uncertainty metrics untuk monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyMetrics {
    pub effective_variance: f32,
    pub aleatoric_uncertainty: f32,
    pub epistemic_uncertainty: f32,
    pub total_uncertainty: f32,
    pub confidence_score: f32,
    pub is_confident: bool,
}

/// Adaptive Variance Thresholding
#[derive(Debug, Clone)]
pub struct AdaptiveVarianceThreshold {
    variance_stabilizer: VarianceStabilizer,
    /// Dynamic threshold
    current_threshold: f32,
    /// Threshold adaptation rate
    adaptation_rate: f32,
    /// Minimum threshold
    min_threshold: f32,
    /// Maximum threshold
    max_threshold: f32,
}

impl AdaptiveVarianceThreshold {
    /// Buat adaptive variance threshold
    pub fn new(config: VarianceConfig, initial_threshold: f32) -> Self {
        Self {
            variance_stabilizer: VarianceStabilizer::new(config),
            current_threshold: initial_threshold,
            adaptation_rate: 0.01,
            min_threshold: 0.1,
            max_threshold: 5.0,
        }
    }

    /// Update threshold berdasarkan variance patterns
    pub fn update_threshold(&mut self, predictions: &Array2<f32>) -> bool {
        let effective_variance = self.variance_stabilizer.compute_effective_variance(predictions);
        let stats = self.variance_stabilizer.get_statistics();
        
        // Adapt threshold based on variance trend
        if stats.variance_trend > 0.1 {
            // Variance increasing - increase threshold
            self.current_threshold += self.adaptation_rate;
        } else if stats.variance_trend < -0.1 {
            // Variance decreasing - decrease threshold
            self.current_threshold -= self.adaptation_rate;
        }
        
        // Clamp threshold
        self.current_threshold = self.current_threshold.clamp(self.min_threshold, self.max_threshold);
        
        // Return true if variance exceeds threshold
        effective_variance > self.current_threshold
    }

    /// Get current threshold
    pub fn get_threshold(&self) -> f32 {
        self.current_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_variance_stabilizer() {
        let config = VarianceConfig::default();
        let mut stabilizer = VarianceStabilizer::new(config);
        
        let predictions = Array::from_elem((2, 3), 0.5);
        let effective_variance = stabilizer.compute_effective_variance(&predictions);
        
        assert!(effective_variance > 0.0);
    }

    #[test]
    fn test_entropy_computation() {
        let config = VarianceConfig::default();
        let stabilizer = VarianceStabilizer::new(config);
        
        // Very confident predictions (low entropy)
        let confident = Array::from_vec(vec![10.0, 0.0, 0.0]);
        let entropy = stabilizer.compute_sample_entropy(&confident.view());
        assert!(entropy < 0.1);
        
        // Uniform predictions (high entropy)
        let uniform = Array::from_vec(vec![0.33, 0.33, 0.34]);
        let entropy = stabilizer.compute_sample_entropy(&uniform.view());
        assert!(entropy > 1.0);
    }

    #[test]
    fn test_uncertainty_quantification() {
        let config = VarianceConfig::default();
        let mut quantifier = UncertaintyQuantifier::new(config, 0.7);
        
        let predictions = Array::from_elem((2, 3), 0.5);
        let metrics = quantifier.quantify_uncertainty(&predictions);
        
        assert!(metrics.total_uncertainty >= 0.0);
        assert!(metrics.confidence_score >= 0.0 && metrics.confidence_score <= 1.0);
    }

    #[test]
    fn test_adaptive_variance_threshold() {
        let config = VarianceConfig::default();
        let mut threshold = AdaptiveVarianceThreshold::new(config, 1.0);
        
        let predictions = Array::from_elem((2, 3), 0.5);
        let exceeds = threshold.update_threshold(&predictions);
        
        assert!(threshold.get_threshold() > 0.0);
    }

    #[test]
    fn test_adaptive_alpha() {
        let mut config = VarianceConfig::default();
        config.enable_adaptive_alpha = true;
        let mut stabilizer = VarianceStabilizer::new(config);
        
        let predictions = Array::from_elem((2, 3), 0.5);
        
        // Multiple updates to trigger adaptation
        for _ in 0..20 {
            stabilizer.compute_effective_variance(&predictions);
        }
        
        let stats = stabilizer.get_statistics();
        assert!(stats.history_length > 0);
    }
}
