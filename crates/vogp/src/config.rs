//! Konfigurasi untuk VOGP+ (Variance-Optimized Gradient Penalty Plus)
//!
//! Berisi semua hyperparameter yang dapat dikonfigurasi untuk mengontrol
//! perilaku training dengan dataset kecil.

use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::{debug, info};
use crate::utils::AugmentationType;

/// Konfigurasi lengkap untuk VOGP+ training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOGPConfig {
    /// Bobot untuk smoothness penalty (λ)
    /// Default: 0.1
    /// Range: 0.0 - 1.0
    pub lambda_smooth: f32,
    
    /// Bobot untuk consistency learning (γ)
    /// Default: 0.05
    /// Range: 0.0 - 1.0
    pub gamma_consistency: f32,
    
    /// Threshold adaptif untuk smoothness penalty (τ)
    /// Default: 1.0
    /// Range: 0.1 - 10.0
    pub adaptive_threshold: f32,
    
    /// EMA decay rate untuk gradien historis (β)
    /// Default: 0.99
    /// Range: 0.9 - 0.999
    pub ema_beta: f32,
    
    /// Konstanta stabilitas numerik (ϵ)
    /// Default: 1e-8
    /// Range: 1e-12 - 1e-4
    pub epsilon: f32,
    
    /// Bobot untuk variance vs entropy (α)
    /// Default: 0.7
    /// Range: 0.0 - 1.0
    pub alpha_variance: f32,
    
    /// Mean gradien batch untuk normalisasi (μ_grad)
    /// Default: 0.0
    pub mean_gradient_batch: f32,
    
    /// Jumlah sample untuk stochastic gradient approximation
    /// Default: 10
    /// Range: 1 - 100
    pub stochastic_samples: usize,
    
    /// Jenis augmentasi yang digunakan
    pub augmentation_config: AugmentationConfig,
    
    /// Konfigurasi untuk batch kecil
    pub small_batch_config: SmallBatchConfig,
}

impl Default for VOGPConfig {
    fn default() -> Self {
        Self {
            lambda_smooth: 0.1,
            gamma_consistency: 0.05,
            adaptive_threshold: 1.0,
            ema_beta: 0.99,
            epsilon: 1e-8,
            alpha_variance: 0.7,
            mean_gradient_batch: 0.0,
            stochastic_samples: 10,
            augmentation_config: AugmentationConfig::default(),
            small_batch_config: SmallBatchConfig::default(),
        }
    }
}

impl VOGPConfig {
    /// Buat konfigurasi untuk dataset sangat kecil (< 100 samples)
    pub fn for_very_small_dataset() -> Self {
        Self {
            lambda_smooth: 0.2,        // Lebih tinggi untuk regularisasi kuat
            gamma_consistency: 0.1,     // Lebih tinggi untuk consistency learning
            adaptive_threshold: 0.5,    // Lebih rendah untuk lebih sensitif
            ema_beta: 0.95,             // Lebih rendah untuk adaptasi cepat
            alpha_variance: 0.8,        // Lebih fokus ke variance
            ..Default::default()
        }
    }
    
    /// Buat konfigurasi untuk batch size sangat kecil (1-4)
    pub fn for_micro_batch() -> Self {
        Self {
            lambda_smooth: 0.15,
            gamma_consistency: 0.08,
            adaptive_threshold: 0.8,
            ema_beta: 0.98,             // EMA lebih tinggi untuk stabilitas
            stochastic_samples: 20,      // Lebih banyak sample untuk akurasi
            small_batch_config: SmallBatchConfig {
                enable_gradient_accumulation: true,
                accumulation_steps: 8,
                enable_virtual_batching: true,
                virtual_batch_size: 16,
                enable_memory_optimization: true,
                max_memory_mb: 4096,
                enable_mixed_precision: false,
            },
            ..Default::default()
        }
    }
    
    /// Buat konfigurasi untuk edge device / low-memory
    pub fn for_edge_device() -> Self {
        Self {
            lambda_smooth: 0.05,        // Lebih rendah untuk hemat compute
            gamma_consistency: 0.03,
            adaptive_threshold: 1.5,
            stochastic_samples: 5,       // Lebih sedikit sample untuk kecepatan
            small_batch_config: SmallBatchConfig {
                enable_memory_optimization: true,
                max_memory_mb: 512,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Validasi konfigurasi
    pub fn validate(&self) -> Result<(), VOGPConfigError> {
        if self.lambda_smooth < 0.0 || self.lambda_smooth > 1.0 {
            return Err(VOGPConfigError::InvalidLambdaSmooth);
        }
        
        if self.gamma_consistency < 0.0 || self.gamma_consistency > 1.0 {
            return Err(VOGPConfigError::InvalidGammaConsistency);
        }
        
        if self.adaptive_threshold <= 0.0 {
            return Err(VOGPConfigError::InvalidAdaptiveThreshold);
        }
        
        if self.ema_beta <= 0.0 || self.ema_beta >= 1.0 {
            return Err(VOGPConfigError::InvalidEMABeta);
        }
        
        if self.alpha_variance < 0.0 || self.alpha_variance > 1.0 {
            return Err(VOGPConfigError::InvalidAlphaVariance);
        }
        
        if self.stochastic_samples == 0 {
            return Err(VOGPConfigError::InvalidStochasticSamples);
        }
        
        Ok(())
    }
    
    /// Optimalkan hyperparameter berdasarkan karakteristik dataset
    pub fn optimize_for_dataset(&mut self, dataset_size: usize, batch_size: usize, num_features: usize) {
        info!("Optimizing VOGP+ config for dataset_size={}, batch_size={}, num_features={}", 
              dataset_size, batch_size, num_features);
        
        // Adjust based on dataset size
        if dataset_size < 100 {
            self.lambda_smooth = (self.lambda_smooth * 1.5).min(0.3);
            self.gamma_consistency = (self.gamma_consistency * 1.2).min(0.15);
            self.adaptive_threshold = (self.adaptive_threshold * 0.7).max(0.3);
        } else if dataset_size > 10000 {
            self.lambda_smooth = (self.lambda_smooth * 0.7).max(0.02);
            self.gamma_consistency = (self.gamma_consistency * 0.8).max(0.01);
        }
        
        // Adjust based on batch size
        if batch_size <= 4 {
            self.ema_beta = (self.ema_beta + 0.99) / 2.0; // Increase EMA for stability
            self.stochastic_samples = (self.stochastic_samples * 2).min(50);
        } else if batch_size >= 32 {
            self.ema_beta = (self.ema_beta * 0.95).max(0.9); // Decrease EMA for faster adaptation
        }
        
        // Adjust based on feature dimension
        if num_features > 10000 {
            self.stochastic_samples = ((self.stochastic_samples as f32 * 0.5) as usize).max(3);
        }
        
        debug!("Optimized config: {:?}", self);
    }
}

/// Konfigurasi untuk augmentasi data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Enable consistency learning
    pub enable_consistency: bool,
    
    /// Augmentation types dan probabilitas
    pub augmentations: Vec<AugmentationType>,
    
    /// Jumlah augmentasi per sample
    pub num_augmentations: usize,
    
    /// Strength of augmentation (0.0 - 1.0)
    pub augmentation_strength: f32,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            enable_consistency: true,
            augmentations: vec![
                AugmentationType::GaussianNoise { std: 0.1 },
                AugmentationType::Flip,
            ],
            num_augmentations: 2,
            augmentation_strength: 0.5,
        }
    }
}

/// Konfigurasi khusus untuk batch kecil
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmallBatchConfig {
    /// Enable gradient accumulation
    pub enable_gradient_accumulation: bool,
    
    /// Number of steps for accumulation
    pub accumulation_steps: usize,
    
    /// Enable virtual batching
    pub enable_virtual_batching: bool,
    
    /// Virtual batch size for statistics
    pub virtual_batch_size: usize,
    
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    
    /// Enable mixed precision for small batches
    pub enable_mixed_precision: bool,
}

impl Default for SmallBatchConfig {
    fn default() -> Self {
        Self {
            enable_gradient_accumulation: false,
            accumulation_steps: 4,
            enable_virtual_batching: false,
            virtual_batch_size: 16,
            enable_memory_optimization: false,
            max_memory_mb: 1024,
            enable_mixed_precision: false,
        }
    }
}

/// Error types untuk konfigurasi VOGP+
#[derive(Debug, Clone)]
pub enum VOGPConfigError {
    InvalidLambdaSmooth,
    InvalidGammaConsistency,
    InvalidAdaptiveThreshold,
    InvalidEMABeta,
    InvalidAlphaVariance,
    InvalidStochasticSamples,
    InvalidAugmentationConfig,
}

impl fmt::Display for VOGPConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VOGPConfigError::InvalidLambdaSmooth => {
                write!(f, "lambda_smooth harus dalam range [0.0, 1.0]")
            }
            VOGPConfigError::InvalidGammaConsistency => {
                write!(f, "gamma_consistency harus dalam range [0.0, 1.0]")
            }
            VOGPConfigError::InvalidAdaptiveThreshold => {
                write!(f, "adaptive_threshold harus > 0.0")
            }
            VOGPConfigError::InvalidEMABeta => {
                write!(f, "ema_beta harus dalam range (0.0, 1.0)")
            }
            VOGPConfigError::InvalidAlphaVariance => {
                write!(f, "alpha_variance harus dalam range [0.0, 1.0]")
            }
            VOGPConfigError::InvalidStochasticSamples => {
                write!(f, "stochastic_samples harus > 0")
            }
            VOGPConfigError::InvalidAugmentationConfig => {
                write!(f, "konfigurasi augmentasi tidak valid")
            }
        }
    }
}

impl std::error::Error for VOGPConfigError {}

/// Preset konfigurasi untuk use case yang umum
pub struct VOGPPresets;

impl VOGPPresets {
    /// Medical imaging dengan dataset langka
    pub fn medical_imaging() -> VOGPConfig {
        VOGPConfig {
            lambda_smooth: 0.25,
            gamma_consistency: 0.15,
            adaptive_threshold: 0.3,
            ema_beta: 0.97,
            alpha_variance: 0.9,
            augmentation_config: AugmentationConfig {
                enable_consistency: true,
                augmentations: vec![
                    AugmentationType::GaussianNoise { std: 0.05 },
                    AugmentationType::Crop { ratio: 0.9 },
                ],
                num_augmentations: 3,
                augmentation_strength: 0.3,
            },
            small_batch_config: SmallBatchConfig {
                enable_gradient_accumulation: true,
                accumulation_steps: 8,
                enable_virtual_batching: true,
                virtual_batch_size: 32,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Fault detection industri
    pub fn fault_detection() -> VOGPConfig {
        VOGPConfig {
            lambda_smooth: 0.2,
            gamma_consistency: 0.1,
            adaptive_threshold: 0.8,
            ema_beta: 0.99,
            alpha_variance: 0.6,
            ..Default::default()
        }
    }
    
    /// NLP low-resource language
    pub fn low_resource_nlp() -> VOGPConfig {
        VOGPConfig {
            lambda_smooth: 0.15,
            gamma_consistency: 0.12,
            adaptive_threshold: 0.6,
            ema_beta: 0.98,
            alpha_variance: 0.75,
            augmentation_config: AugmentationConfig {
                enable_consistency: true,
                augmentations: vec![
                    AugmentationType::GaussianNoise { std: 0.08 },
                ],
                num_augmentations: 2,
                augmentation_strength: 0.4,
            },
            ..Default::default()
        }
    }
    
    /// Edge AI deployment
    pub fn edge_ai() -> VOGPConfig {
        VOGPConfig::for_edge_device()
    }
    
    /// Scientific computing dengan data kecil
    pub fn scientific_computing() -> VOGPConfig {
        VOGPConfig {
            lambda_smooth: 0.3,
            gamma_consistency: 0.2,
            adaptive_threshold: 0.4,
            ema_beta: 0.96,
            alpha_variance: 0.85,
            stochastic_samples: 15,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VOGPConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = VOGPConfig::default();
        config.lambda_smooth = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_presets() {
        let medical = VOGPPresets::medical_imaging();
        assert!(medical.validate().is_ok());
        
        let edge = VOGPPresets::edge_ai();
        assert!(edge.validate().is_ok());
    }

    #[test]
    fn test_dataset_optimization() {
        let mut config = VOGPConfig::default();
        config.optimize_for_dataset(50, 2, 1000);
        
        // Should increase regularization for small dataset
        assert!(config.lambda_smooth > 0.1);
        assert!(config.gamma_consistency > 0.05);
    }
}
