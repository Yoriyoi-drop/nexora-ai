//! Configuration types for ATQS (Attention Tensor Quantum System)

use serde::{Deserialize, Serialize};
use crate::compression::SparseConfig;

/// Main ATQS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATQSConfig {
    /// Calibration settings
    pub calibration: CalibrationConfig,
    /// Compression settings
    pub compression: CompressionConfig,
    /// Profiling settings
    pub profiling: ProfilingConfig,
    /// Core settings
    pub core: CoreConfig,
    /// Maximum accuracy drop allowed
    pub max_accuracy_drop: f32,
    /// Target sparse ratio
    pub sparse_ratio: f32,
    /// Validation tolerance
    pub tolerance: f32,
    /// Number of calibration samples
    pub calibration_samples: usize,
    /// Target compression ratio
    pub target_compression_ratio: f32,
}

/// Calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub lora: LoRACalibrationConfig,
    pub accuracy_recovery: AccuracyRecoveryConfig,
    pub optimizer: CalibrationOptimizerConfig,
}

/// LoRA calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRACalibrationConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

/// Accuracy recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyRecoveryConfig {
    pub threshold: f32,
    pub max_iterations: usize,
    pub convergence_tolerance: f32,
}

/// Calibration optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationOptimizerConfig {
    pub algorithm: String,
    pub population_size: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub adaptive_rank: AdaptiveRankConfig,
    pub quantum_sparse: QuantumSparseConfig,
    pub sparse_augmentation: SparseAugmentationConfig,
}

/// Adaptive rank configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRankConfig {
    pub target_compression: f32,
    pub min_rank: usize,
    pub max_rank: usize,
    pub adaptation_rate: f32,
}

/// Quantum sparse configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSparseConfig {
    pub sparsity_threshold: f32,
    pub quantum_layers: Vec<String>,
    pub entanglement_threshold: f32,
}

/// Sparse augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseAugmentationConfig {
    pub augmentation_factor: f32,
    pub sparse_method: String,
    pub recovery_method: String,
}

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub entanglement: EntanglementProfilerConfig,
    pub layer_analysis: LayerAnalyzerConfig,
    pub sensitivity_mapping: SensitivityMapperConfig,
}

/// Entanglement profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementProfilerConfig {
    pub sample_size: usize,
    pub entropy_threshold: f32,
    pub correlation_threshold: f32,
}

/// Layer analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAnalyzerConfig {
    pub analysis_depth: usize,
    pub sensitivity_threshold: f32,
    pub rank_estimation_method: String,
}

/// Sensitivity mapper configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityMapperConfig {
    pub mapping_resolution: usize,
    pub sensitivity_metric: String,
    pub visualization_enabled: bool,
}

/// Core configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    pub attention: AttentionConfig,
    pub tensor_ops: TensorOpsConfig,
    pub quantum_networks: QuantumNetworksConfig,
}

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub heads: usize,
    pub dimensions: usize,
    pub dropout: f32,
    pub optimization: AttentionOptimizationConfig,
}

/// Attention optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionOptimizationConfig {
    pub path_optimization: bool,
    pub joint_optimization: bool,
    pub convergence_threshold: f32,
}

/// Tensor operations configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOpsConfig {
    pub precision: String,
    pub parallelization: bool,
    pub memory_optimization: bool,
}

/// Quantum networks configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNetworksConfig {
    pub network_type: String,
    pub entanglement_degree: usize,
    pub coherence_time: f32,
}

impl Default for ATQSConfig {
    fn default() -> Self {
        // Pre-computed default values to avoid nested allocations
        let calibration = CalibrationConfig {
            lora: LoRACalibrationConfig {
                rank: 64,
                alpha: 16.0,
                dropout: 0.1,
                learning_rate: 1e-4,
                batch_size: 32,
                epochs: 10,
            },
            accuracy_recovery: AccuracyRecoveryConfig {
                threshold: 0.95,
                max_iterations: 1000,
                convergence_tolerance: 1e-6,
            },
            optimizer: CalibrationOptimizerConfig {
                algorithm: "genetic".to_string(),
                population_size: 100,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
            },
        };
        
        let compression = CompressionConfig {
            adaptive_rank: AdaptiveRankConfig {
                target_compression: 0.5,
                min_rank: 8,
                max_rank: 512,
                adaptation_rate: 0.1,
            },
            quantum_sparse: QuantumSparseConfig::default(),
            sparse_augmentation: SparseAugmentationConfig::default(),
        };
        
        let profiling = ProfilingConfig {
            entanglement: EntanglementProfilerConfig::default(),
            layer_analysis: LayerAnalyzerConfig::default(),
            sensitivity_mapping: SensitivityMapperConfig::default(),
        };
        
        let core = CoreConfig {
            attention: AttentionConfig {
                heads: 12,
                dimensions: 768,
                dropout: 0.1,
                optimization: AttentionOptimizationConfig {
                    path_optimization: true,
                    joint_optimization: true,
                    convergence_threshold: 1e-6,
                },
            },
            tensor_ops: TensorOpsConfig {
                precision: "fp32".to_string(),
                parallelization: true,
                memory_optimization: true,
            },
            quantum_networks: QuantumNetworksConfig {
                network_type: "ipeps".to_string(),
                entanglement_degree: 4,
                coherence_time: 1e-3,
            },
        };
        
        Self {
            calibration,
            compression,
            profiling,
            core,
            max_accuracy_drop: 0.05,
            sparse_ratio: 0.5,
            tolerance: 1e-6,
            calibration_samples: 1000,
            target_compression_ratio: 0.5,
        }
    }
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            lora: LoRACalibrationConfig::default(),
            accuracy_recovery: AccuracyRecoveryConfig::default(),
            optimizer: CalibrationOptimizerConfig::default(),
        }
    }
}

impl Default for LoRACalibrationConfig {
    fn default() -> Self {
        Self {
            rank: 64,
            alpha: 16.0,
            dropout: 0.1,
            learning_rate: 1e-4,
            batch_size: 32,
            epochs: 10,
        }
    }
}

impl Default for AccuracyRecoveryConfig {
    fn default() -> Self {
        Self {
            threshold: 0.95,
            max_iterations: 1000,
            convergence_tolerance: 1e-6,
        }
    }
}

impl Default for CalibrationOptimizerConfig {
    fn default() -> Self {
        Self {
            algorithm: "genetic".to_string(),
            population_size: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            adaptive_rank: AdaptiveRankConfig::default(),
            quantum_sparse: QuantumSparseConfig::default(),
            sparse_augmentation: SparseAugmentationConfig::default(),
        }
    }
}

impl Default for AdaptiveRankConfig {
    fn default() -> Self {
        Self {
            target_compression: 0.5,
            min_rank: 8,
            max_rank: 512,
            adaptation_rate: 0.1,
        }
    }
}

impl Default for QuantumSparseConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.1,
            quantum_layers: vec!["attention".to_string(), "mlp".to_string()],
            entanglement_threshold: 0.05,
        }
    }
}

impl Default for SparseAugmentationConfig {
    fn default() -> Self {
        Self {
            augmentation_factor: 1.2,
            sparse_method: "magnitude".to_string(),
            recovery_method: "iterative".to_string(),
        }
    }
}


impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            entanglement: EntanglementProfilerConfig::default(),
            layer_analysis: LayerAnalyzerConfig::default(),
            sensitivity_mapping: SensitivityMapperConfig::default(),
        }
    }
}

impl Default for EntanglementProfilerConfig {
    fn default() -> Self {
        Self {
            sample_size: 1000,
            entropy_threshold: 0.1,
            correlation_threshold: 0.8,
        }
    }
}

impl Default for LayerAnalyzerConfig {
    fn default() -> Self {
        Self {
            analysis_depth: 10,
            sensitivity_threshold: 0.01,
            rank_estimation_method: "svd".to_string(),
        }
    }
}

impl Default for SensitivityMapperConfig {
    fn default() -> Self {
        Self {
            mapping_resolution: 100,
            sensitivity_metric: "gradient".to_string(),
            visualization_enabled: true,
        }
    }
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            attention: AttentionConfig::default(),
            tensor_ops: TensorOpsConfig::default(),
            quantum_networks: QuantumNetworksConfig::default(),
        }
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            heads: 12,
            dimensions: 768,
            dropout: 0.1,
            optimization: AttentionOptimizationConfig::default(),
        }
    }
}

impl Default for AttentionOptimizationConfig {
    fn default() -> Self {
        Self {
            path_optimization: true,
            joint_optimization: true,
            convergence_threshold: 1e-6,
        }
    }
}

impl Default for TensorOpsConfig {
    fn default() -> Self {
        Self {
            precision: "fp32".to_string(),
            parallelization: true,
            memory_optimization: true,
        }
    }
}

impl Default for QuantumNetworksConfig {
    fn default() -> Self {
        Self {
            network_type: "ipeps".to_string(),
            entanglement_degree: 4,
            coherence_time: 1e-3,
        }
    }
}

/// Compression levels for ATQS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionLevel {
    Low,
    Medium,
    High,
    Ultra,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel::Medium
    }
}
