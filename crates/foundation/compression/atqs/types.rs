//! Common types used throughout ATQS

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Layer information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub num_parameters: usize,
    pub weights: ArrayD<f32>,
    pub biases: Option<ArrayD<f32>>,
    pub activation: String,
    pub trainable: bool,
}

impl LayerInfo {
    pub fn new(name: String, layer_type: String) -> Self {
        Self {
            name,
            layer_type,
            input_shape: vec![],
            output_shape: vec![],
            num_parameters: 0,
            weights: ArrayD::zeros(vec![1]),
            biases: None,
            activation: "relu".to_string(),
            trainable: true,
        }
    }
    
    pub fn get_weights(&self) -> &ArrayD<f32> {
        &self.weights
    }
    
    pub fn set_weights(&mut self, weights: ArrayD<f32>) {
        let weight_count = weights.len();
        self.weights = weights;
        self.num_parameters = weight_count;
    }
    
    pub fn get_biases(&self) -> Option<&ArrayD<f32>> {
        self.biases.as_ref()
    }
    
    pub fn set_biases(&mut self, biases: Option<ArrayD<f32>>) {
        let bias_count = biases.as_ref().map(|b| b.len()).unwrap_or(0);
        self.biases = biases;
        self.num_parameters += bias_count;
    }
}

impl ModelLayer for LayerInfo {
    fn get_weights(&self) -> &ArrayD<f32> {
        &self.weights
    }
    
    fn set_weights(&mut self, weights: ArrayD<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let weight_count = weights.len();
        self.weights = weights;
        self.num_parameters = weight_count;
        Ok(())
    }
    
    fn get_biases(&self) -> Option<&ArrayD<f32>> {
        self.biases.as_ref()
    }
    
    fn set_biases(&mut self, biases: Option<ArrayD<f32>>) -> Result<(), Box<dyn std::error::Error>> {
        let bias_count = biases.as_ref().map(|b| b.len()).unwrap_or(0);
        self.biases = biases;
        self.num_parameters += bias_count;
        Ok(())
    }
    
    fn get_layer_info(&self) -> &LayerInfo {
        self
    }
    
    fn forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        // Simple forward pass - just return input for now
        // This would need proper implementation based on layer type
        Ok(input.clone())
    }
    
    fn clone_layer(&self) -> Box<dyn ModelLayer> {
        Box::new(self.clone())
    }
}

/// Calibration dataset structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDataset {
    pub inputs: Vec<ArrayD<f32>>,
    pub targets: Vec<ArrayD<f32>>,
    pub metadata: DatasetMetadata,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub description: String,
    pub total_samples: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub data_type: String,
    pub normalization: Option<String>,
}

impl CalibrationDataset {
    pub fn new(name: String) -> Self {
        Self {
            inputs: Vec::new(),
            targets: Vec::new(),
            metadata: DatasetMetadata {
                name,
                description: String::new(),
                total_samples: 0,
                input_shape: vec![],
                output_shape: vec![],
                data_type: "float32".to_string(),
                normalization: None,
            },
        }
    }
    
    pub fn add_sample(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) {
        self.inputs.push(input);
        self.targets.push(target);
        self.metadata.total_samples = self.inputs.len();
        
        // Update shapes if this is the first sample
        if self.metadata.input_shape.is_empty() && !self.inputs.is_empty() {
            self.metadata.input_shape = self.inputs[0].shape().to_vec();
        }
        if self.metadata.output_shape.is_empty() && !self.targets.is_empty() {
            self.metadata.output_shape = self.targets[0].shape().to_vec();
        }
    }
    
    pub fn len(&self) -> usize {
        self.inputs.len().min(self.targets.len())
    }
    
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() || self.targets.is_empty()
    }
}

/// Calibration batch structure
#[derive(Debug, Clone)]
pub struct CalibrationBatch {
    pub inputs: Vec<ArrayD<f32>>,
    pub targets: Vec<ArrayD<f32>>,
}

impl CalibrationBatch {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            targets: Vec::new(),
        }
    }
    
    pub fn batch_size(&self) -> usize {
        self.inputs.len().min(self.targets.len())
    }
    
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() || self.targets.is_empty()
    }
}

/// Recovery method enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryMethod {
    KnowledgeDistillation,
    PostTrainingQuantization,
    LayerwiseFinetuning,
    AdaptiveCalibration,
    Hybrid,
}

impl Default for RecoveryMethod {
    fn default() -> Self {
        Self::KnowledgeDistillation
    }
}

/// Accuracy recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyRecoveryConfig {
    pub recovery_method: RecoveryMethod,
    pub target_accuracy: f32,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl Default for AccuracyRecoveryConfig {
    fn default() -> Self {
        Self {
            recovery_method: RecoveryMethod::default(),
            target_accuracy: 0.95,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            learning_rate: 1e-4,
            batch_size: 32,
        }
    }
}

/// Calibration optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationOptimizerConfig {
    pub algorithm: String,
    pub population_size: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub validation_split: f32,
    pub max_generations: usize,
    pub convergence_threshold: f32,
}

impl Default for CalibrationOptimizerConfig {
    fn default() -> Self {
        Self {
            algorithm: "genetic".to_string(),
            population_size: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            validation_split: 0.2,
            max_generations: 1000,
            convergence_threshold: 1e-6,
        }
    }
}

/// Optimization state
#[derive(Debug, Clone)]
pub struct OptimizationState {
    pub generation: usize,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub worst_fitness: f32,
    pub convergence_rate: f32,
    pub diversity_score: f32,
    pub loss_trend: Vec<f32>,
    pub gradient_norms: Vec<f32>,
    pub parameter_changes: Vec<f32>,
    pub validation_scores: Vec<f32>,
}

impl Default for OptimizationState {
    fn default() -> Self {
        Self {
            generation: 0,
            best_fitness: f32::INFINITY,
            average_fitness: f32::INFINITY,
            worst_fitness: f32::INFINITY,
            convergence_rate: 0.0,
            diversity_score: 1.0,
            loss_trend: Vec::new(),
            gradient_norms: Vec::new(),
            parameter_changes: Vec::new(),
            validation_scores: Vec::new(),
        }
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub success: bool,
    pub final_fitness: f32,
    pub generations_used: usize,
    pub final_state: OptimizationState,
    pub optimization_history: Vec<OptimizationState>,
    pub best_parameters: HashMap<String, ArrayD<f32>>,
    pub total_training_time: f32,
    pub convergence_achieved: bool,
}

/// Accuracy recovery result
#[derive(Debug, Clone)]
pub struct AccuracyRecoveryResult {
    pub success: bool,
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub accuracy_improvement: f32,
    pub iterations_used: usize,
    pub recovery_method: RecoveryMethod,
    pub recovery_cost: f32,
    pub layer_improvements: Vec<LayerImprovement>,
}

/// Layer improvement information
#[derive(Debug, Clone)]
pub struct LayerImprovement {
    pub layer_idx: usize,
    pub layer_name: String,
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub improvement_method: String,
    pub recovery_strength: f32,
}

/// Optimization algorithm types
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    Genetic,
    ParticleSwarm,
    SimulatedAnnealing,
    GradientDescent,
    Adam,
}

impl OptimizationAlgorithm {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "genetic" => Self::Genetic,
            "particle_swarm" | "pso" => Self::ParticleSwarm,
            "simulated_annealing" | "sa" => Self::SimulatedAnnealing,
            "gradient_descent" | "gd" => Self::GradientDescent,
            "adam" => Self::Adam,
            _ => Self::Genetic,
        }
    }
}

/// Model layer interface
pub trait ModelLayer: Send + Sync {
    fn get_weights(&self) -> &ArrayD<f32>;
    fn set_weights(&mut self, weights: ArrayD<f32>) -> Result<(), Box<dyn std::error::Error>>;
    fn get_biases(&self) -> Option<&ArrayD<f32>>;
    fn set_biases(&mut self, biases: Option<ArrayD<f32>>) -> Result<(), Box<dyn std::error::Error>>;
    fn get_layer_info(&self) -> &LayerInfo;
    fn forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>>;
    fn clone_layer(&self) -> Box<dyn ModelLayer>;
}

/// Error severity enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Info = 0,
    Warning = 1,
    Error = 2,
    Critical = 3,
}

impl Default for ErrorSeverity {
    fn default() -> Self {
        Self::Warning
    }
}

/// Layer sensitivity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSensitivity {
    pub layer_idx: usize,
    pub layer_name: String,
    pub sensitivity_score: f32,
    pub sensitivity_type: SensitivityType,
    pub threshold: f32,
    pub recommendations: Vec<String>,
}

/// Sensitivity type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensitivityType {
    Gradient,
    Activation,
    Weight,
    Output,
}

/// Validation error structure
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub error_type: ErrorType,
    pub message: String,
    pub severity: ErrorSeverity,
}

/// Validation warning structure
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub warning_type: WarningType,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Error type enumeration
#[derive(Debug, Clone)]
pub enum ErrorType {
    AccuracyDegradation,
    CompressionFailure,
    NumericalInstability,
    DimensionMismatch,
    MemoryError,
    IoError,
    SerializationError,
    ValidationError,
}

/// Warning type enumeration
#[derive(Debug, Clone)]
pub enum WarningType {
    LowCompressionRatio,
    HighMemoryUsage,
    SlowPerformance,
    PotentialOverfitting,
    NumericalPrecision,
    NumericalInstability,
}

/// Validation result structure
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub metrics: ValidationMetrics,
}

/// Validation metrics
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub compression_ratio: f32,
    pub accuracy_preservation: f32,
    pub numerical_stability_score: f32,
    pub memory_efficiency: f32,
    pub computational_efficiency: f32,
}

impl Default for ValidationMetrics {
    fn default() -> Self {
        Self {
            compression_ratio: 1.0,
            accuracy_preservation: 1.0,
            numerical_stability_score: 1.0,
            memory_efficiency: 1.0,
            computational_efficiency: 1.0,
        }
    }
}

/// Validation summary
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    pub total_validations: usize,
    pub passed_validations: usize,
    pub failed_validations: usize,
    pub total_errors: usize,
    pub total_warnings: usize,
    pub average_accuracy: f32,
    pub average_compression_ratio: f32,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub check_accuracy_preservation: bool,
    pub check_numerical_stability: bool,
    pub check_memory_efficiency: bool,
    pub tolerance: f32,
    pub max_relative_error: f32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_accuracy_preservation: true,
            check_numerical_stability: true,
            check_memory_efficiency: true,
            tolerance: 1e-6,
            max_relative_error: 0.1,
        }
    }
}

/// Sensitivity distribution
#[derive(Debug, Clone)]
pub struct SensitivityDistribution {
    pub layer_sensitivities: Vec<LayerSensitivity>,
    pub global_sensitivity: f32,
    pub distribution_type: String,
}

/// Sensitivity mapping configuration
#[derive(Debug, Clone)]
pub struct SensitivityMappingConfig {
    pub mapping_resolution: usize,
    pub sensitivity_metric: String,
    pub visualization_enabled: bool,
    pub threshold: f32,
}

/// Layer type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    Attention,
    FeedForward,
    Embedding,
    LayerNorm,
    Output,
}

/// Tensor format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorFormat {
    Dense,
    Tucker,
    CP,
    TT,
    TensorTrain,
    Hierarchical,
    Adaptive,
}

/// Tensor validator
pub struct TensorValidator;

impl TensorValidator {
    /// Validate tensor numerical stability
    pub fn validate_numerical_stability(tensor: &ArrayD<f32>) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Check for NaN values
        let nan_count = tensor.iter().filter(|&&x| x.is_nan()).count();
        if nan_count > 0 {
            errors.push(ValidationError {
                error_type: ErrorType::NumericalInstability,
                message: format!("Found {} NaN values in tensor", nan_count),
                severity: ErrorSeverity::Critical,
            });
        }
        
        // Check for infinite values
        let inf_count = tensor.iter().filter(|&&x| x.is_infinite()).count();
        if inf_count > 0 {
            errors.push(ValidationError {
                error_type: ErrorType::NumericalInstability,
                message: format!("Found {} infinite values in tensor", inf_count),
                severity: ErrorSeverity::Critical,
            });
        }
        
        // Check for very large values
        let large_count = tensor.iter().filter(|&&x| x.abs() > 1e6).count();
        if large_count > 0 {
            warnings.push(ValidationWarning {
                warning_type: WarningType::NumericalPrecision,
                message: format!("Found {} values with magnitude > 1e6", large_count),
                suggestion: Some("Consider normalization".to_string()),
            });
        }
        
        let is_valid = errors.is_empty();
        let numerical_stability_score = if is_valid {
            1.0 - (nan_count + inf_count) as f32 / tensor.len() as f32
        } else {
            0.0
        };
        
        ValidationResult {
            is_valid,
            errors,
            warnings,
            metrics: ValidationMetrics {
                numerical_stability_score,
                ..Default::default()
            },
        }
    }
}
