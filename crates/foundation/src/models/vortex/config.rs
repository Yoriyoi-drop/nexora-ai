//! NXR-VORTEX Configuration
//! 
//! Model-specific configuration for NXR-VORTEX

use serde::{Deserialize, Serialize};
use crate::shared::{model_config::NxrModelConfig, deeplearning_integration::DeepLearningConfig};

/// NXR-VORTEX Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Code analysis configuration
    pub code_analysis: CodeAnalysisConfig,
    /// Debugging configuration
    pub debugging: DebuggingConfig,
    /// Architecture configuration
    pub architecture: ArchitectureConfig,
    /// Optimization configuration
    pub optimization: OptimizationConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Deep learning configuration
    pub deep_learning: DeepLearningConfig,
}

/// Code Analysis Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysisConfig {
    /// Maximum analysis depth
    pub max_analysis_depth: u8,
    /// Enable static analysis
    pub enable_static_analysis: bool,
    /// Enable dynamic analysis
    pub enable_dynamic_analysis: bool,
    /// Language detection mode
    pub language_detection: LanguageDetectionMode,
    /// Pattern recognition sensitivity
    pub pattern_sensitivity: f32,
    /// Complexity calculation method
    pub complexity_method: ComplexityMethod,
}

/// Language Detection Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LanguageDetectionMode {
    /// Automatic detection
    Automatic,
    /// Manual specification
    Manual { language: String },
    /// Multi-language support
    MultiLanguage { primary: String, fallback: Vec<String> },
}

/// Complexity Calculation Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityMethod {
    /// Cyclomatic complexity
    Cyclomatic,
    /// Cognitive complexity
    Cognitive,
    /// Halstead complexity
    Halstead,
    /// Hybrid complexity
    Hybrid { weights: ComplexityWeights },
}

/// Complexity Weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityWeights {
    pub cyclomatic_weight: f32,
    pub cognitive_weight: f32,
    pub halstead_weight: f32,
}

/// Debugging Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggingConfig {
    /// Debugging strategy
    pub debugging_strategy: DebuggingStrategy,
    /// Maximum hypotheses
    pub max_hypotheses: usize,
    /// Enable symbolic execution
    pub enable_symbolic_execution: bool,
    /// Error classification mode
    pub error_classification: ErrorClassificationMode,
    /// Fix generation mode
    pub fix_generation_mode: FixGenerationMode,
}

/// Debugging Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebuggingStrategy {
    /// Static analysis only
    StaticOnly,
    /// Dynamic analysis only
    DynamicOnly,
    /// Hybrid debugging
    Hybrid { static_weight: f32, dynamic_weight: f32 },
    /// Adaptive debugging
    Adaptive,
}

/// Error Classification Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorClassificationMode {
    /// Basic classification
    Basic,
    /// Detailed classification
    Detailed,
    /// Machine learning classification
    MLClassification,
    /// Rule-based classification
    RuleBased,
}

/// Fix Generation Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixGenerationMode {
    /// Conservative fixes only
    Conservative,
    /// Aggressive fixes
    Aggressive,
    /// Context-aware fixes
    ContextAware,
    /// Multi-option fixes
    MultiOption { max_options: usize },
}

/// Architecture Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Enable pattern detection
    pub enable_pattern_detection: bool,
    /// Maximum complexity analysis
    pub max_complexity_analysis: f32,
    /// Enable optimization suggestions
    pub enable_optimization_suggestions: bool,
    /// Design pattern database
    pub pattern_database: PatternDatabase,
    /// Architecture validation mode
    pub validation_mode: ValidationMode,
}

/// Pattern Database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternDatabase {
    /// Built-in patterns only
    BuiltIn,
    /// Custom patterns
    Custom { pattern_files: Vec<String> },
    /// Hybrid database
    Hybrid { built_in_weight: f32, custom_weight: f32 },
}

/// Validation Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    /// Basic validation
    Basic,
    /// Strict validation
    Strict,
    /// Adaptive validation
    Adaptive,
    /// Domain-specific validation
    DomainSpecific { domain: String },
}

/// Optimization Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    /// Target performance metrics
    pub target_metrics: Vec<PerformanceMetric>,
    /// Optimization strategies
    pub strategies: Vec<OptimizationStrategy>,
}

/// Optimization Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Aggressive optimization
    Aggressive,
    /// Maximum optimization
    Maximum,
}

/// Performance Metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Execution time
    ExecutionTime { target_ms: u64 },
    /// Memory usage
    MemoryUsage { target_mb: u64 },
    /// CPU utilization
    CPUUtilization { target_percent: f32 },
    /// Throughput
    Throughput { target_ops_per_sec: f64 },
}

/// Optimization Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Loop optimization
    LoopOptimization,
    /// Memory optimization
    MemoryOptimization,
    /// Algorithm optimization
    AlgorithmOptimization,
    /// Parallelization
    Parallelization,
    /// Caching optimization
    CachingOptimization,
}

/// Security Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable vulnerability scanning
    pub enable_vulnerability_scanning: bool,
    /// Security analysis depth
    pub security_analysis_depth: SecurityDepth,
    /// Vulnerability database
    pub vulnerability_database: VulnerabilityDatabase,
    /// Security standards compliance
    pub compliance_standards: Vec<ComplianceStandard>,
}

/// Security Depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityDepth {
    /// Surface level analysis
    Surface,
    /// Deep analysis
    Deep,
    /// Comprehensive analysis
    Comprehensive,
    /// Expert level analysis
    Expert,
}

/// Vulnerability Database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityDatabase {
    /// CVE database only
    CVE,
    /// Custom database
    Custom { database_files: Vec<String> },
    /// Hybrid database
    Hybrid { cve_weight: f32, custom_weight: f32 },
}

/// Compliance Standard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// OWASP standards
    OWASP,
    /// NIST standards
    NIST,
    /// ISO standards
    ISO { standard_number: String },
    /// Custom standards
    Custom { standard_name: String },
}

impl Default for VortexConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Vortex),
            code_analysis: CodeAnalysisConfig::default(),
            debugging: DebuggingConfig::default(),
            architecture: ArchitectureConfig::default(),
            optimization: OptimizationConfig::default(),
            security: SecurityConfig::default(),
            deep_learning: DeepLearningConfig::star_x(),
        }
    }
}

impl Default for CodeAnalysisConfig {
    fn default() -> Self {
        Self {
            max_analysis_depth: 10,
            enable_static_analysis: true,
            enable_dynamic_analysis: false,
            language_detection: LanguageDetectionMode::Automatic,
            pattern_sensitivity: 0.8,
            complexity_method: ComplexityMethod::Hybrid {
                weights: ComplexityWeights {
                    cyclomatic_weight: 0.4,
                    cognitive_weight: 0.4,
                    halstead_weight: 0.2,
                },
            },
        }
    }
}

impl Default for DebuggingConfig {
    fn default() -> Self {
        Self {
            debugging_strategy: DebuggingStrategy::Hybrid {
                static_weight: 0.7,
                dynamic_weight: 0.3,
            },
            max_hypotheses: 5,
            enable_symbolic_execution: false,
            error_classification: ErrorClassificationMode::MLClassification,
            fix_generation_mode: FixGenerationMode::ContextAware,
        }
    }
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            enable_pattern_detection: true,
            max_complexity_analysis: 10.0,
            enable_optimization_suggestions: true,
            pattern_database: PatternDatabase::Hybrid {
                built_in_weight: 0.7,
                custom_weight: 0.3,
            },
            validation_mode: ValidationMode::Adaptive,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Aggressive,
            enable_performance_profiling: true,
            target_metrics: vec![
                PerformanceMetric::ExecutionTime { target_ms: 100 },
                PerformanceMetric::MemoryUsage { target_mb: 256 },
            ],
            strategies: vec![
                OptimizationStrategy::LoopOptimization,
                OptimizationStrategy::MemoryOptimization,
                OptimizationStrategy::AlgorithmOptimization,
            ],
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_vulnerability_scanning: true,
            security_analysis_depth: SecurityDepth::Deep,
            vulnerability_database: VulnerabilityDatabase::Hybrid {
                cve_weight: 0.8,
                custom_weight: 0.2,
            },
            compliance_standards: vec![
                ComplianceStandard::OWASP,
                ComplianceStandard::NIST,
            ],
        }
    }
}

impl VortexConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate code analysis configuration
        if self.code_analysis.max_analysis_depth == 0 {
            return Err("max_analysis_depth must be > 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.code_analysis.pattern_sensitivity) {
            return Err("pattern_sensitivity must be between 0.0 and 1.0".to_string());
        }

        // Validate debugging configuration
        if self.debugging.max_hypotheses == 0 {
            return Err("max_hypotheses must be > 0".to_string());
        }

        // Validate architecture configuration
        if self.architecture.max_complexity_analysis <= 0.0 {
            return Err("max_complexity_analysis must be > 0".to_string());
        }

        // Validate optimization configuration
        if self.optimization.target_metrics.is_empty() {
            return Err("At least one target metric must be specified".to_string());
        }

        // Validate security configuration
        if self.security.compliance_standards.is_empty() {
            return Err("At least one compliance standard must be specified".to_string());
        }

        Ok(())
    }

    /// Get configuration for specific component
    pub fn get_component_config(&self, component: &str) -> Option<serde_json::Value> {
        match component {
            "code_analysis" => Some(serde_json::to_value(&self.code_analysis).unwrap_or_default()),
            "debugging" => Some(serde_json::to_value(&self.debugging).unwrap_or_default()),
            "architecture" => Some(serde_json::to_value(&self.architecture).unwrap_or_default()),
            "optimization" => Some(serde_json::to_value(&self.optimization).unwrap_or_default()),
            "security" => Some(serde_json::to_value(&self.security).unwrap_or_default()),
            _ => None,
        }
    }

    /// Update component configuration
    pub fn update_component_config<T>(&mut self, component: String, config: T) -> Result<(), serde_json::Error>
    where
        T: Serialize,
    {
        let json_value = serde_json::to_value(config)?;
        
        match component.as_str() {
            "code_analysis" => {
                self.code_analysis = serde_json::from_value(json_value)?;
            }
            "debugging" => {
                self.debugging = serde_json::from_value(json_value)?;
            }
            "architecture" => {
                self.architecture = serde_json::from_value(json_value)?;
            }
            "optimization" => {
                self.optimization = serde_json::from_value(json_value)?;
            }
            "security" => {
                self.security = serde_json::from_value(json_value)?;
            }
            _ => {
                return Err(serde_json::Error::syntax(serde_json::error::ErrorCode::ExpectedColon, 0, 0));
            }
        }

        Ok(())
    }

    /// Get optimization strategies as strings
    pub fn get_optimization_strategies(&self) -> Vec<String> {
        self.optimization.strategies.iter().map(|s| format!("{:?}", s)).collect()
    }

    /// Get compliance standards as strings
    pub fn get_compliance_standards(&self) -> Vec<String> {
        self.security.compliance_standards.iter().map(|s| format!("{:?}", s)).collect()
    }

    /// Check if symbolic execution is enabled
    pub fn is_symbolic_execution_enabled(&self) -> bool {
        self.debugging.enable_symbolic_execution
    }

    /// Check if vulnerability scanning is enabled
    pub fn is_vulnerability_scanning_enabled(&self) -> bool {
        self.security.enable_vulnerability_scanning
    }

    /// Get maximum analysis depth
    pub fn get_max_analysis_depth(&self) -> u8 {
        self.code_analysis.max_analysis_depth
    }

    /// Set maximum analysis depth
    pub fn set_max_analysis_depth(&mut self, depth: u8) {
        self.code_analysis.max_analysis_depth = depth;
    }

    /// Get pattern sensitivity
    pub fn get_pattern_sensitivity(&self) -> f32 {
        self.code_analysis.pattern_sensitivity
    }

    /// Set pattern sensitivity
    pub fn set_pattern_sensitivity(&mut self, sensitivity: f32) {
        self.code_analysis.pattern_sensitivity = sensitivity;
    }
}
