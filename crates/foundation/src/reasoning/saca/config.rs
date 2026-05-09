//! Configuration for SACA framework components

use serde::{Deserialize, Serialize};
use nexora_core::async_executor::ExecutorConfig;

/// Main SACA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACAConfig {
    #[serde(skip)]
    pub executor_config: ExecutorConfig,
    pub cot_config: CoTConfig,
    pub decompose_config: DecomposeConfig,
    pub context_config: ContextConfig,
    pub sampling_config: SamplingConfig,
    pub execute_config: ExecuteConfig,
    pub rerank_config: RerankConfig,
    pub feedback_config: FeedbackConfig,
    
    // Global settings
    pub quality_threshold: f32,
    pub max_feedback_loops: u32,
    pub parallel_execution: bool,
    pub enable_caching: bool,
    pub log_level: String,
}

impl Default for SACAConfig {
    fn default() -> Self {
        Self {
            executor_config: ExecutorConfig::default(),
            cot_config: CoTConfig::default(),
            decompose_config: DecomposeConfig::default(),
            context_config: ContextConfig::default(),
            sampling_config: SamplingConfig::default(),
            execute_config: ExecuteConfig::default(),
            rerank_config: RerankConfig::default(),
            feedback_config: FeedbackConfig::default(),
            quality_threshold: 0.8,
            max_feedback_loops: 5,
            parallel_execution: true,
            enable_caching: true,
            log_level: "info".to_string(),
        }
    }
}

/// Chain-of-Thought configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoTConfig {
    pub max_reasoning_steps: u32,
    pub reasoning_depth: ReasoningDepth,
    pub include_edge_cases: bool,
    pub include_assumptions: bool,
    pub include_risks: bool,
    pub structured_output: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningDepth {
    Shallow,    // Basic step-by-step
    Medium,     // Detailed analysis
    Deep,       // Comprehensive reasoning
    Exhaustive, // All possible considerations
}

impl Default for CoTConfig {
    fn default() -> Self {
        Self {
            max_reasoning_steps: 10,
            reasoning_depth: ReasoningDepth::Medium,
            include_edge_cases: true,
            include_assumptions: true,
            include_risks: true,
            structured_output: true,
        }
    }
}

/// Modular Decomposition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposeConfig {
    pub max_modules: u32,
    pub min_module_size: u32,
    pub max_module_size: u32,
    pub dependency_analysis: bool,
    pub interface_specification: bool,
    pub complexity_estimation: bool,
}

impl Default for DecomposeConfig {
    fn default() -> Self {
        Self {
            max_modules: 20,
            min_module_size: 5,
            max_module_size: 500,
            dependency_analysis: true,
            interface_specification: true,
            complexity_estimation: true,
        }
    }
}

/// Repository Context configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub max_files_to_analyze: u32,
    pub include_test_files: bool,
    pub include_dependencies: bool,
    pub analyze_naming_conventions: bool,
    pub detect_patterns: bool,
    pub context_cache_ttl_seconds: u64,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_files_to_analyze: 1000,
            include_test_files: true,
            include_dependencies: true,
            analyze_naming_conventions: true,
            detect_patterns: true,
            context_cache_ttl_seconds: 3600, // 1 hour
        }
    }
}

/// Large-Scale Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub num_candidates: u32,
    pub sampling_strategy: super::types::SamplingStrategy,
    pub diversity_threshold: f32,
    pub quality_filter: bool,
    pub parallel_generation: bool,
    pub algorithm_variety: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_candidates: 5,
            sampling_strategy: super::types::SamplingStrategy::Hybrid,
            diversity_threshold: 0.3,
            quality_filter: true,
            parallel_generation: true,
            algorithm_variety: true,
        }
    }
}

/// Execute-Fail-Fix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteConfig {
    pub execution_strategy: super::types::ExecutionStrategy,
    pub timeout_seconds: u64,
    pub max_fix_attempts: u32,
    pub error_analysis_depth: super::types::ErrorAnalysisDepth,
    pub parallel_execution: bool,
    pub capture_performance_metrics: bool,
}

impl Default for ExecuteConfig {
    fn default() -> Self {
        Self {
            execution_strategy: super::types::ExecutionStrategy::Adaptive,
            timeout_seconds: 30,
            max_fix_attempts: 3,
            error_analysis_depth: super::types::ErrorAnalysisDepth::Medium,
            parallel_execution: true,
            capture_performance_metrics: true,
        }
    }
}

/// Mathematical Reranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    pub criteria: super::types::RerankingCriteria,
    pub normalization_method: NormalizationMethod,
    pub weight_adjustment: bool,
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    RobustScaling,
    None,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            criteria: super::types::RerankingCriteria::default(),
            normalization_method: NormalizationMethod::MinMax,
            weight_adjustment: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Feedback system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    pub max_loops: u32,
    pub quality_threshold: f32,
    pub improvement_threshold: f32,
    pub error_analysis_depth: super::types::ErrorAnalysisDepth,
    pub learning_enabled: bool,
    pub feedback_cache_size: u32,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            max_loops: 5,
            quality_threshold: 0.8,
            improvement_threshold: 0.1,
            error_analysis_depth: super::types::ErrorAnalysisDepth::Medium,
            learning_enabled: true,
            feedback_cache_size: 1000,
        }
    }
}
