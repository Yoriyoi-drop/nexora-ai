//! Core types and data structures for SACA framework

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a coding task to be solved by SACA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTask {
    pub description: String,
    pub requirements: Vec<String>,
    pub constraints: Vec<String>,
    pub context: Option<TaskContext>,
}

/// Additional context for the coding task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub repository_path: Option<String>,
    pub existing_files: Vec<String>,
    pub dependencies: Vec<String>,
    pub coding_standards: HashMap<String, String>,
}

/// SACA session tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACASession {
    pub id: Uuid,
    pub task: CodingTask,
    pub start_time: DateTime<Utc>,
    pub current_phase: SACAPhase,
    pub iterations: u32,
    pub feedback_loops: u32,
}

/// SACA pipeline phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SACAPhase {
    Think,     // Chain-of-Thought
    Decompose, // Modular Decomposition
    Context,   // Repository-Level Awareness
    Sample,    // Large-Scale Sampling
    Execute,   // Execute-Fail-Fix Loop
    Optimize,  // Mathematical Reranking
}

/// Chain-of-Thought reasoning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoTResult {
    pub task_analysis: String,
    pub reasoning_steps: Vec<ReasoningStep>,
    pub edge_cases: Vec<String>,
    pub assumptions: Vec<String>,
    pub risks: Vec<String>,
    pub approach: String,
}

/// Individual reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_number: u32,
    pub description: String,
    pub logic: String,
    pub expected_outcome: String,
}

/// Modular decomposition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub id: String,
    pub name: String,
    pub description: String,
    pub inputs: Vec<ModuleIO>,
    pub outputs: Vec<ModuleIO>,
    pub dependencies: Vec<String>,
    pub complexity: ModuleComplexity,
    pub estimated_lines: u32,
}

/// Module input/output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleIO {
    pub name: String,
    pub data_type: String,
    pub description: String,
    pub optional: bool,
}

/// Module complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleComplexity {
    Low,
    Medium,
    High,
    Critical,
}

/// Repository context analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryContext {
    pub files_analyzed: usize,
    pub functions_found: usize,
    pub dependencies: Vec<String>,
    pub coding_patterns: HashMap<String, usize>,
    pub naming_conventions: NamingConventions,
    pub architectural_patterns: Vec<String>,
    pub test_frameworks: Vec<String>,
}

impl Default for RepositoryContext {
    fn default() -> Self {
        Self {
            files_analyzed: 0,
            functions_found: 0,
            dependencies: Vec::new(),
            coding_patterns: HashMap::new(),
            naming_conventions: NamingConventions::default(),
            architectural_patterns: Vec::new(),
            test_frameworks: Vec::new(),
        }
    }
}

/// Naming conventions detected in repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingConventions {
    pub variable_case: String,  // snake_case, camelCase, etc.
    pub function_case: String,
    pub class_case: String,
    pub constant_case: String,
    pub file_case: String,
}

impl Default for NamingConventions {
    fn default() -> Self {
        Self {
            variable_case: "snake_case".to_string(),
            function_case: "snake_case".to_string(),
            class_case: "PascalCase".to_string(),
            constant_case: "SCREAMING_SNAKE_CASE".to_string(),
            file_case: "snake_case".to_string(),
        }
    }
}

/// Sampling candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingCandidate {
    pub id: Uuid,
    pub module_id: String,
    pub implementation: String,
    pub approach: String,
    pub algorithm: String,
    pub complexity_score: f32,
    pub novelty_score: f32,
}

/// Execution result for a candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACAExecutionResult {
    pub candidate_id: Uuid,
    pub success: bool,
    pub execution_time_ms: u64,
    pub memory_usage_mb: f64,
    pub error_logs: Vec<String>,
    pub test_results: Vec<TestResult>,
    pub performance_metrics: PerformanceMetrics,
}

/// Test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub execution_time_ms: u64,
    pub error_message: Option<String>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub time_complexity: String,
    pub space_complexity: String,
    pub cpu_cycles: u64,
    pub cache_misses: u64,
    pub instructions: u64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            time_complexity: "O(1)".to_string(),
            space_complexity: "O(1)".to_string(),
            cpu_cycles: 0,
            cache_misses: 0,
            instructions: 0,
        }
    }
}

/// Reranked solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACASolution {
    pub session_id: Uuid,
    pub modules: Vec<SolvedModule>,
    pub quality_score: f32,
    pub total_iterations: u32,
    pub total_feedback_loops: u32,
    pub execution_time: chrono::Duration,
    pub final_code: String,
    pub test_coverage: f32,
    pub performance_grade: PerformanceGrade,
}

/// Solved module with final implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolvedModule {
    pub module: Module,
    pub implementation: String,
    pub executed_candidates: Option<Vec<SACAExecutionResult>>,
    pub quality_metrics: ModuleQualityMetrics,
}

/// Quality metrics for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleQualityMetrics {
    pub correctness: f32,
    pub efficiency: f32,
    pub readability: f32,
    pub maintainability: f32,
    pub test_coverage: f32,
    pub documentation_score: f32,
}

/// Performance grade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceGrade {
    Excellent,
    Good,
    Average,
    Poor,
}

/// Feedback for improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACAFeedback {
    pub feedback_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub issues_identified: Vec<String>,
    pub improvement_suggestions: Vec<String>,
    pub new_constraints: Vec<String>,
    pub updated_requirements: Vec<String>,
    pub confidence_score: f32,
}

/// SACA performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SACAMetrics {
    pub total_tasks_processed: u64,
    pub average_quality_score: f32,
    pub average_iterations_per_task: f32,
    pub average_feedback_loops_per_task: f32,
    pub average_execution_time_seconds: f32,
    pub success_rate: f32,
    pub phase_performance: HashMap<SACAPhase, PhaseMetrics>,
}

/// Metrics for individual phases
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseMetrics {
    pub average_time_ms: f64,
    pub success_rate: f32,
    pub average_attempts: f32,
    pub error_rate: f32,
}

/// Sampling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    Random,
    Diverse,
    QualityFocused,
    PerformanceFocused,
    Hybrid,
}

/// Execution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    Adaptive,
}

/// Reranking criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingCriteria {
    pub correctness_weight: f32,
    pub performance_weight: f32,
    pub readability_weight: f32,
    pub maintainability_weight: f32,
    pub test_coverage_weight: f32,
    pub documentation_weight: f32,
}

impl Default for RerankingCriteria {
    fn default() -> Self {
        Self {
            correctness_weight: 0.4,
            performance_weight: 0.25,
            readability_weight: 0.15,
            maintainability_weight: 0.1,
            test_coverage_weight: 0.07,
            documentation_weight: 0.03,
        }
    }
}

/// Feedback loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    pub max_loops: u32,
    pub quality_threshold: f32,
    pub improvement_threshold: f32,
    pub error_analysis_depth: ErrorAnalysisDepth,
}

/// Error analysis depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorAnalysisDepth {
    Shallow,    // Basic error detection
    Medium,     // Pattern analysis
    Deep,       // Root cause analysis
    Comprehensive, // Full contextual analysis
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            max_loops: 5,
            quality_threshold: 0.8,
            improvement_threshold: 0.1,
            error_analysis_depth: ErrorAnalysisDepth::Medium,
        }
    }
}
