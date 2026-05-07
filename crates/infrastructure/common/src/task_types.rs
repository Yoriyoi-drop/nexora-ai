//! Shared Task Types for Nexora-AI
//! 
//! Centralized task type definitions to ensure consistency across modules

use serde::{Serialize, Deserialize};

/// Unified Task Type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    Unknown = 0,
    SingleModel = 1,
    Sequential = 2,
    Parallel = 3,
    Conditional = 4,
    Iterative = 5,
    Hierarchical = 6,
    
    // Specific task types from orchestrator
    Generation = 7,
    Classification = 8,
    Translation = 9,
    Summarization = 10,
    QuestionAnswering = 11,
    CodeGeneration = 12,
    Reasoning = 13,
    Creative = 14,
}

impl Default for TaskType {
    fn default() -> Self {
        TaskType::Unknown
    }
}

impl TaskType {
    pub fn name(&self) -> &'static str {
        match self {
            TaskType::Unknown => "Unknown",
            TaskType::SingleModel => "Single Model",
            TaskType::Sequential => "Sequential",
            TaskType::Parallel => "Parallel",
            TaskType::Conditional => "Conditional",
            TaskType::Iterative => "Iterative",
            TaskType::Hierarchical => "Hierarchical",
            TaskType::Generation => "Generation",
            TaskType::Classification => "Classification",
            TaskType::Translation => "Translation",
            TaskType::Summarization => "Summarization",
            TaskType::QuestionAnswering => "Question Answering",
            TaskType::CodeGeneration => "Code Generation",
            TaskType::Reasoning => "Reasoning",
            TaskType::Creative => "Creative",
        }
    }
    
    /// Get base execution time in milliseconds
    pub fn base_execution_time(&self) -> u64 {
        match self {
            TaskType::Unknown => 1000,
            TaskType::SingleModel => 1000,
            TaskType::Sequential => 3000,
            TaskType::Parallel => 2000,
            TaskType::Conditional => 1500,
            TaskType::Iterative => 2500,
            TaskType::Hierarchical => 5000,
            TaskType::Generation => 1200,
            TaskType::Classification => 800,
            TaskType::Translation => 2000,
            TaskType::Summarization => 1500,
            TaskType::QuestionAnswering => 1000,
            TaskType::CodeGeneration => 1800,
            TaskType::Reasoning => 2500,
            TaskType::Creative => 2000,
        }
    }
    
    /// Check if this task type requires RAG
    pub fn requires_rag(&self) -> bool {
        matches!(self, TaskType::QuestionAnswering | TaskType::Reasoning)
    }
    
    /// Check if this task type is complex
    pub fn is_complex(&self) -> bool {
        matches!(self, TaskType::Hierarchical | TaskType::Reasoning | TaskType::Creative)
    }
}

/// Task analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAnalysis {
    pub task_type: TaskType,
    pub confidence: f32,
    pub reasoning: String,
    pub complexity: f32,
    pub estimated_time_ms: u64,
}

/// Task decomposition structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDecomposition {
    pub task_id: String,
    pub original_query: String,
    pub subtasks: Vec<SubTask>,
    pub overall_type: TaskType,
    pub requires_rag: bool,
    pub requires_context: bool,
    pub complexity_score: f32,
    pub estimated_time_ms: u64,
}

/// Individual subtask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    pub id: String,
    pub description: String,
    pub task_type: TaskType,
    pub dependencies: Vec<String>,
    pub estimated_time_ms: u64,
}

impl TaskDecomposition {
    pub fn new(task_id: String, original_query: String, overall_type: TaskType) -> Self {
        Self {
            task_id,
            original_query,
            subtasks: Vec::new(),
            overall_type,
            requires_rag: overall_type.requires_rag(),
            requires_context: false,
            complexity_score: 0.3,
            estimated_time_ms: overall_type.base_execution_time(),
        }
    }
}
