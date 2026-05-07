//! Configuration for Nexora AI
//! 
//! Global configuration and task analysis settings

use serde::{Deserialize, Serialize};

/// Global configuration for the application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Task classification keywords
    pub task_keywords: TaskKeywords,
    /// Task analysis configuration
    pub task_analysis: TaskAnalysisConfig,
}

/// Keywords for task classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskKeywords {
    /// RAG-related keywords
    pub rag_keywords: Vec<String>,
    /// Generation task keywords
    pub generation: Vec<String>,
    /// Analysis task keywords
    pub analysis: Vec<String>,
    /// Comparison task keywords
    pub comparison: Vec<String>,
    /// Conditional task keywords
    pub conditional: Vec<String>,
    /// Iterative task keywords
    pub iterative: Vec<String>,
    /// Hierarchical task keywords
    pub hierarchical: Vec<String>,
}

/// Task analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAnalysisConfig {
    /// Base complexity score
    pub base_complexity: f32,
    /// Complexity bonus for "complex" keyword
    pub complexity_keyword_bonus: f32,
    /// Complexity bonus for "multi" keyword
    pub complexity_multi_bonus: f32,
    /// Length threshold for complexity bonus
    pub complexity_length_threshold: usize,
    /// Complexity bonus for length
    pub complexity_length_bonus: f32,
    /// Maximum complexity score
    pub max_complexity: f32,
    /// Base time per 50 characters
    pub base_time_per_50_chars: u64,
}

impl Default for TaskKeywords {
    fn default() -> Self {
        Self {
            rag_keywords: vec![
                "what".to_string(),
                "how".to_string(),
                "why".to_string(),
                "explain".to_string(),
                "describe".to_string(),
                "define".to_string(),
            ],
            generation: vec![
                "generate".to_string(),
                "create".to_string(),
                "write".to_string(),
                "produce".to_string(),
            ],
            analysis: vec![
                "analyze".to_string(),
                "examine".to_string(),
                "evaluate".to_string(),
                "assess".to_string(),
            ],
            comparison: vec![
                "compare".to_string(),
                "versus".to_string(),
                "vs".to_string(),
                "difference".to_string(),
            ],
            conditional: vec![
                "if".to_string(),
                "when".to_string(),
                "conditional".to_string(),
                "depending".to_string(),
            ],
            iterative: vec![
                "iterate".to_string(),
                "repeat".to_string(),
                "loop".to_string(),
                "multiple".to_string(),
            ],
            hierarchical: vec![
                "hierarchy".to_string(),
                "nested".to_string(),
                "recursive".to_string(),
                "tree".to_string(),
            ],
        }
    }
}

impl Default for TaskAnalysisConfig {
    fn default() -> Self {
        Self {
            base_complexity: 0.3,
            complexity_keyword_bonus: 0.2,
            complexity_multi_bonus: 0.15,
            complexity_length_threshold: 100,
            complexity_length_bonus: 0.1,
            max_complexity: 1.0,
            base_time_per_50_chars: 100,
        }
    }
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            task_keywords: TaskKeywords::default(),
            task_analysis: TaskAnalysisConfig::default(),
        }
    }
}

impl GlobalConfig {
    /// Get global configuration instance
    pub fn instance() -> Self {
        Self::default()
    }
}
