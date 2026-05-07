//! Tokenizer configuration

use serde::{Deserialize, Serialize};

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub min_frequency: usize,
    pub special_tokens: SpecialTokens,
    pub enable_unicode_normalization: bool,
    pub model_path: Option<String>,
    pub cache_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub unk: String,
    pub pad: String,
    pub bos: String,
    pub eos: String,
    pub sep: String,
    pub cls: String,
    pub mask: String,
    pub domain_coding: String,
    pub domain_memory: String,
    pub domain_debugging: String,
    pub domain_planning: String,
    pub domain_reasoning: String,
    pub domain_ranking: String,
    pub domain_retrieval: String,
    pub domain_validation: String,
    pub domain_personality: String,
    pub domain_optimization: String,
    pub task_code_generation: String,
    pub task_code_analysis: String,
    pub task_code_completion: String,
    pub task_code_review: String,
    pub task_memory_storage: String,
    pub task_memory_retrieval: String,
    pub task_debug_analysis: String,
    pub task_debug_fix: String,
    pub task_plan_creation: String,
    pub task_plan_execution: String,
    pub task_reasoning_logical: String,
    pub task_reasoning_causal: String,
    pub task_ranking_relevance: String,
    pub task_ranking_importance: String,
    pub task_retrieval_search: String,
    pub task_retrieval_synthesis: String,
    pub task_validation_syntax: String,
    pub task_validation_semantic: String,
    pub task_personality_style: String,
    pub task_personality_tone: String,
    pub task_optimization_performance: String,
    pub task_optimization_memory: String,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50000,
            min_frequency: 2,
            special_tokens: SpecialTokens::default(),
            enable_unicode_normalization: true,
            model_path: None,
            cache_size: 1000,
        }
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            unk: "<unk>".to_string(),
            pad: "<pad>".to_string(),
            bos: "<bos>".to_string(),
            eos: "<eos>".to_string(),
            sep: "<sep>".to_string(),
            cls: "<cls>".to_string(),
            mask: "<mask>".to_string(),
            domain_coding: "<domain:coding>".to_string(),
            domain_memory: "<domain:memory>".to_string(),
            domain_debugging: "<domain:debugging>".to_string(),
            domain_planning: "<domain:planning>".to_string(),
            domain_reasoning: "<domain:reasoning>".to_string(),
            domain_ranking: "<domain:ranking>".to_string(),
            domain_retrieval: "<domain:retrieval>".to_string(),
            domain_validation: "<domain:validation>".to_string(),
            domain_personality: "<domain:personality>".to_string(),
            domain_optimization: "<domain:optimization>".to_string(),
            task_code_generation: "<task:code_generation>".to_string(),
            task_code_analysis: "<task:code_analysis>".to_string(),
            task_code_completion: "<task:code_completion>".to_string(),
            task_code_review: "<task:code_review>".to_string(),
            task_memory_storage: "<task:memory_storage>".to_string(),
            task_memory_retrieval: "<task:memory_retrieval>".to_string(),
            task_debug_analysis: "<task:debug_analysis>".to_string(),
            task_debug_fix: "<task:debug_fix>".to_string(),
            task_plan_creation: "<task:plan_creation>".to_string(),
            task_plan_execution: "<task:plan_execution>".to_string(),
            task_reasoning_logical: "<task:reasoning_logical>".to_string(),
            task_reasoning_causal: "<task:reasoning_causal>".to_string(),
            task_ranking_relevance: "<task:ranking_relevance>".to_string(),
            task_ranking_importance: "<task:ranking_importance>".to_string(),
            task_retrieval_search: "<task:retrieval_search>".to_string(),
            task_retrieval_synthesis: "<task:retrieval_synthesis>".to_string(),
            task_validation_syntax: "<task:validation_syntax>".to_string(),
            task_validation_semantic: "<task:validation_semantic>".to_string(),
            task_personality_style: "<task:personality_style>".to_string(),
            task_personality_tone: "<task:personality_tone>".to_string(),
            task_optimization_performance: "<task:optimization_performance>".to_string(),
            task_optimization_memory: "<task:optimization_memory>".to_string(),
        }
    }
}
