//! Query Processing Utilities
//! 
//! Shared utilities for parsing and analyzing queries

use crate::config::GlobalConfig;

/// Query analysis result
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub original: String,
    pub normalized: String,
    pub keywords: Vec<String>,
    pub length: usize,
    pub word_count: usize,
}

impl QueryAnalysis {
    /// Analyze a query string
    pub fn new(query: &str) -> Self {
        let normalized = query.to_lowercase();
        let keywords: Vec<String> = normalized
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let word_count = keywords.len();
        
        Self {
            original: query.to_string(),
            normalized,
            keywords,
            length: query.len(),
            word_count,
        }
    }
    
    /// Check if query contains any of the keywords
    pub fn contains_any(&self, keywords: &[&str]) -> bool {
        keywords.iter().any(|&kw| self.normalized.contains(kw))
    }
    
    /// Check if query contains any of the string keywords
    pub fn contains_any_string(&self, keywords: &[String]) -> bool {
        keywords.iter().any(|kw| self.normalized.contains(kw))
    }
    
    /// Check if query contains all of the keywords
    pub fn contains_all(&self, keywords: &[&str]) -> bool {
        keywords.iter().all(|&kw| self.normalized.contains(kw))
    }
    
    /// Get words that match any of the given keywords
    pub fn matching_words(&self, keywords: &[&str]) -> Vec<String> {
        self.keywords
            .iter()
            .filter(|word| keywords.iter().any(|&kw| word.contains(kw)))
            .cloned()
            .collect()
    }
    
    /// Calculate relevance score against a context
    pub fn relevance_score(&self, context: &str) -> f32 {
        let context_lower = context.to_lowercase();
        let context_words: Vec<&str> = context_lower.split_whitespace().collect();
        
        let mut matches = 0;
        let total_words = self.keywords.len();
        
        if total_words == 0 {
            return 0.0;
        }
        
        for word in &self.keywords {
            if context_words.contains(&word.as_str()) {
                matches += 1;
            }
        }
        
        matches as f32 / total_words as f32
    }
    
    /// Determine if this is a RAG query
    pub fn is_rag_query(&self) -> bool {
        let config = GlobalConfig::instance();
        self.contains_any_string(&config.task_keywords.rag_keywords)
    }
    
    /// Determine task type based on keywords
    pub fn classify_task_type(&self) -> crate::TaskType {
        use crate::TaskType;
        let config = GlobalConfig::instance();
        
        if self.contains_any_string(&config.task_keywords.generation) {
            TaskType::SingleModel
        } else if self.contains_any_string(&config.task_keywords.analysis) {
            TaskType::Sequential
        } else if self.contains_any_string(&config.task_keywords.comparison) {
            TaskType::Parallel
        } else if self.contains_any_string(&config.task_keywords.conditional) {
            TaskType::Conditional
        } else if self.contains_any_string(&config.task_keywords.iterative) {
            TaskType::Iterative
        } else if self.contains_any_string(&config.task_keywords.hierarchical) {
            TaskType::Hierarchical
        } else {
            TaskType::Unknown
        }
    }
    
    /// Calculate complexity based on configuration
    pub fn calculate_complexity(&self) -> f32 {
        let config = GlobalConfig::instance();
        let mut complexity = config.task_analysis.base_complexity;
        
        // Increase complexity based on keywords
        if self.normalized.contains("complex") {
            complexity += config.task_analysis.complexity_keyword_bonus;
        }
        if self.normalized.contains("multi") {
            complexity += config.task_analysis.complexity_multi_bonus;
        }
        if self.length > config.task_analysis.complexity_length_threshold {
            complexity += config.task_analysis.complexity_length_bonus;
        }
        
        complexity.min(config.task_analysis.max_complexity)
    }
    
    /// Estimate execution time
    pub fn estimate_execution_time(&self, task_type: &crate::TaskType) -> u64 {
        let base_time = task_type.base_execution_time();
        let config = GlobalConfig::instance();
        
        // Adjust based on query length
        let length_factor = (self.length / 50) as u64;
        base_time + (length_factor * config.task_analysis.base_time_per_50_chars)
    }
}

/// Utility functions for query processing
pub struct QueryProcessor;

impl QueryProcessor {
    /// Create a query analysis from a string
    pub fn analyze(query: &str) -> QueryAnalysis {
        QueryAnalysis::new(query)
    }
    
    /// Extract keywords from query using configuration
    pub fn extract_keywords(query: &str) -> Vec<String> {
        let analysis = QueryAnalysis::new(query);
        analysis.keywords
    }
    
    /// Simple keyword matching for backward compatibility
    pub fn contains_keyword(query: &str, keyword: &str) -> bool {
        query.to_lowercase().contains(&keyword.to_lowercase())
    }
    
    /// Check if query requires RAG
    pub fn requires_rag(query: &str) -> bool {
        let analysis = QueryAnalysis::new(query);
        analysis.is_rag_query()
    }
}
