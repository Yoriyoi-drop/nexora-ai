//! Context analyzer untuk Nexora Core

use crate::types::{InputData, ContextInfo, ModelId};
use crate::error::CoreResult;
use tracing::{debug, info};

/// Context analyzer untuk menganalisis konteks sistem
pub struct ContextAnalyzer {
    memory_enabled: bool,
}

impl ContextAnalyzer {
    pub fn new() -> Self {
        Self {
            memory_enabled: true,
        }
    }
    
    pub fn with_memory(mut self, enabled: bool) -> Self {
        self.memory_enabled = enabled;
        self
    }
    
    /// Analyze current system context
    pub async fn analyze_context(&self, input_data: &InputData, last_active_model: ModelId) -> CoreResult<ContextInfo> {
        debug!("Analyzing context for input: {}", &input_data.raw_input[..input_data.raw_input.len().min(50)]);
        
        let mut context = ContextInfo::new(input_data.raw_input.clone(), last_active_model);
        
        // Check for memory relevance
        if self.memory_enabled {
            context.has_memory = self.has_memory_relevance(&input_data.raw_input);
        }
        
        // Calculate context relevance (simplified version)
        context.context_relevance = 0.5; // Default relevance since we don't store context in controller
        
        info!("Context analyzed: has_memory={}, relevance={:.2}", context.has_memory, context.context_relevance);
        
        Ok(context)
    }
    
    /// Update context dengan informasi baru
    pub async fn update_context(&self, context: &mut ContextInfo, new_context: String) {
        debug!("Updating context");
        context.update_context(new_context);
        info!("Context updated successfully");
    }
    
    /// Check context relevance dengan query
    pub fn check_context_relevance(&self, context: &ContextInfo, query: &str) -> f32 {
        if !query.is_empty() && !context.current_context.is_empty() {
            self.calculate_relevance(&context.current_context, query)
        } else {
            0.0
        }
    }
    
    fn has_memory_relevance(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();
        input_lower.contains("sebelumnya") || 
        input_lower.contains("tadi") ||
        input_lower.contains("ingat") ||
        input_lower.contains("remember")
    }
    
    fn calculate_relevance(&self, context: &str, query: &str) -> f32 {
        // Simple keyword overlap check
        let context_lower = context.to_lowercase();
        let query_lower = query.to_lowercase();
        
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let mut overlap_count = 0;
        
        for word in &query_words {
            if context_lower.contains(word) {
                overlap_count += 1;
            }
        }
        
        if query_words.is_empty() {
            0.0
        } else {
            overlap_count as f32 / query_words.len() as f32
        }
    }
}

impl Default for ContextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_context_analysis() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new("test input".to_string(), crate::types::InputType::Text);
        
        let context = analyzer.analyze_context(&input_data, ModelId::Controller).await.unwrap();
        assert_eq!(context.current_context, "test input");
        assert_eq!(context.active_model, ModelId::Controller);
    }
    
    #[tokio::test]
    async fn test_memory_relevance() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new("ingat yang tadi".to_string(), crate::types::InputType::Text);
        
        let context = analyzer.analyze_context(&input_data, ModelId::Controller).await.unwrap();
        assert!(context.has_memory);
    }
    
    #[test]
    fn test_context_relevance() {
        let analyzer = ContextAnalyzer::new();
        let context = ContextInfo::new("coding rust program".to_string(), ModelId::Controller);
        
        let relevance = analyzer.check_context_relevance(&context, "rust coding");
        assert!(relevance > 0.5);
        
        let relevance = analyzer.check_context_relevance(&context, "python machine learning");
        assert!(relevance < 0.5);
    }
}
