//! Context analyzer untuk Nexora Core

use crate::error::CoreResult;
use crate::types::{ContextInfo, InputData, ModelId};
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
    pub async fn analyze_context(
        &self,
        input_data: &InputData,
        last_active_model: ModelId,
    ) -> CoreResult<ContextInfo> {
        debug!(
            "Analyzing context for input: {}",
            input_data.raw_input.chars().take(50).collect::<String>()
        );

        let mut context = ContextInfo::new(input_data.raw_input.clone(), last_active_model);

        // Check for memory relevance
        if self.memory_enabled {
            context.has_memory = self.has_memory_relevance(&input_data.raw_input);
        }

        // Calculate context relevance (simplified version)
        context.context_relevance = 0.5; // Default relevance since we don't store context in controller

        info!(
            "Context analyzed: has_memory={}, relevance={:.2}",
            context.has_memory, context.context_relevance
        );

        Ok(context)
    }

    /// Update context dengan informasi baru
    pub async fn update_context(&self, context: &mut ContextInfo, new_context: String) {
        debug!("Updating context");
        context.update_context(new_context);
        info!("Context updated successfully");
    }

    fn has_memory_relevance(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();
        let memory_triggers = [
            "sebelumnya", "tadi", "ingat", "remember", "recall", "dulu", "lalu",
            "episodic", "kenangan", "pengalaman", "memori", "memory",
            "sejarah", "history", "previous", "sebelum", "lampau",
        ];
        memory_triggers.iter().any(|k| input_lower.contains(k))
    }

    fn calculate_relevance(&self, context: &str, query: &str) -> f32 {
        let context_lower = context.to_lowercase();
        let query_lower = query.to_lowercase();

        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let context_words: std::collections::HashSet<&str> =
            context_lower.split_whitespace().collect();

        if query_words.is_empty() || context_words.is_empty() {
            return 0.0;
        }

        // TF-IDF-like: count overlapping non-stop-word terms
        let stop_words = ["yang", "dan", "di", "ke", "dari", "the", "and", "in", "to", "of"];
        let significant_terms: Vec<&&str> = query_words
            .iter()
            .filter(|w| w.len() > 3 && !stop_words.contains(w))
            .collect();

        if significant_terms.is_empty() {
            return 0.0;
        }

        let overlap = significant_terms
            .iter()
            .filter(|w| context_words.contains(**w))
            .count();
        overlap as f32 / significant_terms.len() as f32
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

        let context = analyzer
            .analyze_context(&input_data, ModelId::Controller)
            .await
            .unwrap();
        assert_eq!(context.current_context, "test input");
        assert_eq!(context.active_model, ModelId::Controller);
    }

    #[tokio::test]
    async fn test_memory_relevance() {
        let analyzer = ContextAnalyzer::new();
        let input_data =
            InputData::new("ingat yang tadi".to_string(), crate::types::InputType::Text);

        let context = analyzer
            .analyze_context(&input_data, ModelId::Controller)
            .await
            .unwrap();
        assert!(context.has_memory);
    }

    #[tokio::test]
    async fn test_context_without_memory() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new(
            "buat program rust".to_string(),
            crate::types::InputType::Text,
        );
        let context = analyzer
            .analyze_context(&input_data, ModelId::Coding)
            .await
            .unwrap();
        assert_eq!(context.active_model, ModelId::Coding);
    }

    #[tokio::test]
    async fn test_context_memory_disabled() {
        let analyzer = ContextAnalyzer::new().with_memory(false);
        let input_data = InputData::new(
            "ingat yang tadi".to_string(),
            crate::types::InputType::Text,
        );
        let context = analyzer
            .analyze_context(&input_data, ModelId::Controller)
            .await
            .unwrap();
        assert!(!context.has_memory);
    }

    #[tokio::test]
    async fn test_context_update() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new(
            "test".to_string(),
            crate::types::InputType::Text,
        );
        let mut context = analyzer
            .analyze_context(&input_data, ModelId::Controller)
            .await
            .unwrap();
        analyzer.update_context(&mut context, "updated context".to_string()).await;
        assert_eq!(context.current_context, "updated context");
    }

    #[tokio::test]
    async fn test_recall_triggers_memory() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new(
            "recall previous conversation".to_string(),
            crate::types::InputType::Text,
        );
        let context = analyzer
            .analyze_context(&input_data, ModelId::Controller)
            .await
            .unwrap();
        assert!(context.has_memory);
    }

    #[tokio::test]
    async fn test_dulu_triggers_memory() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new(
            "cerita tentang dulu".to_string(),
            crate::types::InputType::Text,
        );
        let context = analyzer
            .analyze_context(&input_data, ModelId::Controller)
            .await
            .unwrap();
        assert!(context.has_memory);
    }

    #[test]
    fn test_context_default_relevance() {
        let analyzer = ContextAnalyzer::new();
        // Default relevance for non-matching input
        let relevance = analyzer.calculate_relevance("rust programming", "buat fungsi");
        // Some words may overlap
        assert!(relevance >= 0.0);
    }

    #[test]
    fn test_context_empty_query_relevance() {
        let analyzer = ContextAnalyzer::new();
        let relevance = analyzer.calculate_relevance("some context", "");
        assert_eq!(relevance, 0.0);
    }
}
