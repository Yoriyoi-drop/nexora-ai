//! Foundation Reasoning Framework (SACA)
//!
//! Re-exports from `nexora-reasoning` crate, plus SacaAetherIntegration
//! that depends on NXR-ÆTHER (model series still in foundation).

pub use nexora_reasoning::saca::*;
pub use nexora_reasoning::*;

use nexora_models::aether::NxrAetherModel;
use nexora_shared::base_model::{InputData, NxrInput, NxrModel, NxrOutput};

/// Enhanced SACA with NXR-ÆTHER integration
pub struct SacaAetherIntegration {
    pub saca_engine: nexora_reasoning::saca::SacaEngine,
    pub aether_model: NxrAetherModel,
    pub integration_config: SacaAetherConfig,
}

#[derive(Debug, Clone)]
pub struct SacaAetherConfig {
    pub enable_emotional_reasoning: bool,
    pub empathy_threshold: f32,
    pub cultural_adaptation: bool,
}

impl Default for SacaAetherConfig {
    fn default() -> Self {
        Self {
            enable_emotional_reasoning: true,
            empathy_threshold: 0.7,
            cultural_adaptation: true,
        }
    }
}

impl SacaAetherIntegration {
    pub fn new() -> Self {
        Self {
            saca_engine: nexora_reasoning::saca::SacaEngine::new(),
            aether_model: NxrAetherModel::new(),
            integration_config: SacaAetherConfig::default(),
        }
    }

    pub async fn enhanced_reasoning(&self, query: &str) -> EnhancedReasoningResult {
        let mut result = EnhancedReasoningResult::new();
        if self.integration_config.enable_emotional_reasoning {
            let emotional_context = self.analyze_emotional_context(query);
            let aether_input = NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: InputData::Text(query.to_string()),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };
            if let Ok(aether_output) = self.aether_model.infer(&aether_input).await {
                result.emotional_analysis = Some(aether_output);
            }
            result.emotional_context = Some(emotional_context);
        }
        result
    }

    fn analyze_emotional_context(&self, query: &str) -> String {
        let lower = query.to_lowercase();
        let positive_words = [
            "happy",
            "great",
            "excellent",
            "good",
            "love",
            "wonderful",
            "amazing",
            "fantastic",
            "beautiful",
            "joy",
            "hope",
            "grateful",
        ];
        let negative_words = [
            "sad",
            "angry",
            "bad",
            "terrible",
            "horrible",
            "hate",
            "awful",
            "depressed",
            "anxious",
            "fear",
            "scared",
            "angry",
            "frustrated",
        ];
        let urgent_words = [
            "urgent",
            "emergency",
            "critical",
            "immediately",
            "asap",
            "important",
            "deadline",
            "crisis",
        ];
        let analytical_words = [
            "analyze",
            "compare",
            "evaluate",
            "assess",
            "calculate",
            "determine",
            "investigate",
            "examine",
            "review",
        ];

        let pos_count = positive_words.iter().filter(|w| lower.contains(*w)).count();
        let neg_count = negative_words.iter().filter(|w| lower.contains(*w)).count();
        let urg_count = urgent_words.iter().filter(|w| lower.contains(*w)).count();
        let ana_count = analytical_words
            .iter()
            .filter(|w| lower.contains(*w))
            .count();

        if urg_count > 0 {
            format!("urgent (urgency_score: {})", urg_count)
        } else if pos_count > neg_count {
            format!(
                "positive (positivity: {:.1})",
                pos_count as f32 / (pos_count + neg_count + 1) as f32
            )
        } else if neg_count > pos_count {
            format!(
                "negative (negativity: {:.1})",
                neg_count as f32 / (pos_count + neg_count + 1) as f32
            )
        } else if ana_count > 0 {
            format!("analytical (analytical_score: {})", ana_count)
        } else {
            format!("neutral (query_length: {})", query.len())
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnhancedReasoningResult {
    pub emotional_analysis: Option<NxrOutput>,
    pub emotional_context: Option<String>,
}

impl EnhancedReasoningResult {
    pub fn new() -> Self {
        Self {
            emotional_analysis: None,
            emotional_context: None,
        }
    }
}

impl Default for SacaAetherIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saca_aether_config_default() {
        let config = SacaAetherConfig::default();
        assert!(config.enable_emotional_reasoning);
        assert!((config.empathy_threshold - 0.7).abs() < 1e-6);
        assert!(config.cultural_adaptation);
    }

    #[test]
    fn test_enhanced_reasoning_result_new() {
        let result = EnhancedReasoningResult::new();
        assert!(result.emotional_analysis.is_none());
        assert!(result.emotional_context.is_none());
    }

    #[test]
    fn test_analyze_emotional_context_positive() {
        let integration = SacaAetherIntegration::new();
        let context = integration.analyze_emotional_context("I feel great and happy today");
        assert!(context.starts_with("positive"));
    }

    #[test]
    fn test_analyze_emotional_context_negative() {
        let integration = SacaAetherIntegration::new();
        let context = integration.analyze_emotional_context("I feel sad and angry");
        assert!(context.starts_with("negative"));
    }

    #[test]
    fn test_analyze_emotional_context_urgent() {
        let integration = SacaAetherIntegration::new();
        let context = integration.analyze_emotional_context("This is urgent and critical");
        assert!(context.starts_with("urgent"));
    }

    #[test]
    fn test_analyze_emotional_context_analytical() {
        let integration = SacaAetherIntegration::new();
        let context = integration.analyze_emotional_context("Please analyze and evaluate this");
        assert!(context.starts_with("analytical"));
    }

    #[test]
    fn test_analyze_emotional_context_neutral() {
        let integration = SacaAetherIntegration::new();
        let context = integration.analyze_emotional_context("The weather is nice");
        assert!(context.starts_with("neutral"));
    }

    #[test]
    fn test_analyze_emotional_context_empty() {
        let integration = SacaAetherIntegration::new();
        let context = integration.analyze_emotional_context("");
        assert!(context.starts_with("neutral"));
    }

    #[test]
    fn test_saca_aether_config_debug() {
        let config = SacaAetherConfig::default();
        let _debug = format!("{:?}", config);
    }

    #[test]
    fn test_enhanced_reasoning_result_debug() {
        let result = EnhancedReasoningResult::new();
        let _debug = format!("{:?}", result);
    }
}
