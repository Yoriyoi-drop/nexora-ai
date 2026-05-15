//! Harmony Weaver Configuration Module

pub mod intelligence;
pub mod cultural;
pub mod strategies;

use serde::{Deserialize, Serialize};
use crate::shared::base_agent::BaseAgentConfig;

pub use intelligence::*;
pub use cultural::*;
pub use strategies::*;

/// Harmony Weaver Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyWeaverConfig {
    /// Base agent configuration
    pub base_config: BaseAgentConfig,
    /// Emotional intelligence model
    pub emotional_intelligence_model: EmotionalIntelligenceModel,
    /// Cultural context
    pub cultural_context: CulturalContext,
    /// Social dynamics
    pub social_dynamics: SocialDynamics,
    /// Harmony strategies
    pub harmony_strategies: Vec<HarmonyStrategy>,
}

impl Default for HarmonyWeaverConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            emotional_intelligence_model: EmotionalIntelligenceModel::GolemanModel,
            cultural_context: CulturalContext::default(),
            social_dynamics: SocialDynamics::default(),
            harmony_strategies: vec![
                HarmonyStrategy {
                    strategy_id: "emotional_regulation".to_string(),
                    strategy_name: "Emotional Regulation".to_string(),
                    strategy_description: "Promote emotional self-regulation and awareness".to_string(),
                    strategy_type: HarmonyStrategyType::EmotionalRegulation,
                    effectiveness: 0.85,
                    cultural_applicability: vec!["Western".to_string(), "Eastern".to_string()],
                },
                HarmonyStrategy {
                    strategy_id: "communication_enhancement".to_string(),
                    strategy_name: "Communication Enhancement".to_string(),
                    strategy_description: "Improve communication patterns and understanding".to_string(),
                    strategy_type: HarmonyStrategyType::CommunicationEnhancement,
                    effectiveness: 0.9,
                    cultural_applicability: vec!["Universal".to_string()],
                },
            ],
        }
    }
}
