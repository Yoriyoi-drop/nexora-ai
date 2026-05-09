//! Empathy Catalyst Configuration Module

pub mod empathy_model;
pub mod cultural;
pub mod intelligence;

use serde::{Deserialize, Serialize};
use crate::shared::base_agent::BaseAgentConfig;

pub use empathy_model::*;
pub use cultural::*;
pub use intelligence::*;

/// Empathy Catalyst Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathyCatalystConfig {
    /// Base agent configuration
    pub base_config: BaseAgentConfig,
    /// Empathy model
    pub empathy_model: EmpathyModel,
    /// Cultural empathy settings
    pub cultural_empathy_settings: CulturalEmpathySettings,
    /// Emotional intelligence framework
    pub emotional_intelligence_framework: EmotionalIntelligenceFramework,
    /// Response personalization
    pub response_personalization: ResponsePersonalization,
}

impl Default for EmpathyCatalystConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            empathy_model: EmpathyModel::IntegratedEmpathyModel { 
                models: vec![
                    EmpathyModel::CognitiveEmpathyModel,
                    EmpathyModel::AffectiveEmpathyModel,
                    EmpathyModel::CompassionateEmpathyModel,
                ]
            },
            cultural_empathy_settings: CulturalEmpathySettings::default(),
            emotional_intelligence_framework: EmotionalIntelligenceFramework::default(),
            response_personalization: ResponsePersonalization::default(),
        }
    }
}
