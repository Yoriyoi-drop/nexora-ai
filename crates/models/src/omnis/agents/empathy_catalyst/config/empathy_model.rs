//! Empathy Model Types

use serde::{Deserialize, Serialize};

/// Empathy Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmpathyModel {
    /// Cognitive empathy model
    CognitiveEmpathyModel,
    /// Affective empathy model
    AffectiveEmpathyModel,
    /// Compassionate empathy model
    CompassionateEmpathyModel,
    /// Social empathy model
    SocialEmpathyModel,
    /// Cultural empathy model
    CulturalEmpathyModel,
    /// Integrated empathy model
    IntegratedEmpathyModel { models: Vec<EmpathyModel> },
}

/// Empathy Model Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathyModelConfig {
    /// Model type
    pub model_type: EmpathyModel,
    /// Sensitivity level
    pub sensitivity_level: f32,
    /// Adaptation speed
    pub adaptation_speed: f32,
    /// Cultural awareness factor
    pub cultural_awareness_factor: f32,
    /// Emotional depth
    pub emotional_depth: f32,
}

impl Default for EmpathyModelConfig {
    fn default() -> Self {
        Self {
            model_type: EmpathyModel::CompassionateEmpathyModel,
            sensitivity_level: 0.8,
            adaptation_speed: 0.7,
            cultural_awareness_factor: 0.9,
            emotional_depth: 0.85,
        }
    }
}

/// Empathy Model Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathyModelParameters {
    /// Emotional recognition threshold
    pub emotional_recognition_threshold: f32,
    /// Context sensitivity
    pub context_sensitivity: f32,
    /// Response empathy factor
    pub response_empathy_factor: f32,
    /// Personalization weight
    pub personalization_weight: f32,
}

impl Default for EmpathyModelParameters {
    fn default() -> Self {
        Self {
            emotional_recognition_threshold: 0.6,
            context_sensitivity: 0.8,
            response_empathy_factor: 0.9,
            personalization_weight: 0.7,
        }
    }
}
