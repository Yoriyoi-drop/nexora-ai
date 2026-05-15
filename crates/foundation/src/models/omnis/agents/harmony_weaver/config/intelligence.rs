//! Emotional Intelligence Models

use serde::{Deserialize, Serialize};

/// Emotional Intelligence Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionalIntelligenceModel {
    /// Goleman model
    GolemanModel,
    /// Mayer-Salovey model
    MayerSaloveyModel,
    /// Bar-On model
    BarOnModel,
    /// Trait model
    TraitModel,
    /// Ability model
    AbilityModel,
    /// Hybrid model
    HybridModel { models: Vec<EmotionalIntelligenceModel> },
}

/// Emotional Intelligence Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalIntelligenceConfig {
    /// Model type
    pub model_type: EmotionalIntelligenceModel,
    /// Assessment accuracy
    pub assessment_accuracy: f32,
    /// Cultural adaptation level
    pub cultural_adaptation_level: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Sensitivity threshold
    pub sensitivity_threshold: f32,
}

impl Default for EmotionalIntelligenceConfig {
    fn default() -> Self {
        Self {
            model_type: EmotionalIntelligenceModel::GolemanModel,
            assessment_accuracy: 0.85,
            cultural_adaptation_level: 0.8,
            learning_rate: 0.1,
            sensitivity_threshold: 0.6,
        }
    }
}

/// Emotional Intelligence Dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalIntelligenceDimensions {
    /// Self-awareness dimension
    pub self_awareness: SelfAwarenessDimension,
    /// Self-regulation dimension
    pub self_regulation: SelfRegulationDimension,
    /// Social awareness dimension
    pub social_awareness: SocialAwarenessDimension,
    /// Relationship management dimension
    pub relationship_management: RelationshipManagementDimension,
}

impl Default for EmotionalIntelligenceDimensions {
    fn default() -> Self {
        Self {
            self_awareness: SelfAwarenessDimension::default(),
            self_regulation: SelfRegulationDimension::default(),
            social_awareness: SocialAwarenessDimension::default(),
            relationship_management: RelationshipManagementDimension::default(),
        }
    }
}

/// Self-Awareness Dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAwarenessDimension {
    /// Emotional recognition accuracy
    pub emotional_recognition_accuracy: f32,
    /// Self-reflection depth
    pub self_reflection_depth: f32,
    /// Emotional vocabulary
    pub emotional_vocabulary: Vec<String>,
    /// Pattern recognition
    pub pattern_recognition: f32,
}

impl Default for SelfAwarenessDimension {
    fn default() -> Self {
        Self {
            emotional_recognition_accuracy: 0.85,
            self_reflection_depth: 0.8,
            emotional_vocabulary: vec![
                "joy".to_string(),
                "sadness".to_string(),
                "anger".to_string(),
                "fear".to_string(),
                "surprise".to_string(),
                "disgust".to_string(),
                "anticipation".to_string(),
                "trust".to_string(),
            ],
            pattern_recognition: 0.75,
        }
    }
}

/// Self-Regulation Dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfRegulationDimension {
    /// Impulse control
    pub impulse_control: f32,
    /// Stress management
    pub stress_management: f32,
    /// Adaptability
    pub adaptability: f32,
    /// Achievement orientation
    pub achievement_orientation: f32,
    /// Positive outlook
    pub positive_outlook: f32,
}

impl Default for SelfRegulationDimension {
    fn default() -> Self {
        Self {
            impulse_control: 0.8,
            stress_management: 0.75,
            adaptability: 0.85,
            achievement_orientation: 0.7,
            positive_outlook: 0.8,
        }
    }
}

/// Social Awareness Dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialAwarenessDimension {
    /// Empathy accuracy
    pub empathy_accuracy: f32,
    /// Organizational awareness
    pub organizational_awareness: f32,
    /// Service orientation
    pub service_orientation: f32,
    /// Cultural awareness
    pub cultural_awareness: f32,
}

impl Default for SocialAwarenessDimension {
    fn default() -> Self {
        Self {
            empathy_accuracy: 0.85,
            organizational_awareness: 0.75,
            service_orientation: 0.8,
            cultural_awareness: 0.9,
        }
    }
}

/// Relationship Management Dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipManagementDimension {
    /// Influence effectiveness
    pub influence_effectiveness: f32,
    /// Communication skills
    pub communication_skills: f32,
    /// Leadership capabilities
    pub leadership_capabilities: f32,
    /// Change catalyst
    pub change_catalyst: f32,
    /// Conflict management
    pub conflict_management: f32,
    /// Teamwork capabilities
    pub teamwork_capabilities: f32,
    /// Building bonds
    pub building_bonds: f32,
}

impl Default for RelationshipManagementDimension {
    fn default() -> Self {
        Self {
            influence_effectiveness: 0.75,
            communication_skills: 0.85,
            leadership_capabilities: 0.7,
            change_catalyst: 0.65,
            conflict_management: 0.8,
            teamwork_capabilities: 0.9,
            building_bonds: 0.85,
        }
    }
}

/// Emotional Intelligence Assessment Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalIntelligenceAssessmentParameters {
    /// Assessment frequency
    pub assessment_frequency: AssessmentFrequency,
    /// Assessment depth
    pub assessment_depth: AssessmentDepth,
    /// Cultural considerations
    pub cultural_considerations: Vec<CulturalConsideration>,
    /// Context factors
    pub context_factors: Vec<ContextFactor>,
}

impl Default for EmotionalIntelligenceAssessmentParameters {
    fn default() -> Self {
        Self {
            assessment_frequency: AssessmentFrequency::Weekly,
            assessment_depth: AssessmentDepth::Comprehensive,
            cultural_considerations: vec![
                CulturalConsideration {
                    consideration_type: "communication_style".to_string(),
                    weight: 0.3,
                    description: "Consider cultural communication differences".to_string(),
                },
                CulturalConsideration {
                    consideration_type: "emotional_expression".to_string(),
                    weight: 0.25,
                    description: "Account for cultural emotional expression norms".to_string(),
                },
            ],
            context_factors: vec![
                ContextFactor {
                    factor_type: "social_environment".to_string(),
                    weight: 0.2,
                    description: "Consider social context impact".to_string(),
                },
                ContextFactor {
                    factor_type: "professional_setting".to_string(),
                    weight: 0.15,
                    description: "Account for professional environment".to_string(),
                },
            ],
        }
    }
}

/// Assessment Frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentFrequency {
    /// Daily assessment
    Daily,
    /// Weekly assessment
    Weekly,
    /// Monthly assessment
    Monthly,
    /// Quarterly assessment
    Quarterly,
    /// Annual assessment
    Annual,
    /// On-demand assessment
    OnDemand,
}

/// Assessment Depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentDepth {
    /// Surface level
    Surface,
    /// Moderate depth
    Moderate,
    /// Deep analysis
    Deep,
    /// Comprehensive analysis
    Comprehensive,
}

/// Cultural Consideration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalConsideration {
    /// Consideration type
    pub consideration_type: String,
    /// Weight
    pub weight: f32,
    /// Description
    pub description: String,
}

/// Context Factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFactor {
    /// Factor type
    pub factor_type: String,
    /// Weight
    pub weight: f32,
    /// Description
    pub description: String,
}
