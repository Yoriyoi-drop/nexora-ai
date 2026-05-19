//! Empathy Catalyst Results Types

use serde::{Deserialize, Serialize};

/// Analysis Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResults {
    /// Emotional analysis results
    pub emotional_analysis_results: EmotionalAnalysisResults,
    /// Cultural analysis results
    pub cultural_analysis_results: CulturalAnalysisResults,
    /// Context analysis results
    pub context_analysis_results: ContextAnalysisResults,
    /// Personalization analysis results
    pub personalization_analysis_results: PersonalizationAnalysisResults,
}

/// Cultural Analysis Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalAnalysisResults {
    /// Cultural context detected
    pub cultural_context_detected: Vec<String>,
    /// Cultural appropriateness score
    pub cultural_appropriateness_score: f32,
    /// Cultural adaptation recommendations
    pub cultural_adaptation_recommendations: Vec<CulturalAdaptationRecommendation>,
    /// Cross-cultural considerations
    pub cross_cultural_considerations: Vec<CrossCulturalConsideration>,
}

/// Cultural Adaptation Recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalAdaptationRecommendation {
    /// Recommendation type
    pub recommendation_type: String,
    /// Description
    pub description: String,
    /// Priority
    pub priority: f32,
    /// Applicability
    pub applicability: Vec<String>,
    /// Expected impact
    pub expected_impact: f32,
}

/// Cross Cultural Consideration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCulturalConsideration {
    /// Consideration type
    pub consideration_type: String,
    /// Description
    pub description: String,
    /// Relevance score
    pub relevance_score: f32,
    /// Action required
    pub action_required: Option<String>,
}

/// Context Analysis Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnalysisResults {
    /// Situational context
    pub situational_context: SituationalContext,
    /// Social context
    pub social_context: SocialContext,
    /// Environmental context
    pub environmental_context: EnvironmentalContext,
    /// Temporal context
    pub temporal_context: TemporalContext,
}

/// Situational Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationalContext {
    /// Situation type
    pub situation_type: String,
    /// Formality level
    pub formality_level: f32,
    /// Urgency level
    pub urgency_level: f32,
    /// Complexity level
    pub complexity_level: f32,
    /// Emotional intensity
    pub emotional_intensity: f32,
}

/// Social Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialContext {
    /// Relationship type
    pub relationship_type: String,
    /// Social distance
    pub social_distance: f32,
    /// Power dynamics
    pub power_dynamics: PowerDynamics,
    /// Group dynamics
    pub group_dynamics: GroupDynamics,
}

/// Power Dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDynamics {
    /// Power imbalance
    pub power_imbalance: f32,
    /// Authority level
    pub authority_level: f32,
    /// Influence level
    pub influence_level: f32,
    /// Hierarchy position
    pub hierarchy_position: String,
}

/// Group Dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDynamics {
    /// Group size
    pub group_size: u32,
    /// Group cohesion
    pub group_cohesion: f32,
    /// Social norms
    pub social_norms: Vec<String>,
    /// Group culture
    pub group_culture: String,
}

/// Environmental Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalContext {
    /// Physical environment
    pub physical_environment: String,
    /// Digital environment
    pub digital_environment: String,
    /// Noise level
    pub noise_level: f32,
    /// Privacy level
    pub privacy_level: f32,
}

/// Temporal Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Time of day
    pub time_of_day: String,
    /// Day of week
    pub day_of_week: String,
    /// Season
    pub season: Option<String>,
    /// Time pressure
    pub time_pressure: f32,
    /// Available time
    pub available_time: Option<f32>,
}

/// Personalization Analysis Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationAnalysisResults {
    /// User preferences analysis
    pub user_preferences_analysis: UserPreferencesAnalysis,
    /// Communication style analysis
    pub communication_style_analysis: CommunicationStyleAnalysis,
    /// Learning pattern analysis
    pub learning_pattern_analysis: LearningPatternAnalysis,
    /// Adaptation effectiveness
    pub adaptation_effectiveness: AdaptationEffectiveness,
}

/// User Preferences Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferencesAnalysis {
    /// Detected preferences
    pub detected_preferences: Vec<DetectedPreference>,
    /// Preference confidence scores
    pub preference_confidence_scores: std::collections::HashMap<String, f32>,
    /// Preference stability
    pub preference_stability: f32,
    /// Preference evolution
    pub preference_evolution: PreferenceEvolution,
}

/// Detected Preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPreference {
    /// Preference name
    pub preference_name: String,
    /// Preference value
    pub preference_value: String,
    /// Confidence score
    pub confidence_score: f32,
    /// Context
    pub context: String,
    /// Frequency
    pub frequency: f32,
}

/// Preference Evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceEvolution {
    /// Change direction
    pub change_direction: String,
    /// Change magnitude
    pub change_magnitude: f32,
    /// Time period
    pub time_period: f32,
    /// Stability trend
    pub stability_trend: f32,
}

/// Communication Style Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyleAnalysis {
    /// Preferred style
    pub preferred_style: String,
    /// Style consistency
    pub style_consistency: f32,
    /// Adaptability score
    pub adaptability_score: f32,
    /// Context variations
    pub context_variations: Vec<ContextVariation>,
}

/// Context Variation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextVariation {
    /// Context type
    pub context_type: String,
    /// Style variation
    pub style_variation: String,
    /// Frequency
    pub frequency: f32,
    /// Effectiveness
    pub effectiveness: f32,
}

/// Learning Pattern Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPatternAnalysis {
    /// Learning speed
    pub learning_speed: f32,
    /// Retention rate
    pub retention_rate: f32,
    /// Adaptation strategies
    pub adaptation_strategies: Vec<String>,
    /// Feedback responsiveness
    pub feedback_responsiveness: f32,
}

/// Adaptation Effectiveness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEffectiveness {
    /// Overall effectiveness
    pub overall_effectiveness: f32,
    /// Specific adaptations
    pub specific_adaptations: Vec<SpecificAdaptation>,
    /// Improvement trend
    pub improvement_trend: f32,
    /// User satisfaction
    pub user_satisfaction: f32,
}

/// Specific Adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificAdaptation {
    /// Adaptation type
    pub adaptation_type: String,
    /// Effectiveness score
    pub effectiveness_score: f32,
    /// Usage frequency
    pub usage_frequency: f32,
    /// User feedback
    pub user_feedback: Option<String>,
}

// Re-export types from other modules
pub use super::super::capabilities::analysis::{
    EmotionalAnalysisResults, DetectedEmotion, SentimentAnalysisResult,
    EmotionalStateAssessmentResult, StateTransition
};

pub use super::super::capabilities::response::{
    ResponseQualityMetrics, PersonalizationDetails, UserPreferences,
    CulturalContext, AdaptationStrategy
};
