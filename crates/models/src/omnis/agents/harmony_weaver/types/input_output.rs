use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyWeaverTaskInput {
    pub social_context: String,
    pub participants: Vec<ParticipantProfile>,
    pub interaction_history: Vec<InteractionEvent>,
    pub current_dynamics: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantProfile {
    pub id: String,
    pub name: String,
    pub role: String,
    pub communication_style: CommunicationStyle,
    pub emotional_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStyle {
    Direct,
    Indirect,
    Analytical,
    Expressive,
    Reserved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub participants: Vec<String>,
    pub content: String,
    pub sentiment: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyWeaverTaskOutput {
    pub harmony_assessment: super::analysis::HarmonyAssessment,
    pub emotional_intelligence_analysis: super::analysis::EmotionalIntelligenceAnalysis,
    pub intervention_recommendations: super::results::InterventionRecommendations,
    pub conflict_resolution_strategies: super::results::ConflictResolutionStrategies,
    pub harmony_optimization_plan: super::results::HarmonyOptimizationPlan,
    pub processing_time_ms: u64,
}
