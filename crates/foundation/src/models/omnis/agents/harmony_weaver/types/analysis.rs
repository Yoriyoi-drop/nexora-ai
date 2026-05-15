use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyAssessment {
    pub emotional_landscape: EmotionalLandscape,
    pub social_dynamics: SocialDynamics,
    pub harmony_indicators: Vec<HarmonyIndicator>,
    pub harmony_score: f32,
    pub assessment_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalLandscape {
    pub dominant_emotions: Vec<EmotionEntry>,
    pub emotional_intensity: f32,
    pub emotional_diversity: f32,
    pub emotional_stability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionEntry {
    pub emotion: String,
    pub intensity: f32,
    pub source_participant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialDynamics {
    pub group_cohesion: f32,
    pub communication_patterns: Vec<CommunicationPattern>,
    pub power_dynamics: Vec<PowerDynamic>,
    pub social_network_structure: Vec<SocialConnection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPattern {
    pub from: String,
    pub to: String,
    pub frequency: u32,
    pub average_sentiment: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDynamic {
    pub participant_id: String,
    pub influence_score: f32,
    pub dominance_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialConnection {
    pub participant_a: String,
    pub participant_b: String,
    pub connection_strength: f32,
    pub interaction_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyIndicator {
    pub name: String,
    pub score: f32,
    pub trend: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalIntelligenceAnalysis {
    pub emotional_awareness: f32,
    pub emotional_regulation: f32,
    pub social_awareness: f32,
    pub relationship_management: f32,
    pub overall_ei_score: f32,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}
