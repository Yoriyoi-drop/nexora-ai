//! Empathy Catalyst Input and Output Types

use serde::{Deserialize, Serialize};

/// Empathy Catalyst Task Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathyCatalystTaskInput {
    /// User input text
    pub user_input: String,
    /// Emotional context
    pub emotional_context: EmotionalContext,
    /// User profile
    pub user_profile: UserProfile,
    /// Cultural context
    pub cultural_context: CulturalContext,
    /// Conversation history
    pub conversation_history: Vec<ConversationEntry>,
    /// Context information
    pub context_information: ContextInformation,
    /// Task requirements
    pub task_requirements: TaskRequirements,
}

/// Emotional Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalContext {
    /// Current emotional state
    pub current_emotional_state: String,
    /// Emotional intensity
    pub emotional_intensity: f32,
    /// Emotional triggers
    pub emotional_triggers: Vec<String>,
    /// Emotional history
    pub emotional_history: Vec<EmotionalStateEntry>,
    /// Emotional needs
    pub emotional_needs: Vec<String>,
    /// Emotional goals
    pub emotional_goals: Vec<String>,
}

/// Emotional State Entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateEntry {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Emotional state
    pub emotional_state: String,
    /// Intensity
    pub intensity: f32,
    /// Duration
    pub duration: f32,
    /// Context
    pub context: String,
}

/// User Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User ID
    pub user_id: String,
    /// Demographics
    pub demographics: Demographics,
    /// Preferences
    pub preferences: UserPreferences,
    /// Communication style
    pub communication_style: CommunicationStyle,
    /// Cultural background
    pub cultural_background: CulturalBackground,
    /// Personality traits
    pub personality_traits: Vec<PersonalityTrait>,
    /// Emotional intelligence level
    pub emotional_intelligence_level: f32,
    /// Previous interactions
    pub previous_interactions: Vec<PreviousInteraction>,
}

/// Demographics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demographics {
    /// Age
    pub age: Option<u32>,
    /// Gender
    pub gender: Option<String>,
    /// Location
    pub location: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Education level
    pub education_level: Option<String>,
    /// Occupation
    pub occupation: Option<String>,
}

/// Communication Style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    /// Preferred style
    pub preferred_style: String,
    /// Formality level
    pub formality_level: f32,
    /// Directness level
    pub directness_level: f32,
    /// Expressiveness level
    pub expressiveness_level: f32,
    /// Response length preference
    pub response_length_preference: String,
}

/// Cultural Background
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalBackground {
    /// Primary culture
    pub primary_culture: String,
    /// Secondary cultures
    pub secondary_cultures: Vec<String>,
    /// Cultural values
    pub cultural_values: Vec<String>,
    /// Cultural norms
    pub cultural_norms: Vec<String>,
    /// Cultural experiences
    pub cultural_experiences: Vec<String>,
}

/// Personality Trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTrait {
    /// Trait name
    pub trait_name: String,
    /// Trait value
    pub trait_value: f32,
    /// Confidence
    pub confidence: f32,
}

/// Previous Interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviousInteraction {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Interaction type
    pub interaction_type: String,
    /// Outcome
    pub outcome: String,
    /// Satisfaction score
    pub satisfaction_score: Option<f32>,
    /// Notes
    pub notes: Option<String>,
}

/// Cultural Context (re-export from response module)
pub use super::super::capabilities::response::CulturalContext;

/// Conversation Entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEntry {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Speaker
    pub speaker: Speaker,
    /// Message
    pub message: String,
    /// Emotional tone
    pub emotional_tone: Option<String>,
    /// Context
    pub context: Option<String>,
}

/// Speaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Speaker {
    /// User
    User,
    /// Agent
    Agent,
    /// System
    System,
}

/// Context Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInformation {
    /// Situation type
    pub situation_type: String,
    /// Environment
    pub environment: String,
    /// Time of day
    pub time_of_day: Option<String>,
    /// Location
    pub location: Option<String>,
    /// Social context
    pub social_context: Option<String>,
    /// Professional context
    pub professional_context: Option<String>,
    /// Personal context
    pub personal_context: Option<String>,
}

/// Task Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    /// Response type
    pub response_type: String,
    /// Empathy level required
    pub empathy_level_required: f32,
    /// Cultural sensitivity required
    pub cultural_sensitivity_required: f32,
    /// Personalization level
    pub personalization_level: String,
    /// Response length
    pub response_length: Option<String>,
    /// Tone requirements
    pub tone_requirements: Vec<String>,
    /// Constraints
    pub constraints: Vec<String>,
}

/// Empathy Catalyst Task Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathyCatalystTaskOutput {
    /// Empathetic response
    pub empathetic_response: EmpatheticResponse,
    /// Emotional analysis results
    pub emotional_analysis_results: EmotionalAnalysisResults,
    /// Personalization details
    pub personalization_details: PersonalizationDetails,
    /// Response quality metrics
    pub response_quality_metrics: ResponseQualityMetrics,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Additional insights
    pub additional_insights: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

// Re-export types from capabilities
pub use super::super::capabilities::response::{
    EmpatheticResponse, ResponseTone, PersonalizationLevel, ResponseQualityMetrics,
    PersonalizationDetails, UserPreferences
};

pub use super::super::capabilities::analysis::{
    EmotionalAnalysisResults, DetectedEmotion, SentimentAnalysisResult,
    EmotionalStateAssessmentResult, StateTransition
};
