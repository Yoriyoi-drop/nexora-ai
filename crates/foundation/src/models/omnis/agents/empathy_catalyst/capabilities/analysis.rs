//! Emotional Analysis Capabilities

use serde::{Deserialize, Serialize};
use crate::shared::agent_types::AgentResult;

/// Emotional Analysis
#[derive(Debug, Clone)]
pub struct EmotionalAnalysis {
    /// Emotion detection capabilities
    pub emotion_detection: EmotionDetection,
    /// Sentiment analysis capabilities
    pub sentiment_analysis: SentimentAnalysis,
    /// Emotional state assessment capabilities
    pub emotional_state_assessment: EmotionalStateAssessment,
    /// Context analysis capabilities
    pub context_analysis: ContextAnalysis,
}

impl Default for EmotionalAnalysis {
    fn default() -> Self {
        Self {
            emotion_detection: EmotionDetection::default(),
            sentiment_analysis: SentimentAnalysis::default(),
            emotional_state_assessment: EmotionalStateAssessment::default(),
            context_analysis: ContextAnalysis::default(),
        }
    }
}

/// Emotion Detection
#[derive(Debug, Clone)]
pub struct EmotionDetection {
    /// Detection accuracy
    pub detection_accuracy: f32,
    /// Supported emotions
    pub supported_emotions: Vec<String>,
    /// Confidence threshold
    pub confidence_threshold: f32,
}

impl Default for EmotionDetection {
    fn default() -> Self {
        Self {
            detection_accuracy: 0.85,
            supported_emotions: vec![
                "joy".to_string(),
                "sadness".to_string(),
                "anger".to_string(),
                "fear".to_string(),
                "surprise".to_string(),
                "disgust".to_string(),
                "anticipation".to_string(),
                "trust".to_string(),
                "love".to_string(),
                "hope".to_string(),
            ],
            confidence_threshold: 0.7,
        }
    }
}

/// Sentiment Analysis
#[derive(Debug, Clone)]
pub struct SentimentAnalysis {
    /// Polarity detection accuracy
    pub polarity_detection_accuracy: f32,
    /// Subjectivity detection accuracy
    pub subjectivity_detection_accuracy: f32,
    /// Emotional intensity detection
    pub emotional_intensity_detection: f32,
}

impl Default for SentimentAnalysis {
    fn default() -> Self {
        Self {
            polarity_detection_accuracy: 0.8,
            subjectivity_detection_accuracy: 0.75,
            emotional_intensity_detection: 0.85,
        }
    }
}

/// Emotional State Assessment
#[derive(Debug, Clone)]
pub struct EmotionalStateAssessment {
    /// State recognition accuracy
    pub state_recognition_accuracy: f32,
    /// State stability assessment
    pub state_stability_assessment: f32,
    /// State transition prediction
    pub state_transition_prediction: f32,
}

impl Default for EmotionalStateAssessment {
    fn default() -> Self {
        Self {
            state_recognition_accuracy: 0.8,
            state_stability_assessment: 0.75,
            state_transition_prediction: 0.7,
        }
    }
}

/// Context Analysis
#[derive(Debug, Clone)]
pub struct ContextAnalysis {
    /// Situational awareness
    pub situational_awareness: f32,
    /// Cultural context understanding
    pub cultural_context_understanding: f32,
    /// Social context analysis
    pub social_context_analysis: f32,
}

impl Default for ContextAnalysis {
    fn default() -> Self {
        Self {
            situational_awareness: 0.85,
            cultural_context_understanding: 0.8,
            social_context_analysis: 0.75,
        }
    }
}

/// Emotional Analysis Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalAnalysisResults {
    /// Detected emotions
    pub detected_emotions: Vec<DetectedEmotion>,
    /// Sentiment analysis
    pub sentiment_analysis: SentimentAnalysisResult,
    /// Emotional state assessment
    pub emotional_state_assessment: EmotionalStateAssessmentResult,
    /// Confidence scores
    pub confidence_scores: std::collections::HashMap<String, f32>,
}

/// Detected Emotion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEmotion {
    /// Emotion name
    pub emotion_name: String,
    /// Confidence score
    pub confidence_score: f32,
    /// Intensity
    pub intensity: f32,
    /// Duration
    pub duration: Option<f32>,
    /// Triggers
    pub triggers: Vec<String>,
}

/// Sentiment Analysis Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysisResult {
    /// Polarity (-1 to 1)
    pub polarity: f32,
    /// Subjectivity (0 to 1)
    pub subjectivity: f32,
    /// Confidence
    pub confidence: f32,
    /// Emotional valence
    pub emotional_valence: f32,
    /// Arousal level
    pub arousal_level: f32,
}

/// Emotional State Assessment Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateAssessmentResult {
    /// Current state
    pub current_state: String,
    /// State intensity
    pub intensity: f32,
    /// State stability
    pub stability: f32,
    /// Predicted transitions
    pub predicted_transitions: Vec<StateTransition>,
    /// State duration
    pub state_duration: Option<f32>,
}

/// State Transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// From state
    pub from_state: String,
    /// To state
    pub to_state: String,
    /// Probability
    pub probability: f32,
    /// Timeframe
    pub timeframe: Option<f32>,
    /// Triggers
    pub triggers: Vec<String>,
}
