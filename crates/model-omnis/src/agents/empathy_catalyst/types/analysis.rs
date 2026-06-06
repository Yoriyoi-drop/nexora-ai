//! Analysis Types for Empathy Catalyst

use serde::{Deserialize, Serialize};

/// Sentiment Analysis (simplified version for types module)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    /// Polarity score (-1 to 1)
    pub polarity: f32,
    /// Subjectivity score (0 to 1)
    pub subjectivity: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Emotional State Assessment (simplified version for types module)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateAssessment {
    /// Current state
    pub current_state: String,
    /// Intensity level
    pub intensity: f32,
    /// Stability level
    pub stability: f32,
}

/// Analysis Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetrics {
    /// Accuracy score
    pub accuracy_score: f32,
    /// Processing time
    pub processing_time_ms: u64,
    /// Confidence score
    pub confidence_score: f32,
    /// Data quality score
    pub data_quality_score: f32,
}

/// Analysis Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfiguration {
    /// Analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Sensitivity level
    pub sensitivity_level: f32,
    /// Cultural awareness enabled
    pub cultural_awareness_enabled: bool,
    /// Context analysis enabled
    pub context_analysis_enabled: bool,
    /// Real-time analysis enabled
    pub real_time_analysis_enabled: bool,
}

/// Analysis Depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Surface level analysis
    Surface,
    /// Moderate depth analysis
    Moderate,
    /// Deep analysis
    Deep,
    /// Comprehensive analysis
    Comprehensive,
}

impl Default for AnalysisConfiguration {
    fn default() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Moderate,
            sensitivity_level: 0.7,
            cultural_awareness_enabled: true,
            context_analysis_enabled: true,
            real_time_analysis_enabled: false,
        }
    }
}

/// Analysis Result Summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResultSummary {
    /// Overall sentiment
    pub overall_sentiment: String,
    /// Primary emotions
    pub primary_emotions: Vec<String>,
    /// Emotional intensity
    pub emotional_intensity: f32,
    /// Cultural context detected
    pub cultural_context_detected: Vec<String>,
    /// Personalization recommendations
    pub personalization_recommendations: Vec<String>,
    /// Confidence scores
    pub confidence_scores: std::collections::HashMap<String, f32>,
}
