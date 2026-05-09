//! Empathy Catalyst Types Module

pub mod input_output;
pub mod results;
pub mod analysis;

pub use input_output::*;
pub use results::*;
pub use analysis::*;

// Re-export commonly used types from capabilities
pub use super::capabilities::analysis::{
    EmotionalAnalysisResults, DetectedEmotion, SentimentAnalysisResult, 
    EmotionalStateAssessmentResult, StateTransition
};

pub use super::capabilities::response::{
    EmpatheticResponse, ResponseTone, PersonalizationLevel, ResponseQualityMetrics,
    PersonalizationDetails, UserPreferences, CulturalContext, AdaptationStrategy
};
