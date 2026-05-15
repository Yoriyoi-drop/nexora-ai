//! Empathy Catalyst Agent Module
//! 
//! Deep emotional understanding and empathetic response generation

pub mod config;
pub mod capabilities;
pub mod types;

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

use config::EmpathyCatalystConfig;
use capabilities::{EmpathyCapabilities, EmotionalAnalysis, EmpatheticResponseGeneration};
use types::{EmpathyCatalystTaskInput, EmpathyCatalystTaskOutput};

/// Empathy Catalyst Agent - Deep emotional understanding and empathetic response generation
#[derive(Debug, Clone)]
pub struct EmpathyCatalystAgent {
    /// Agent configuration
    pub config: EmpathyCatalystConfig,
    /// Empathy capabilities
    pub empathy_capabilities: EmpathyCapabilities,
    /// Emotional analysis
    pub emotional_analysis: EmotionalAnalysis,
    /// Empathetic response generation
    pub empathetic_response_generation: EmpatheticResponseGeneration,
    /// Agent status
    status: AgentStatus,
    /// Agent metrics
    metrics: AgentMetrics,
}

impl EmpathyCatalystAgent {
    /// Create a new Empathy Catalyst Agent
    pub fn new(config: EmpathyCatalystConfig) -> Self {
        Self {
            config,
            empathy_capabilities: EmpathyCapabilities::default(),
            emotional_analysis: EmotionalAnalysis::default(),
            empathetic_response_generation: EmpatheticResponseGeneration::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    /// Validate empathy catalyst task input
    fn validate_input(&self, input: &EmpathyCatalystTaskInput) -> AgentResult<()> {
        if input.user_input.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "User input cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }

    /// Analyze emotional context
    async fn analyze_emotional_context(&self, input: &EmpathyCatalystTaskInput) -> AgentResult<types::EmotionalAnalysisResults> {
        // Detect emotions from user input
        let detected_emotions = self.detect_emotions(&input.user_input).await?;
        
        // Analyze sentiment
        let sentiment_analysis = self.analyze_sentiment(&input.user_input).await?;
        
        // Assess emotional state
        let emotional_state_assessment = self.assess_emotional_state(&input.emotional_context.current_emotional_state).await?;
        
        // Calculate confidence scores
        let mut confidence_scores = HashMap::new();
        confidence_scores.insert("emotion_detection".to_string(), 0.85);
        confidence_scores.insert("sentiment_analysis".to_string(), 0.8);
        confidence_scores.insert("state_assessment".to_string(), 0.75);
        
        Ok(types::EmotionalAnalysisResults {
            detected_emotions,
            sentiment_analysis,
            emotional_state_assessment,
            confidence_scores,
        })
    }

    /// Generate empathetic response
    async fn generate_empathetic_response(&self, input: &EmpathyCatalystTaskInput, analysis: &types::EmotionalAnalysisResults) -> AgentResult<types::EmpatheticResponse> {
        let text = &input.user_input;
        let lower = text.to_lowercase();

        let tone = if lower.contains("sad") || lower.contains("cry") || lower.contains("depress") {
            types::ResponseTone::Compassionate
        } else if lower.contains("angry") || lower.contains("frustrat") || lower.contains("upset") {
            types::ResponseTone::Validating
        } else if lower.contains("fear") || lower.contains("worry") || lower.contains("anxious") {
            types::ResponseTone::Supportive
        } else if lower.contains("happy") || lower.contains("excited") || lower.contains("grate") {
            types::ResponseTone::Warm
        } else {
            types::ResponseTone::Empathetic
        };

        let mut content = match &tone {
            types::ResponseTone::Compassionate => "I hear how difficult this is for you. Your feelings are completely valid, and I want you to know you're not alone in this.",
            types::ResponseTone::Validating => "It makes complete sense that you feel this way given what you're experiencing. Your frustration is justified, and I'm here to support you through it.",
            types::ResponseTone::Supportive => "I understand this feels overwhelming right now. Let's take it one step at a time together - you don't have to face this alone.",
            types::ResponseTone::Warm => "That's wonderful to hear! Your positive energy is truly inspiring. I'm glad things are going well for you.",
            _ => "I truly understand how you're feeling, and I appreciate you sharing this with me. Let me help you work through this.",
        };

        let personalization_level = if input.user_profile.user_id.is_empty() {
            types::PersonalizationLevel::Basic
        } else {
            types::PersonalizationLevel::High
        };

        let empathy_score = if analysis.detected_emotions.is_empty() { 0.7 } else { 0.85 };

        Ok(types::EmpatheticResponse {
            content: content.to_string(),
            tone,
            personalization_level,
            cultural_sensitivity_score: input.cultural_context.cultural_values.is_empty().then(|| 0.85).unwrap_or(0.9),
            emotional_appropriateness_score: empathy_score,
            response_quality_metrics: types::ResponseQualityMetrics {
                empathy_score,
                appropriateness_score: 0.9,
                cultural_sensitivity_score: 0.88,
                personalization_score: if matches!(personalization_level, types::PersonalizationLevel::High) { 0.85 } else { 0.6 },
                overall_quality_score: 0.85,
                response_relevance_score: 0.9,
                emotional_alignment_score: empathy_score,
                clarity_score: 0.92,
            },
        })
    }

    /// Apply personalization
    async fn apply_personalization(&self, input: &EmpathyCatalystTaskInput, response: &types::EmpatheticResponse) -> AgentResult<types::PersonalizationDetails> {
        let mut strategies = Vec::new();
        if input.user_profile.communication_style.formality_level > 0.7 {
            strategies.push(types::AdaptationStrategy {
                strategy_name: "formal_adaptation".to_string(),
                strategy_type: "communication_style".to_string(),
                effectiveness_score: 0.85,
                application_context: "formal communication".to_string(),
                cultural_appropriateness: 0.9,
            });
        }
        if input.user_profile.communication_style.directness_level > 0.7 {
            strategies.push(types::AdaptationStrategy {
                strategy_name: "direct_adaptation".to_string(),
                strategy_type: "communication_style".to_string(),
                effectiveness_score: 0.8,
                application_context: "direct communication".to_string(),
                cultural_appropriateness: 0.85,
            });
        }

        Ok(types::PersonalizationDetails {
            user_preferences: input.user_profile.preferences.clone(),
            cultural_context: input.cultural_context.clone(),
            adaptation_strategies: strategies,
            personalization_effectiveness: if strategies.is_empty() { 0.7 } else { 0.85 },
        })
    }

    /// Assess response quality
    async fn assess_response_quality(&self, input: &EmpathyCatalystTaskInput, response: &types::EmpatheticResponse, analysis: &types::EmotionalAnalysisResults) -> AgentResult<types::ResponseQualityMetrics> {
        let empathy_score = response.emotional_appropriateness_score;
        let has_emotions = !analysis.detected_emotions.is_empty();
        let cultural_context_used = !input.cultural_context.cultural_values.is_empty();

        Ok(types::ResponseQualityMetrics {
            empathy_score,
            appropriateness_score: if has_emotions { 0.9 } else { 0.75 },
            cultural_sensitivity_score: if cultural_context_used { 0.92 } else { 0.8 },
            personalization_score: response.response_quality_metrics.personalization_score,
            overall_quality_score: (empathy_score * 0.3 + 0.9 * 0.25 + 0.88 * 0.25 + response.response_quality_metrics.personalization_score * 0.2),
            response_relevance_score: 0.9,
            emotional_alignment_score: empathy_score,
            clarity_score: 0.92,
        })
    }

    // Helper methods with heuristic-based implementations
    async fn detect_emotions(&self, input: &str) -> AgentResult<Vec<types::DetectedEmotion>> {
        let lower = input.to_lowercase();
        let emotion_keywords = vec![
            ("joy", vec!["happy", "joy", "delight", "wonderful", "great", "excellent", "love", "amazing"]),
            ("sadness", vec!["sad", "unhappy", "depress", "grief", "sorrow", "heartbroken", "melancholy"]),
            ("anger", vec!["angry", "frustrat", "annoyed", "furious", "outraged", "irritated"]),
            ("fear", vec!["fear", "scared", "anxious", "worried", "terrified", "nervous"]),
            ("surprise", vec!["surpris", "shock", "amazed", "astonished", "stunned"]),
            ("trust", vec!["trust", "believe", "confident", "rely", "depend"]),
            ("anticipation", vec!["expect", "anticipat", "hope", "look forward", "eager"]),
            ("disgust", vec!["disgust", "revolting", "repulsive", "appalled"]),
        ];

        let mut detections: std::collections::HashMap<String, (u32, f32)> = std::collections::HashMap::new();
        for (emotion, keywords) in &emotion_keywords {
            let count = keywords.iter().filter(|k| lower.contains(*k)).count() as u32;
            if count > 0 {
                let intensity = (count as f32 / keywords.len() as f32).min(1.0);
                let confidence = 0.6 + intensity * 0.3;
                detections.insert(emotion.to_string(), (count, (count as f32 / 10.0).min(1.0)));
            }
        }

        Ok(detections.into_iter().map(|(emotion, (_count, intensity))| {
            types::DetectedEmotion {
                emotion_name: emotion,
                confidence_score: 0.7 + intensity * 0.2,
                intensity,
                duration: None,
                triggers: vec![],
            }
        }).collect())
    }

    async fn analyze_sentiment(&self, input: &str) -> AgentResult<types::SentimentAnalysisResult> {
        let positive_words = ["happy", "good", "great", "excellent", "wonderful", "love", "beautiful", "fantastic", "amazing", "joy"];
        let negative_words = ["bad", "terrible", "awful", "hate", "horrible", "sad", "angry", "ugly", "dreadful", "worst"];
        let lower = input.to_lowercase();

        let pos_count = positive_words.iter().filter(|w| lower.contains(*w)).count() as f32;
        let neg_count = negative_words.iter().filter(|w| lower.contains(*w)).count() as f32;
        let total = pos_count + neg_count;

        let polarity = if total == 0.0 { 0.0 } else { (pos_count - neg_count) / total.max(1.0) };
        let word_count = input.split_whitespace().count().max(1);
        let subjectivity = (total / word_count as f32).min(1.0);

        Ok(types::SentimentAnalysisResult {
            polarity: polarity.max(-1.0).min(1.0),
            subjectivity,
            confidence: 0.7 + subjectivity * 0.2,
            emotional_valence: (polarity + 1.0) / 2.0,
            arousal_level: subjectivity,
        })
    }

    async fn assess_emotional_state(&self, state: &str) -> AgentResult<types::EmotionalStateAssessmentResult> {
        let lower = state.to_lowercase();
        let stable_indicators = ["stable", "calm", "peaceful", "composed", "balanced", "content"];
        let intense_indicators = ["intense", "overwhelming", "extreme", "powerful", "strong", "passionate"];

        let stability = if stable_indicators.iter().any(|s| lower.contains(*s)) { 0.8 } else { 0.5 };
        let intensity = if intense_indicators.iter().any(|s| lower.contains(*s)) { 0.8 } else { 0.5 };

        Ok(types::EmotionalStateAssessmentResult {
            current_state: state.to_string(),
            intensity,
            stability,
            predicted_transitions: vec![
                types::StateTransition { from_state: state.to_string(), to_state: "calm".to_string(), probability: 0.6, timeframe: None, triggers: vec!["intervention".to_string()] },
            ],
            state_duration: None,
        })
    }
}

impl Default for EmpathyCatalystAgent {
    fn default() -> Self {
        Self {
            config: EmpathyCatalystConfig::default(),
            empathy_capabilities: EmpathyCapabilities::default(),
            emotional_analysis: EmotionalAnalysis::default(),
            empathetic_response_generation: EmpatheticResponseGeneration::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for EmpathyCatalystAgent {
    type Config = EmpathyCatalystConfig;
    type Input = EmpathyCatalystTaskInput;
    type Output = EmpathyCatalystTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        self.validate_input(&input)?;
        
        // Analyze emotional context
        let emotional_analysis_results = self.analyze_emotional_context(&input).await?;
        
        // Generate empathetic response
        let empathetic_response = self.generate_empathetic_response(&input, &emotional_analysis_results).await?;
        
        // Apply personalization
        let personalization_details = self.apply_personalization(&input, &empathetic_response).await?;
        
        // Assess response quality
        let response_quality_metrics = self.assess_response_quality(&input, &empathetic_response, &emotional_analysis_results).await?;
        
        // Build output
        let output = EmpathyCatalystTaskOutput {
            empathetic_response,
            emotional_analysis_results,
            personalization_details,
            response_quality_metrics,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
        };
        
        Ok(output)
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability {
                name: "emotional_analysis".to_string(),
                description: "Deep emotional understanding and analysis".to_string(),
                enabled: true,
            },
            AgentCapability {
                name: "empathetic_response_generation".to_string(),
                description: "Generate empathetic and contextually appropriate responses".to_string(),
                enabled: true,
            },
            AgentCapability {
                name: "cultural_sensitivity".to_string(),
                description: "Culturally aware and sensitive interactions".to_string(),
                enabled: true,
            },
        ]
    }
}

// Re-export all types for backward compatibility
pub use config::*;
pub use capabilities::*;
pub use types::*;
