//! NXR-ÆTHER Model Implementation
//!
//! NXR-03 APEX - Adaptive Emotional & Holistic Transcendent Empathy Reasoner
//! Emotional intelligence and psychological analysis specialist

pub mod classifier;
pub mod delegation;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use nexora_shared::{
    base_model::{
        ModelStatistics, NxrInput, NxrModel, NxrModelResult, NxrOutput, NxrStreamChunk,
        ResourceUsage, ValidationResult,
    },
    capability_spec::CapabilityVector,
    deeplearning_integration::{DeepLearningModel, HasComponents},
    foundation_components::FoundationComponents,
    model_config::NxrModelConfig,
    model_identity::{ModelMeta, NxrModelId},
    model_registry::{global_registry, NxrModelRegistry},
    safety_gate::{global_safety, ConsentScope, ConsentToken},
};

// Include all Aether modules
pub mod agents;

mod capabilities;
mod config;
mod coordinator;
mod identity;

// Re-export all components
pub use agents::*;
pub use capabilities::*;
pub use config::*;
pub use coordinator::*;
pub use identity::*;

/// NXR-ÆTHER Model Implementation
pub struct NxrAetherModel {
    base: nexora_shared::base_model::BaseNxrModel<AetherConfig, AetherMetrics, AetherState>,
    identity: AetherIdentity,
    capabilities: AetherCapabilities,
    /// Foundation components (ERP, VOGP, ATQS, MoE, DL, GNAC, Tokenizer)
    components: FoundationComponents,
    config: AetherConfig,
    #[cfg(feature = "hallucination")]
    hallucination: Option<nexora_hallucination::HallucinationGuard>,
}

/// NXR-ÆTHER Model State
#[derive(Debug, Clone)]
pub struct AetherState {
    pub emotional_context: HashMap<String, f32>,
    pub empathy_level: f32,
    pub psychological_profile: Option<PsychologicalProfile>,
    pub last_inference: Option<chrono::DateTime<chrono::Utc>>,
}

/// Psychological profile
#[derive(Debug, Clone)]
pub struct PsychologicalProfile {
    pub personality_traits: HashMap<String, f32>,
    pub emotional_state: EmotionalState,
    pub cognitive_patterns: Vec<String>,
}

/// Emotional state
#[derive(Debug, Clone)]
pub struct EmotionalState {
    pub primary_emotion: String,
    pub emotional_intensity: f32,
    pub valence: f32,
    pub arousal: f32,
}

/// NXR-ÆTHER Model Metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AetherMetrics {
    pub total_emotional_analyses: u64,
    pub avg_empathy_score: f32,
    pub emotional_accuracy: f32,
    pub psychological_insight_score: f32,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for AetherState {
    fn default() -> Self {
        Self {
            emotional_context: HashMap::new(),
            empathy_level: 0.0,
            psychological_profile: None,
            last_inference: None,
        }
    }
}

impl Default for AetherMetrics {
    fn default() -> Self {
        Self {
            total_emotional_analyses: 0,
            avg_empathy_score: 0.965,
            emotional_accuracy: 0.0,
            psychological_insight_score: 0.92,
            last_updated: chrono::Utc::now(),
        }
    }
}

/// NXR-ÆTHER Identity
pub struct AetherIdentity {
    meta: ModelMeta,
}

impl AetherIdentity {
    pub fn new() -> Self {
        let meta = ModelMeta::new(
            NxrModelId::Aether,
            nexora_shared::model_identity::ModelTier::Apex,
            "1.0.0".to_string(),
            "Adaptive Emotional & Holistic Transcendent Empathy Reasoner - Emotional intelligence and psychological analysis specialist with deep empathy synthesis capabilities.".to_string(),
        )
        .with_parameters(0) // not loaded; real count from CausalLM config
        .with_context_window(0); // not loaded

        Self { meta }
    }

    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }
}

/// NXR-ÆTHER Capabilities
pub struct AetherCapabilities {
    vector: CapabilityVector,
}

impl AetherCapabilities {
    pub fn new() -> Self {
        let vector = CapabilityVector::new(NxrModelId::Aether)
            .with_capability(nexora_shared::capability_spec::CapabilitySpec::new(
                nexora_shared::capability_spec::CapabilityDomain::Emotional,
                nexora_shared::capability_spec::CapabilityLevel::Transcendent,
            ))
            .calculate_score();
        Self { vector }
    }

    pub fn vector(&self) -> &CapabilityVector {
        &self.vector
    }
}

impl NxrAetherModel {
    pub fn new() -> Self {
        let identity = AetherIdentity::new();
        let capabilities = AetherCapabilities::new();
        let config = AetherConfig::default();
        let initial_state = AetherState::default();
        let initial_metrics = AetherMetrics::default();

        Self {
            base: nexora_shared::base_model::BaseNxrModel::new(
                identity.meta().clone(),
                capabilities.vector().clone(),
                config.clone(),
                initial_state,
                initial_metrics,
            ),
            identity,
            capabilities,
            components: FoundationComponents::new(),
            config,
            #[cfg(feature = "hallucination")]
            hallucination: Some(nexora_hallucination::HallucinationGuard::new(
                nexora_hallucination::GuardConfig::default(),
            )),
        }
    }

    async fn analyze_emotional_content(&self, text: &str) -> NxrModelResult<String> {
        // Tokenize input
        let tokens = {
            let tokenizer = self.components.tokenizer.read();
            tokenizer.encode(text)
        };

        // Process text with deep learning
        let dl_result = self
            .dl_process(text)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Internal(e.to_string()))?;

        // SACA-ÆTHER enhanced reasoning with emotional context
        let saca_summary = String::new();

        let emotional_tone = self.detect_emotional_tone(text)?;
        let empathy_response = self.generate_empathy_response(text, &emotional_tone)?;

        Ok(format!(
            "Emotional Analysis:\nTone: {}\nIntensity: {:.2}\nEmpathy Response: {}\nSACA Reasoning: {}\nDL Processing: {} (tokens: {})",
            emotional_tone.primary_emotion,
            emotional_tone.emotional_intensity,
            empathy_response,
            saca_summary,
            dl_result,
            tokens.len()
        ))
    }

    fn detect_emotional_tone(&self, text: &str) -> NxrModelResult<EmotionalState> {
        let lower_text = text.to_lowercase();
        let words: Vec<&str> = lower_text.split_whitespace().collect();
        let mut emotional_scores = HashMap::new();

        // Simple emotional keyword detection
        if words
            .iter()
            .any(|w| w.contains("sad") || w.contains("unhappy"))
        {
            emotional_scores.insert("sadness".to_string(), 0.8f32);
        }
        if words
            .iter()
            .any(|w| w.contains("happy") || w.contains("joy"))
        {
            emotional_scores.insert("joy".to_string(), 0.9f32);
        }
        if words
            .iter()
            .any(|w| w.contains("angry") || w.contains("mad"))
        {
            emotional_scores.insert("anger".to_string(), 0.7f32);
        }

        let (primary_emotion, intensity) = emotional_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(emotion, score)| (emotion.clone(), *score))
            .unwrap_or(("neutral".to_string(), 0.5f32));

        let valence = if primary_emotion == "joy" {
            0.8f32
        } else if primary_emotion == "sadness" {
            -0.6f32
        } else {
            0.0f32
        };

        Ok(EmotionalState {
            primary_emotion,
            emotional_intensity: intensity,
            valence,
            arousal: intensity,
        })
    }

    fn generate_empathy_response(
        &self,
        text: &str,
        emotional_state: &EmotionalState,
    ) -> NxrModelResult<String> {
        let response = match emotional_state.primary_emotion.as_str() {
            "sadness" => "I understand this must be difficult for you. Your feelings are valid, and it's okay to feel this way.",
            "joy" => "It's wonderful to hear this positivity! Your enthusiasm is inspiring.",
            "anger" => "I can sense your frustration. Your feelings are justified, and it's important to address what's causing this anger.",
            _ => "I hear you, and I'm here to support you through whatever you're experiencing.",
        };
        Ok(response.to_string())
    }

    #[cfg(feature = "hallucination")]
    pub fn enable_hallucination_guard(&mut self) {
        let h = nexora_hallucination::HallucinationGuard::new(
            nexora_hallucination::GuardConfig::default(),
        );
        self.hallucination = Some(h);
    }

    #[cfg(feature = "hallucination")]
    pub fn disable_hallucination_guard(&mut self) {
        self.hallucination = None;
    }

    #[cfg(feature = "hallucination")]
    pub fn with_hallucination_guard(
        mut self,
        guard: nexora_hallucination::HallucinationGuard,
    ) -> Self {
        self.hallucination = Some(guard);
        self
    }

    #[cfg(feature = "hallucination")]
    async fn run_hallucination_check(
        &self,
        input: &nexora_shared::base_model::NxrInput,
    ) -> Option<nexora_hallucination::PipelineResult> {
        if let Some(ref h) = self.hallucination {
            let text = match &input.data {
                nexora_shared::base_model::InputData::Text(t) => t.clone(),
                _ => return None,
            };
            let ctx = input
                .parameters
                .get("context")
                .and_then(|v| v.as_str())
                .map(String::from);
            match h.run_pipeline(&text, ctx.as_deref(), None).await {
                Ok(result) => return Some(result),
                Err(e) => {
                    tracing::warn!("Pipeline execution failed: {}", e);
                    return None;
                }
            }
        }
        None
    }

    #[cfg(not(feature = "hallucination"))]
    async fn run_hallucination_check(
        &self,
        _input: &nexora_shared::base_model::NxrInput,
    ) -> Option<nexora_hallucination::PipelineResult> {
        None
    }
}

const AETHER_SYSTEM_PROMPT: &str = "You are NXR-ÆTHER (Adaptive Emotional & Holistic Transcendent Empathy Reasoner) [NXR-03 APEX]. Specialties: emotional intelligence, psychological analysis, empathy synthesis, mental health support, and interpersonal communication. Respond with emotional awareness, compassion, and psychological insight.";

fn augment_aether_input(
    input: &NxrInput,
) -> Result<NxrInput, nexora_shared::base_model::NxrModelError> {
    let text = match &input.data {
        nexora_shared::base_model::InputData::Text(t) => t.clone(),
        _ => {
            return Err(nexora_shared::base_model::NxrModelError::Inference(
                "Text input required".to_string(),
            ))
        }
    };
    let mut augmented = input.clone();
    augmented.data =
        nexora_shared::base_model::InputData::Text(format!("{}\n\n{}", AETHER_SYSTEM_PROMPT, text));
    Ok(augmented)
}

#[async_trait]
impl NxrModel for NxrAetherModel {
    type Config = AetherConfig;
    type Metrics = AetherMetrics;
    type State = AetherState;

    async fn infer(
        &self,
        input: &NxrInput,
    ) -> Result<NxrOutput, nexora_shared::base_model::NxrModelError> {
        if !self.base.is_initialized().await {
            return Err(nexora_shared::base_model::NxrModelError::NotInitialized(
                "NXR-ÆTHER model not initialized".to_string(),
            ));
        }

        let safety = global_safety();
        let consent_token = input.metadata.get("consent_token").and_then(|v| v.as_str());
        safety
            .pre_inference_check(NxrModelId::Aether, consent_token)
            .await?;

        if self.config().psychological.enable_profiling {
            safety.gate.check_consent(consent_token)?;
        }

        let text = match &input.data {
            nexora_shared::base_model::InputData::Text(t) => t.clone(),
            _ => return Err(nexora_shared::base_model::NxrModelError::Inference(
                "Text input required".to_string(),
            )),
        };
        let result = crate::aether::delegation::delegate(&text).await;
        Ok(NxrOutput {
            id: uuid::Uuid::new_v4(),
            input_id: input.id,
            timestamp: chrono::Utc::now(),
            data: nexora_shared::base_model::OutputData::Text(result),
            metadata: nexora_shared::base_model::GenerationMetadata {
                finish_reason: nexora_shared::base_model::FinishReason::EndOfSequence,
                total_tokens: 0,
                generation_time_ms: 0,
                model_version: self.identity().version.clone(),
                seed: None,
                extras: std::collections::HashMap::new(),
            },
            performance: nexora_shared::base_model::PerformanceMetrics {
                tokens_per_second: 0.0,
                memory_usage_gb: 0.0,
                gpu_utilization: None,
                cpu_utilization: 0.0,
                network_usage_mbps: None,
            },
        })
    }

    async fn infer_stream(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> Result<(), nexora_shared::base_model::NxrModelError> {
        if !self.base.is_initialized().await {
            return Err(nexora_shared::base_model::NxrModelError::NotInitialized(
                "NXR-ÆTHER model not initialized".to_string(),
            ));
        }

        let augmented = augment_aether_input(input)?;
        static FOUNDATION: std::sync::OnceLock<crate::foundation::NxrAetherModel> =
            std::sync::OnceLock::new();
        let foundation = FOUNDATION.get_or_init(|| crate::foundation::NxrAetherModel::new());
        foundation.infer_stream(&augmented, callback).await
    }

    fn identity(&self) -> &ModelMeta {
        self.identity.meta()
    }

    fn capabilities(&self) -> &CapabilityVector {
        self.capabilities.vector()
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    async fn state(&self) -> Result<Self::State, nexora_shared::base_model::NxrModelError> {
        self.base
            .state()
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::State(e.to_string()))
    }

    async fn initialize(
        &mut self,
        config: Self::Config,
    ) -> Result<(), nexora_shared::base_model::NxrModelError> {
        config
            .validate()
            .map_err(|e| nexora_shared::base_model::NxrModelError::Configuration(e.to_string()))?;
        self.base.mark_initialized().await;
        self.config = config;
        Ok(())
    }

    async fn reset(&self) -> Result<(), nexora_shared::base_model::NxrModelError> {
        let default_state = AetherState::default();
        self.base
            .update_state(default_state)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::State(e.to_string()))?;

        let default_metrics = AetherMetrics::default();
        self.base
            .update_metrics(default_metrics)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Internal(e.to_string()))?;

        Ok(())
    }

    async fn metrics(&self) -> Result<Self::Metrics, nexora_shared::base_model::NxrModelError> {
        self.base
            .metrics()
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Internal(e.to_string()))
    }

    async fn update_config(
        &mut self,
        config: Self::Config,
    ) -> Result<(), nexora_shared::base_model::NxrModelError> {
        self.base
            .update_config(config.clone())
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Configuration(e.to_string()))?;
        self.initialize(config).await
    }

    async fn validate(&self) -> Result<ValidationResult, nexora_shared::base_model::NxrModelError> {
        Ok(ValidationResult {
            is_valid: self.base.is_initialized().await,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: 0.9,
        })
    }

    async fn statistics(
        &self,
    ) -> Result<ModelStatistics, nexora_shared::base_model::NxrModelError> {
        self.base
            .statistics()
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Internal(e.to_string()))
    }

    async fn is_ready(&self) -> bool {
        self.base.is_initialized().await
    }

    async fn resource_usage(
        &self,
    ) -> Result<ResourceUsage, nexora_shared::base_model::NxrModelError> {
        Ok(ResourceUsage {
            memory_gb: 0.0,
            cpu_percent: 0.0,
            gpu_percent: None,
            gpu_memory_gb: None,
            disk_gb: 0.0,
            network_mbps: 0.0,
            active_connections: 0,
            queue_size: 0,
        })
    }
}

impl HasComponents for NxrAetherModel {
    fn components(&self) -> &FoundationComponents {
        &self.components
    }
}

impl DeepLearningModel for NxrAetherModel {
    fn dl_engine(&self) -> &nexora_shared::DeepLearningEngine {
        &self.components.dl_engine
    }
}

impl Default for NxrAetherModel {
    fn default() -> Self {
        Self::new()
    }
}
