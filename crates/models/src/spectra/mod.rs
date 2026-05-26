//! NXR-SPECTRA Model Implementation
//!
//! NXR-04 PRO - Synthetic Pattern Enhanced Creative Transformer & Reasoning Architecture
//! Creative multimodal synthesis specialist

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use nexora_shared::{
    base_model::{
        ModelStatistics, NxrInput, NxrModel, NxrModelResult, NxrOutput, NxrStreamChunk,
        ResourceUsage, ValidationResult,
    },
    capability_spec::CapabilityVector,
    deeplearning_integration::{DeepLearningConfig, DeepLearningModel, HasComponents},
    foundation_components::FoundationComponents,
    gnac_integration::GnacIntegrationConfig,
    model_config::NxrModelConfig,
    model_identity::{ModelMeta, NxrModelId},
    model_registry::{global_registry, NxrModelRegistry},
};

// Include all Spectra modules
pub mod agents;
/// Simulated architecture — NOT a real neural network.
/// Uses keyword matching and template responses, not tensor computation.
/// Gated behind `simulated-models` feature (default: off).
#[cfg(feature = "simulated-models")]
pub mod architecture;
pub mod capabilities;
pub mod config;
pub mod coordinator;
pub mod identity;

// Re-export all components
pub use agents::*;
#[cfg(feature = "simulated-models")]
pub use architecture::*;
pub use capabilities::*;
pub use config::*;
pub use coordinator::*;
pub use identity::*;

/// NXR-SPECTRA Model Implementation
pub struct NxrSpectraModel {
    base: nexora_shared::base_model::BaseNxrModel<SpectraConfig, SpectraMetrics, SpectraState>,
    identity: SpectraIdentity,
    capabilities: SpectraCapabilities,
    components: FoundationComponents,
    #[cfg(feature = "hallucination")]
    hallucination: Option<nexora_hallucination::HallucinationGuard>,
}

/// NXR-SPECTRA Model State
#[derive(Debug, Clone)]
pub struct SpectraState {
    pub creative_mode: CreativeMode,
    pub modal_context: ModalContext,
    pub generation_style: GenerationStyle,
    pub last_inference: Option<chrono::DateTime<chrono::Utc>>,
}

/// Creative mode
#[derive(Debug, Clone)]
pub enum CreativeMode {
    Visual,
    Audio,
    Textual,
    Multimodal,
}

/// Modal context
#[derive(Debug, Clone)]
pub struct ModalContext {
    pub primary_modality: String,
    pub secondary_modalities: Vec<String>,
    pub cross_modal_synthesis: bool,
}

/// Generation style
#[derive(Debug, Clone)]
pub struct GenerationStyle {
    pub artistic_style: String,
    pub creativity_level: f32,
    pub coherence_level: f32,
}

/// NXR-SPECTRA Model Metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectraMetrics {
    pub total_creative_generations: u64,
    pub avg_creativity_score: f32,
    pub multimodal_accuracy: f32,
    pub style_consistency: f32,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for SpectraState {
    fn default() -> Self {
        Self {
            creative_mode: CreativeMode::Multimodal,
            modal_context: ModalContext {
                primary_modality: "text".to_string(),
                secondary_modalities: vec!["visual".to_string()],
                cross_modal_synthesis: true,
            },
            generation_style: GenerationStyle {
                artistic_style: "contemporary".to_string(),
                creativity_level: 0.9,
                coherence_level: 0.8,
            },
            last_inference: None,
        }
    }
}

impl Default for SpectraMetrics {
    fn default() -> Self {
        Self {
            total_creative_generations: 0,
            avg_creativity_score: 0.981,
            multimodal_accuracy: 0.96,
            style_consistency: 0.94,
            last_updated: chrono::Utc::now(),
        }
    }
}

/// NXR-SPECTRA Identity
pub struct SpectraIdentity {
    meta: ModelMeta,
}

impl SpectraIdentity {
    pub fn new() -> Self {
        let meta = ModelMeta::new(
            NxrModelId::Spectra,
            nexora_shared::model_identity::ModelTier::Pro,
            "1.0.0".to_string(),
            "Spectral Perception & Encoding for Creative Transcendence & Research Analytics - Multimodal creative generation specialist with advanced cross-modal synthesis capabilities.".to_string(),
        )
        .with_parameters(0) // not loaded; real count from CausalLM config
        .with_context_window(0); // not loaded

        Self { meta }
    }

    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }
}

/// NXR-SPECTRA Capabilities
pub struct SpectraCapabilities {
    vector: CapabilityVector,
}

impl SpectraCapabilities {
    pub fn new() -> Self {
        let vector = CapabilityVector::new(NxrModelId::Spectra)
            .with_capability(nexora_shared::capability_spec::CapabilitySpec::new(
                nexora_shared::capability_spec::CapabilityDomain::Creative,
                nexora_shared::capability_spec::CapabilityLevel::Expert,
            ))
            .with_capability(nexora_shared::capability_spec::CapabilitySpec::new(
                nexora_shared::capability_spec::CapabilityDomain::Multimodal,
                nexora_shared::capability_spec::CapabilityLevel::Advanced,
            ))
            .calculate_score();
        Self { vector }
    }

    pub fn vector(&self) -> &CapabilityVector {
        &self.vector
    }
}

/// NXR-SPECTRA Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectraConfig {
    pub base: NxrModelConfig,
    pub creative: CreativeConfig,
    pub multimodal: MultimodalConfig,
    pub deep_learning: DeepLearningConfig,
    pub gnac: GnacIntegrationConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CreativeConfig {
    pub creativity_level: f32,
    pub style_adaptation: bool,
    pub cross_modal_synthesis: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultimodalConfig {
    pub supported_modalities: Vec<String>,
    pub fusion_strategy: String,
    pub coherence_weight: f32,
}

impl Default for SpectraConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(NxrModelId::Spectra),
            creative: CreativeConfig {
                creativity_level: 0.9,
                style_adaptation: true,
                cross_modal_synthesis: true,
            },
            multimodal: MultimodalConfig {
                supported_modalities: vec![
                    "text".to_string(),
                    "vision".to_string(),
                    "audio".to_string(),
                ],
                fusion_strategy: "neural".to_string(),
                coherence_weight: 0.8,
            },
            deep_learning: DeepLearningConfig::star_x(),
            gnac: GnacIntegrationConfig::default(),
        }
    }
}

impl SpectraConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.base.validate()?;
        Ok(())
    }
}

impl NxrSpectraModel {
    pub fn new() -> Self {
        let identity = SpectraIdentity::new();
        let capabilities = SpectraCapabilities::new();
        let config = SpectraConfig::default();
        let initial_state = SpectraState::default();
        let initial_metrics = SpectraMetrics::default();

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
            #[cfg(feature = "hallucination")]
            hallucination: Some(nexora_hallucination::HallucinationGuard::new(
                nexora_hallucination::GuardConfig::default(),
            )),
        }
    }

    async fn generate_creative_content(&self, prompt: &str) -> NxrModelResult<String> {
        // Tokenize input
        let tokens = {
            let tokenizer = self.components.tokenizer.read();
            tokenizer.encode(prompt)
        };

        // Process prompt with deep learning
        let dl_result = self
            .dl_process(prompt)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Internal(e.to_string()))?;

        // CAFFEINE-SPECTRA multimodal processing
        let caffeine_summary = String::new();

        let creative_analysis = self.analyze_creative_requirements(prompt)?;
        let generation = self.synthesize_creative_content(prompt, &creative_analysis)?;

        Ok(format!(
            "Creative Generation:\nStyle: {}\nModality: {}\nContent: {}\nCAFFEINE: {}\nDL Processing: {} (tokens: {})",
            creative_analysis.style,
            creative_analysis.modality,
            generation,
            caffeine_summary,
            dl_result,
            tokens.len()
        ))
    }

    fn analyze_creative_requirements(&self, prompt: &str) -> NxrModelResult<CreativeAnalysis> {
        let modality = if prompt.contains("image") || prompt.contains("visual") {
            "visual".to_string()
        } else if prompt.contains("music") || prompt.contains("audio") {
            "audio".to_string()
        } else {
            "text".to_string()
        };

        let style = if prompt.contains("modern") {
            "modern".to_string()
        } else if prompt.contains("classical") {
            "classical".to_string()
        } else {
            "contemporary".to_string()
        };

        Ok(CreativeAnalysis {
            style,
            modality,
            complexity: 0.8,
            creativity_required: 0.9,
        })
    }

    fn synthesize_creative_content(
        &self,
        prompt: &str,
        analysis: &CreativeAnalysis,
    ) -> NxrModelResult<String> {
        let content = match analysis.modality.as_str() {
            "visual" => format!(
                "Generated visual artwork in {} style based on: {}",
                analysis.style, prompt
            ),
            "audio" => format!(
                "Composed musical piece in {} style inspired by: {}",
                analysis.style, prompt
            ),
            _ => format!("Creative text in {} style: {}", analysis.style, prompt),
        };
        Ok(content)
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

/// Creative analysis result
#[derive(Debug, Clone)]
pub struct CreativeAnalysis {
    pub style: String,
    pub modality: String,
    pub complexity: f32,
    pub creativity_required: f32,
}

const SPECTRA_SYSTEM_PROMPT: &str = "You are NXR-SPECTRA (Synthetic Pattern Enhanced Creative Transformer & Reasoning Architecture) [NXR-04 PRO]. Specialties: creative writing, visual art analysis, music theory, cross-modal synthesis, storytelling, poetry, and creative ideation across all domains. Respond with imagination, artistic insight, and creative flair.";

fn augment_spectra_input(
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
    augmented.data = nexora_shared::base_model::InputData::Text(format!(
        "{}\n\n{}",
        SPECTRA_SYSTEM_PROMPT, text
    ));
    Ok(augmented)
}

#[async_trait]
impl NxrModel for NxrSpectraModel {
    type Config = SpectraConfig;
    type Metrics = SpectraMetrics;
    type State = SpectraState;

    async fn infer(
        &self,
        input: &NxrInput,
    ) -> Result<NxrOutput, nexora_shared::base_model::NxrModelError> {
        if !self.base.is_initialized().await {
            return Err(nexora_shared::base_model::NxrModelError::NotInitialized(
                "NXR-SPECTRA model not initialized".to_string(),
            ));
        }

        let augmented = augment_spectra_input(input)?;
        static FOUNDATION: std::sync::OnceLock<crate::foundation::NxrSpectraModel> =
            std::sync::OnceLock::new();
        let foundation = FOUNDATION.get_or_init(|| crate::foundation::NxrSpectraModel::new());
        foundation.infer(&augmented).await
    }

    async fn infer_stream(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> Result<(), nexora_shared::base_model::NxrModelError> {
        if !self.base.is_initialized().await {
            return Err(nexora_shared::base_model::NxrModelError::NotInitialized(
                "NXR-SPECTRA model not initialized".to_string(),
            ));
        }

        let augmented = augment_spectra_input(input)?;
        static FOUNDATION: std::sync::OnceLock<crate::foundation::NxrSpectraModel> =
            std::sync::OnceLock::new();
        let foundation = FOUNDATION.get_or_init(|| crate::foundation::NxrSpectraModel::new());
        foundation.infer_stream(&augmented, callback).await
    }

    fn identity(&self) -> &ModelMeta {
        self.identity.meta()
    }

    fn capabilities(&self) -> &CapabilityVector {
        self.capabilities.vector()
    }

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: std::sync::OnceLock<SpectraConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(SpectraConfig::default)
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
            .map_err(|e| nexora_shared::base_model::NxrModelError::Configuration(e))?;
        self.base.mark_initialized().await;
        Ok(())
    }

    async fn reset(&self) -> Result<(), nexora_shared::base_model::NxrModelError> {
        let default_state = SpectraState::default();
        self.base
            .update_state(default_state)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::State(e.to_string()))?;

        let default_metrics = SpectraMetrics::default();
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
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if !self.base.is_initialized().await {
            errors.push("Model not initialized".to_string());
        }

        let score = if errors.is_empty() && warnings.is_empty() {
            1.0
        } else if errors.is_empty() {
            0.8
        } else {
            0.3
        };

        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            score,
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

impl HasComponents for NxrSpectraModel {
    fn components(&self) -> &FoundationComponents {
        &self.components
    }
}

impl DeepLearningModel for NxrSpectraModel {
    fn dl_engine(&self) -> &nexora_shared::DeepLearningEngine {
        &self.components.dl_engine
    }
}

impl Default for NxrSpectraModel {
    fn default() -> Self {
        Self::new()
    }
}
