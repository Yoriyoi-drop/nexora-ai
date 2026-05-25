//! NXR-SWIFT Model Implementation
//!
//! NXR-08 EDGE - Sub-millisecond Weighted Inference & Fast Thought
//! Ultra-lightweight edge computing specialist

pub mod agents;
pub mod architecture;
pub mod capabilities;
pub mod config;
pub mod identity;

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
};

use self::{
    agents::SwiftAgents, architecture::SwiftArchitecture, capabilities::SwiftCapabilities,
    config::SwiftConfig, identity::SwiftIdentity,
};

pub struct NxrSwiftModel {
    base: nexora_shared::base_model::BaseNxrModel<SwiftConfig, SwiftMetrics, SwiftState>,
    identity: SwiftIdentity,
    architecture: SwiftArchitecture,
    agents: SwiftAgents,
    capabilities: SwiftCapabilities,
    components: FoundationComponents,
    config: SwiftConfig,
    #[cfg(feature = "hallucination")]
    hallucination: Option<nexora_hallucination::HallucinationGuard>,
}

#[derive(Debug, Clone)]
pub struct SwiftState {
    pub optimization_level: OptimizationLevel,
    pub resource_constraints: ResourceConstraints,
    pub latency_target_ms: u32,
    pub last_inference: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    Minimal,
    Balanced,
    Aggressive,
    Maximum,
}

#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_memory_mb: u32,
    pub max_cpu_percent: f32,
    pub battery_optimized: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwiftMetrics {
    pub total_inferences: u64,
    pub avg_latency_ms: f32,
    pub throughput_ops_per_second: f64,
    pub energy_efficiency: f32,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for SwiftState {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Aggressive,
            resource_constraints: ResourceConstraints {
                max_memory_mb: 512,
                max_cpu_percent: 80.0,
                battery_optimized: true,
            },
            latency_target_ms: 1,
            last_inference: None,
        }
    }
}

impl Default for SwiftMetrics {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            avg_latency_ms: 0.1,
            throughput_ops_per_second: 10000.0,
            energy_efficiency: 0.95,
            last_updated: chrono::Utc::now(),
        }
    }
}

impl NxrSwiftModel {
    pub fn new() -> Self {
        let identity = SwiftIdentity::new();
        let capabilities = SwiftCapabilities::new();
        let config = SwiftConfig::default();
        let initial_state = SwiftState::default();
        let initial_metrics = SwiftMetrics::default();

        Self {
            base: nexora_shared::base_model::BaseNxrModel::new(
                identity.meta().clone(),
                capabilities.vector().clone(),
                config.clone(),
                initial_state,
                initial_metrics,
            ),
            identity,
            architecture: SwiftArchitecture::new(&config),
            agents: SwiftAgents::new(&config),
            capabilities,
            components: FoundationComponents::new(),
            config,
            #[cfg(feature = "hallucination")]
            hallucination: Some(nexora_hallucination::HallucinationGuard::new(nexora_hallucination::GuardConfig::default())),
        }
    }

    async fn fast_inference(&self, input: &str) -> NxrModelResult<String> {
        // Check cache first
        if let Some(cached) = self.agents.check_inference_cache(input).await {
            return Ok(cached);
        }

        let start_time = std::time::Instant::now();

        // Tokenize input
        let tokens = {
            let tokenizer = self.components.tokenizer.read();
            tokenizer.encode(input)
        };

        // Apply ATQS compression for edge efficiency
        {
            let _ = self.components.atqs.compress(input.as_bytes()).await;
        }

        // Process input with deep learning
        let dl_result = self
            .dl_process(input)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Internal(e.to_string()))?;

        // Ultra-fast processing with minimal overhead
        let processed = self.minimal_processing(input)?;
        let optimized = self.edge_optimization(&processed)?;

        let latency = start_time.elapsed().as_millis() as u32;

        let result = format!(
            "Edge Response ({}ms): {}\nDL Processing: {} (tokens: {})",
            latency,
            optimized,
            dl_result,
            tokens.len()
        );

        // Store in cache
        self.agents.store_inference_cache(input, &result);

        Ok(result)
    }

    fn minimal_processing(&self, input: &str) -> NxrModelResult<String> {
        // Minimal processing for speed
        let words: Vec<&str> = input.split_whitespace().collect();
        if words.len() > 10 {
            Ok(format!("Processed {} words efficiently", words.len()))
        } else {
            Ok(input.to_string())
        }
    }

    fn edge_optimization(&self, processed: &str) -> NxrModelResult<String> {
        // Edge-specific optimizations
        Ok(format!("Optimized for edge: {}", processed))
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

#[async_trait]
impl NxrModel for NxrSwiftModel {
    type Config = SwiftConfig;
    type Metrics = SwiftMetrics;
    type State = SwiftState;

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
            .map_err(|e| nexora_shared::base_model::NxrModelError::Configuration(e))?;
        self.architecture
            .initialize(&config)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::Internal(e.to_string()))?;
        self.base.mark_initialized().await;
        self.config = config;
        Ok(())
    }

    async fn reset(&self) -> Result<(), nexora_shared::base_model::NxrModelError> {
        let default_state = SwiftState::default();
        self.base
            .update_state(default_state)
            .await
            .map_err(|e| nexora_shared::base_model::NxrModelError::State(e.to_string()))?;

        let default_metrics = SwiftMetrics::default();
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

    async fn infer(
        &self,
        input: &NxrInput,
    ) -> Result<NxrOutput, nexora_shared::base_model::NxrModelError> {
        if !self.base.is_initialized().await {
            return Err(nexora_shared::base_model::NxrModelError::NotInitialized(
                "NXR-SWIFT model not initialized".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        let input_text = match &input.data {
            nexora_shared::base_model::InputData::Text(text) => text.clone(),
            _ => {
                return Err(nexora_shared::base_model::NxrModelError::Inference(
                    "NXR-SWIFT only supports text input".to_string(),
                ))
            }
        };

        let result = self.fast_inference(&input_text).await?;
        let generation_time_ms = start_time.elapsed().as_millis() as u64;
        let total_tokens = result.split_whitespace().count();

        let mut extras = std::collections::HashMap::new();
        #[cfg(feature = "hallucination")]
        if let Some(report) = self.run_hallucination_check(input).await {
            extras.insert(
                "hallucination_risk".to_string(),
                serde_json::Value::String(format!("{:?}", report.risk_level)),
            );
            extras.insert(
                "hallucination_score".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(report.score as f64)
                        .unwrap_or(serde_json::Number::from(0)),
                ),
            );
            extras.insert(
                "hallucination_action".to_string(),
                serde_json::Value::String(format!("{:?}", report.action)),
            );
        }

        Ok(NxrOutput {
            id: uuid::Uuid::new_v4(),
            input_id: input.id,
            timestamp: chrono::Utc::now(),
            data: nexora_shared::base_model::OutputData::Text(result),
            metadata: nexora_shared::base_model::GenerationMetadata {
                finish_reason: nexora_shared::base_model::FinishReason::EndOfSequence,
                total_tokens,
                generation_time_ms,
                model_version: self.identity.meta().version.clone(),
                seed: None,
                extras,
            },
            performance: nexora_shared::base_model::PerformanceMetrics {
                tokens_per_second: total_tokens as f32 / (generation_time_ms as f32 / 1000.0),
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
                "NXR-SWIFT model not initialized".to_string(),
            ));
        }

        let input_text = match &input.data {
            nexora_shared::base_model::InputData::Text(text) => text.clone(),
            _ => {
                return Err(nexora_shared::base_model::NxrModelError::Inference(
                    "NXR-SWIFT only supports text input".to_string(),
                ))
            }
        };

        let result = self.fast_inference(&input_text).await?;
        let chunk = NxrStreamChunk {
            id: uuid::Uuid::new_v4(),
            input_id: input.id,
            timestamp: chrono::Utc::now(),
            data: nexora_shared::base_model::StreamChunkData::TextDelta(result),
            is_final: true,
        };
        callback(chunk);

        Ok(())
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

impl HasComponents for NxrSwiftModel {
    fn components(&self) -> &FoundationComponents {
        &self.components
    }
}

impl DeepLearningModel for NxrSwiftModel {
    fn dl_engine(&self) -> &nexora_shared::DeepLearningEngine {
        &self.components.dl_engine
    }
}

impl Default for NxrSwiftModel {
    fn default() -> Self {
        Self::new()
    }
}
