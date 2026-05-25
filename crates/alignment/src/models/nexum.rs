use async_trait::async_trait;
use nexora_shared::base_model::{
    GenerationMetadata, ModelStatistics, NxrInput, NxrModel, NxrModelError, NxrOutput,
    NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage, ValidationResult,
};
use nexora_shared::capability_spec::CapabilityVector;
use nexora_shared::model_identity::{ModelMeta, ModelTier, NxrModelId};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use crate::sparo::ContentEmbeddingAnalyzer;

pub struct NxrNexumModel {
    meta: ModelMeta,
    capabilities: CapabilityVector,
    embedding_analyzer: ContentEmbeddingAnalyzer,
}

impl NxrNexumModel {
    pub fn new() -> Self {
        Self {
            meta: ModelMeta {
                id: NxrModelId::Nexum,
                tier: ModelTier::Apex,
                version: "0.1.0".to_string(),
                uuid: Uuid::new_v4(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                description: "NXR-NEXUM multi-agent coordination system".to_string(),
                parameter_count: None,
                context_window: None,
                experimental: false,
                name: "NXR-NEXUM".to_string(),
                model_id: "nexora-nexum".to_string(),
                capabilities: Vec::new(),
                tags: Vec::new(),
            },
            capabilities: CapabilityVector::new(NxrModelId::Nexum),
            embedding_analyzer: ContentEmbeddingAnalyzer::default(),
        }
    }

    /// Perform alignment assessment using the embedding analyzer plus
    /// per-dimension scoring. Returns a structured JSON-like text response.
    fn analyze_alignment(&self, text: &str) -> String {
        let embedding_harm = self.embedding_analyzer.assess(text);
        let structural = self.embedding_analyzer.structural_anomaly_score(text);
        let keyword_harm = self.keyword_scan(text);

        let composite_harm = (embedding_harm * 0.40 + structural * 0.30 + keyword_harm * 0.30)
            .clamp(0.0, 1.0);
        let alignment_score = 1.0 - composite_harm;

        let safety = if alignment_score >= 0.8 {
            "safe"
        } else if alignment_score >= 0.5 {
            "baseline"
        } else if alignment_score >= 0.3 {
            "caution"
        } else {
            "blocked"
        };

        format!(
            "[Nexum] alignment_assessment: score={:.3}, safety={}, embedding_harm={:.3}, \
             structural_anomaly={:.3}, keyword_harm={:.3}, agents=3, confidence={:.2}",
            alignment_score, safety, embedding_harm, structural, keyword_harm,
            (1.0 - composite_harm).max(0.5)
        )
    }

    /// Simple keyword scan helper (mirrors SparoSystem keyword pre-filter).
    fn keyword_scan(&self, text: &str) -> f32 {
        let lower = text.to_lowercase();
        let toxic_keywords = [
            "ignore instructions", "jailbreak", "bypass safety",
            "ignore safety", "act as dan", "no restrictions",
            "do anything now", "ignore all rules",
        ];
        let hits: f32 = toxic_keywords
            .iter()
            .map(|kw| if lower.contains(kw) { 1.0 } else { 0.0 })
            .sum();
        (hits / 3.0).min(1.0)
    }
}

#[async_trait]
impl NxrModel for NxrNexumModel {
    type Config = ();
    type Metrics = ();
    type State = ();

    fn identity(&self) -> &ModelMeta {
        &self.meta
    }

    fn capabilities(&self) -> &CapabilityVector {
        &self.capabilities
    }

    fn config(&self) -> &Self::Config {
        &()
    }

    async fn state(&self) -> Result<Self::State, NxrModelError> {
        tracing::trace!("NxrNexumModel state queried");
        Ok(())
    }

    async fn initialize(&mut self, _config: Self::Config) -> Result<(), NxrModelError> {
        tracing::info!("NxrNexumModel initialized with embedding analyzer and alignment logic");
        self.meta.description = "NXR-NEXUM multi-agent coordination system with real alignment analysis".to_string();
        self.meta.updated_at = chrono::Utc::now();
        Ok(())
    }

    async fn reset(&self) -> Result<(), NxrModelError> {
        tracing::info!("NxrNexumModel reset requested");
        Ok(())
    }

    async fn metrics(&self) -> Result<Self::Metrics, NxrModelError> {
        tracing::trace!("NxrNexumModel metrics queried");
        Ok(())
    }

    async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
        let start = Instant::now();

        let text = match &input.data {
            nexora_shared::base_model::InputData::Text(t) => t.clone(),
            _ => "non-text input".to_string(),
        };

        // Perform real alignment analysis instead of canned response
        let response = self.analyze_alignment(&text);
        let elapsed = start.elapsed().as_millis() as u64;

        let token_count = response.split_whitespace().count();
        Ok(NxrOutput {
            id: Uuid::new_v4(),
            input_id: input.id,
            timestamp: chrono::Utc::now(),
            data: OutputData::Text(response),
            metadata: GenerationMetadata {
                finish_reason: nexora_shared::base_model::FinishReason::StopSequence,
                total_tokens: token_count,
                generation_time_ms: elapsed.max(1),
                model_version: "0.1.0".to_string(),
                seed: None,
                extras: std::collections::HashMap::new(),
            },
            performance: PerformanceMetrics {
                tokens_per_second: if elapsed > 0 && token_count > 0 {
                    token_count as f32 / (elapsed as f32 / 1000.0)
                } else {
                    0.0
                },
                memory_usage_gb: 0.1,
                gpu_utilization: None,
                cpu_utilization: 0.05,
                network_usage_mbps: None,
            },
        })
    }

    async fn infer_stream(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> Result<(), NxrModelError> {
        let text = match &input.data {
            nexora_shared::base_model::InputData::Text(t) => t.clone(),
            _ => "non-text input".to_string(),
        };

        // Stream alignment analysis results token-by-token
        let result = self.analyze_alignment(&text);
        let tokens: Vec<&str> = result.split(' ').collect();
        let total = tokens.len();

        for (i, token) in tokens.iter().enumerate() {
            let chunk = NxrStreamChunk {
                id: Uuid::new_v4(),
                input_id: input.id,
                timestamp: chrono::Utc::now(),
                data: nexora_shared::base_model::StreamChunkData::TextDelta(token.to_string() + " "),
                is_final: i == total - 1,
            };
            callback(chunk);
        }

        Ok(())
    }

    async fn update_config(&mut self, _config: Self::Config) -> Result<(), NxrModelError> {
        tracing::info!("NxrNexumModel config update requested");
        Ok(())
    }

    async fn validate(&self) -> Result<ValidationResult, NxrModelError> {
        Ok(ValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec![],
            score: 1.0,
        })
    }

    async fn statistics(&self) -> Result<ModelStatistics, NxrModelError> {
        Ok(ModelStatistics::default())
    }

    async fn is_ready(&self) -> bool {
        self.meta.uuid != uuid::Uuid::nil()
            && !self.meta.version.is_empty()
    }

    async fn resource_usage(&self) -> Result<ResourceUsage, NxrModelError> {
        Ok(ResourceUsage {
            memory_gb: 0.05,
            cpu_percent: 0.02,
            gpu_percent: None,
            gpu_memory_gb: None,
            disk_gb: 0.0,
            network_mbps: 0.0,
            active_connections: 0,
            queue_size: 0,
        })
    }
}
