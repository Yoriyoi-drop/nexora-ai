//! Foundation model adapter — the primary NxrModel implementation.
//!
//! Single responsibility: bridge between the `NxrModel` trait (shared interface)
//! and the underlying `CausalLM` transformer backbone. This module owns model
//! lifecycle, inference dispatch, and metrics tracking — and nothing else.
//!
//! Related concerns live in submodules:
//! - `config`: model-to-TransformerConfig mapping
//! - `tokenizer`: fallback byte-level encode/decode
//! - `error`: FoundationError type
//! - `helpers`: call_model, extract_params

pub mod config;
pub mod error;
pub mod helpers;
pub mod tokenizer;

// ── Re-exports for backward compatibility ──────────────────────────────────
pub use self::config::{model_tier_for, transformer_config_for};
pub use self::error::FoundationError;
pub use self::helpers::{call_model, extract_params};
pub use self::tokenizer::{byte_decode, byte_encode};

use async_trait::async_trait;
use ndarray::Array1;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;
use tracing::instrument;

use nexora_shared::{
    base_model::{
        FinishReason, GenerationMetadata, InputData, ModelStatistics, NxrInput, NxrModel,
        NxrModelError, NxrOutput, NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage,
        StreamChunkData, ValidationResult,
    },
    capability_spec::CapabilityVector,
    model_identity::{ModelMeta, NxrModelId},
    tokenizer_integration::NxrTokenizerRef,
};
use nexora_transformer::{CausalLM, TransformerConfig};

// ── FoundationModel ────────────────────────────────────────────────────────

/// Primary foundation model adapter.
///
/// Owns:
/// - Model identity + lifecycle (`model_id`, `model`, `model_config`)
/// - Inference counters / metrics
/// - Hallucination guard (optional)
/// - Cached identity, capabilities, config
pub struct FoundationModel {
    pub model_id: NxrModelId,
    pub tokenizer: Option<NxrTokenizerRef>,
    pub model: Arc<Mutex<Option<Arc<CausalLM>>>>,
    model_config: TransformerConfig,
    inference_count: AtomicU64,
    total_generated: AtomicU64,
    total_time_ms: AtomicU64,
    start_time: chrono::DateTime<chrono::Utc>,
    hallucination: Option<nexora_alignment::hallucination::HallucinationGuard>,
    identity_cache: OnceLock<ModelMeta>,
    capabilities_cache: OnceLock<CapabilityVector>,
    config_cache: OnceLock<Value>,
}

impl FoundationModel {
    pub fn new(model_id: NxrModelId) -> Self {
        let config = transformer_config_for(model_id);
        Self {
            model_id,
            tokenizer: None,
            model: Arc::new(Mutex::new(None)),
            model_config: config,
            inference_count: AtomicU64::new(0),
            total_generated: AtomicU64::new(0),
            total_time_ms: AtomicU64::new(0),
            start_time: chrono::Utc::now(),
            hallucination: Some(nexora_alignment::hallucination::HallucinationGuard::new(
                nexora_alignment::hallucination::GuardConfig::default(),
            )),
            identity_cache: OnceLock::new(),
            capabilities_cache: OnceLock::new(),
            config_cache: OnceLock::new(),
        }
    }

    pub fn with_tokenizer(t: NxrTokenizerRef) -> Self {
        let mut m = Self::new(NxrModelId::Omnis);
        m.tokenizer = Some(t);
        m
    }

    pub fn enable_hallucination_guard(&mut self) {
        self.hallucination = Some(nexora_alignment::hallucination::HallucinationGuard::new(
            nexora_alignment::hallucination::GuardConfig::default(),
        ));
    }

    pub fn disable_hallucination_guard(&mut self) {
        self.hallucination = None;
    }

    pub fn with_hallucination_guard(
        mut self,
        guard: nexora_alignment::hallucination::HallucinationGuard,
    ) -> Self {
        self.hallucination = Some(guard);
        self
    }

    async fn run_hallucination_check(
        &self,
        input: &NxrInput,
    ) -> Option<nexora_alignment::hallucination::PipelineResult> {
        if let Some(ref h) = self.hallucination {
            let text = match &input.data {
                InputData::Text(t) => t.clone(),
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

    pub fn set_model_arc(&self, model: Arc<CausalLM>) {
        let mut guard = self.model.lock().unwrap_or_else(|e| {
            tracing::warn!("Mutex poisoned in set_model_arc, recovering");
            e.into_inner()
        });
        *guard = Some(model);
    }

    fn encode_text(&self, text: &str) -> Vec<u32> {
        if let Some(ref tk) = self.tokenizer {
            tk.read().encode(text)
        } else {
            byte_encode(text)
        }
    }

    fn decode_ids(&self, ids: &[u32]) -> String {
        if let Some(ref tk) = self.tokenizer {
            tk.read().decode(ids)
        } else {
            byte_decode(ids)
        }
    }

    pub async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
        <Self as NxrModel>::infer(self, input).await
    }

    pub fn reload(&mut self, path: &str) -> Result<(), FoundationError> {
        let config = transformer_config_for(self.model_id);
        let loaded = CausalLM::from_checkpoint(config, path)?;
        *self.model.lock().unwrap_or_else(|e| {
            tracing::warn!("Mutex poisoned in reload, recovering");
            e.into_inner()
        }) = Some(Arc::new(loaded));
        Ok(())
    }

    pub fn omnis() -> Self { Self::new(NxrModelId::Omnis) }
    pub fn vortex() -> Self { Self::new(NxrModelId::Vortex) }
    pub fn aether() -> Self { Self::new(NxrModelId::Aether) }
    pub fn spectra() -> Self { Self::new(NxrModelId::Spectra) }
    pub fn nexum() -> Self { Self::new(NxrModelId::Nexum) }
    pub fn axiom() -> Self { Self::new(NxrModelId::Axiom) }
    pub fn cipher() -> Self { Self::new(NxrModelId::Cipher) }
    pub fn swift() -> Self { Self::new(NxrModelId::Swift) }
    pub fn kronos() -> Self { Self::new(NxrModelId::Kronos) }
    pub fn genesis() -> Self { Self::new(NxrModelId::Genesis) }

    pub fn from_checkpoint(path: &str) -> Self {
        let model_id = NxrModelId::Omnis;
        let config = transformer_config_for(model_id);
        match CausalLM::from_checkpoint(config.clone(), path) {
            Ok(loaded) => Self {
                model_id,
                tokenizer: None,
                model: Arc::new(Mutex::new(Some(Arc::new(loaded)))),
                model_config: config,
                inference_count: AtomicU64::new(0),
                total_generated: AtomicU64::new(0),
                total_time_ms: AtomicU64::new(0),
                start_time: chrono::Utc::now(),
                hallucination: Some(nexora_alignment::hallucination::HallucinationGuard::new(
                    nexora_alignment::hallucination::GuardConfig::default(),
                )),
                identity_cache: OnceLock::new(),
                capabilities_cache: OnceLock::new(),
                config_cache: OnceLock::new(),
            },
            Err(e) => {
                tracing::warn!("Checkpoint load failed (will use random init): {}", e);
                Self::new(model_id)
            }
        }
    }
}

impl std::fmt::Debug for FoundationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_initialized = self
            .model
            .lock()
            .ok()
            .map(|g| g.is_some())
            .unwrap_or(false);
        f.debug_struct("FoundationModel")
            .field("model_id", &self.model_id)
            .field("tokenizer", &self.tokenizer.is_some())
            .field("model_initialized", &model_initialized)
            .finish()
    }
}

impl Clone for FoundationModel {
    fn clone(&self) -> Self {
        Self {
            model_id: self.model_id,
            tokenizer: self.tokenizer.clone(),
            model: Arc::new(Mutex::new(None)),
            model_config: self.model_config.clone(),
            inference_count: AtomicU64::new(self.inference_count.load(Ordering::Relaxed)),
            total_generated: AtomicU64::new(self.total_generated.load(Ordering::Relaxed)),
            total_time_ms: AtomicU64::new(self.total_time_ms.load(Ordering::Relaxed)),
            start_time: self.start_time,
            hallucination: Some(nexora_alignment::hallucination::HallucinationGuard::new(
                nexora_alignment::hallucination::GuardConfig::default(),
            )),
            identity_cache: OnceLock::new(),
            capabilities_cache: OnceLock::new(),
            config_cache: OnceLock::new(),
        }
    }
}

// ── NxrModel trait implementation ─────────────────────────────────────────

#[async_trait]
impl NxrModel for FoundationModel {
    type Config = Value;
    type Metrics = Value;
    type State = Value;

    fn identity(&self) -> &ModelMeta {
        self.identity_cache.get_or_init(|| {
            ModelMeta::new(
                self.model_id,
                model_tier_for(self.model_id),
                "0.1.0".to_string(),
                format!("Foundation model for {}", self.model_id.fullname()),
            )
            .with_parameters(self.model_config.parameter_count() as u64)
            .with_context_window(self.model_config.max_seq_len)
        })
    }

    fn capabilities(&self) -> &CapabilityVector {
        self.capabilities_cache
            .get_or_init(|| CapabilityVector::new(self.model_id))
    }

    fn config(&self) -> &Self::Config {
        self.config_cache.get_or_init(|| {
            serde_json::json!({
                "model": self.model_id.to_string(),
                "parameters": self.model_config.parameter_count(),
                "hidden_size": self.model_config.hidden_size,
                "num_layers": self.model_config.num_layers,
            })
        })
    }

    async fn state(&self) -> Result<Self::State, NxrModelError> {
        let model = self.model.clone();
        let model_id = self.model_id.to_string();
        let inferences = self.inference_count.load(Ordering::Relaxed);
        let status = tokio::task::spawn_blocking(move || {
            model.lock()
                .ok()
                .map(|g| if g.is_some() { "ready" } else { "uninitialized" })
                .unwrap_or("poisoned")
                .to_string()
        })
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("state() blocking task failed: {e}");
            "blocking_failed".to_string()
        });
        Ok(serde_json::json!({
            "status": status,
            "model": model_id,
            "inferences": inferences,
        }))
    }

    async fn initialize(&mut self, _config: Self::Config) -> Result<(), NxrModelError> {
        let model = self.model.clone();
        let config = self.model_config.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().unwrap_or_else(|e| {
                tracing::warn!("Mutex poisoned in initialize, recovering");
                e.into_inner()
            });
            if guard.is_none() {
                match nexora_transformer::resolve_single_backbone() {
                    Ok(backbone) => *guard = Some(backbone),
                    Err(_) => {
                        tracing::warn!("Backbone resolve failed, retrying once...");
                        match nexora_transformer::resolve_single_backbone() {
                            Ok(backbone) => *guard = Some(backbone),
                            Err(_) => {
                                *guard = Some(Arc::new(CausalLM::new(config)));
                                tracing::info!("Backbone retry failed, created standalone model");
                            }
                        }
                    }
                }
            }
        })
        .await
        .map_err(|e| NxrModelError::Inference(format!("initialize blocking task failed: {e}")))?;
        Ok(())
    }

    async fn reset(&self) -> Result<(), NxrModelError> {
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().unwrap_or_else(|e| {
                tracing::warn!("Mutex poisoned in reset, recovering");
                e.into_inner()
            });
            *guard = None;
        })
        .await
        .map_err(|e| NxrModelError::Inference(format!("reset blocking task failed: {e}")))?;
        self.inference_count.store(0, Ordering::Relaxed);
        self.total_generated.store(0, Ordering::Relaxed);
        self.total_time_ms.store(0, Ordering::Relaxed);
        Ok(())
    }

    async fn metrics(&self) -> Result<Self::Metrics, NxrModelError> {
        let count = self.inference_count.load(Ordering::Relaxed);
        let tokens = self.total_generated.load(Ordering::Relaxed);
        let time_ms = self.total_time_ms.load(Ordering::Relaxed);
        Ok(serde_json::json!({
            "inferences": count,
            "tokens_generated": tokens,
            "total_time_ms": time_ms,
            "avg_tokens_per_second": if time_ms > 0 { tokens as f64 / time_ms as f64 * 1000.0 } else { 0.0 },
        }))
    }

    #[instrument(skip_all, fields(model_id = %self.model_id))]
    async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
        let cb_manager = nexora_core::error_recovery::global_error_recovery();
        let cb_breaker = cb_manager
            .get_circuit_breaker("model_inference")
            .await
            .map_err(|e| NxrModelError::Inference(e.to_string()))?;
        if cb_breaker.get_state() == nexora_core::error_recovery::CircuitState::Open {
            cb_breaker.record_failure();
            return Err(NxrModelError::Inference(
                "Circuit breaker 'model_inference' is open — request rejected".to_string(),
            ));
        }

        let text = match &input.data {
            InputData::Text(t) => t.clone(),
            InputData::Tokens(tokens) => self.decode_ids(tokens),
            _ => {
                return Err(NxrModelError::Inference(
                    "Unsupported input type".to_string(),
                ))
            }
        };

        let (max_tokens, temperature, top_k) = extract_params(input);
        let prompt_ids = self.encode_text(&text);

        let model_arc = self.model.clone();
        let (elapsed_ms, output_ids, memory_gb) = tokio::task::spawn_blocking(move || {
            let guard = model_arc.lock().unwrap_or_else(|e| {
                tracing::warn!("Mutex poisoned during infer, recovering — model state may be inconsistent");
                e.into_inner()
            });
            let model = guard.as_deref().ok_or_else(|| {
                NxrModelError::NotInitialized("Model failed to initialize".to_string())
            })?;
            let start = Instant::now();
            let (output_ids, _cache) =
                model.generate(&prompt_ids, max_tokens, temperature, top_k);
            let elapsed_ms = start.elapsed().as_millis() as u64;
            let memory_gb = model.memory_bytes() as f32 / (1024.0 * 1024.0 * 1024.0);
            Ok::<_, NxrModelError>((elapsed_ms, output_ids, memory_gb))
        })
        .await
        .map_err(|e| NxrModelError::Inference(format!("Blocking task failed: {e}")))??;

        let output_text = self.decode_ids(&output_ids);
        let n_tokens = output_ids.len();

        self.inference_count.fetch_add(1, Ordering::Relaxed);
        self.total_generated
            .fetch_add(n_tokens as u64, Ordering::Relaxed);
        self.total_time_ms
            .fetch_add(elapsed_ms, Ordering::Relaxed);

        let mut output = NxrOutput {
            id: uuid::Uuid::new_v4(),
            input_id: input.id,
            timestamp: chrono::Utc::now(),
            data: OutputData::Text(output_text),
            metadata: GenerationMetadata {
                finish_reason: if n_tokens < max_tokens {
                    FinishReason::EndOfSequence
                } else {
                    FinishReason::MaxTokens
                },
                total_tokens: n_tokens,
                generation_time_ms: elapsed_ms,
                model_version: self.identity().version.clone(),
                seed: None,
                extras: HashMap::new(),
            },
            performance: PerformanceMetrics {
                tokens_per_second: if elapsed_ms > 0 {
                    n_tokens as f32 / elapsed_ms as f32 * 1000.0
                } else {
                    0.0
                },
                memory_usage_gb: memory_gb,
                gpu_utilization: None,
                cpu_utilization: 100.0,
                network_usage_mbps: None,
            },
        };

        if let Some(report) = self.run_hallucination_check(input).await {
            let mut meta = std::collections::HashMap::new();
            meta.insert(
                "hallucination_risk".to_string(),
                serde_json::json!(format!("{:?}", report.risk_level)),
            );
            meta.insert(
                "hallucination_score".to_string(),
                serde_json::json!(report.score),
            );
            meta.insert(
                "hallucination_action".to_string(),
                serde_json::json!(format!("{:?}", report.action)),
            );
            output.metadata.extras = meta;
        }

        cb_breaker.record_success();
        Ok(output)
    }

    #[instrument(skip_all, fields(model_id = %self.model_id))]
    async fn infer_stream(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> Result<(), NxrModelError> {
        let text = match &input.data {
            InputData::Text(t) => t.clone(),
            InputData::Tokens(tokens) => self.decode_ids(tokens),
            _ => {
                return Err(NxrModelError::Inference(
                    "Unsupported input type".to_string(),
                ))
            }
        };

        let (max_tokens, temperature, top_k) = extract_params(input);
        let prompt_ids = self.encode_text(&text);
        let input_id = input.id;

        let model_arc = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let cb = callback.clone();

        let count = tokio::task::spawn_blocking(move || {
            let guard = model_arc.lock().unwrap_or_else(|e| {
                tracing::warn!("Mutex poisoned during infer_stream, recovering");
                e.into_inner()
            });
            let model = match guard.as_deref() {
                Some(m) => m,
                None => {
                    return Err(NxrModelError::NotInitialized(
                        "Model not initialized".to_string(),
                    ))
                }
            };

            let mut cache = model.reset_cache();
            for &token_id in &prompt_ids {
                let _ = model.forward(&[token_id], &mut cache);
            }

            let mut last_id = *prompt_ids.last().unwrap_or(&0);
            let mut count = 0usize;

            for pos in 0..max_tokens {
                let logits = model
                    .forward(&[last_id], &mut cache)
                    .unwrap_or_else(|_| Array1::zeros(model.config.vocab_size));
                let next_id = nexora_transformer::sample_token(&logits, temperature, top_k);
                last_id = next_id;

                let token_text = if let Some(ref tk) = tokenizer {
                    tk.read().decode(&[next_id])
                } else {
                    byte_decode(&[next_id])
                };
                let is_final = next_id == 0 || pos == max_tokens - 1;

                cb(NxrStreamChunk {
                    id: uuid::Uuid::new_v4(),
                    input_id,
                    timestamp: chrono::Utc::now(),
                    data: StreamChunkData::TextDelta(token_text),
                    is_final,
                });

                count += 1;
                if next_id == 0 {
                    break;
                }
            }

            Ok::<_, NxrModelError>(count)
        })
        .await
        .map_err(|e| NxrModelError::Inference(format!("Blocking task failed: {e}")))??;

        self.inference_count.fetch_add(1, Ordering::Relaxed);
        self.total_generated
            .fetch_add(count as u64, Ordering::Relaxed);

        if count == 0 {
            callback(NxrStreamChunk {
                id: uuid::Uuid::new_v4(),
                input_id,
                timestamp: chrono::Utc::now(),
                data: StreamChunkData::TextDelta(String::new()),
                is_final: true,
            });
        }

        Ok(())
    }

    async fn update_config(&mut self, config: Self::Config) -> Result<(), NxrModelError> {
        if let Some(v) = config.get("hidden_size").and_then(|v| v.as_u64()) {
            self.model_config.hidden_size = v as usize;
        }
        if let Some(v) = config.get("num_layers").and_then(|v| v.as_u64()) {
            self.model_config.num_layers = v as usize;
        }
        if let Some(v) = config.get("num_heads").and_then(|v| v.as_u64()) {
            self.model_config.num_heads = v as usize;
        }
        if let Some(v) = config.get("num_kv_heads").and_then(|v| v.as_u64()) {
            self.model_config.num_kv_heads = v as usize;
        }
        if let Some(v) = config.get("max_seq_len").and_then(|v| v.as_u64()) {
            self.model_config.max_seq_len = v as usize;
        }
        if let Some(v) = config.get("intermediate_size").and_then(|v| v.as_u64()) {
            self.model_config.intermediate_size = v as usize;
        }
        if let Some(v) = config.get("vocab_size").and_then(|v| v.as_u64()) {
            self.model_config.vocab_size = v as usize;
        }
        if let Some(v) = config.get("rope_theta").and_then(|v| v.as_f64()) {
            self.model_config.rope_theta = v as f32;
        }
        if let Some(v) = config.get("norm_eps").and_then(|v| v.as_f64()) {
            self.model_config.norm_eps = v as f32;
        }
        Ok(())
    }

    async fn validate(&self) -> Result<ValidationResult, NxrModelError> {
        Ok(ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: 1.0,
        })
    }

    async fn statistics(&self) -> Result<ModelStatistics, NxrModelError> {
        let count = self.inference_count.load(Ordering::Relaxed);
        Ok(ModelStatistics {
            total_requests: count,
            successful_requests: count,
            failed_requests: 0,
            avg_response_time_ms: if count > 0 {
                self.total_time_ms.load(Ordering::Relaxed) as f64 / count as f64
            } else {
                0.0
            },
            total_tokens_generated: self.total_generated.load(Ordering::Relaxed),
            uptime_seconds: (chrono::Utc::now() - self.start_time).num_seconds() as u64,
            last_activity: Some(chrono::Utc::now()),
        })
    }

    async fn is_ready(&self) -> bool {
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || {
            model.lock()
                .ok()
                .map(|g| g.is_some())
                .unwrap_or(false)
        })
        .await
        .unwrap_or(false)
    }

    async fn resource_usage(&self) -> Result<ResourceUsage, NxrModelError> {
        let params = self.model_config.parameter_count();
        let mem_gb = (params as f64 * self.model_config.bytes_per_param()) as f32
            / (1024.0 * 1024.0 * 1024.0);
        Ok(ResourceUsage {
            memory_gb: mem_gb,
            cpu_percent: 5.0,
            gpu_percent: None,
            gpu_memory_gb: None,
            disk_gb: 0.0,
            network_mbps: 0.0,
            active_connections: 0,
            queue_size: 0,
        })
    }
}

// ── Type aliases ───────────────────────────────────────────────────────────

pub type NxrOmnisModel = FoundationModel;
pub type NxrVortexModel = FoundationModel;
pub type NxrAetherModel = FoundationModel;
pub type NxrSpectraModel = FoundationModel;
pub type NxrNexumModel = FoundationModel;
pub type NxrAxiomModel = FoundationModel;
pub type NxrCipherModel = FoundationModel;
pub type NxrSwiftModel = FoundationModel;
pub type NxrKronosModel = FoundationModel;
pub type NxrGenesisModel = FoundationModel;
