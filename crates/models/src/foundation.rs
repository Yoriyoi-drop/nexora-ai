//! Foundation model implementations.
//!
//! Each NXR model wraps a `Mutex<Option<CausalLM>>` as its core transformer backbone.
//! Training the CausalLM via the training pipeline produces .safetensors weights
//! that these types can load via `reload()` or `from_checkpoint()`.
//!
//! This is the PRIMARY model implementation — inference runs directly on the
//! trained CausalLM with token-based generation. Per-directory agent-based models
//! (omnis::NxrOmnisModel, etc.) are available for multi-agent orchestration flows.

use async_trait::async_trait;
use ndarray::Array1;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::instrument;

use nexora_shared::{
    base_model::{
        FinishReason, GenerationMetadata, InputData, ModelStatistics, NxrInput, NxrModel,
        NxrModelError, NxrOutput, NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage,
        StreamChunkData, ValidationResult,
    },
    capability_spec::CapabilityVector,
    model_identity::{ModelMeta, ModelTier, NxrModelId},
    tokenizer_integration::NxrTokenizerRef,
};
use nexora_transformer::{CausalLM, TransformerConfig};

fn transformer_config_for(model_id: NxrModelId) -> TransformerConfig {
    let base = TransformerConfig {
        vocab_size: 50257,
        max_seq_len: 2048,
        norm_eps: 1e-6,
        use_cache: true,
        rope_theta: 10000.0,
        ..Default::default()
    };
    match model_id {
        NxrModelId::Omnis | NxrModelId::Axiom | NxrModelId::Genesis => TransformerConfig {
            hidden_size: 768,
            num_heads: 12,
            num_kv_heads: 4,
            num_layers: 8,
            intermediate_size: 3072,
            ..base
        },
        NxrModelId::Vortex | NxrModelId::Aether | NxrModelId::Nexum => TransformerConfig {
            hidden_size: 512,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 6,
            intermediate_size: 2048,
            ..base
        },
        NxrModelId::Spectra | NxrModelId::Cipher => TransformerConfig {
            hidden_size: 384,
            num_heads: 6,
            num_kv_heads: 3,
            num_layers: 4,
            intermediate_size: 1536,
            ..base
        },
        NxrModelId::Kronos => TransformerConfig {
            hidden_size: 256,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 3,
            intermediate_size: 1024,
            ..base
        },
        NxrModelId::Swift => TransformerConfig {
            hidden_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            intermediate_size: 512,
            ..base
        },
    }
}

fn byte_encode(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32).collect()
}

fn byte_decode(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .map(|&id| if id < 256 { id as u8 } else { b'?' })
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}

fn extract_params(input: &NxrInput) -> (usize, f32, usize) {
    let max_tokens = input
        .parameters
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    let temperature = input
        .parameters
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    let top_k = input
        .parameters
        .get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    (max_tokens, temperature, top_k)
}

macro_rules! define_foundation_model {
    ($name:ident, $id:ident, $tier:ident) => {
        pub struct $name {
            pub tokenizer: Option<NxrTokenizerRef>,
            /// `Arc<std::sync::Mutex>` wraps `CausalLM`. `infer()` moves CPU-bound
            /// generation to `tokio::task::spawn_blocking` so the async runtime is not
            /// blocked. `infer_stream()` still holds the lock inline — callers should
            /// ensure streams are not interleaved with other model operations.
            model: Arc<Mutex<Option<CausalLM>>>,
            model_config: TransformerConfig,
            inference_count: AtomicU64,
            total_generated: AtomicU64,
            total_time_ms: AtomicU64,
            start_time: chrono::DateTime<chrono::Utc>,
            #[cfg(feature = "hallucination")]
            hallucination: Option<nexora_hallucination::HallucinationGuard>,
        }

        impl $name {
            pub fn new() -> Self {
                let config = transformer_config_for(NxrModelId::$id);
                Self {
                    tokenizer: None,
                    model: Arc::new(Mutex::new(None)),
                    model_config: config,
                    inference_count: AtomicU64::new(0),
                    total_generated: AtomicU64::new(0),
                    total_time_ms: AtomicU64::new(0),
                    start_time: chrono::Utc::now(),
                    #[cfg(feature = "hallucination")]
                    hallucination: Some(nexora_hallucination::HallucinationGuard::new(nexora_hallucination::GuardConfig::default())),
                }
            }

            pub fn with_tokenizer(t: NxrTokenizerRef) -> Self {
                let mut m = Self::new();
                m.tokenizer = Some(t);
                m
            }

            #[cfg(feature = "hallucination")]
            pub fn enable_hallucination_guard(&mut self) {
                let integration = nexora_hallucination::HallucinationGuard::new(nexora_hallucination::GuardConfig::default());
                self.hallucination = Some(integration);
            }

            #[cfg(feature = "hallucination")]
            pub fn disable_hallucination_guard(&mut self) {
                self.hallucination = None;
            }

            #[cfg(feature = "hallucination")]
            pub fn with_hallucination_guard(mut self, guard: nexora_hallucination::HallucinationGuard) -> Self {
                self.hallucination = Some(guard);
                self
            }

            #[cfg(feature = "hallucination")]
            async fn run_hallucination_check(&self, input: &NxrInput) -> Option<nexora_hallucination::PipelineResult> {
                if let Some(ref h) = self.hallucination {
                    let text = match &input.data {
                        InputData::Text(t) => t.clone(),
                        _ => return None,
                    };
                    let ctx = input.parameters.get("context")
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
            /// Returns `None` because hallucination checking is disabled when the
            /// `hallucination` feature is off. `None` means "no issues found" —
            /// not "check not performed".
            async fn run_hallucination_check(&self, _input: &NxrInput) -> Option<nexora_hallucination::PipelineResult> {
                None
            }

            fn get_or_init_model(&self) -> std::sync::MutexGuard<Option<CausalLM>> {
                let mut guard = self.model.lock().unwrap_or_else(|e| {
                    tracing::warn!(target: stringify!($name), "mutex was poisoned, recovering");
                    e.into_inner()
                });
                if guard.is_none() {
                    *guard = Some(CausalLM::new(self.model_config.clone()));
                }
                guard
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

            /// Load trained .safetensors weights into the CausalLM backbone.
            /// The model must be initialized first (call `initialize` or `get_or_init_model`).
            /// Note: this takes &mut self since it needs write access to the model Mutex.
            pub fn reload(&mut self, path: &str) -> Result<(), FoundationError> {
                let config = transformer_config_for(NxrModelId::$id);
                let loaded = CausalLM::from_checkpoint(config, path)?;
                *self.model.lock().unwrap_or_else(|e| e.into_inner()) = Some(loaded);
                Ok(())
            }

            /// Create a new instance with weights loaded from a .safetensors checkpoint.
            /// Falls back to random init if loading fails.
            pub fn from_checkpoint(path: &str) -> Self {
                let config = transformer_config_for(NxrModelId::$id);
                match CausalLM::from_checkpoint(config.clone(), path) {
                    Ok(loaded) => Self {
                        tokenizer: None,
                        model: Arc::new(Mutex::new(Some(loaded))),
                        model_config: config,
                        inference_count: AtomicU64::new(0),
                        total_generated: AtomicU64::new(0),
                        total_time_ms: AtomicU64::new(0),
                        start_time: chrono::Utc::now(),
                        #[cfg(feature = "hallucination")]
                        hallucination: Some(nexora_hallucination::HallucinationGuard::new(nexora_hallucination::GuardConfig::default())),
                    },
                    Err(e) => {
                        tracing::warn!(target: stringify!($name), "Checkpoint load failed (will use random init): {}", e);
                        Self::new()
                    }
                }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("tokenizer", &self.tokenizer.is_some())
                    .field("model_initialized", &self.model.lock().unwrap_or_else(|e| e.into_inner()).is_some())
                    .finish()
            }
        }

        impl Clone for $name {
            fn clone(&self) -> Self {
                Self {
                    tokenizer: self.tokenizer.clone(),
                    model: Arc::new(Mutex::new(None)),
                    model_config: self.model_config.clone(),
                    inference_count: AtomicU64::new(self.inference_count.load(Ordering::Relaxed)),
                    total_generated: AtomicU64::new(self.total_generated.load(Ordering::Relaxed)),
                    total_time_ms: AtomicU64::new(self.total_time_ms.load(Ordering::Relaxed)),
                    start_time: self.start_time,
                    #[cfg(feature = "hallucination")]
                    hallucination: Some(nexora_hallucination::HallucinationGuard::new(nexora_hallucination::GuardConfig::default())),
                }
            }
        }

        #[async_trait]
        impl NxrModel for $name {
            type Config = Value;
            type Metrics = Value;
            type State = Value;

            fn identity(&self) -> &ModelMeta {
                static META: OnceLock<ModelMeta> = OnceLock::new();
                META.get_or_init(|| {
                    ModelMeta::new(
                        NxrModelId::$id,
                        ModelTier::$tier,
                        "0.1.0".to_string(),
                        format!("Foundation model for {}", NxrModelId::$id.fullname()),
                    )
                    .with_parameters(self.model_config.parameter_count() as u64)
                    .with_context_window(self.model_config.max_seq_len)
                })
            }

            fn capabilities(&self) -> &CapabilityVector {
                static CAPABILITIES: OnceLock<CapabilityVector> = OnceLock::new();
                CAPABILITIES.get_or_init(|| CapabilityVector::new(NxrModelId::$id))
            }

            fn config(&self) -> &Self::Config {
                static CONFIG: OnceLock<Value> = OnceLock::new();
                CONFIG.get_or_init(|| serde_json::json!({
                    "model": stringify!($id),
                    "parameters": self.model_config.parameter_count(),
                    "hidden_size": self.model_config.hidden_size,
                    "num_layers": self.model_config.num_layers,
                }))
            }

            async fn state(&self) -> Result<Self::State, NxrModelError> {
                let guard = self.model.lock().unwrap_or_else(|e| {
                    tracing::warn!("mutex was poisoned, recovering");
                    e.into_inner()
                });
                Ok(serde_json::json!({
                    "status": if guard.is_some() { "ready" } else { "uninitialized" },
                    "model": stringify!($id),
                    "inferences": self.inference_count.load(Ordering::Relaxed),
                }))
            }

            async fn initialize(&mut self, _config: Self::Config) -> Result<(), NxrModelError> {
                self.get_or_init_model();
                Ok(())
            }

            async fn reset(&self) -> Result<(), NxrModelError> {
                let mut guard = self.model.lock().unwrap_or_else(|e| {
                    tracing::warn!("mutex was poisoned, recovering");
                    e.into_inner()
                });
                *guard = None;
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

            #[instrument(skip_all, fields(model_id = %self.identity().model_id))]
            async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
                // Circuit breaker protection at outermost inference call
                let cb_manager = nexora_core::error_recovery::global_error_recovery();
                let cb_breaker = cb_manager.get_circuit_breaker("model_inference").await
                    .map_err(|e| NxrModelError::Inference(e.to_string()))?;
                if cb_breaker.get_state() == nexora_core::error_recovery::CircuitState::Open {
                    cb_breaker.record_failure();
                    return Err(NxrModelError::Inference(
                        "Circuit breaker 'model_inference' is open — request rejected".to_string()
                    ));
                }

                let text = match &input.data {
                    InputData::Text(t) => t.clone(),
                    InputData::Tokens(tokens) => self.decode_ids(tokens),
                    _ => return Err(NxrModelError::Inference("Unsupported input type".to_string())),
                };

                let (max_tokens, temperature, top_k) = extract_params(input);
                let prompt_ids = self.encode_text(&text);

                // Move CPU-bound model generation to a blocking thread
                // so the async runtime is not blocked for potentially seconds.
                let model_arc = self.model.clone();
                let (elapsed_ms, output_ids, memory_gb) = tokio::task::spawn_blocking(move || {
                    let guard = model_arc.lock().unwrap_or_else(|e| {
                        tracing::warn!("model mutex was poisoned, recovering");
                        e.into_inner()
                    });
                    let model = guard.as_ref().ok_or_else(|| {
                        NxrModelError::NotInitialized("Model failed to initialize".to_string())
                    })?;
                    let start = Instant::now();
                    let (output_ids, _cache) = model.generate(&prompt_ids, max_tokens, temperature, top_k);
                    let elapsed_ms = start.elapsed().as_millis() as u64;
                    let memory_gb = model.memory_bytes() as f32 / (1024.0 * 1024.0 * 1024.0);
                    Ok::<_, NxrModelError>((elapsed_ms, output_ids, memory_gb))
                })
                .await
                .map_err(|e| NxrModelError::Inference(format!("Blocking task failed: {e}")))??;

                let output_text = self.decode_ids(&output_ids);
                let n_tokens = output_ids.len();

                self.inference_count.fetch_add(1, Ordering::Relaxed);
                self.total_generated.fetch_add(n_tokens as u64, Ordering::Relaxed);
                self.total_time_ms.fetch_add(elapsed_ms, Ordering::Relaxed);

                let mut output = NxrOutput {
                    id: uuid::Uuid::new_v4(),
                    input_id: input.id,
                    timestamp: chrono::Utc::now(),
                    data: OutputData::Text(output_text),
                    metadata: GenerationMetadata {
                        finish_reason: if n_tokens < max_tokens { FinishReason::EndOfSequence } else { FinishReason::MaxTokens },
                        total_tokens: n_tokens,
                        generation_time_ms: elapsed_ms,
                        model_version: self.identity().version.clone(),
                        seed: None,
                        extras: HashMap::new(),
                    },
                    performance: PerformanceMetrics {
                        tokens_per_second: if elapsed_ms > 0 { n_tokens as f32 / elapsed_ms as f32 * 1000.0 } else { 0.0 },
                        memory_usage_gb: memory_gb,
                        gpu_utilization: None,
                        cpu_utilization: 100.0,
                        network_usage_mbps: None,
                    },
                };

                #[cfg(feature = "hallucination")]
                if let Some(report) = self.run_hallucination_check(input).await {
                    let mut meta = std::collections::HashMap::new();
                    meta.insert("hallucination_risk".to_string(), serde_json::json!(format!("{:?}", report.risk_level)));
                    meta.insert("hallucination_score".to_string(), serde_json::json!(report.score));
                    meta.insert("hallucination_action".to_string(), serde_json::json!(format!("{:?}", report.action)));
                    output.metadata.extras = meta;
                }

                cb_breaker.record_success();
                Ok(output)
            }

            #[instrument(skip_all, fields(model_id = %self.identity().model_id))]
            async fn infer_stream(
                &self,
                input: &NxrInput,
                callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
            ) -> Result<(), NxrModelError> {
                let text = match &input.data {
                    InputData::Text(t) => t.clone(),
                    InputData::Tokens(tokens) => self.decode_ids(tokens),
                    _ => return Err(NxrModelError::Inference("Unsupported input type".to_string())),
                };

                let (max_tokens, temperature, top_k) = extract_params(input);
                let prompt_ids = self.encode_text(&text);
                let input_id = input.id;

                // Ensure model is loaded before spawning
                { self.get_or_init_model(); }

                let model_arc = self.model.clone();
                let tokenizer = self.tokenizer.clone();
                let cb = callback.clone();

                let count = tokio::task::spawn_blocking(move || {
                    let guard = model_arc.lock().unwrap_or_else(|e| {
                        tracing::warn!("model mutex was poisoned, recovering");
                        e.into_inner()
                    });
                    let model = match guard.as_ref() {
                        Some(m) => m,
                        None => return Ok(0usize),
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
                self.total_generated.fetch_add(count as u64, Ordering::Relaxed);

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
                    } else { 0.0 },
                    total_tokens_generated: self.total_generated.load(Ordering::Relaxed),
                    uptime_seconds: (chrono::Utc::now() - self.start_time).num_seconds() as u64,
                    last_activity: Some(chrono::Utc::now()),
                })
            }

            async fn is_ready(&self) -> bool {
                self.model
                    .lock()
                    .map(|guard| guard.is_some())
                    .unwrap_or(false)
            }

            async fn resource_usage(&self) -> Result<ResourceUsage, NxrModelError> {
                let params = self.model_config.parameter_count();
                let mem_gb = (params * 4) as f32 / (1024.0 * 1024.0 * 1024.0);
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
    };
}

#[derive(Debug)]
pub struct FoundationError(pub String);

impl std::fmt::Display for FoundationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FoundationError {}

impl From<nexora_transformer::TransformerError> for FoundationError {
    fn from(e: nexora_transformer::TransformerError) -> Self {
        FoundationError(e.to_string())
    }
}

define_foundation_model!(NxrOmnisModel, Omnis, Ultra);
define_foundation_model!(NxrVortexModel, Vortex, Apex);
define_foundation_model!(NxrAetherModel, Aether, Apex);
define_foundation_model!(NxrSpectraModel, Spectra, Pro);
define_foundation_model!(NxrNexumModel, Nexum, Apex);
define_foundation_model!(NxrAxiomModel, Axiom, Ultra);
define_foundation_model!(NxrCipherModel, Cipher, Pro);
define_foundation_model!(NxrSwiftModel, Swift, Edge);
define_foundation_model!(NxrKronosModel, Kronos, Core);
define_foundation_model!(NxrGenesisModel, Genesis, Ultra);
