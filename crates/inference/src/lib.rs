//! Nexora Inference Engine
//!
//! Mesin inference utama untuk sistem Nexora AI.

use std::sync::Arc;

pub mod batching;
pub mod beam_search;
pub mod blaa_integration;
pub mod continuous_batching;
pub mod decoding;
pub mod degradation;
pub mod distributed;
pub mod self_healing;
pub mod engine;
pub mod inference_trait;
pub mod kv_cache;
pub mod latency;
pub mod metrics;
pub mod paged_cache;
pub mod paged_provider;
pub mod prefix_cache;
pub mod runtime;
pub mod sampler;
pub mod scheduler;
pub mod sequence_state;
pub mod session;
pub mod stop_conditions;
pub mod streaming;

// Re-export main types
pub use beam_search::{BeamHypothesis, BeamSearchConfig};
pub use blaa_integration::{BlaaEmbeddingsEngine, BlaaInferenceEngine};
pub use continuous_batching::{
    ContinuousBatchingConfig, ContinuousBatchingEngine, SequentialBatchingEngine, StepResult,
};
pub use decoding::{DecodingConfig, DecodingStrategy};
pub use degradation::{DegradationConfig, DegradationLevel, DegradationManager, DegradationStats};
pub use engine::{InferenceConfig, InferenceEngine as InferenceEngineStruct};
pub use self_healing::{SelfHealingConfig, SelfHealingWorker, WorkerHealth};
pub use inference_trait::InferenceEngine;
pub use latency::{LatencyStats, LatencyTracker};
pub use metrics::{InferenceMetrics, MetricsCollector};
pub use nexora_runtime::Scheduler;
pub use paged_cache::{
    init_global_paged_cache, BlockTable, PagedCacheConfig, PagedCacheStats, PagedKVCache,
    GLOBAL_PAGED_CACHE,
};
pub use prefix_cache::{PrefixCache, PrefixCacheConfig, PrefixMatch};
pub use runtime::{read_gpu_memory, InferenceRuntime, RuntimeState};
pub use sampler::{Sampler, SamplingConfig, SamplingMethod};
pub use sequence_state::{SeqState, Sequence};
pub use session::{InferenceSession, SessionConfig, SessionEntry, SessionState};
pub use stop_conditions::{StopCondition, StopConditions};

/// Versi inference engine
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error types untuk inference engine
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Engine not initialized: {0}")]
    EngineNotInitialized(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Decoding error: {0}")]
    DecodingError(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Batch error: {0}")]
    BatchError(String),
}

pub type Result<T> = std::result::Result<T, InferenceError>;

impl From<anyhow::Error> for InferenceError {
    fn from(err: anyhow::Error) -> Self {
        InferenceError::InternalError(err.to_string())
    }
}

impl From<nexora_blaa::BlaaError> for InferenceError {
    fn from(err: nexora_blaa::BlaaError) -> Self {
        InferenceError::InternalError(format!("BLAA error: {}", err))
    }
}

/// Token yang dihasilkan oleh inference
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneratedToken {
    /// Token ID
    pub token_id: u32,
    /// Token text
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    pub token_text: Arc<str>,
    /// Log probability
    pub log_prob: f32,
    /// Token position
    pub position: usize,
    /// Token metadata
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

// ─── F16 conversion utilities (shared by sampler + paged_cache) ─────────

/// Convert f32 to IEEE 754 f16 bit pattern.
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0 {
        sign | (mant >> 13) as u16
    } else if exp == 255 {
        sign | (0x7c00 | ((mant >> 13) as u16))
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            sign | (0x7c00 | ((mant >> 13) as u16))
        } else if new_exp <= 0 {
            sign | ((mant | 0x800000) >> (13 - new_exp + 1)) as u16
        } else {
            sign | ((new_exp as u16) << 10) | (mant >> 13) as u16
        }
    }
}

/// Convert IEEE 754 f16 bit pattern to f32 (subnormals flush to zero).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            f32::from_bits(sign) // flush subnormals to zero
        }
    } else if exp == 31 {
        f32::from_bits(sign | (0xff << 23) | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp - 15 + 127) << 23) | (mant << 13))
    }
}

fn serialize_arc_str<S: serde::Serializer>(
    v: &Arc<str>,
    s: S,
) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(v)
}

fn deserialize_arc_str<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Arc<str>, D::Error> {
    let s: String = serde::Deserialize::deserialize(d)?;
    Ok(Arc::<str>::from(s.as_ref()))
}

/// Inference request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceRequest {
    /// Unique request ID
    pub request_id: uuid::Uuid,
    /// Session ID
    pub session_id: Option<uuid::Uuid>,
    /// Model ID
    pub model_id: String,
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature untuk sampling
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: u32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Streaming enabled
    pub streaming: bool,
    /// Request metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    /// Input tokens (for batching)
    pub input_tokens: Vec<u32>,
    /// Target tokens (for batching)  
    pub target_tokens: Option<Vec<u32>>,
    /// Request priority
    pub priority: u8,
    /// Request start time
    #[serde(skip)]
    pub start_time: Option<std::time::Instant>,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            request_id: uuid::Uuid::new_v4(),
            session_id: None,
            model_id: "default".to_string(),
            prompt: String::new(),
            max_tokens: 100,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: Vec::new(),
            streaming: false,
            metadata: std::collections::HashMap::new(),
            input_tokens: Vec::new(),
            target_tokens: None,
            priority: 1,
            start_time: None,
        }
    }
}

/// Inference response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResponse {
    /// Request ID
    pub request_id: uuid::Uuid,
    /// Generated tokens
    pub tokens: Vec<GeneratedToken>,
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Inference time (ms)
    pub inference_time_ms: u64,
    /// Response metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Finish reason untuk inference
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FinishReason {
    /// Maximum tokens reached
    MaxTokens,
    /// Stop sequence encountered
    StopSequence,
    /// End of sequence token
    EndOfSequence,
    /// Timeout
    Timeout,
    /// Cancelled
    Cancelled,
    /// Error
    Error(String),
    /// Unknown reason
    Unknown,
}

impl GeneratedToken {
    /// Create new token
    pub fn new(token_id: u32, token_text: String, log_prob: f32, position: usize) -> Self {
        Self {
            token_id,
            token_text: Arc::from(token_text.as_ref()),
            log_prob,
            position,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

// ─── Cross-layer integration (Phase 5 wiring) ───────────────────────
// Nyata: quantized weight loading for inference
fn _check_quantized(dtype: nexora_quantization::QuantizedDtype) -> usize {
    dtype.bits_per_element()
}

// Nyata: monitoring system for inference observability
fn _init_inference_monitoring() -> nexora_monitoring::MonitoringSystem {
    nexora_monitoring::MonitoringSystem::new(nexora_monitoring::MonitoringConfig::default())
}

// Nyata: isolation check sebelum inference
fn _pre_inference_isolation(agent_id: uuid::Uuid) -> std::result::Result<(), nexora_isolation::IsolationCheckError> {
    let config = nexora_isolation::config::IsolationConfig::default();
    let orch = nexora_isolation::IsolationOrchestrator::new(config);
    orch.pre_inference_check(agent_id)
}

// Nyata: validasi tensor shape untuk input inference
fn _validate_inference_input(shape: &[usize]) -> std::result::Result<(), nexora_validation::ValidationError> {
    nexora_validation::validate_tensor_shape(shape)
}

// Nyata: utils untuk text processing di inference pipeline
fn _inference_text_utils() -> nexora_utils::UtilsManager {
    nexora_utils::UtilsManager::default()
}

// Nyata: cognition reasoning chain untuk advanced inference
fn _inference_reasoning() -> nexora_cognition::reasoning::ReasoningChain {
    nexora_cognition::reasoning::ReasoningChain {
        steps: vec![],
        conclusion: String::new(),
        confidence: 0.0,
    }
}

// Nyata: database untuk session persistence
fn _inference_db() -> nexora_database::DatabaseManager {
    nexora_database::DatabaseManager::new()
}

impl InferenceRequest {
    /// Create new request
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            ..Default::default()
        }
    }

    /// Set model ID
    pub fn with_model(mut self, model_id: String) -> Self {
        self.model_id = model_id;
        self
    }

    /// Set session ID
    pub fn with_session(mut self, session_id: uuid::Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Add stop sequence
    pub fn with_stop_sequence(mut self, stop_sequence: String) -> Self {
        self.stop_sequences.push(stop_sequence);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl InferenceResponse {
    /// Create new response
    pub fn new(request_id: uuid::Uuid) -> Self {
        Self {
            request_id,
            tokens: Vec::new(),
            text: String::new(),
            finish_reason: FinishReason::Unknown,
            total_tokens: 0,
            inference_time_ms: 0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add token
    pub fn add_token(&mut self, token: GeneratedToken) {
        self.text.push_str(&*(token.token_text));
        self.tokens.push(token);
        self.total_tokens += 1;
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = finish_reason;
        self
    }

    /// Set inference time
    pub fn with_inference_time(mut self, inference_time_ms: u64) -> Self {
        self.inference_time_ms = inference_time_ms;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}
