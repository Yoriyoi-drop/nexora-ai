//! Nexora Inference Engine
//! 
//! Mesin inference utama untuk sistem Nexora AI.

use std::sync::Arc;

pub mod batching;
pub mod engine;
pub mod runtime;
pub mod session;
pub mod decoding;
pub mod beam_search;
pub mod speculative_decoding;
pub mod sampler;
pub mod stop_conditions;
pub mod token_loop;
pub mod latency;
pub mod metrics;
pub mod scheduler;
pub mod kv_cache;
pub mod prefix_cache;
pub mod paged_cache;
pub mod streaming;
pub mod blaa_integration;
pub mod inference_trait;
pub mod sequence_state;
pub mod continuous_batching;

// Re-export main types
pub use engine::{InferenceEngine as InferenceEngineStruct, InferenceConfig};
pub use inference_trait::InferenceEngine;
pub use runtime::{InferenceRuntime, RuntimeState};
pub use session::{InferenceSession, SessionConfig, SessionState};
pub use decoding::{DecodingStrategy, DecodingConfig};
pub use beam_search::{BeamSearchConfig, BeamHypothesis};
pub use sampler::{Sampler, SamplingConfig, SamplingMethod};
pub use stop_conditions::{StopCondition, StopConditions};
pub use token_loop::{TokenLoop, TokenLoopConfig};
pub use latency::{LatencyTracker, LatencyStats};
pub use metrics::{InferenceMetrics, MetricsCollector};
pub use blaa_integration::{BlaaInferenceEngine, BlaaEmbeddingsEngine};
pub use prefix_cache::{PrefixCache, PrefixCacheConfig, PrefixMatch};
pub use paged_cache::{PagedKVCache, PagedCacheConfig, PagedCacheStats, BlockTable};
pub use sequence_state::{Sequence, SeqState};
pub use continuous_batching::{ContinuousBatchingEngine, StepResult};

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
    #[serde(serialize_with = "serialize_arc_str", deserialize_with = "deserialize_arc_str")]
    pub token_text: Arc<str>,
    /// Log probability
    pub log_prob: f32,
    /// Token position
    pub position: usize,
    /// Token metadata
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

fn serialize_arc_str<S: serde::Serializer>(v: &Arc<str>, s: S) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(v)
}

fn deserialize_arc_str<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Arc<str>, D::Error> {
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
