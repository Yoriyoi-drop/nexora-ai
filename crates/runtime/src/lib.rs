//! Nexora Runtime Infrastructure Layer
//! 
//! Shared runtime utilities for all AI frameworks and services
//! 
//! ## Features

//! 
//! Provides shared runtime utilities for:
//! - Async execution patterns with concurrency control
//! - Priority-based task scheduling
//! - Resource pooling and memory management
//! - Performance monitoring and health checks
//! - Batch processing and request batching
//! - KV cache for inference optimization
//! - Streaming for real-time inference
//! 
//! This layer sits above foundation AI frameworks and provides
//! infrastructure needed for production-grade AI services.

use anyhow::Result as AnyhowResult;
use serde::{Serialize, Deserialize};

// Common type definitions
pub type Result<T> = AnyhowResult<T>;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("Batch error: {0}")]
    BatchError(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Invalid state: {0}")]
    InvalidState(String),
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model_id: String,
    pub inputs: Vec<u8>,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    pub request_id: Option<String>,
    pub input_tokens: Vec<u32>,
    pub target_tokens: Option<Vec<u32>>,
    pub priority: u8,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub model_id: String,
    pub outputs: Vec<u8>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    pub request_id: Option<String>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub token_id: u32,
    pub text: String,
    pub logprob: f32,
    pub is_special: bool,
}

pub mod executor;
pub mod scheduler;
pub mod resource;
pub mod monitoring;
pub mod batching;
pub mod kv_cache;
pub mod streaming;

// Re-export main components
pub use executor::*;
pub use scheduler::*;
pub use resource::*;
pub use monitoring::*;
pub use batching::*;
pub use kv_cache::*;
pub use streaming::*;
