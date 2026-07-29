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
use serde::{Deserialize, Serialize};

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

pub mod batching;
pub mod executor;
pub mod kv_cache;
pub mod monitoring;
pub mod resource;
pub mod scheduler;
pub mod vram_budget;
pub mod scheduler_trait;
pub mod streaming;
pub mod gpu_runtime;

// Scheduler-v2 module (merged from nexora-scheduler-v2 crate)
pub mod scheduler_v2;

// Distributed scheduler modules — Phase 5c+
pub mod cluster;
pub mod gossip;
pub mod distributed;

// Re-export main components
pub use batching::*;
pub use executor::*;
pub use kv_cache::*;
pub use monitoring::*;
pub use resource::*;
pub use scheduler::*;
pub use vram_budget::*;
pub use scheduler_trait::Scheduler;
pub use streaming::*;
pub use cluster::*;
pub use distributed::*;

// ─── Cross-layer integration (Phase 5 wiring) ───────────────────────
// Nyata: core interaction untuk scheduler
pub fn runtime_core() {
    let _config = nexora_core::CoreController::new();
}

// Nyata: monitoring untuk scheduler observability
pub fn runtime_monitoring() -> nexora_monitoring::MonitoringSystem {
    nexora_monitoring::MonitoringSystem::new(nexora_monitoring::MonitoringConfig::default())
}

// Nyata: utils untuk time formatting dan performance
pub fn runtime_utils() -> nexora_utils::UtilsManager {
    nexora_utils::UtilsManager::default()
}
