//! Nexora Runtime Infrastructure Layer
//! 
//! Shared runtime utilities for all AI frameworks and services

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

/// Runtime infrastructure layer
/// 
/// Provides shared runtime utilities for:
/// - Async execution patterns with concurrency control
/// - Priority-based task scheduling
/// - Resource pooling and memory management
/// - Performance monitoring and health checks
/// - Batch processing and request batching
/// - KV cache for inference optimization
/// - Streaming for real-time inference
/// 
/// This layer sits above foundation AI frameworks and provides
/// infrastructure needed for production-grade AI services.
