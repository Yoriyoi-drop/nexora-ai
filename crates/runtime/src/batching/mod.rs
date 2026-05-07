//! Batching module for efficient inference

pub mod config;
pub mod types;
pub mod processor;

// Re-export all batching components for easier access
pub use config::{BatchConfig, PaddingStrategy};
pub use types::{BatchItem, Batch, BatchStats};
pub use processor::{BatchProcessor, ProcessorState};
