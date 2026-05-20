//! Batching module for efficient inference

pub mod config;
pub mod processor;
pub mod types;

// Re-export all batching components for easier access
pub use config::{BatchConfig, PaddingStrategy};
pub use processor::{BatchProcessor, ProcessorState};
pub use types::{Batch, BatchItem, BatchStats};
