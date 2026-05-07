//! Foundation Traits - Core trait definitions for modular framework
//!
//! This crate defines the core traits that all foundation frameworks must implement.
//! This enables true modularity and backend swapping without rewriting the system.

use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::pin::Pin;
use std::future::Future;

pub mod reasoning;
pub mod compression;
pub mod memory;
pub mod scheduler;
pub mod tokenizer;
pub mod multimodal;

pub use reasoning::*;
pub use compression::*;
pub use memory::*;
pub use scheduler::*;
pub use tokenizer::*;
pub use multimodal::*;

/// Common error type for all foundation traits
pub type FoundationResult<T> = Result<T, FoundationError>;

#[derive(Debug, thiserror::Error)]
pub enum FoundationError {
    #[error("Implementation error: {0}")]
    Implementation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Resource error: {0}")]
    Resource(String),
    
    #[error("Timeout error")]
    Timeout,
    
    #[error("Not implemented")]
    NotImplemented,
}
