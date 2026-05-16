//! Foundation AI Components for Nexora
//! 
//! Shared tensor operations, validation, and core utilities
//! for all AI frameworks in the Nexora ecosystem.
//! Now includes NXR Model Series foundation implementations.

#![allow(deprecated)]

// Include framework modules

pub mod validation;
pub mod safetensors;
pub mod reasoning;
pub mod atqs;
pub mod compression;
pub mod multimodal;

// Include research modules
pub mod has_moe_ffn;
pub mod oracle;
pub mod alignment;
pub mod traits;

// Include HLDVA-T module
pub mod hldva_t;

// Include VOGP+ module
pub mod vogp;

// Include ERP module
pub mod erp;

// Include training module
pub mod training;

// Include NXR Model Series
pub mod shared;
pub mod models;
pub mod clustering_orchestrator;

// Re-export main components for easier access

pub use validation::*;
pub use hldva_t::*;
pub use vogp::*;
pub use erp::*;

// Re-export external framework modules
pub use crate::reasoning::*;
pub use crate::compression::*;
pub use crate::multimodal::*;

// Re-export NXR Model Series components
pub use shared::*;
pub use models::*;

// Define FoundationResult directly to avoid nested structure issues
pub type FoundationResult<T> = std::result::Result<T, FoundationError>;

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
