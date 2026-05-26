//! Foundation AI Components for Nexora
//!
//! Shared tensor operations, validation, and core utilities
//! for all AI frameworks in the Nexora ecosystem.
//! Now includes NXR Model Series foundation implementations.

// Include framework modules

pub mod atqs;
pub mod reasoning;
pub mod safetensors;
pub mod validation;
pub mod validation_modules;

pub mod multimodal;

// Include research modules
pub mod alignment;
pub mod has_moe_ffn;
pub mod oracle;

// Include HLDVA-T module
pub mod hldva_t;

// Include VOGP+ module
pub mod vogp;

// Include ERP module
pub mod erp;

// Include training module
pub mod training;

// Include NXR Model Series
pub mod causal_lm_model;
pub mod clustering_orchestrator;
pub mod echo_net_injector;
#[cfg(feature = "gpu")]
pub mod gpu_cluster_ops;
pub mod init;
pub mod model_agent_manager;
pub mod models;
pub mod quantization;
pub mod shared;

// Re-export main components for easier access

pub use erp::*;
pub use hldva_t::*;
pub use validation::*;
pub use vogp::*;

// Re-export external framework modules
pub use crate::atqs::*;
pub use crate::multimodal::*;
pub use crate::reasoning::*;

// Re-export NXR Model Series components
pub use models::*;
pub use shared::*;

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

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Timeout error")]
    Timeout,
}

impl From<nexora_transformer::TransformerError> for FoundationError {
    fn from(e: nexora_transformer::TransformerError) -> Self {
        FoundationError::Processing(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foundation_error_implementation() {
        let e = FoundationError::Implementation("test".into());
        assert_eq!(e.to_string(), "Implementation error: test");
    }

    #[test]
    fn test_foundation_error_configuration() {
        let e = FoundationError::Configuration("bad config".into());
        assert_eq!(e.to_string(), "Configuration error: bad config");
    }

    #[test]
    fn test_foundation_error_processing() {
        let e = FoundationError::Processing("error".into());
        assert_eq!(e.to_string(), "Processing error: error");
    }

    #[test]
    fn test_foundation_error_timeout() {
        let e = FoundationError::Timeout;
        assert_eq!(e.to_string(), "Timeout error");
    }

    #[test]
    fn test_foundation_error_from_transformer_error() {
        let te = nexora_transformer::TransformerError::Configuration("invalid".into());
        let fe: FoundationError = te.into();
        assert!(fe.to_string().contains("invalid"));
    }
}
