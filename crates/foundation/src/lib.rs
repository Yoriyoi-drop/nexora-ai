//! Foundation AI Components for Nexora
//!
//! Shared tensor operations, validation, and core utilities
//! for all AI frameworks in the Nexora ecosystem.
//! Now includes NXR Model Series foundation implementations.

pub use nexora_foundation_types::{FoundationError, FoundationResult};

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

// Nyata: wire monitoring system for observability (Phase 5d)
fn _init_monitoring() -> nexora_monitoring::MonitoringSystem {
    let cfg = nexora_monitoring::MonitoringConfig::default();
    nexora_monitoring::MonitoringSystem::new(cfg)
}

// Nyata: wire memory manager for memory-augmented generation
fn _init_memory() -> nexora_memory::MemoryManager {
    nexora_memory::MemoryManager::new()
}

// Nyata: wire isolation orchestrator for pre-inference security checks
fn _init_isolation(cfg: nexora_isolation::config::IsolationConfig) -> nexora_isolation::IsolationOrchestrator {
    nexora_isolation::IsolationOrchestrator::new(cfg)
}

// Nyata: wire database manager for model checkpoint persistence
async fn _init_database() -> anyhow::Result<nexora_database::DatabaseManager> {
    Ok(nexora_database::DatabaseManager::new())
}

// Nyata: wire utils manager for text processing, crypto, validation
fn _init_utils() -> nexora_utils::UtilsManager {
    nexora_utils::UtilsManager::default()
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
}
