//! Foundation AI Components for Nexora
//!
//! Shared tensor operations, validation, and core utilities
//! for all AI frameworks in the Nexora ecosystem.
//! Now includes NXR Model Series foundation implementations.

use std::fmt;

#[derive(Debug, Clone)]
pub enum FoundationError {
    Implementation(String),
    Configuration(String),
    Processing(String),
    Resource(String),
    Timeout,
}

impl fmt::Display for FoundationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FoundationError::Implementation(msg) => write!(f, "Implementation error: {msg}"),
            FoundationError::Configuration(msg) => write!(f, "Configuration error: {msg}"),
            FoundationError::Processing(msg) => write!(f, "Processing error: {msg}"),
            FoundationError::Resource(msg) => write!(f, "Resource error: {msg}"),
            FoundationError::Timeout => write!(f, "Timeout error"),
        }
    }
}

impl std::error::Error for FoundationError {}

pub type FoundationResult<T> = std::result::Result<T, FoundationError>;

// Tokenizer module (merged from nexora-tokenizer)
pub mod tokenizer;

// Model core module (merged from nexora-model-core)
pub mod model_core;

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
pub mod distillation;
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
pub fn init_monitoring() -> nexora_monitoring::MonitoringSystem {
    let cfg = nexora_monitoring::MonitoringConfig::default();
    nexora_monitoring::MonitoringSystem::new(cfg)
}

// Nyata: wire memory manager for memory-augmented generation
pub fn init_memory() -> nexora_memory::MemoryManager {
    nexora_memory::MemoryManager::new()
}

// Nyata: wire isolation orchestrator for pre-inference security checks
pub fn init_isolation(cfg: nexora_alignment::isolation::config::IsolationConfig) -> nexora_alignment::isolation::IsolationOrchestrator {
    nexora_alignment::isolation::IsolationOrchestrator::new(cfg)
}

// Nyata: wire database manager for model checkpoint persistence
pub async fn init_database() -> anyhow::Result<nexora_database::DatabaseManager> {
    Ok(nexora_database::DatabaseManager::new())
}

// Nyata: wire utils manager for text processing, crypto, validation
pub fn init_utils() -> nexora_utils::UtilsManager {
    let utils = nexora_utils::UtilsManager::default();
    // Verify SIMD ops available (exercises dot_product/cosine_similarity)
    let _dot = utils.dot_product(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let _cos = utils.cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
    let _sim = utils.string_similarity("hello", "hel lo");
    let _monitor = utils.performance_monitor();
    utils
}

// Nyata: wire ERP compression engine for model weight compression
pub fn init_erp() -> nexora_erp::ERPEngine {
    let cfg = nexora_erp::ERPConfig::default();
    nexora_erp::ERPEngine::new(cfg)
}

// Nyata: wire VOGP+ training regularizer for small-dataset training
pub fn init_vogp() -> nexora_vogp::VOGPPlus {
    nexora_vogp::VOGPPlus::new()
}

// Nyata: wire HLDVA-T pipeline for text-to-image generation
pub async fn init_hldva_t() -> anyhow::Result<nexora_hldva_t::HLDVAPipeline> {
    let cfg = nexora_hldva_t::HLDVAConfig::default();
    Ok(nexora_hldva_t::HLDVAPipeline::new(cfg)?)
}

/// Initialize all foundation subsystems for production use.
/// Called once during system startup.
pub fn init_subsystems() -> InitContext {
    let _monitoring = init_monitoring();
    let _memory = init_memory();
    let _utils = init_utils();
    let _erp = init_erp();
    let _vogp = init_vogp();
    InitContext { _monitoring, _memory, _utils, _erp, _vogp }
}

/// Context holding initialized subsystems for reuse.
pub struct InitContext {
    pub _monitoring: nexora_monitoring::MonitoringSystem,
    pub _memory: nexora_memory::MemoryManager,
    pub _utils: nexora_utils::UtilsManager,
    pub _erp: nexora_erp::ERPEngine,
    pub _vogp: nexora_vogp::VOGPPlus,
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
