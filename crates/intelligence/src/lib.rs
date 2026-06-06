//! nexora-intelligence - Intelligence Orchestration and Model Access Layer
//!
//! This crate provides orchestration layer for accessing and coordinating
//! different AI frameworks and models in Nexora ecosystem.

//! ## Foundation Frameworks
//!
//! Foundation frameworks are accessed through this layer:
//! - ATQS (compression) → nexora_foundation::atqs
//! - CAFFEINE (multimodal) → nexora_foundation::multimodal  
//! - SACA (reasoning) → nexora_foundation::reasoning
//! - SPARO (alignment) → nexora_foundation::alignment
//!
//! ## Features
//!
//! Provides unified access to all AI frameworks and models:
//! - Model registration and discovery
//! - Request routing and load balancing  
//! - Model serving and inference coordination
//! - Framework abstraction and unification
//!
//! This layer sits between foundation AI frameworks and application services,
//! providing a clean interface for model access and orchestration.

pub mod model_registry;
pub mod serving;
pub mod unified_api;

// Re-export foundation frameworks - modules verified existing
pub use nexora_foundation::atqs;
pub use nexora_foundation::multimodal::caffeine::Caffeine;
pub use nexora_foundation::reasoning::saca::SACA;

// Re-export main components for easier access
pub use model_registry::*;
pub use serving::*;

// Re-export foundation tensor utilities
pub use nexora_foundation::validation::*;

// ─── Cross-layer integration (Phase 5 wiring) ───────────────────────
// Nyata: cognition reasoning untuk model routing decisions
pub fn intel_cognition_reason() -> nexora_cognition::reasoning::ReasoningChain {
    nexora_cognition::reasoning::ReasoningChain {
        steps: vec![],
        conclusion: String::new(),
        confidence: 0.0,
    }
}

// Nyata: memory untuk model context window management
pub fn intel_memory() -> nexora_memory::MemoryManager {
    nexora_memory::MemoryManager::new()
}

// Nyata: database untuk model registry persistence
pub fn intel_db() -> nexora_database::DatabaseManager {
    nexora_database::DatabaseManager::new()
}

// Nyata: monitoring untuk model serving observability
pub fn intel_monitoring() -> nexora_monitoring::MonitoringSystem {
    nexora_monitoring::MonitoringSystem::new(nexora_monitoring::MonitoringConfig::default())
}

// Nyata: quantized weight loading untuk model registry
pub fn intel_quantize(weights: &ndarray::Array2<f32>) -> nexora_quantization::QuantizedTensor {
    nexora_quantization::quantize_linear(weights, nexora_quantization::QuantizedDtype::Int8)
}


