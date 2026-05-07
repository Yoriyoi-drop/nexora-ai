//! nexora-intelligence - Intelligence Orchestration and Model Access Layer
//! 
//! This crate provides the orchestration layer for accessing and coordinating
//! different AI frameworks and models in the Nexora ecosystem.
//! 
//! Foundation frameworks are accessed through this layer:
//! - ATQS (compression) → nexora_foundation::compression
//! - CAFFEINE (multimodal) → nexora_foundation::multimodal  
//! - SACA (reasoning) → nexora_foundation::reasoning
//! - SPARO (alignment) → nexora_foundation::alignment

pub mod model_registry;
pub mod serving;

// Re-export main components for easier access
pub use model_registry::*;
pub use serving::*;

// Re-export foundation frameworks through orchestration layer
pub use nexora_foundation::compression::*;
pub use nexora_foundation::multimodal::*;
pub use nexora_foundation::reasoning::*;
pub use nexora_foundation::alignment::*;

// Re-export foundation tensor utilities
pub use nexora_foundation::tensor::*;

/// Intelligence Orchestration Layer
/// 
/// Provides unified access to all AI frameworks and models:
/// - Model registration and discovery
/// - Request routing and load balancing  
/// - Model serving and inference coordination
/// - Framework abstraction and unification
/// 
/// This layer sits between foundation AI frameworks and application services,
/// providing a clean interface for model access and orchestration.

