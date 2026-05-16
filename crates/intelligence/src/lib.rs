//! nexora-intelligence - Intelligence Orchestration and Model Access Layer
//! 
//! This crate provides orchestration layer for accessing and coordinating
//! different AI frameworks and models in Nexora ecosystem.

//! ## Foundation Frameworks
//! 
//! Foundation frameworks are accessed through this layer:
//! - ATQS (compression) → nexora_foundation::compression
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
pub use nexora_foundation::reasoning::saca::SACA;
pub use nexora_foundation::compression;
pub use nexora_foundation::multimodal::caffeine::Caffeine;

// Re-export main components for easier access
pub use model_registry::*;
pub use serving::*;

// Re-export foundation tensor utilities
pub use nexora_foundation::traits::tensor_traits::*;
pub use nexora_foundation::validation::*;
