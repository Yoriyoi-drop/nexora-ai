//! NXR-NEXUM Agents Module
//! 
//! Individual agent implementations for multi-agent orchestration

pub mod orchestrator_prime;
pub mod consensus_builder;
pub mod alignment_arbiter;
pub mod resource_optimizer;

// Re-export all agents
pub use orchestrator_prime::*;
pub use consensus_builder::*;
pub use alignment_arbiter::*;
pub use resource_optimizer::*;
