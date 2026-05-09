//! NXR-AXIOM Agents Module
//! 
//! Individual agent implementations for logic and truth

pub mod axiom_prime;
pub mod logic_core;
pub mod truth_validator;
pub mod axiom_discoverer;

// Re-export all agents
pub use axiom_prime::*;
pub use logic_core::*;
pub use truth_validator::*;
pub use axiom_discoverer::*;
