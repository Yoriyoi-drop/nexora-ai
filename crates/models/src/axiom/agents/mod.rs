//! NXR-AXIOM Agents Module
//!
//! Individual agent implementations for logic and truth

pub mod axiom_discoverer;
pub mod axiom_prime;
pub mod logic_core;
pub mod truth_validator;

// Re-export all agents
pub use axiom_discoverer::*;
pub use axiom_prime::*;
pub use logic_core::*;
pub use truth_validator::*;

#[derive(Debug, Clone, Default)]
pub struct AxiomAgents;

impl AxiomAgents {
    pub fn new(_config: &super::config::AxiomConfig) -> Self {
        Self
    }
}
