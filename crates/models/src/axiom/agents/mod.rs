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

#[derive(Debug, Clone)]
pub struct AxiomAgents {
    config: super::config::AxiomConfig,
}

impl Default for AxiomAgents {
    fn default() -> Self {
        Self {
            config: super::config::AxiomConfig::default(),
        }
    }
}

impl AxiomAgents {
    pub fn new(config: &super::config::AxiomConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}
