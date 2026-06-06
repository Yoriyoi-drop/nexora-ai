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
    pub logic_core: LogicCoreAgent,
    pub truth_validator: TruthValidatorAgent,
}

impl Default for AxiomAgents {
    fn default() -> Self {
        Self {
            config: super::config::AxiomConfig::default(),
            logic_core: LogicCoreAgent::default(),
            truth_validator: TruthValidatorAgent::default(),
        }
    }
}

impl AxiomAgents {
    pub fn new(config: &super::config::AxiomConfig) -> Self {
        Self {
            config: config.clone(),
            logic_core: LogicCoreAgent::default(),
            truth_validator: TruthValidatorAgent::default(),
        }
    }

    pub fn logic_core(&self) -> &LogicCoreAgent {
        &self.logic_core
    }

    pub fn truth_validator(&self) -> &TruthValidatorAgent {
        &self.truth_validator
    }
}
