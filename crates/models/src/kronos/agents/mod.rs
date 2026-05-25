//! NXR-KRONOS Agents Module
//!
//! Individual agent implementations for temporal analysis and optimization

pub mod chronos_prime;
pub mod temporal_optimizer;
pub mod temporal_orchestrator;
pub mod time_analyzer;

// Re-export all agents
pub use chronos_prime::*;
pub use temporal_optimizer::*;
pub use temporal_orchestrator::*;
pub use time_analyzer::*;

#[derive(Debug, Clone)]
pub struct KronosAgents {
    config: super::config::KronosConfig,
}

impl Default for KronosAgents {
    fn default() -> Self {
        Self {
            config: super::config::KronosConfig::default(),
        }
    }
}

impl KronosAgents {
    pub fn new(config: &super::config::KronosConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}
