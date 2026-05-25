//! NXR-CIPHER Agents Module
//!
//! Individual agent implementations for cryptography and security

pub mod crypto_prime;
pub mod encryption_master;
pub mod security_guardian;

// Re-export all agents
pub use crypto_prime::*;
pub use encryption_master::*;
pub use security_guardian::*;

#[derive(Debug, Clone)]
pub struct CipherAgents {
    config: super::config::CipherConfig,
}

impl Default for CipherAgents {
    fn default() -> Self {
        Self {
            config: super::config::CipherConfig::default(),
        }
    }
}

impl CipherAgents {
    pub fn new(config: &super::config::CipherConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}
