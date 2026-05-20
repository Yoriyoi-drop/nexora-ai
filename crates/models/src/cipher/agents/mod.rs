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

#[derive(Debug, Clone, Default)]
pub struct CipherAgents;

impl CipherAgents {
    pub fn new(_config: &super::config::CipherConfig) -> Self {
        Self
    }
}
