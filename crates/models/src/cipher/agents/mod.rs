//! NXR-CIPHER Agents Module
//! 
//! Individual agent implementations for cryptography and security

pub mod crypto_prime;
pub mod security_guardian;
pub mod encryption_master;

// Re-export all agents
pub use crypto_prime::*;
pub use security_guardian::*;
pub use encryption_master::*;

#[derive(Debug, Clone, Default)]
pub struct CipherAgents;

impl CipherAgents {
    pub fn new(_config: &super::config::CipherConfig) -> Self {
        Self
    }
}
