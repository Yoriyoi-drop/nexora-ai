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
    pub security_guardian: SecurityGuardianAgent,
    pub encryption_master: EncryptionMasterAgent,
}

impl Default for CipherAgents {
    fn default() -> Self {
        Self {
            config: super::config::CipherConfig::default(),
            security_guardian: SecurityGuardianAgent::default(),
            encryption_master: EncryptionMasterAgent::default(),
        }
    }
}

impl CipherAgents {
    pub fn new(config: &super::config::CipherConfig) -> Self {
        Self {
            config: config.clone(),
            security_guardian: SecurityGuardianAgent::default(),
            encryption_master: EncryptionMasterAgent::default(),
        }
    }

    pub fn security_guardian(&self) -> &SecurityGuardianAgent {
        &self.security_guardian
    }

    pub fn encryption_master(&self) -> &EncryptionMasterAgent {
        &self.encryption_master
    }
}
