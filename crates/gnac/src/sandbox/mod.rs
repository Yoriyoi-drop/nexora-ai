//! Secure Tensor Sandbox
//!
//! Untuk kebutuhan enterprise dan medis: encrypted tensor transport,
//! isolated execution domain, policy-based dataset access, audit logging,
//! compliance verification, dan model behavior verification sebelum deployment.

pub mod security;
pub mod verification;

pub use security::*;
pub use verification::*;


/// Policy akses dataset
#[derive(Debug, Clone)]
pub struct DataAccessPolicy {
    pub allowed_users: Vec<String>,
    pub allowed_roles: Vec<String>,
    pub dataset_id: String,
    pub encryption_required: bool,
    pub audit_enabled: bool,
}
