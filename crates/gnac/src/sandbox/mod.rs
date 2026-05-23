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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_access_policy() {
        let p = DataAccessPolicy {
            allowed_users: vec!["alice".to_string()],
            allowed_roles: vec!["admin".to_string()],
            dataset_id: "ds1".to_string(),
            encryption_required: true,
            audit_enabled: true,
        };
        assert_eq!(p.dataset_id, "ds1");
        assert!(p.encryption_required);
    }
}
