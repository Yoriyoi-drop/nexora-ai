use crate::canvas::NeuralGraph;
use crate::sandbox::DataAccessPolicy;
use sha2::{Digest, Sha256};

/// Security manager untuk tensor sandbox
pub struct SecurityManager;

impl SecurityManager {
    /// Verifikasi akses user terhadap dataset
    pub fn verify_access(policy: &DataAccessPolicy, user: &str, role: &str) -> bool {
        let user_ok =
            policy.allowed_users.is_empty() || policy.allowed_users.contains(&user.to_string());
        let role_ok =
            policy.allowed_roles.is_empty() || policy.allowed_roles.contains(&role.to_string());
        user_ok && role_ok
    }

    /// Hash tensor untuk audit trail
    pub fn tensor_fingerprint(tensor_data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(tensor_data);
        format!("{:x}", hasher.finalize())
    }

    /// Verifikasi integritas graf
    pub fn verify_graph_integrity(graph: &NeuralGraph) -> bool {
        !graph.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_access_empty_policy() {
        let p = DataAccessPolicy {
            allowed_users: vec![],
            allowed_roles: vec![],
            dataset_id: "ds".to_string(),
            encryption_required: false,
            audit_enabled: false,
        };
        assert!(SecurityManager::verify_access(&p, "anyone", "anyrole"));
    }

    #[test]
    fn test_verify_access_restricted() {
        let p = DataAccessPolicy {
            allowed_users: vec!["alice".to_string()],
            allowed_roles: vec!["admin".to_string()],
            dataset_id: "ds".to_string(),
            encryption_required: false,
            audit_enabled: false,
        };
        assert!(SecurityManager::verify_access(&p, "alice", "admin"));
        assert!(!SecurityManager::verify_access(&p, "bob", "admin"));
    }

    #[test]
    fn test_tensor_fingerprint() {
        let fp = SecurityManager::tensor_fingerprint(b"hello");
        assert_eq!(fp.len(), 64);
    }

    #[test]
    fn test_verify_graph_integrity_empty() {
        let g = NeuralGraph::new("empty");
        assert!(!SecurityManager::verify_graph_integrity(&g));
    }

    #[test]
    fn test_verify_graph_integrity_non_empty() {
        let mut g = NeuralGraph::new("g");
        g.add_node(crate::canvas::GraphNode::new(
            crate::NodeType::Input,
            "in",
            0.0,
            0.0,
        ));
        assert!(SecurityManager::verify_graph_integrity(&g));
    }
}
