use crate::gnac::canvas::NeuralGraph;
use crate::gnac::sandbox::DataAccessPolicy;
use md5;

/// Security manager untuk tensor sandbox
pub struct SecurityManager;

impl SecurityManager {
    /// Verifikasi akses user terhadap dataset
    pub fn verify_access(policy: &DataAccessPolicy, user: &str, role: &str) -> bool {
        let user_ok = policy.allowed_users.is_empty() || policy.allowed_users.contains(&user.to_string());
        let role_ok = policy.allowed_roles.is_empty() || policy.allowed_roles.contains(&role.to_string());
        user_ok && role_ok
    }

    /// Hash tensor untuk audit trail
    pub fn tensor_fingerprint(tensor_data: &[u8]) -> String {
        let digest = md5::compute(tensor_data);
        format!("{:x}", digest)
    }

    /// Verifikasi integritas graf
    pub fn verify_graph_integrity(graph: &NeuralGraph) -> bool {
        !graph.nodes.is_empty()
    }
}
