use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Capability {
    InternetAccess,
    VectorDbAccess,
    ShellAccess,
    CompilerAccess,
    FileSystemRead,
    FileSystemWrite,
    NetworkScan,
    DatabaseQuery,
    ExternalApiCall,
    AgentSpawn,
    MemoryRead(Uuid),
    MemoryWrite(Uuid),
    ToolExecution(String),
    ModelInference(String),
    Custom(String),
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternetAccess => write!(f, "cap:internet"),
            Self::VectorDbAccess => write!(f, "cap:vector-db"),
            Self::ShellAccess => write!(f, "cap:shell"),
            Self::CompilerAccess => write!(f, "cap:compiler"),
            Self::FileSystemRead => write!(f, "cap:fs-read"),
            Self::FileSystemWrite => write!(f, "cap:fs-write"),
            Self::NetworkScan => write!(f, "cap:network-scan"),
            Self::DatabaseQuery => write!(f, "cap:db-query"),
            Self::ExternalApiCall => write!(f, "cap:external-api"),
            Self::AgentSpawn => write!(f, "cap:spawn-agent"),
            Self::MemoryRead(id) => write!(f, "cap:mem-read:{id}"),
            Self::MemoryWrite(id) => write!(f, "cap:mem-write:{id}"),
            Self::ToolExecution(t) => write!(f, "cap:tool:{t}"),
            Self::ModelInference(m) => write!(f, "cap:model:{m}"),
            Self::Custom(c) => write!(f, "cap:custom:{c}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PermissionEffect {
    Allow,
    Deny,
    Audit,
    Conditional { condition: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRule {
    pub id: Uuid,
    pub effect: PermissionEffect,
    pub resource: String,
    pub action: String,
    pub priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPermissions {
    pub agent_id: Uuid,
    pub role: AgentRole,
    pub allowed_capabilities: HashSet<Capability>,
    pub denied_capabilities: HashSet<Capability>,
    pub custom_rules: Vec<PermissionRule>,
    pub resource_quotas: ResourceQuota,
    pub isolation_label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    System,
    Admin,
    Specialist,
    Observer,
    Restricted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    pub max_memory_mb: u64,
    pub max_cpu_cores: f64,
    pub max_network_bandwidth_mbps: u64,
    pub max_storage_mb: u64,
    pub max_concurrent_tasks: u32,
    pub max_tokens_per_minute: u64,
}

const MAX_AUDIT_LOG: usize = 1000;

pub type SharedPermissionLayer = Arc<RwLock<PermissionLayer>>;

pub struct PermissionLayer {
    agents: HashMap<Uuid, AgentPermissions>,
    role_defaults: HashMap<AgentRole, HashSet<Capability>>,
    audit_log: VecDeque<AccessAuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessAuditEntry {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub capability: Capability,
    pub granted: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub reason: String,
}

impl PermissionLayer {
    pub fn new() -> Self {
        let mut role_defaults = HashMap::new();

        role_defaults.insert(AgentRole::System, HashSet::new());
        role_defaults.insert(AgentRole::Admin, HashSet::from([
            Capability::InternetAccess,
            Capability::VectorDbAccess,
            Capability::FileSystemRead,
            Capability::AgentSpawn,
        ]));
        role_defaults.insert(AgentRole::Specialist, HashSet::from([
            Capability::VectorDbAccess,
            Capability::ModelInference("default".into()),
        ]));
        role_defaults.insert(AgentRole::Observer, HashSet::from([
            Capability::VectorDbAccess,
            Capability::FileSystemRead,
        ]));
        role_defaults.insert(AgentRole::Restricted, HashSet::new());

        Self {
            agents: HashMap::new(),
            role_defaults,
            audit_log: VecDeque::new(),
        }
    }

    pub fn register_agent(
        &mut self,
        agent_id: Uuid,
        name: &str,
        role: AgentRole,
    ) -> AgentPermissions {
        let defaults = self.role_defaults.get(&role).cloned().unwrap_or_default();
        let perms = AgentPermissions {
            agent_id,
            role: role.clone(),
            allowed_capabilities: defaults,
            denied_capabilities: HashSet::new(),
            custom_rules: Vec::new(),
            resource_quotas: ResourceQuota {
                max_memory_mb: 1024,
                max_cpu_cores: 1.0,
                max_network_bandwidth_mbps: 10,
                max_storage_mb: 100,
                max_concurrent_tasks: 5,
                max_tokens_per_minute: 100000,
            },
            isolation_label: format!("agent:{}:{}", role_name(&role), name),
        };
        self.agents.insert(agent_id, perms.clone());
        perms
    }

    pub fn check_capability(&mut self, agent_id: Uuid, capability: &Capability) -> bool {
        let Some(agent) = self.agents.get(&agent_id) else {
            self.push_audit_entry(AccessAuditEntry {
                id: Uuid::new_v4(),
                agent_id,
                capability: capability.clone(),
                granted: false,
                timestamp: chrono::Utc::now(),
                reason: "Agent not registered".into(),
            });
            return false;
        };

        if agent.denied_capabilities.contains(capability) {
            self.push_audit_entry(AccessAuditEntry {
                id: Uuid::new_v4(),
                agent_id,
                capability: capability.clone(),
                granted: false,
                timestamp: chrono::Utc::now(),
                reason: "Capability explicitly denied".into(),
            });
            return false;
        }

        let granted = agent.allowed_capabilities.contains(capability);
        self.push_audit_entry(AccessAuditEntry {
            id: Uuid::new_v4(),
            agent_id,
            capability: capability.clone(),
            granted,
            timestamp: chrono::Utc::now(),
            reason: if granted { "Allowed by role".into() } else { "Not in allowed capabilities".into() },
        });
        granted
    }

    pub fn grant_capability(&mut self, agent_id: Uuid, capability: Capability) -> Result<(), PermissionError> {
        let agent = self.agents.get_mut(&agent_id)
            .ok_or(PermissionError::AgentNotFound(agent_id))?;
        agent.denied_capabilities.remove(&capability);
        agent.allowed_capabilities.insert(capability);
        Ok(())
    }

    pub fn revoke_capability(&mut self, agent_id: Uuid, capability: &Capability) -> Result<(), PermissionError> {
        let agent = self.agents.get_mut(&agent_id)
            .ok_or(PermissionError::AgentNotFound(agent_id))?;
        agent.allowed_capabilities.remove(capability);
        agent.denied_capabilities.insert(capability.clone());
        Ok(())
    }

    pub fn set_agent_role(&mut self, agent_id: Uuid, role: AgentRole) -> Result<(), PermissionError> {
        let agent = self.agents.get_mut(&agent_id)
            .ok_or(PermissionError::AgentNotFound(agent_id))?;
        agent.role = role.clone();
        if let Some(defaults) = self.role_defaults.get(&role) {
            agent.allowed_capabilities = defaults.clone();
        }
        Ok(())
    }

    pub fn get_agent_permissions(&self, agent_id: Uuid) -> Option<&AgentPermissions> {
        self.agents.get(&agent_id)
    }

    fn push_audit_entry(&mut self, entry: AccessAuditEntry) {
        self.audit_log.push_back(entry);
        if self.audit_log.len() > MAX_AUDIT_LOG {
            self.audit_log.pop_front();
        }
    }

    pub fn get_audit_log(&self, since: chrono::DateTime<chrono::Utc>) -> Vec<&AccessAuditEntry> {
        self.audit_log.iter().filter(|e| e.timestamp >= since).collect()
    }

    pub fn verify_tool_access(&mut self, agent_id: Uuid, tool: &str) -> bool {
        self.check_capability(agent_id, &Capability::ToolExecution(tool.to_string()))
    }

    pub fn verify_model_access(&mut self, agent_id: Uuid, model: &str) -> bool {
        self.check_capability(agent_id, &Capability::ModelInference(model.to_string()))
    }

    pub fn verify_memory_access(&mut self, agent_id: Uuid, memory_id: Uuid, write: bool) -> bool {
        if write {
            self.check_capability(agent_id, &Capability::MemoryWrite(memory_id))
        } else {
            self.check_capability(agent_id, &Capability::MemoryRead(memory_id))
        }
    }
}

fn role_name(role: &AgentRole) -> &'static str {
    match role {
        AgentRole::System => "sys",
        AgentRole::Admin => "admin",
        AgentRole::Specialist => "specialist",
        AgentRole::Observer => "observer",
        AgentRole::Restricted => "restricted",
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum PermissionError {
    #[error("Agent not found: {0}")]
    AgentNotFound(Uuid),
    #[error("Capability already denied: {0}")]
    CapabilityAlreadyDenied(Capability),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_permission_layer() -> PermissionLayer {
        PermissionLayer::new()
    }

    #[test]
    fn test_oracle_permissions() {
        let mut layer = make_permission_layer();
        let oracle_id = Uuid::new_v4();
        layer.register_agent(oracle_id, "oracle", AgentRole::Specialist);

        assert!(layer.check_capability(oracle_id, &Capability::VectorDbAccess));
        assert!(!layer.check_capability(oracle_id, &Capability::ShellAccess));
        assert!(!layer.check_capability(oracle_id, &Capability::InternetAccess));
    }

    #[test]
    fn test_code_sentinel_permissions() {
        let mut layer = make_permission_layer();
        let sentinel_id = Uuid::new_v4();
        let _perms = layer.register_agent(sentinel_id, "code-sentinel", AgentRole::Restricted);

        layer.grant_capability(sentinel_id, Capability::ShellAccess).unwrap();
        layer.grant_capability(sentinel_id, Capability::CompilerAccess).unwrap();

        assert!(layer.check_capability(sentinel_id, &Capability::ShellAccess));
        assert!(layer.check_capability(sentinel_id, &Capability::CompilerAccess));
        assert!(!layer.check_capability(sentinel_id, &Capability::InternetAccess));
    }
}
