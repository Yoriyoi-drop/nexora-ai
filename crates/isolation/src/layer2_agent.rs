use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::layer1_mode::ModeId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPod {
    pub id: Uuid,
    pub name: String,
    pub mode_id: ModeId,
    pub agent_type: AgentType,
    pub memory_buffer: MemoryBufferSpec,
    pub runtime_spec: AgentRuntimeSpec,
    pub status: PodStatus,
    pub health: PodHealth,
    pub permissions: Vec<String>,
    pub isolation_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    Oracle,
    Scanner,
    WebMind,
    FactChecker,
    CodeSentinel,
    Watcher,
    Sentinel,
    VectorEngine,
    MemoryFilter,
    Planner,
    Validator,
    Inference,
    Router,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBufferSpec {
    pub max_entries: usize,
    pub ttl_seconds: u64,
    pub isolated: bool,
    pub quarantine_on_corruption: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntimeSpec {
    pub cpu_limit: f64,
    pub memory_limit_mb: u64,
    pub network_access: bool,
    pub filesystem_access: bool,
    pub execution_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PodStatus {
    Pending,
    Running,
    Degraded,
    Quarantined,
    Terminated,
    Killed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodHealth {
    pub cpu_usage_pct: f64,
    pub memory_usage_mb: u64,
    pub error_count: u64,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub anomaly_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGroup {
    pub id: Uuid,
    pub mode_id: ModeId,
    pub name: String,
    pub pods: Vec<AgentPod>,
    pub shared_memory_keys: Vec<String>,
    pub isolation_policy: GroupIsolationPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupIsolationPolicy {
    pub can_read_other_agent_memory: bool,
    pub can_write_other_agent_memory: bool,
    pub can_see_other_agent_reasoning: bool,
    pub require_gateway_for_external: bool,
}

pub type SharedAgentIsolation = Arc<RwLock<AgentIsolationLayer>>;

pub struct AgentIsolationLayer {
    groups: HashMap<Uuid, AgentGroup>,
    pods: HashMap<Uuid, AgentPod>,
}

impl AgentIsolationLayer {
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            pods: HashMap::new(),
        }
    }

    pub fn create_group(&mut self, mode_id: ModeId, name: &str) -> AgentGroup {
        let group = AgentGroup {
            id: Uuid::new_v4(),
            mode_id,
            name: name.to_string(),
            pods: Vec::new(),
            shared_memory_keys: Vec::new(),
            isolation_policy: GroupIsolationPolicy {
                can_read_other_agent_memory: false,
                can_write_other_agent_memory: false,
                can_see_other_agent_reasoning: false,
                require_gateway_for_external: true,
            },
        };
        let id = group.id;
        self.groups.insert(id, group);
        self.groups.get(&id).expect("group was just inserted").clone()
    }

    pub fn spawn_pod(
        &mut self,
        group_id: Uuid,
        name: &str,
        agent_type: AgentType,
        runtime: AgentRuntimeSpec,
    ) -> Result<AgentPod, AgentIsolationError> {
        let group = self.groups.get_mut(&group_id)
            .ok_or(AgentIsolationError::GroupNotFound(group_id))?;

        let pod = AgentPod {
            id: Uuid::new_v4(),
            name: format!("{}-{}", name, Uuid::new_v4().to_string().split('-').next().unwrap_or("x")),
            mode_id: group.mode_id.clone(),
            agent_type,
            memory_buffer: MemoryBufferSpec {
                max_entries: 1000,
                ttl_seconds: 3600,
                isolated: true,
                quarantine_on_corruption: true,
            },
            runtime_spec: runtime,
            status: PodStatus::Pending,
            health: PodHealth {
                cpu_usage_pct: 0.0,
                memory_usage_mb: 0,
                error_count: 0,
                last_heartbeat: chrono::Utc::now(),
                anomaly_score: 0.0,
            },
            permissions: Vec::new(),
            isolation_label: format!("agent:{}:{}", group.mode_id.0, name),
        };
        let pod_id = pod.id;
        self.pods.insert(pod_id, pod.clone());
        group.pods.push(pod.clone());
        Ok(pod)
    }

    pub fn get_pod(&self, id: Uuid) -> Option<&AgentPod> {
        self.pods.get(&id)
    }

    pub fn get_pod_mut(&mut self, id: Uuid) -> Option<&mut AgentPod> {
        self.pods.get_mut(&id)
    }

    pub fn get_group(&self, id: Uuid) -> Option<&AgentGroup> {
        self.groups.get(&id)
    }

    pub fn set_pod_health(&mut self, pod_id: Uuid, health: PodHealth) {
        if let Some(pod) = self.pods.get_mut(&pod_id) {
            pod.health = health;
        }
    }

    pub fn quarantine_pod(&mut self, pod_id: Uuid) -> Result<(), AgentIsolationError> {
        let pod = self.pods.get_mut(&pod_id)
            .ok_or(AgentIsolationError::PodNotFound(pod_id))?;
        pod.status = PodStatus::Quarantined;
        pod.memory_buffer.quarantine_on_corruption = true;
        pod.runtime_spec.network_access = false;
        Ok(())
    }

    pub fn terminate_pod(&mut self, pod_id: Uuid) -> Result<(), AgentIsolationError> {
        let pod = self.pods.get_mut(&pod_id)
            .ok_or(AgentIsolationError::PodNotFound(pod_id))?;
        pod.status = PodStatus::Terminated;
        Ok(())
    }

    pub fn list_pods_by_mode(&self, mode_id: &ModeId) -> Vec<&AgentPod> {
        self.pods.values().filter(|p| p.mode_id == *mode_id).collect()
    }

    pub fn list_pods_by_status(&self, status: PodStatus) -> Vec<&AgentPod> {
        self.pods.values().filter(|p| p.status == status).collect()
    }

    pub fn pods_with_anomaly_above(&self, threshold: f64) -> Vec<&AgentPod> {
        self.pods.values()
            .filter(|p| p.health.anomaly_score > threshold)
            .collect()
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum AgentIsolationError {
    #[error("Agent group not found: {0}")]
    GroupNotFound(Uuid),
    #[error("Agent pod not found: {0}")]
    PodNotFound(Uuid),
    #[error("Agent pod not running: {0}")]
    PodNotRunning(Uuid),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer1_mode::ModeId;

    #[test]
    fn test_agent_group_creation() {
        let mut layer = AgentIsolationLayer::new();
        let mode_id = ModeId::new("research");
        let group = layer.create_group(mode_id, "oracle-group");
        assert_eq!(group.name, "oracle-group");
        assert!(!group.isolation_policy.can_read_other_agent_memory);
    }

    #[test]
    fn test_spawn_and_quarantine_pod() {
        let mut layer = AgentIsolationLayer::new();
        let mode_id = ModeId::new("research");
        let group = layer.create_group(mode_id, "test-group");
        let pod = layer.spawn_pod(
            group.id,
            "oracle",
            AgentType::Oracle,
            AgentRuntimeSpec {
                cpu_limit: 2.0,
                memory_limit_mb: 4096,
                network_access: true,
                filesystem_access: false,
                execution_timeout_seconds: 300,
            },
        ).unwrap();
        assert_eq!(pod.status, PodStatus::Pending);
        layer.quarantine_pod(pod.id).unwrap();
        let pod = layer.get_pod(pod.id).unwrap();
        assert_eq!(pod.status, PodStatus::Quarantined);
        assert!(!pod.runtime_spec.network_access);
    }
}
