use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::config::{GpuQuota, NetworkPolicy};

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct ModeId(pub String);

impl ModeId {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl std::fmt::Display for ModeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mode-{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModeKind {
    Reasoning,
    Coding,
    Memory,
    Research,
    Defense,
    Creative,
    Analysis,
    System,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mode {
    pub id: ModeId,
    pub uid: Uuid,
    pub kind: ModeKind,
    pub namespace: String,
    pub network_policy: NetworkPolicy,
    pub gpu_quota: GpuQuota,
    pub memory_quota_mb: u64,
    pub agent_ids: Vec<Uuid>,
    pub isolation_policy: ModeIsolationPolicy,
    pub status: ModeStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeIsolationPolicy {
    pub inter_mode_communication: bool,
    pub allow_egress_to_modes: Vec<String>,
    pub allow_ingress_from_modes: Vec<String>,
    pub memory_isolation: bool,
    pub gpu_isolation: bool,
    pub network_isolation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModeStatus {
    Creating,
    Active,
    Idle,
    Quarantined,
    Terminating,
    Terminated,
    KillSwitch,
}

pub type SharedModeIsolation = Arc<RwLock<ModeIsolationLayer>>;

pub struct ModeIsolationLayer {
    modes: HashMap<ModeId, Mode>,
    default_policy: NetworkPolicy,
    default_gpu_quota: GpuQuota,
    default_memory_quota_mb: u64,
}

impl ModeIsolationLayer {
    pub fn new(
        default_policy: NetworkPolicy,
        default_gpu_quota: GpuQuota,
        default_memory_quota_mb: u64,
    ) -> Self {
        Self {
            modes: HashMap::new(),
            default_policy,
            default_gpu_quota,
            default_memory_quota_mb,
        }
    }

    pub fn create_mode(&mut self, id: ModeId, kind: ModeKind) -> &Mode {
        let namespace = format!("nexora-{}", id.0);
        let mode = Mode {
            id: id.clone(),
            uid: Uuid::new_v4(),
            kind,
            namespace,
            network_policy: self.default_policy.clone(),
            gpu_quota: self.default_gpu_quota.clone(),
            memory_quota_mb: self.default_memory_quota_mb,
            agent_ids: Vec::new(),
            isolation_policy: ModeIsolationPolicy {
                inter_mode_communication: false,
                allow_egress_to_modes: Vec::new(),
                allow_ingress_from_modes: Vec::new(),
                memory_isolation: true,
                gpu_isolation: true,
                network_isolation: true,
            },
            status: ModeStatus::Creating,
            created_at: chrono::Utc::now(),
        };
        self.modes.entry(id).or_insert(mode)
    }

    pub fn get_mode(&self, id: &ModeId) -> Option<&Mode> {
        self.modes.get(id)
    }

    pub fn get_mode_mut(&mut self, id: &ModeId) -> Option<&mut Mode> {
        self.modes.get_mut(id)
    }

    pub fn activate_mode(&mut self, id: &ModeId) -> Result<(), IsolationError> {
        let mode = self.modes.get_mut(id).ok_or(IsolationError::ModeNotFound(id.clone()))?;
        mode.status = ModeStatus::Active;
        Ok(())
    }

    pub fn register_agent(&mut self, mode_id: &ModeId, agent_uid: Uuid) -> Result<(), IsolationError> {
        let mode = self.modes.get_mut(mode_id).ok_or(IsolationError::ModeNotFound(mode_id.clone()))?;
        mode.agent_ids.push(agent_uid);
        Ok(())
    }

    pub fn remove_agent(&mut self, mode_id: &ModeId, agent_uid: Uuid) -> Result<(), IsolationError> {
        let mode = self.modes.get_mut(mode_id).ok_or(IsolationError::ModeNotFound(mode_id.clone()))?;
        mode.agent_ids.retain(|id| *id != agent_uid);
        Ok(())
    }

    pub fn set_mode_policy(
        &mut self,
        id: &ModeId,
        policy: ModeIsolationPolicy,
    ) -> Result<(), IsolationError> {
        let mode = self.modes.get_mut(id).ok_or(IsolationError::ModeNotFound(id.clone()))?;
        mode.isolation_policy = policy;
        Ok(())
    }

    pub fn can_communicate(&self, from: &ModeId, to: &ModeId) -> bool {
        if from == to {
            return true;
        }
        let Some(from_mode) = self.modes.get(from) else { return false };
        if !from_mode.isolation_policy.inter_mode_communication {
            return false;
        }
        from_mode.isolation_policy.allow_egress_to_modes.contains(&to.0)
    }

    pub fn list_modes(&self) -> Vec<&Mode> {
        self.modes.values().collect()
    }

    pub fn list_modes_by_status(&self, status: ModeStatus) -> Vec<&Mode> {
        self.modes.values().filter(|m| m.status == status).collect()
    }

    pub fn terminate_mode(&mut self, id: &ModeId) -> Result<Vec<Uuid>, IsolationError> {
        let mode = self.modes.get_mut(id).ok_or(IsolationError::ModeNotFound(id.clone()))?;
        let agents = mode.agent_ids.clone();
        mode.status = ModeStatus::Terminating;
        Ok(agents)
    }

    pub fn quarantine_mode(&mut self, id: &ModeId) -> Result<(), IsolationError> {
        let mode = self.modes.get_mut(id).ok_or(IsolationError::ModeNotFound(id.clone()))?;
        mode.status = ModeStatus::Quarantined;
        mode.isolation_policy.inter_mode_communication = false;
        mode.isolation_policy.allow_egress_to_modes.clear();
        mode.isolation_policy.allow_ingress_from_modes.clear();
        Ok(())
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum IsolationError {
    #[error("Mode not found: {0}")]
    ModeNotFound(ModeId),
    #[error("Mode already exists: {0}")]
    ModeAlreadyExists(ModeId),
    #[error("Mode is not active: {0}")]
    ModeNotActive(ModeId),
    #[error("Communication denied: {0} -> {1}")]
    CommunicationDenied(ModeId, ModeId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_creation_and_activation() {
        let mut layer = ModeIsolationLayer::new(
            NetworkPolicy::DenyAll,
            GpuQuota::default(),
            16384,
        );
        let id = ModeId::new("research");
        layer.create_mode(id.clone(), ModeKind::Research);
        assert_eq!(layer.get_mode(&id).unwrap().status, ModeStatus::Creating);
        layer.activate_mode(&id).unwrap();
        assert_eq!(layer.get_mode(&id).unwrap().status, ModeStatus::Active);
    }

    #[test]
    fn test_inter_mode_communication_blocked() {
        let mut layer = ModeIsolationLayer::new(
            NetworkPolicy::DenyAll,
            GpuQuota::default(),
            16384,
        );
        let research = ModeId::new("research");
        let defense = ModeId::new("defense");
        layer.create_mode(research.clone(), ModeKind::Research);
        layer.create_mode(defense.clone(), ModeKind::Defense);
        assert!(!layer.can_communicate(&research, &defense));
    }
}
