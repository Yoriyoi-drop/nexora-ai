use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::layer1_mode::ModeId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitch {
    pub enabled: bool,
    pub confirm_required: bool,
    pub grace_period_seconds: u64,
    pub history: Vec<KillEvent>,
    pub protection: KillProtection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillEvent {
    pub id: Uuid,
    pub target: KillTarget,
    pub reason: String,
    pub triggered_by: KillTrigger,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: KillStatus,
    pub affected_agents: Vec<Uuid>,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillTarget {
    Agent(Uuid),
    Mode(ModeId),
    Group(Uuid),
    Tool(ToolTarget),
    Cluster,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolTarget {
    pub tool_type: String,
    pub pod_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillTrigger {
    Manual { user: String },
    AutoQuarantine { anomaly_score: f64 },
    SecurityViolation { capability: String },
    ResourceExhaustion { resource: String },
    HallucinationOutbreak { chain_count: usize },
    KillChain { parent_event: Uuid },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KillStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillProtection {
    pub protected_agents: Vec<Uuid>,
    pub protected_modes: Vec<String>,
    pub max_kills_per_minute: u32,
    pub cooldown_seconds: u64,
    pub require_escalation_for: Vec<String>,
}

pub type SharedKillSwitch = Arc<RwLock<KillSwitch>>;

impl KillSwitch {
    pub fn new() -> Self {
        Self {
            enabled: true,
            confirm_required: true,
            grace_period_seconds: 30,
            history: Vec::new(),
            protection: KillProtection {
                protected_agents: Vec::new(),
                protected_modes: vec!["system".into(), "security-core".into()],
                max_kills_per_minute: 10,
                cooldown_seconds: 60,
                require_escalation_for: vec!["defense".into(), "system".into()],
            },
        }
    }

    pub fn kill_agent(&mut self, agent_id: Uuid, reason: &str, trigger: KillTrigger) -> Result<KillEvent, KillSwitchError> {
        if !self.enabled {
            return Err(KillSwitchError::KillSwitchDisabled);
        }

        if self.protection.protected_agents.contains(&agent_id) {
            return Err(KillSwitchError::AgentProtected(agent_id));
        }

        let event = KillEvent {
            id: Uuid::new_v4(),
            target: KillTarget::Agent(agent_id),
            reason: reason.to_string(),
            triggered_by: trigger,
            timestamp: chrono::Utc::now(),
            status: KillStatus::Pending,
            affected_agents: vec![agent_id],
            execution_time_ms: 0,
        };

        self.history.push(event.clone());
        Ok(event)
    }

    pub fn kill_mode(&mut self, mode_id: ModeId, reason: &str, trigger: KillTrigger) -> Result<KillEvent, KillSwitchError> {
        if !self.enabled {
            return Err(KillSwitchError::KillSwitchDisabled);
        }

        if self.protection.protected_modes.contains(&mode_id.0) {
            return Err(KillSwitchError::ModeProtected(mode_id.clone()));
        }

        let event = KillEvent {
            id: Uuid::new_v4(),
            target: KillTarget::Mode(mode_id),
            reason: reason.to_string(),
            triggered_by: trigger,
            timestamp: chrono::Utc::now(),
            status: KillStatus::Pending,
            affected_agents: Vec::new(),
            execution_time_ms: 0,
        };

        self.history.push(event.clone());
        Ok(event)
    }

    pub fn kill_cluster(&mut self, reason: &str) -> Result<KillEvent, KillSwitchError> {
        if !self.enabled {
            return Err(KillSwitchError::KillSwitchDisabled);
        }

        let event = KillEvent {
            id: Uuid::new_v4(),
            target: KillTarget::Cluster,
            reason: reason.to_string(),
            triggered_by: KillTrigger::Manual { user: "system".into() },
            timestamp: chrono::Utc::now(),
            status: KillStatus::Pending,
            affected_agents: Vec::new(),
            execution_time_ms: 0,
        };

        self.history.push(event.clone());
        Ok(event)
    }

    pub fn complete_kill(&mut self, event_id: Uuid, affected: Vec<Uuid>, duration_ms: u64) {
        if let Some(event) = self.history.iter_mut().find(|e| e.id == event_id) {
            event.status = KillStatus::Completed;
            event.affected_agents = affected;
            event.execution_time_ms = duration_ms;
        }
    }

    pub fn fail_kill(&mut self, event_id: Uuid, error: &str) {
        if let Some(event) = self.history.iter_mut().find(|e| e.id == event_id) {
            event.status = KillStatus::Failed(error.to_string());
        }
    }

    pub fn protect_agent(&mut self, agent_id: Uuid) {
        self.protection.protected_agents.push(agent_id);
    }

    pub fn unprotect_agent(&mut self, agent_id: Uuid) {
        self.protection.protected_agents.retain(|id| *id != agent_id);
    }

    pub fn get_recent_kills(&self, count: usize) -> Vec<&KillEvent> {
        self.history.iter().rev().take(count).collect()
    }

    pub fn get_kills_by_trigger(&self, trigger: &KillTrigger) -> Vec<&KillEvent> {
        self.history.iter()
            .filter(|e| {
                std::mem::discriminant(&e.triggered_by) == std::mem::discriminant(trigger)
            })
            .collect()
    }

    pub fn is_mode_protected(&self, mode_id: &ModeId) -> bool {
        self.protection.protected_modes.contains(&mode_id.0)
    }
}

impl std::fmt::Display for KillTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Agent(id) => write!(f, "agent:{id}"),
            Self::Mode(mode) => write!(f, "mode:{}", mode.0),
            Self::Group(id) => write!(f, "group:{id}"),
            Self::Tool(t) => write!(f, "tool:{}:{}", t.tool_type, t.pod_id),
            Self::Cluster => write!(f, "cluster:global"),
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum KillSwitchError {
    #[error("Kill switch is disabled")]
    KillSwitchDisabled,
    #[error("Agent is protected: {0}")]
    AgentProtected(Uuid),
    #[error("Mode is protected: {0}")]
    ModeProtected(ModeId),
    #[error("Rate limit exceeded. Max kills per minute: {0}")]
    RateLimitExceeded(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_agent() {
        let mut ks = KillSwitch::new();
        let agent_id = Uuid::new_v4();
        let event = ks.kill_agent(agent_id, "Anomaly detected", KillTrigger::AutoQuarantine { anomaly_score: 0.95 }).unwrap();
        assert_eq!(event.status, KillStatus::Pending);
        assert_eq!(event.affected_agents, vec![agent_id]);
    }

    #[test]
    fn test_kill_mode() {
        let mut ks = KillSwitch::new();
        let mode_id = ModeId::new("research");
        let event = ks.kill_mode(mode_id, "Memory corruption outbreak", KillTrigger::HallucinationOutbreak { chain_count: 15 }).unwrap();
        assert_eq!(event.status, KillStatus::Pending);
    }

    #[test]
    fn test_protected_mode_rejected() {
        let mut ks = KillSwitch::new();
        let system_mode = ModeId::new("system");
        let result = ks.kill_mode(system_mode, "test", KillTrigger::Manual { user: "admin".into() });
        assert!(matches!(result, Err(KillSwitchError::ModeProtected(_))));
    }

    #[test]
    fn test_kill_completion() {
        let mut ks = KillSwitch::new();
        let agent_id = Uuid::new_v4();
        let event = ks.kill_agent(agent_id, "test", KillTrigger::Manual { user: "admin".into() }).unwrap();
        ks.complete_kill(event.id, vec![agent_id], 150);
        let recent = ks.get_recent_kills(1);
        assert!(matches!(recent[0].status, KillStatus::Completed));
    }

    #[test]
    fn test_kill_chaining() {
        let mut ks = KillSwitch::new();
        let first = ks.kill_agent(Uuid::new_v4(), "primary", KillTrigger::AutoQuarantine { anomaly_score: 0.9 }).unwrap();
        let second = ks.kill_agent(Uuid::new_v4(), "secondary", KillTrigger::KillChain { parent_event: first.id }).unwrap();
        assert!(matches!(second.triggered_by, KillTrigger::KillChain { .. }));
    }
}
