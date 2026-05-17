use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::layer3_tool::ToolKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuarantine {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub reason: QuarantineReason,
    pub severity: QuarantineSeverity,
    pub isolated_regions: Vec<Uuid>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: QuarantineStatus,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuarantineReason {
    AnomalyDetected { score: f64, details: String },
    MemoryCorruption { region_id: Uuid, checksum_mismatch: bool },
    HallucinationSpread { affected_chains: Vec<Uuid> },
    SecurityViolation { capability: String, action: String },
    ResourceExhaustion { resource: String, usage_pct: f64 },
    ManualIntervention { triggered_by: String },
    ChainContamination { source_chain: Uuid },
    ToolAbuse { tool: ToolKind, violations: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuarantineSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuarantineStatus {
    Active,
    Investigating,
    Resolved,
    Escalated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineManager {
    pub active_quarantines: Vec<MemoryQuarantine>,
    pub quarantine_history: Vec<MemoryQuarantine>,
    pub max_active: usize,
    pub auto_resolve_after_seconds: u64,
    pub escalation_mode: EscalationMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationMode {
    AutoKill,
    NotifyAdmin,
    PauseMode(String),
    FullClusterAlert,
}

impl QuarantineManager {
    pub fn new() -> Self {
        Self {
            active_quarantines: Vec::new(),
            quarantine_history: Vec::new(),
            max_active: 100,
            auto_resolve_after_seconds: 3600,
            escalation_mode: EscalationMode::NotifyAdmin,
        }
    }

    pub fn quarantine_agent(
        &mut self,
        agent_id: Uuid,
        reason: QuarantineReason,
        severity: QuarantineSeverity,
        regions: Vec<Uuid>,
    ) -> MemoryQuarantine {
        let q = MemoryQuarantine {
            id: Uuid::new_v4(),
            agent_id,
            reason,
            severity,
            isolated_regions: regions,
            timestamp: chrono::Utc::now(),
            status: QuarantineStatus::Active,
            metadata: HashMap::new(),
        };

        if self.active_quarantines.len() >= self.max_active {
            let oldest_idx = self.active_quarantines
                .iter()
                .enumerate()
                .min_by_key(|(_, q)| q.timestamp)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let old = self.active_quarantines.remove(oldest_idx);
            self.quarantine_history.push(old);
        }

        self.active_quarantines.push(q.clone());
        q
    }

    pub fn resolve_quarantine(&mut self, quarantine_id: Uuid) -> Result<(), QuarantineError> {
        let idx = self.active_quarantines.iter().position(|q| q.id == quarantine_id)
            .ok_or(QuarantineError::QuarantineNotFound(quarantine_id))?;
        let mut q = self.active_quarantines.remove(idx);
        q.status = QuarantineStatus::Resolved;
        self.quarantine_history.push(q);
        Ok(())
    }

    pub fn escalate_quarantine(&mut self, quarantine_id: Uuid) -> Result<(), QuarantineError> {
        let q = self.active_quarantines.iter_mut()
            .find(|q| q.id == quarantine_id)
            .ok_or(QuarantineError::QuarantineNotFound(quarantine_id))?;
        q.status = QuarantineStatus::Escalated;
        q.metadata.insert("escalated_at".into(), chrono::Utc::now().to_rfc3339());
        Ok(())
    }

    pub fn get_active_for_agent(&self, agent_id: Uuid) -> Vec<&MemoryQuarantine> {
        self.active_quarantines.iter().filter(|q| q.agent_id == agent_id).collect()
    }

    pub fn is_agent_quarantined(&self, agent_id: Uuid) -> bool {
        self.active_quarantines.iter().any(|q| q.agent_id == agent_id)
    }

    pub fn get_critical_quarantines(&self) -> Vec<&MemoryQuarantine> {
        self.active_quarantines.iter()
            .filter(|q| matches!(q.severity, QuarantineSeverity::Critical | QuarantineSeverity::High))
            .collect()
    }

    pub fn cleanup_expired(&mut self) {
        let now = chrono::Utc::now();
        let mut to_remove = Vec::with_capacity(self.active_quarantines.len());
        for (i, q) in self.active_quarantines.iter().enumerate() {
            if (now - q.timestamp).num_seconds() > self.auto_resolve_after_seconds as i64 {
                to_remove.push(i);
            }
        }
        for i in to_remove.into_iter().rev() {
            let mut q = self.active_quarantines.remove(i);
            q.status = QuarantineStatus::Resolved;
            self.quarantine_history.push(q);
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum QuarantineError {
    #[error("Quarantine not found: {0}")]
    QuarantineNotFound(Uuid),
    #[error("Agent already quarantined: {0}")]
    AlreadyQuarantined(Uuid),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quarantine_create_and_resolve() {
        let mut manager = QuarantineManager::new();
        let agent_id = Uuid::new_v4();

        let q = manager.quarantine_agent(
            agent_id,
            QuarantineReason::AnomalyDetected { score: 0.95, details: "unusual memory access pattern".into() },
            QuarantineSeverity::High,
            vec![],
        );

        assert!(manager.is_agent_quarantined(agent_id));
        assert_eq!(q.status, QuarantineStatus::Active);

        manager.resolve_quarantine(q.id).unwrap();
        assert!(!manager.is_agent_quarantined(agent_id));
    }

    #[test]
    fn test_critical_quarantine_escalation() {
        let mut manager = QuarantineManager::new();
        let agent_id = Uuid::new_v4();

        let q = manager.quarantine_agent(
            agent_id,
            QuarantineReason::SecurityViolation {
                capability: "shell".into(),
                action: "rm -rf /".into(),
            },
            QuarantineSeverity::Critical,
            vec![],
        );

        manager.escalate_quarantine(q.id).unwrap();
        let critical = manager.get_critical_quarantines();
        assert_eq!(critical.len(), 1);
        assert!(matches!(critical[0].status, QuarantineStatus::Escalated));
    }
}
