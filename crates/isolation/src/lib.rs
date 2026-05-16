pub mod config;
pub mod layer0_global;
pub mod layer1_mode;
pub mod layer2_agent;
pub mod layer3_tool;
pub mod layer4_runtime;
pub mod layer5_cognitive;
pub mod layer6_permission;
pub mod firewall;
pub mod quarantine;
pub mod killswitch;
pub mod multicluster;

use parking_lot::RwLock;
use std::sync::Arc;

use layer0_global::{GlobalSystemIsolation, SharedGlobalCluster};
use layer1_mode::{ModeIsolationLayer, SharedModeIsolation};
use layer2_agent::{AgentIsolationLayer, SharedAgentIsolation};
use layer3_tool::{ToolIsolationLayer, SharedToolGateway};
use layer5_cognitive::{CognitiveIsolationLayer, SharedCognitiveIsolation};
use layer6_permission::{PermissionLayer, SharedPermissionLayer};
use firewall::{InterAgentFirewall, SharedFirewall, AgentBus, SharedAgentBus};
use quarantine::QuarantineManager;
use killswitch::{KillSwitch, SharedKillSwitch};

#[derive(Clone)]
pub struct IsolationOrchestrator {
    pub global: SharedGlobalCluster,
    pub mode: SharedModeIsolation,
    pub agent: SharedAgentIsolation,
    pub tool: SharedToolGateway,
    pub cognitive: SharedCognitiveIsolation,
    pub permission: SharedPermissionLayer,
    pub firewall: SharedFirewall,
    pub agent_bus: SharedAgentBus,
    pub kill_switch: SharedKillSwitch,
    pub quarantine: Arc<RwLock<QuarantineManager>>,
}

impl IsolationOrchestrator {
    pub fn new(config: config::IsolationConfig) -> Self {
        let mode_layer = Arc::new(RwLock::new(ModeIsolationLayer::new(
            config.mode.default_network_policy,
            config.mode.default_gpu_quota,
            config.mode.default_memory_quota_mb,
        )));

        Self {
            global: Arc::new(RwLock::new(
                GlobalSystemIsolation::new(&config.global.cluster_name).cluster().read().clone()
            )),
            mode: mode_layer,
            agent: Arc::new(RwLock::new(AgentIsolationLayer::new())),
            tool: Arc::new(RwLock::new(ToolIsolationLayer::new(
                config.tool.allowed_tools.iter().map(|t| match t.as_str() {
                    "python" => crate::layer3_tool::ToolKind::Python,
                    "browser" => crate::layer3_tool::ToolKind::Browser,
                    "terminal" => crate::layer3_tool::ToolKind::Terminal,
                    "filesystem" => crate::layer3_tool::ToolKind::FileSystem,
                    "shell" => crate::layer3_tool::ToolKind::Shell,
                    "compiler" => crate::layer3_tool::ToolKind::Compiler,
                    _ => crate::layer3_tool::ToolKind::Custom(t.clone()),
                }).collect(),
            ))),
            cognitive: Arc::new(RwLock::new(CognitiveIsolationLayer::new())),
            permission: Arc::new(RwLock::new(PermissionLayer::new())),
            firewall: Arc::new(RwLock::new(InterAgentFirewall::new())),
            agent_bus: Arc::new(RwLock::new(AgentBus::new(10000))),
            kill_switch: Arc::new(RwLock::new(KillSwitch::new())),
            quarantine: Arc::new(RwLock::new(QuarantineManager::new())),
        }
    }

    pub fn pre_inference_check(&self, agent_id: uuid::Uuid) -> Result<(), IsolationCheckError> {
        {
            let quarantine = self.quarantine.read();
            if quarantine.is_agent_quarantined(agent_id) {
                return Err(IsolationCheckError::AgentQuarantined(agent_id));
            }
        }

        {
            let mut perm = self.permission.write();
            if !perm.check_capability(agent_id, &layer6_permission::Capability::ModelInference("default".into())) {
                return Err(IsolationCheckError::CapabilityDenied("model_inference".into()));
            }
        }

        Ok(())
    }

    pub fn check_agent_communication(
        &self,
        source_id: uuid::Uuid,
        source_label: &str,
        dest_id: uuid::Uuid,
        dest_label: &str,
        msg_type: &str,
        payload: &[u8],
    ) -> Result<(), IsolationCheckError> {
        let mut firewall = self.firewall.write();
        let action = firewall.evaluate(source_id, source_label, dest_id, dest_label, msg_type, payload);

        match action {
            firewall::FirewallAction::Allow | firewall::FirewallAction::AuditAllow => {
                let mut bus = self.agent_bus.write();
                bus.send(firewall::AgentBusMessage {
                    id: uuid::Uuid::new_v4(),
                    source_id,
                    destination_id: dest_id,
                    message_type: firewall::AgentBusMsgType::Query,
                    payload: payload.to_vec(),
                    size_bytes: payload.len() as u64,
                    priority: firewall::MessagePriority::Normal,
                    ttl_seconds: 60,
                });
                Ok(())
            }
            firewall::FirewallAction::Deny | firewall::FirewallAction::AuditDeny => {
                Err(IsolationCheckError::CommunicationDenied { from: source_id, to: dest_id })
            }
            firewall::FirewallAction::QuarantineSource => {
                let mut q = self.quarantine.write();
                q.quarantine_agent(
                    source_id,
                    quarantine::QuarantineReason::SecurityViolation {
                        capability: "inter-agent-comm".into(),
                        action: format!("blocked by firewall: {msg_type}"),
                    },
                    quarantine::QuarantineSeverity::High,
                    vec![],
                );
                Err(IsolationCheckError::AgentQuarantined(source_id))
            }
            firewall::FirewallAction::RateLimit(_) => {
                Err(IsolationCheckError::RateLimitExceeded(source_id))
            }
        }
    }

    pub fn verify_tool_access(
        &self,
        agent_id: uuid::Uuid,
        tool: &str,
    ) -> Result<(), IsolationCheckError> {
        let mut perm = self.permission.write();
        if !perm.verify_tool_access(agent_id, tool) {
            return Err(IsolationCheckError::CapabilityDenied(format!("tool:{tool}")));
        }
        Ok(())
    }

    pub fn trigger_kill_switch(
        &self,
        target: killswitch::KillTarget,
        reason: &str,
        trigger: killswitch::KillTrigger,
    ) -> Result<killswitch::KillEvent, IsolationCheckError> {
        let mut ks = self.kill_switch.write();
        match target {
            killswitch::KillTarget::Agent(id) => {
                ks.kill_agent(id, reason, trigger)
                    .map_err(|e| IsolationCheckError::KillSwitchError(e.to_string()))
            }
            killswitch::KillTarget::Mode(mode_id) => {
                ks.kill_mode(mode_id, reason, trigger)
                    .map_err(|e| IsolationCheckError::KillSwitchError(e.to_string()))
            }
            killswitch::KillTarget::Cluster => {
                ks.kill_cluster(reason)
                    .map_err(|e| IsolationCheckError::KillSwitchError(e.to_string()))
            }
            _ => Err(IsolationCheckError::KillSwitchError("unsupported target".into())),
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum IsolationCheckError {
    #[error("Agent is quarantined: {0}")]
    AgentQuarantined(uuid::Uuid),
    #[error("Capability denied: {0}")]
    CapabilityDenied(String),
    #[error("Communication denied: {from} -> {to}")]
    CommunicationDenied { from: uuid::Uuid, to: uuid::Uuid },

    #[error("Rate limit exceeded for agent: {0}")]
    RateLimitExceeded(uuid::Uuid),
    #[error("Kill switch error: {0}")]
    KillSwitchError(String),
    #[error("Internal isolation error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_orchestrator_creation() {
        let config = config::IsolationConfig::default();
        let orch = IsolationOrchestrator::new(config);
        assert!(orch.global.read().api_gateway.enabled);
    }

    #[test]
    fn test_agent_comm_blocked_by_default() {
        let config = config::IsolationConfig::default();
        let orch = IsolationOrchestrator::new(config);

        let agent_a = Uuid::new_v4();
        let agent_b = Uuid::new_v4();

        let result = orch.check_agent_communication(
            agent_a, "agent:research:oracle",
            agent_b, "agent:defense:sentinel",
            "query", b"hello",
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_pre_inference_check_blocks_unknown_agent() {
        let config = config::IsolationConfig::default();
        let orch = IsolationOrchestrator::new(config);
        let unknown = Uuid::new_v4();
        let result = orch.pre_inference_check(unknown);
        assert!(result.is_err());
    }

    #[test]
    fn test_full_isolation_flow() {
        let config = config::IsolationConfig::default();
        let orch = IsolationOrchestrator::new(config);

        let agent_id = Uuid::new_v4();
        orch.permission.write().register_agent(agent_id, "test-agent", layer6_permission::AgentRole::Restricted);
        orch.permission.write().grant_capability(agent_id, layer6_permission::Capability::ModelInference("default".into())).unwrap();

        assert!(orch.pre_inference_check(agent_id).is_ok());

        orch.permission.write().revoke_capability(agent_id, &layer6_permission::Capability::ModelInference("default".into())).unwrap();
        assert!(orch.pre_inference_check(agent_id).is_err());
    }
}

pub mod prelude {
    pub use crate::config::IsolationConfig;
    pub use crate::layer0_global::*;
    pub use crate::layer1_mode::{Mode, ModeId, ModeKind, ModeStatus, ModeIsolationPolicy, ModeIsolationLayer};
    pub use crate::layer2_agent::*;
    pub use crate::layer3_tool::{
        ToolKind, ToolPod, ToolGateway, ToolExecutionRequest, ToolExecutionResult,
        SandboxSpec, SandboxKind as ToolSandboxKind, ToolStatus, ToolIsolationLayer,
    };
    pub use crate::layer4_runtime::*;
    pub use crate::layer5_cognitive::*;
    pub use crate::layer6_permission::*;
    pub use crate::firewall::{
        InterAgentFirewall, AgentBus, AgentBusMessage, AgentBusMsgType,
        FirewallRule, FirewallMatch, FirewallAction, FirewallAuditEntry,
        FirewallRateLimiter, SuspiciousPattern,
    };
    pub use crate::quarantine::*;
    pub use crate::killswitch::*;
    pub use crate::multicluster::*;
    pub use crate::{IsolationOrchestrator, IsolationCheckError};
}
