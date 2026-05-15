//! DECISION-X Agent
//!
//! Final decision execution with audit trail

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

#[derive(Debug, Clone)]
pub struct DecisionXAgent {
    pub config: DecisionXConfig,
    pub decision_capabilities: DecisionCapabilities,
    pub audit_trail: AuditTrail,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionXConfig {
    pub base_config: BaseAgentConfig,
    pub decision_framework: DecisionFramework,
    pub execution_mode: ExecutionMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionFramework {
    Utilitarian,
    Deontological,
    Pragmatic,
    ConsensusBased,
    HybridFramework { frameworks: Vec<DecisionFramework> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionMode {
    Automatic,
    SemiAutomatic,
    ManualApproval,
    Supervised,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCapabilities {
    pub option_ranking: bool,
    pub decision_finalization: bool,
    pub audit_logging: bool,
    pub rollback_planning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    pub audit_entries: Vec<AuditEntry>,
    pub immutable_log: bool,
    pub signing_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub action: String,
    pub agent: String,
    pub details: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionXTaskInput {
    pub decision_context: String,
    pub options: Vec<DecisionOption>,
    pub constraints: Vec<String>,
    pub authorization_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    pub id: String,
    pub label: String,
    pub expected_value: f32,
    pub risk_score: f32,
    pub feasibility: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionXTaskOutput {
    pub final_decision: DecisionOption,
    pub execution_plan: Vec<String>,
    pub audit_entries: Vec<AuditEntry>,
    pub rollback_strategy: Vec<String>,
    pub decision_hash: String,
}

impl Default for DecisionXConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            decision_framework: DecisionFramework::HybridFramework {
                frameworks: vec![
                    DecisionFramework::Pragmatic,
                    DecisionFramework::ConsensusBased,
                ],
            },
            execution_mode: ExecutionMode::SemiAutomatic,
        }
    }
}

impl Default for DecisionCapabilities {
    fn default() -> Self {
        Self {
            option_ranking: true,
            decision_finalization: true,
            audit_logging: true,
            rollback_planning: true,
        }
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self {
            audit_entries: Vec::new(),
            immutable_log: true,
            signing_method: "sha256".to_string(),
        }
    }
}

impl Default for DecisionXAgent {
    fn default() -> Self {
        Self {
            config: DecisionXConfig::default(),
            decision_capabilities: DecisionCapabilities::default(),
            audit_trail: AuditTrail::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for DecisionXAgent {
    type Config = DecisionXConfig;
    type Input = DecisionXTaskInput;
    type Output = DecisionXTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let final_decision = self.select_decision(&input).await?;
        let execution_plan = self.create_execution_plan(&input, &final_decision).await?;
        let audit_entries = self.log_audit_trail(&input, &final_decision).await?;
        let rollback_strategy = self.create_rollback_strategy(&input, &final_decision).await?;
        let decision_hash = self.sign_decision(&audit_entries).await?;

        Ok(DecisionXTaskOutput {
            final_decision,
            execution_plan,
            audit_entries,
            rollback_strategy,
            decision_hash,
        })
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability {
                name: "decision_x".to_string(),
                description: "Final decision execution with audit trail".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["decision_context".to_string(), "options".to_string(), "constraints".to_string()],
                output_types: vec!["final_decision".to_string(), "execution_plan".to_string(), "audit_log".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.94,
                    avg_latency: 2800.0,
                    resource_usage: 0.5,
                    reliability: 0.97,
                },
            },
        ]
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }

    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()> {
        self.config = config;
        self.status = AgentStatus::Idle;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        self.status = AgentStatus::Disabled;
        Ok(())
    }
}

impl DecisionXAgent {
    pub fn new(config: DecisionXConfig) -> Self {
        Self {
            config,
            decision_capabilities: DecisionCapabilities::default(),
            audit_trail: AuditTrail::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    async fn select_decision(&self, input: &DecisionXTaskInput) -> AgentResult<DecisionOption> {
        input.options.iter()
            .max_by(|a, b| {
                let score_a = a.expected_value * 0.5 + (1.0 - a.risk_score) * 0.3 + a.feasibility * 0.2;
                let score_b = b.expected_value * 0.5 + (1.0 - b.risk_score) * 0.3 + b.feasibility * 0.2;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| crate::shared::agent_types::AgentError::ProcessingFailed("No options to decide from".to_string()))
    }

    async fn create_execution_plan(&self, input: &DecisionXTaskInput, decision: &DecisionOption) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Phase 1: Authorize decision '{}' with level {}", decision.label, input.authorization_level),
            "Phase 2: Notify stakeholders and allocate resources".to_string(),
            "Phase 3: Execute primary actions per decision framework".to_string(),
            "Phase 4: Monitor outcomes and trigger rollback if needed".to_string(),
        ])
    }

    async fn log_audit_trail(&self, input: &DecisionXTaskInput, decision: &DecisionOption) -> AgentResult<Vec<AuditEntry>> {
        let now = chrono::Utc::now();
        let entries = vec![
            AuditEntry {
                timestamp: now,
                action: "DECISION_INITIATED".to_string(),
                agent: self.config.base_config.agent_id.clone(),
                details: format!("Decision context: {}", input.decision_context),
                signature: format!("sig_{}", now.timestamp()),
            },
            AuditEntry {
                timestamp: now,
                action: "OPTION_SELECTED".to_string(),
                agent: self.config.base_config.agent_id.clone(),
                details: format!("Selected option '{}' with value {:.2}", decision.label, decision.expected_value),
                signature: format!("sig_{}_final", now.timestamp()),
            },
        ];
        Ok(entries)
    }

    async fn create_rollback_strategy(&self, _input: &DecisionXTaskInput, _decision: &DecisionOption) -> AgentResult<Vec<String>> {
        Ok(vec![
            "Trigger: Decision outcome deviates >20% from expected value".to_string(),
            "Action 1: Halt all active execution threads".to_string(),
            "Action 2: Restore pre-decision state from checkpoint".to_string(),
            "Action 3: Notify oversight committee".to_string(),
        ])
    }

    async fn sign_decision(&self, entries: &[AuditEntry]) -> AgentResult<String> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for entry in entries {
            entry.action.hash(&mut hasher);
            entry.details.hash(&mut hasher);
        }
        Ok(format!("DEC-{:016x}", hasher.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_x_agent_creation() {
        let agent = DecisionXAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_decision_x_task_processing() {
        let agent = DecisionXAgent::default();
        let input = DecisionXTaskInput {
            decision_context: "Market entry strategy selection".to_string(),
            options: vec![
                DecisionOption {
                    id: "OPT-1".to_string(),
                    label: "Full market entry".to_string(),
                    expected_value: 0.85,
                    risk_score: 0.60,
                    feasibility: 0.70,
                },
                DecisionOption {
                    id: "OPT-2".to_string(),
                    label: "Phased entry".to_string(),
                    expected_value: 0.65,
                    risk_score: 0.30,
                    feasibility: 0.90,
                },
            ],
            constraints: vec!["Regulatory approval required".to_string()],
            authorization_level: "executive".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.execution_plan.is_empty());
        assert!(!output.audit_entries.is_empty());
        assert!(!output.rollback_strategy.is_empty());
        assert!(!output.decision_hash.is_empty());
    }
}
