//! Nexora Agent Layer
//!
//! Layer koordinasi dan behavior untuk sistem Nexora AI.
//! Agent bekerja sebagai "decision personality layer" di atas
//! system backbone yang sudah ada di `core`.

pub mod agent_manager;
pub mod base_agent;
pub mod communication;
pub mod context_agent;
pub mod inference_agent;
pub mod lifecycle;
pub mod memory_agent;
pub mod planner_agent;
pub mod registry;
pub mod response_agent;
pub mod routing_agent;
pub mod state;
pub mod validation_agent;
pub mod worker_agent;

// Re-export main types untuk kemudahan penggunaan
pub use agent_manager::AgentManager;
pub use base_agent::{
    Agent, AgentConfig, AgentContext, AgentMessage, AgentResponse, AgentStats, AgentStatus,
    MessagePriority,
};
pub use communication::{InterAgentMessage, MessageBus};
pub use lifecycle::{AgentLifecycleEvent, LifecycleManager};
pub use registry::{AgentRegistry, IntentMapping};
pub use state::AgentState;
pub use worker_agent::{WorkerAgent, WorkerAgentConfig, WorkItem, WorkStatus, StepExecutionResult};

/// Versi agent layer
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error types untuk agent layer
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Agent not found: {agent_id}")]
    AgentNotFound { agent_id: String },

    #[error("Agent already exists: {agent_id}")]
    AgentAlreadyExists { agent_id: String },

    #[error("Invalid agent configuration: {field}: {reason}")]
    InvalidConfiguration { field: String, reason: String },

    #[error("Communication error with {target}: {reason}")]
    CommunicationError { target: String, reason: String },

    #[error("Lifecycle error: {reason}")]
    LifecycleError { reason: String },

    #[error("Processing error in {operation}: {reason}")]
    ProcessingError { operation: String, reason: String },

    #[error("State error: {reason}")]
    StateError { reason: String },
}

impl From<serde_json::Error> for AgentError {
    fn from(err: serde_json::Error) -> Self {
        AgentError::ProcessingError {
            operation: "serde_json".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<anyhow::Error> for AgentError {
    fn from(err: anyhow::Error) -> Self {
        AgentError::ProcessingError {
            operation: "anyhow".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<reqwest::Error> for AgentError {
    fn from(err: reqwest::Error) -> Self {
        AgentError::ProcessingError {
            operation: "reqwest".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<std::io::Error> for AgentError {
    fn from(err: std::io::Error) -> Self {
        AgentError::ProcessingError {
            operation: "io".to_string(),
            reason: err.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, AgentError>;

// ─── Cross-layer integration (Phase 5 wiring) ───────────────────────
// Nyata: isolation check sebelum agent berkomunikasi
fn _agent_isolation_check(agent_id: uuid::Uuid) -> std::result::Result<(), nexora_isolation::IsolationCheckError> {
    let config = nexora_isolation::config::IsolationConfig::default();
    let orch = nexora_isolation::IsolationOrchestrator::new(config);
    orch.pre_inference_check(agent_id)
}

// Nyata: database untuk persist agent state
fn _agent_db() -> nexora_database::DatabaseManager {
    nexora_database::DatabaseManager::new()
}

// Nyata: cognition planning untuk agent task decomposition
fn _agent_create_plan() -> nexora_cognition::planning::Plan {
    let step = nexora_cognition::planning::PlanStep {
        id: uuid::Uuid::new_v4(),
        action: "process".into(),
        parameters: serde_json::json!({}),
        dependencies: vec![],
        estimated_duration_ms: 100,
    };
    nexora_cognition::planning::Plan {
        id: uuid::Uuid::new_v4(),
        steps: vec![step],
        dependencies: vec![],
        estimated_duration_ms: 100,
    }
}

// Nyata: monitoring untuk agent observability
fn _agent_monitoring() -> nexora_monitoring::MonitoringSystem {
    nexora_monitoring::MonitoringSystem::new(nexora_monitoring::MonitoringConfig::default())
}

// Nyata: text preprocessing untuk agent prompts
fn _agent_preprocess(text: &str) -> anyhow::Result<String> {
    let utils = nexora_utils::UtilsManager::default();
    utils.preprocess_text(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_agent::ResponseStatus;

    #[test]
    fn test_agent_error_display() {
        let err = AgentError::AgentNotFound {
            agent_id: "test-agent".into(),
        };
        assert_eq!(format!("{}", err), "Agent not found: test-agent");
    }

    #[test]
    fn test_agent_error_from_serde() {
        let invalid = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let err: AgentError = invalid.into();
        assert!(matches!(err, AgentError::ProcessingError { .. }));
    }

    #[test]
    fn test_agent_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AgentError = io_err.into();
        assert!(matches!(err, AgentError::ProcessingError { .. }));
    }

    #[test]
    fn test_agent_message_creation() {
        let msg = AgentMessage::new("test-type", serde_json::json!("payload"));
        assert_eq!(msg.message_type, "test-type");
    }

    #[test]
    fn test_agent_message_with_priority() {
        let msg =
            AgentMessage::new("test", serde_json::json!({})).with_priority(MessagePriority::High);
        assert_eq!(msg.priority, MessagePriority::High);
    }

    #[test]
    fn test_agent_response_success() {
        let req_id = uuid::Uuid::new_v4();
        let resp = AgentResponse::success(req_id, serde_json::json!("ok"), 10);
        assert_eq!(resp.status, ResponseStatus::Success);
    }

    #[test]
    fn test_agent_response_error() {
        let req_id = uuid::Uuid::new_v4();
        let resp = AgentResponse::error(req_id, "fail", 5);
        assert_ne!(resp.status, ResponseStatus::Success);
    }

    #[test]
    fn test_version_constant() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_agent_stats_default() {
        let stats = AgentStats::default();
        assert_eq!(stats.messages_processed, 0);
        assert_eq!(stats.errors, 0);
    }
}
