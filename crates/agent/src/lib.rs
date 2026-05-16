//! Nexora Agent Layer
//! 
//! Layer koordinasi dan behavior untuk sistem Nexora AI.
//! Agent bekerja sebagai "decision personality layer" di atas
//! system backbone yang sudah ada di `core`.

pub mod base_agent;
pub mod agent_manager;
pub mod registry;
pub mod lifecycle;
pub mod communication;
pub mod context_agent;
pub mod routing_agent;
pub mod inference_agent;
pub mod planner_agent;
pub mod memory_agent;
pub mod validation_agent;
pub mod response_agent;
pub mod state;

// Re-export main types untuk kemudahan penggunaan
pub use base_agent::{Agent, AgentMessage, AgentResponse, AgentStatus, AgentConfig, AgentStats, AgentContext, MessagePriority};
pub use agent_manager::AgentManager;
pub use registry::{AgentRegistry, IntentMapping};
pub use lifecycle::{LifecycleManager, AgentLifecycleEvent};
pub use communication::{MessageBus, InterAgentMessage};
pub use state::AgentState;

/// Versi agent layer
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error types untuk agent layer
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Agent not found: {0}")]
    AgentNotFound(String),
    
    #[error("Agent already exists: {0}")]
    AgentAlreadyExists(String),
    
    #[error("Invalid agent configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Communication error: {0}")]
    CommunicationError(String),
    
    #[error("Lifecycle error: {0}")]
    LifecycleError(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("State error: {0}")]
    StateError(String),
}

impl From<serde_json::Error> for AgentError {
    fn from(err: serde_json::Error) -> Self {
        AgentError::ProcessingError(format!("JSON error: {}", err))
    }
}

impl From<anyhow::Error> for AgentError {
    fn from(err: anyhow::Error) -> Self {
        AgentError::ProcessingError(format!("Processing error: {}", err))
    }
}

impl From<reqwest::Error> for AgentError {
    fn from(err: reqwest::Error) -> Self {
        AgentError::ProcessingError(format!("HTTP request error: {}", err))
    }
}

impl From<std::io::Error> for AgentError {
    fn from(err: std::io::Error) -> Self {
        AgentError::ProcessingError(format!("I/O error: {}", err))
    }
}

pub type Result<T> = std::result::Result<T, AgentError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_error_display() {
        let err = AgentError::AgentNotFound("test-agent".into());
        assert_eq!(format!("{}", err), "Agent not found: test-agent");
    }

    #[test]
    fn test_agent_error_from_serde() {
        let invalid = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let err: AgentError = invalid.into();
        assert!(matches!(err, AgentError::ProcessingError(_)));
    }

    #[test]
    fn test_agent_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AgentError = io_err.into();
        assert!(matches!(err, AgentError::ProcessingError(_)));
    }

    #[test]
    fn test_agent_message_creation() {
        let msg = AgentMessage::new("test-type", serde_json::json!("payload"));
        assert_eq!(msg.message_type(), "test-type");
    }

    #[test]
    fn test_agent_message_with_priority() {
        let msg = AgentMessage::new("test", serde_json::json!({}))
            .with_priority(MessagePriority::High);
        assert_eq!(msg.priority(), MessagePriority::High);
    }

    #[test]
    fn test_agent_response_success() {
        let req_id = uuid::Uuid::new_v4();
        let resp = AgentResponse::success(req_id, serde_json::json!("ok"), 10);
        assert!(resp.is_success());
    }

    #[test]
    fn test_agent_response_error() {
        let req_id = uuid::Uuid::new_v4();
        let resp = AgentResponse::error(req_id, "fail", 5);
        assert!(!resp.is_success());
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
