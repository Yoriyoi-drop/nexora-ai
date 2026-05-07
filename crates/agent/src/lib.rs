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

pub type Result<T> = std::result::Result<T, AgentError>;
