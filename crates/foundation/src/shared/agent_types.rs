//! Shared Agent Types
//! 
//! Common types and enums used across all NXR models

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Agent Status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    /// Agent is idle and ready to process
    Idle,
    /// Agent is currently processing a task
    Processing,
    /// Agent encountered an error
    Error(String),
    /// Agent is under maintenance
    Maintenance,
    /// Agent is disabled
    Disabled,
}

/// Task Routing Rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRoutingRule {
    /// Pattern to match against task descriptions
    pub pattern: String,
    /// Target agent for this pattern
    pub target_agent: String,
    /// Priority of this rule (higher = more important)
    pub priority: u8,
    /// Weight for probabilistic routing
    pub weight: Option<f32>,
}

/// Communication Channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationChannel {
    /// Channel identifier
    pub id: String,
    /// Channel type
    pub channel_type: ChannelType,
    /// Connected agents
    pub connected_agents: Vec<String>,
    /// Channel configuration
    pub config: HashMap<String, String>,
}

/// Channel Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    /// Direct point-to-point communication
    Direct,
    /// Broadcast to multiple agents
    Broadcast,
    /// Topic-based pub/sub
    Topic(String),
    /// Request-response pattern
    RequestResponse,
}

/// Agent Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Total tasks processed
    pub tasks_processed: u64,
    /// Average processing time (ms)
    pub avg_processing_time: f64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Current load (0.0 - 1.0)
    pub current_load: f64,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Task Priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 2,
    /// High priority
    High = 3,
    /// Critical priority
    Critical = 4,
}

/// Agent Capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapability {
    /// Capability name
    pub name: String,
    /// Capability description
    pub description: String,
    /// Capability version
    pub version: String,
    /// Supported input types
    pub input_types: Vec<String>,
    /// Supported output types
    pub output_types: Vec<String>,
    /// Performance metrics for this capability
    pub metrics: CapabilityMetrics,
}

/// Capability Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMetrics {
    /// Accuracy score (0.0 - 1.0)
    pub accuracy: f32,
    /// Average latency (ms)
    pub avg_latency: f32,
    /// Resource usage (0.0 - 1.0)
    pub resource_usage: f32,
    /// Reliability score (0.0 - 1.0)
    pub reliability: f32,
}

/// Task Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult<T> {
    /// Result data
    pub data: T,
    /// Success status
    pub success: bool,
    /// Processing time (ms)
    pub processing_time: u64,
    /// Agent that processed the task
    pub agent_id: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Error Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentError {
    /// Task processing failed
    ProcessingFailed(String),
    /// Agent not available
    AgentUnavailable(String),
    /// Invalid input
    InvalidInput(String),
    /// Timeout occurred
    Timeout,
    /// Resource constraint
    ResourceConstraint(String),
    /// Configuration error
    ConfigurationError(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::ProcessingFailed(msg) => write!(f, "Processing failed: {}", msg),
            AgentError::AgentUnavailable(id) => write!(f, "Agent unavailable: {}", id),
            AgentError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AgentError::Timeout => write!(f, "Operation timeout"),
            AgentError::ResourceConstraint(msg) => write!(f, "Resource constraint: {}", msg),
            AgentError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {}

/// Result type for agent operations
pub type AgentResult<T> = Result<T, AgentError>;
