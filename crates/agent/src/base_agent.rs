//! Base Agent Trait
//!
//! Interface standar untuk semua agent di Nexora.
//! Semua agent harus mengikuti pattern: receive() -> process() -> respond()

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::{AgentError, Result};

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_id: String,
    pub agent_type: String,
    pub max_concurrent_tasks: usize,
    pub timeout_seconds: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            agent_id: "default".to_string(),
            agent_type: "unknown".to_string(),
            max_concurrent_tasks: 1,
            timeout_seconds: 30,
        }
    }
}

/// Status dari sebuah agent
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent sedang diinisialisasi
    Initializing,
    /// Agent siap menerima request
    Ready,
    /// Agent sedang memproses request
    Processing,
    /// Agent sedang di-pause
    Paused,
    /// Agent mengalami error
    Error(String),
    /// Agent sedang shutdown
    ShuttingDown,
    /// Agent sudah shutdown
    Shutdown,
}

/// Message yang bisa diterima agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Unique ID untuk message
    pub id: Uuid,
    /// Tipe message
    pub message_type: String,
    /// Payload message
    pub payload: serde_json::Value,
    /// Metadata tambahan
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp pembuatan
    pub timestamp: DateTime<Utc>,
    /// Sender agent ID (jika applicable)
    pub sender: Option<Uuid>,
    /// Priority level
    pub priority: MessagePriority,
}

/// Priority level untuk message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessagePriority {
    Normal,
    High,
}

/// Response dari agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Unique ID untuk response
    pub id: Uuid,
    /// ID message yang di-response
    pub request_id: Uuid,
    /// Status response
    pub status: ResponseStatus,
    /// Payload response
    pub payload: serde_json::Value,
    /// Metadata tambahan
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp pembuatan
    pub timestamp: DateTime<Utc>,
    /// Processing time dalam milliseconds
    pub processing_time_ms: u64,
}

/// Status dari response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    Error(String),
}

/// Context yang diberikan ke agent saat processing
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// Session ID
    pub session_id: Uuid,
    /// User ID (jika applicable)
    pub user_id: Option<Uuid>,
    /// Request parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// State dari session
    pub session_state: HashMap<String, serde_json::Value>,
    /// Metadata tambahan
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Base trait untuk semua agent
#[async_trait]
pub trait Agent: Send + Sync {
    /// Get unique ID agent
    fn id(&self) -> Uuid;

    /// Get nama agent
    fn name(&self) -> &str;

    /// Get tipe agent
    fn agent_type(&self) -> &str;

    /// Get current status
    fn status(&self) -> AgentStatus;

    /// Initialize agent
    async fn initialize(&mut self, config: AgentConfig) -> Result<()>;

    /// Receive message dari external atau agent lain
    async fn receive(&mut self, message: AgentMessage) -> Result<()>;

    /// Process message yang sudah diterima
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse>;

    /// Send response ke requester
    async fn respond(&mut self, response: AgentResponse) -> Result<()>;

    /// Shutdown agent gracefully
    async fn shutdown(&mut self) -> Result<()>;

    /// Health check agent
    async fn health_check(&self) -> Result<bool>;

    /// Get agent statistics
    fn get_stats(&self) -> AgentStats;

    /// Get agent configuration
    fn get_config(&self) -> AgentConfig;

    /// Reset agent state (jika supported)
    async fn reset(&mut self) -> Result<()> {
        Err(AgentError::ProcessingError {
            operation: "reset".to_string(),
            reason: "Reset not supported for this agent".to_string(),
        })
    }
}

/// Statistics untuk agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    /// Total messages processed
    pub messages_processed: u64,
    /// Total errors
    pub errors: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Uptime dalam seconds
    pub uptime_seconds: u64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
}

impl Default for AgentStats {
    fn default() -> Self {
        Self {
            messages_processed: 0,
            errors: 0,
            avg_processing_time_ms: 0.0,
            uptime_seconds: 0,
            memory_usage_bytes: 0,
            last_activity: Utc::now(),
        }
    }
}

impl AgentMessage {
    /// Create new message
    pub fn new(message_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            message_type: message_type.into(),
            payload,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            sender: None,
            priority: MessagePriority::Normal,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set sender
    pub fn with_sender(mut self, sender: Uuid) -> Self {
        self.sender = Some(sender);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl AgentResponse {
    /// Create new success response
    pub fn success(request_id: Uuid, payload: serde_json::Value, processing_time_ms: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            request_id,
            status: ResponseStatus::Success,
            payload,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            processing_time_ms,
        }
    }

    /// Create new error response
    pub fn error(request_id: Uuid, error: impl Into<String>, processing_time_ms: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            request_id,
            status: ResponseStatus::Error(error.into()),
            payload: serde_json::Value::Null,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            processing_time_ms,
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl AgentContext {
    /// Create new context
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            user_id: None,
            parameters: HashMap::new(),
            session_state: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set user ID
    pub fn with_user_id(mut self, user_id: Uuid) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Add session state
    pub fn with_session_state(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.session_state.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.agent_id, "default");
        assert_eq!(config.agent_type, "unknown");
        assert_eq!(config.max_concurrent_tasks, 1);
        assert_eq!(config.timeout_seconds, 30);
    }

    #[test]
    fn test_agent_config_serialize_roundtrip() {
        let config = AgentConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.agent_id, deserialized.agent_id);
    }

    #[test]
    fn test_agent_status_variants() {
        assert_ne!(AgentStatus::Ready, AgentStatus::Initializing);
        assert_eq!(
            AgentStatus::Error("msg".into()),
            AgentStatus::Error("msg".into())
        );
        assert_ne!(AgentStatus::Shutdown, AgentStatus::ShuttingDown);
    }

    #[test]
    fn test_agent_status_debug_clone() {
        let status = AgentStatus::Processing;
        let cloned = status.clone();
        assert_eq!(format!("{:?}", status), format!("{:?}", cloned));
    }

    #[test]
    fn test_agent_status_serialize() {
        let status = AgentStatus::Error("test error".into());
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: AgentStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_message_priority_default() {
        assert_eq!(MessagePriority::Normal, MessagePriority::Normal);
        assert_ne!(MessagePriority::High, MessagePriority::Normal);
    }

    #[test]
    fn test_agent_message_new() {
        let msg = AgentMessage::new("test_type", serde_json::json!({"key": "value"}));
        assert_eq!(msg.message_type, "test_type");
        assert_eq!(msg.priority, MessagePriority::Normal);
        assert!(msg.sender.is_none());
    }

    #[test]
    fn test_agent_message_builder() {
        let sender_id = Uuid::new_v4();
        let msg = AgentMessage::new("type", serde_json::json!(42))
            .with_priority(MessagePriority::High)
            .with_sender(sender_id)
            .with_metadata("source", serde_json::json!("test"));
        assert_eq!(msg.priority, MessagePriority::High);
        assert_eq!(msg.sender, Some(sender_id));
        assert_eq!(msg.metadata.get("source"), Some(&serde_json::json!("test")));
    }

    #[test]
    fn test_agent_response_success() {
        let req_id = Uuid::new_v4();
        let resp = AgentResponse::success(req_id, serde_json::json!("payload"), 42);
        assert_eq!(resp.status, ResponseStatus::Success);
        assert_eq!(resp.request_id, req_id);
        assert_eq!(resp.processing_time_ms, 42);
    }

    #[test]
    fn test_agent_response_error() {
        let req_id = Uuid::new_v4();
        let resp = AgentResponse::error(req_id, "something failed", 10);
        assert!(matches!(resp.status, ResponseStatus::Error(_)));
        if let ResponseStatus::Error(msg) = &resp.status {
            assert_eq!(msg, "something failed");
        }
    }

    #[test]
    fn test_agent_response_with_metadata() {
        let req_id = Uuid::new_v4();
        let resp = AgentResponse::success(req_id, serde_json::json!(true), 5)
            .with_metadata("trace_id", serde_json::json!("abc"));
        assert_eq!(
            resp.metadata.get("trace_id"),
            Some(&serde_json::json!("abc"))
        );
    }

    #[test]
    fn test_agent_context_new() {
        let session_id = Uuid::new_v4();
        let ctx = AgentContext::new(session_id);
        assert_eq!(ctx.session_id, session_id);
        assert!(ctx.user_id.is_none());
        assert!(ctx.parameters.is_empty());
    }

    #[test]
    fn test_agent_context_with_user_id() {
        let session_id = Uuid::new_v4();
        let user_id = Uuid::new_v4();
        let ctx = AgentContext::new(session_id).with_user_id(user_id);
        assert_eq!(ctx.user_id, Some(user_id));
    }

    #[test]
    fn test_agent_context_with_parameter() {
        let ctx =
            AgentContext::new(Uuid::new_v4()).with_parameter("key", serde_json::json!("value"));
        assert_eq!(ctx.parameters.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_agent_stats_default() {
        let stats = AgentStats::default();
        assert_eq!(stats.messages_processed, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.avg_processing_time_ms, 0.0);
        assert_eq!(stats.uptime_seconds, 0);
    }

    #[test]
    fn test_response_status_serialize() {
        let status = ResponseStatus::Success;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"Success\"");
        let deserialized: ResponseStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ResponseStatus::Success);
    }
}
