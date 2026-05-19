//! Error types for SACA framework

use thiserror::Error;
use nexora_common::error::NexoraError;

/// SACA-specific errors
#[derive(Error, Debug)]
pub enum SACAError {
    #[error("Chain-of-Thought reasoning failed: {0}")]
    CoTError(String),
    
    #[error("Modular decomposition failed: {0}")]
    DecomposeError(String),
    
    #[error("Context analysis failed: {0}")]
    ContextError(String),
    
    #[error("Sampling failed: {0}")]
    SamplingError(String),
    
    #[error("Execution failed: {0}")]
    ExecuteError(String),
    
    #[error("Reranking failed: {0}")]
    RerankError(String),
    
    #[error("Feedback system error: {0}")]
    FeedbackError(String),
    
    #[error("Pipeline error in phase {phase}: {message}")]
    PipelineError { phase: String, message: String },
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Session error: {0}")]
    SessionError(String),
    
    #[error("Quality threshold not met: {current:.3} < {threshold:.3}")]
    QualityThresholdNotMet { current: f32, threshold: f32 },
    
    #[error("Maximum feedback loops exceeded: {0}")]
    MaxFeedbackLoopsExceeded(u32),
    
    #[error("Timeout during {operation}: {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },
    
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Core error: {0}")]
    CoreError(String),
    
    #[error("Task cancelled")]
    Cancelled,
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl SACAError {
    /// Create a pipeline error for a specific phase
    pub fn pipeline_error(phase: &str, message: impl Into<String>) -> Self {
        Self::PipelineError {
            phase: phase.to_string(),
            message: message.into(),
        }
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::CoTError(_) => true,
            Self::DecomposeError(_) => true,
            Self::SamplingError(_) => true,
            Self::ExecuteError(_) => true,
            Self::ContextError(_) => false, // Context errors are usually not recoverable
            Self::RerankError(_) => true,
            Self::FeedbackError(_) => true,
            Self::PipelineError { .. } => true,
            Self::ConfigError(_) => false,
            Self::SessionError(_) => false,
            Self::QualityThresholdNotMet { .. } => true,
            Self::MaxFeedbackLoopsExceeded(_) => false,
            Self::Timeout { .. } => true,
            Self::ResourceExhaustion(_) => false,
            Self::InvalidInput(_) => false,
            Self::IoError(_) => false,
            Self::SerializationError(_) => false,
            Self::CoreError(_) => false,
            Self::Cancelled => false,
            Self::Unknown(_) => true,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::CoTError(_) => ErrorSeverity::Medium,
            Self::DecomposeError(_) => ErrorSeverity::Medium,
            Self::ContextError(_) => ErrorSeverity::High,
            Self::SamplingError(_) => ErrorSeverity::Medium,
            Self::ExecuteError(_) => ErrorSeverity::Medium,
            Self::RerankError(_) => ErrorSeverity::Low,
            Self::FeedbackError(_) => ErrorSeverity::Medium,
            Self::PipelineError { .. } => ErrorSeverity::High,
            Self::ConfigError(_) => ErrorSeverity::Critical,
            Self::SessionError(_) => ErrorSeverity::High,
            Self::QualityThresholdNotMet { .. } => ErrorSeverity::Low,
            Self::MaxFeedbackLoopsExceeded(_) => ErrorSeverity::Medium,
            Self::Timeout { .. } => ErrorSeverity::Medium,
            Self::ResourceExhaustion(_) => ErrorSeverity::Critical,
            Self::InvalidInput(_) => ErrorSeverity::Medium,
            Self::IoError(_) => ErrorSeverity::Medium,
            Self::SerializationError(_) => ErrorSeverity::Low,
            Self::CoreError(_) => ErrorSeverity::High,
            Self::Cancelled => ErrorSeverity::Low,
            Self::Unknown(_) => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,       // Minor issues, can continue
    Medium,    // Significant but recoverable
    High,      // Serious issues, may need intervention
    Critical,  // System-level failures
}

/// Result type for SACA operations
pub type SACAResult<T> = Result<T, SACAError>;

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub phase: String,
    pub operation: String,
    pub session_id: Option<uuid::Uuid>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(phase: &str, operation: &str) -> Self {
        Self {
            phase: phase.to_string(),
            operation: operation.to_string(),
            session_id: None,
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_session(mut self, session_id: uuid::Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }
    
    pub fn with_info(mut self, key: &str, value: &str) -> Self {
        self.additional_info.insert(key.to_string(), value.to_string());
        self
    }
}

/// Enhanced error with context
#[derive(Error, Debug)]
pub struct ContextualError {
    #[source]
    pub error: SACAError,
    pub context: ErrorContext,
}

impl From<anyhow::Error> for SACAError {
    fn from(err: anyhow::Error) -> Self {
        Self::Unknown(err.to_string())
    }
}

impl ContextualError {
    pub fn new(error: SACAError, context: ErrorContext) -> Self {
        Self { error, context }
    }
}

impl std::fmt::Display for ContextualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}:{}] {}",
            self.context.phase,
            self.context.operation,
            self.error
        )
    }
}
