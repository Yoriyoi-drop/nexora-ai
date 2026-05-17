//! Custom error types untuk Nexora AI
//! 
//! Module ini berisi definisi error types yang spesifik dan terstruktur
//! untuk menggantikan generic error handling

use thiserror::Error;

/// Error types utama untuk Nexora AI
#[derive(Debug, Error)]
pub enum NexoraError {
    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Initialization error: {message}")]
    Initialization { message: String },

    #[error("Processing error: {message}")]
    Processing { message: String },

    #[error("Agent error: {message}")]
    Agent { message: String },

    #[error("Memory error: {message}")]
    Memory { message: String },

    #[error("Model error: {message}")]
    Model { message: String },

    #[error("IO error: {source}")]
    Io { #[from] source: std::io::Error },

    #[error("Serialization error: {message}")]
    Serialization { message: String },

    #[error("System error: {message}")]
    System { message: String },

    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },

    #[error("Timeout error: operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    #[error("Permission denied: {action}")]
    PermissionDenied { action: String },

    #[error("Rate limit exceeded: limit={limit}, window={window}s")]
    RateLimit { limit: u32, window: u64 },
}

impl NexoraError {
    /// Helper untuk membuat configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Helper untuk membuat processing error
    pub fn processing(message: impl Into<String>) -> Self {
        Self::Processing {
            message: message.into(),
        }
    }

    /// Helper untuk membuat initialization error
    pub fn initialization(message: impl Into<String>) -> Self {
        Self::Initialization {
            message: message.into(),
        }
    }

    /// Helper untuk membuat IO error
    pub fn io(source: std::io::Error) -> Self {
        Self::Io { source }
    }

    /// Helper untuk membuat serialization error
    pub fn serialization(source: impl std::fmt::Display) -> Self {
        Self::Serialization {
            message: source.to_string(),
        }
    }

    /// Helper untuk membuat system error
    pub fn system(message: impl Into<String>) -> Self {
        Self::System {
            message: message.into(),
        }
    }

    /// Helper untuk membuat model error
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
        }
    }

    /// Helper untuk membuat validation error
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Helper untuk membuat not found error
    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// Get error code untuk API responses
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Config { .. } => "CONFIG_ERROR",
            Self::Initialization { .. } => "INIT_ERROR",
            Self::Processing { .. } => "PROCESSING_ERROR",
            Self::Agent { .. } => "AGENT_ERROR",
            Self::Memory { .. } => "MEMORY_ERROR",
            Self::Model { .. } => "MODEL_ERROR",
            Self::Io { .. } => "IO_ERROR",
            Self::Serialization { .. } => "SERIALIZATION_ERROR",
            Self::System { .. } => "SYSTEM_ERROR",
            Self::Validation { .. } => "VALIDATION_ERROR",
            Self::Timeout { .. } => "TIMEOUT_ERROR",
            Self::NotFound { .. } => "NOT_FOUND",
            Self::PermissionDenied { .. } => "PERMISSION_DENIED",
            Self::RateLimit { .. } => "RATE_LIMIT_EXCEEDED",
        }
    }

    /// Get HTTP status code untuk API errors
    pub fn http_status(&self) -> u16 {
        match self {
            Self::Config { .. } | Self::Initialization { .. } | Self::System { .. } => 500,
            Self::Processing { .. } | Self::Agent { .. } | Self::Model { .. } => 422,
            Self::Memory { .. } => 503,
            Self::Io { .. } => 500,
            Self::Serialization { .. } => 400,
            Self::Validation { .. } => 400,
            Self::Timeout { .. } => 408,
            Self::NotFound { .. } => 404,
            Self::PermissionDenied { .. } => 403,
            Self::RateLimit { .. } => 429,
        }
    }
}

/// Result type yang menggunakan NexoraError
pub type NexoraResult<T> = Result<T, NexoraError>;

impl From<anyhow::Error> for NexoraError {
    fn from(error: anyhow::Error) -> Self {
        Self::system(error.to_string())
    }
}


