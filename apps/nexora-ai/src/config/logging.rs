//! Logging configuration

use serde::{Deserialize, Serialize};

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub enable_file_logging: bool,
    pub file_path: Option<String>,
    pub max_file_size_mb: usize,
    pub max_files: usize,
    pub enable_console_logging: bool,
    pub enable_structured_logging: bool,
    pub enable_tracing: bool,
    pub tracing_endpoint: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "compact".to_string(),
            enable_file_logging: true,
            file_path: Some("./logs/nexora.log".to_string()),
            max_file_size_mb: 100,
            max_files: 10,
            enable_console_logging: true,
            enable_structured_logging: false,
            enable_tracing: true,
            tracing_endpoint: None,
        }
    }
}
