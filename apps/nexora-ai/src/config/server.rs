//! Server configuration

use serde::{Deserialize, Serialize};

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub enable_tls: bool,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    pub max_connections: usize,
    pub request_timeout_seconds: u64,
    pub enable_cors: bool,
    pub cors_origins: Vec<String>,
    pub api_keys: Vec<String>,
    pub enable_auth: bool,
    /// Optional rate limit in requests per minute (0 or None = disabled).
    /// Applies per client IP using a sliding window counter.
    pub rate_limit_rpm: Option<u64>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            enable_tls: true,
            cert_path: None,
            key_path: None,
            max_connections: 1000,
            request_timeout_seconds: 30,
            enable_cors: true,
            cors_origins: vec![],
            api_keys: vec![],
            enable_auth: true,
            rate_limit_rpm: Some(60),
        }
    }
}
