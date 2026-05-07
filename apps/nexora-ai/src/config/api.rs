//! API configuration

use serde::{Deserialize, Serialize};

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub enable_rate_limiting: bool,
    pub requests_per_minute: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8080".to_string(),
            api_key: None,
            timeout_seconds: 30,
            max_retries: 3,
            enable_rate_limiting: true,
            requests_per_minute: 1000,
        }
    }
}
