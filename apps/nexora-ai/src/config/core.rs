//! Core configuration

use serde::{Deserialize, Serialize};

/// Core controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    pub enable_ml_intent: bool,
    pub enable_coordination: bool,
    pub enable_error_recovery: bool,
    pub enable_monitoring: bool,
    pub max_concurrent_requests: usize,
    pub request_timeout_ms: u64,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            enable_ml_intent: true,
            enable_coordination: true,
            enable_error_recovery: true,
            enable_monitoring: true,
            max_concurrent_requests: 100,
            request_timeout_ms: 30000,
        }
    }
}
