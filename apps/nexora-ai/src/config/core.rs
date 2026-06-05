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
    pub enable_distributed: bool,
    pub distributed_listen_address: String,
    pub distributed_seed_nodes: Vec<String>,
    pub distributed_gossip_interval_ms: u64,
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
            enable_distributed: false,
            distributed_listen_address: "127.0.0.1:8080".to_string(),
            distributed_seed_nodes: Vec::new(),
            distributed_gossip_interval_ms: 1000,
        }
    }
}
