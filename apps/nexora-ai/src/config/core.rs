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
    /// Max tiers that can be loaded simultaneously in VRAM
    #[serde(default = "default_max_tiers_in_vram")]
    pub max_tiers_in_vram: usize,
    /// VRAM budget in MB — when exceeded, LRU tiers are evicted
    #[serde(default = "default_vram_budget_mb")]
    pub vram_budget_mb: u64,
    /// Eviction threshold (0.0–1.0) — fraction of VRAM budget that triggers eviction
    #[serde(default = "default_eviction_threshold")]
    pub eviction_threshold: f32,
}

fn default_max_tiers_in_vram() -> usize { 1 }
fn default_vram_budget_mb() -> u64 { 24000 }
fn default_eviction_threshold() -> f32 { 0.85 }

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
            max_tiers_in_vram: default_max_tiers_in_vram(),
            vram_budget_mb: default_vram_budget_mb(),
            eviction_threshold: default_eviction_threshold(),
        }
    }
}
