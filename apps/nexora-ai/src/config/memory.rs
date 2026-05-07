//! Memory configuration

use serde::{Deserialize, Serialize};

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub short_term_capacity: usize,
    pub session_capacity: usize,
    pub long_term_capacity: usize,
    pub knowledge_capacity: usize,
    pub enable_compression: bool,
    pub compression_threshold: f32,
    pub enable_persistence: bool,
    pub persistence_path: Option<String>,
    pub cleanup_interval_seconds: u64,
    pub max_age_hours: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_term_capacity: 1000,
            session_capacity: 10000,
            long_term_capacity: 100000,
            knowledge_capacity: 1000000,
            enable_compression: true,
            compression_threshold: 0.8,
            enable_persistence: true,
            persistence_path: Some("./data/memory".to_string()),
            cleanup_interval_seconds: 3600,
            max_age_hours: 24 * 7, // 1 week
        }
    }
}
