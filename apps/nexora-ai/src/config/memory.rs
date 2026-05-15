//! Memory configuration

use serde::{Deserialize, Serialize};

/// Eviction strategy untuk memory layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Hapus yang paling lama tidak diakses
    LRU,
    /// Hapus yang paling lama dibuat
    FIFO,
    /// Hapus berdasarkan umur maksimal
    TTL,
    /// Kombinasi LRU + TTL
    LruTtl,
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        EvictionStrategy::LruTtl
    }
}

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
    pub eviction_strategy: EvictionStrategy,
    pub max_memory_mb: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_term_capacity: 500,
            session_capacity: 2000,
            long_term_capacity: 10000,
            knowledge_capacity: 50000,
            enable_compression: true,
            compression_threshold: 0.8,
            enable_persistence: true,
            persistence_path: Some("./data/memory".to_string()),
            cleanup_interval_seconds: 300,
            max_age_hours: 24,
            eviction_strategy: EvictionStrategy::LruTtl,
            max_memory_mb: 512,
        }
    }
}
