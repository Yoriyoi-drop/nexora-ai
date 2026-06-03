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

impl EvictionStrategy {
    /// Score an item for eviction: higher = more likely to evict.
    /// - LRU: `1.0 / (last_access_secs_ago + 1.0)` — recently accessed items score lower
    /// - FIFO: `1.0 / (created_secs_ago + 1.0)` — older items score higher
    /// - TTL: `(age_secs / max_age_secs).clamp(0, 2)` — items past TTL score >1.0
    /// - LruTtl: LRU + TTL combined, items past TTL get double weight
    pub fn eviction_score(
        &self,
        last_access_secs_ago: f64,
        created_secs_ago: f64,
        max_age_secs: f64,
    ) -> f64 {
        match self {
            EvictionStrategy::LRU => 1.0 / (last_access_secs_ago + 1.0),
            EvictionStrategy::FIFO => created_secs_ago,
            EvictionStrategy::TTL => {
                if max_age_secs <= 0.0 {
                    1.0
                } else {
                    (created_secs_ago / max_age_secs).clamp(0.0, 2.0)
                }
            }
            EvictionStrategy::LruTtl => {
                let lru = 1.0 / (last_access_secs_ago + 1.0);
                let ttl_bonus = if max_age_secs > 0.0 && created_secs_ago > max_age_secs {
                    2.0
                } else {
                    0.0
                };
                lru + ttl_bonus
            }
        }
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
