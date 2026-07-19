use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheLayer {
    Prompt,
    Embedding,
    Retrieval,
    Tool,
    Http,
    Token,
    Model,
}

impl CacheLayer {
    pub fn name(&self) -> &'static str {
        match self {
            CacheLayer::Prompt => "prompt",
            CacheLayer::Embedding => "embedding",
            CacheLayer::Retrieval => "retrieval",
            CacheLayer::Tool => "tool",
            CacheLayer::Http => "http",
            CacheLayer::Token => "token",
            CacheLayer::Model => "model",
        }
    }

    pub fn priority(&self) -> u8 {
        match self {
            CacheLayer::Prompt => 7,
            CacheLayer::Embedding => 6,
            CacheLayer::Retrieval => 5,
            CacheLayer::Tool => 4,
            CacheLayer::Http => 3,
            CacheLayer::Token => 2,
            CacheLayer::Model => 1,
        }
    }
}

pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub layer: CacheLayer,
    pub size_bytes: usize,
    pub created_at: Instant,
    pub access_count: u64,
    pub ttl: Option<Duration>,
    pub last_access: AtomicU64,
}

impl CacheEntry {
    pub fn new(key: String, value: Vec<u8>, layer: CacheLayer, ttl: Option<Duration>) -> Self {
        Self {
            size_bytes: value.len(),
            key,
            value,
            layer,
            created_at: Instant::now(),
            access_count: 0,
            ttl,
            last_access: AtomicU64::new(0),
        }
    }

    pub fn is_expired(&self) -> bool {
        self.ttl
            .map(|ttl| self.created_at.elapsed() > ttl)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub layer: CacheLayer,
    pub entries: usize,
    pub total_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

impl CacheStats {
    pub fn new(layer: CacheLayer) -> Self {
        Self {
            layer,
            entries: 0,
            total_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            hit_rate: 0.0,
        }
    }
}
