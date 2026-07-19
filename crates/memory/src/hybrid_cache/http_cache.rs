use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::cache_layer::{CacheLayer, CacheStats};

/// HTTP cache — cache response HTTP API calls
pub struct HttpCache {
    entries: DashMap<String, HttpCacheEntry>,
    max_entries: usize,
    default_ttl: Duration,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

struct HttpCacheEntry {
    response: Vec<u8>,
    status: u16,
    created_at: Instant,
    ttl: Duration,
}

impl HttpCache {
    pub fn new(max_entries: usize, default_ttl_secs: u64) -> Self {
        Self {
            entries: DashMap::new(),
            max_entries,
            default_ttl: Duration::from_secs(default_ttl_secs),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn get(&self, key: &str) -> Option<(Vec<u8>, u16)> {
        if let Some(entry) = self.entries.get(key) {
            if entry.created_at.elapsed() > entry.ttl {
                drop(entry);
                self.entries.remove(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some((entry.response.clone(), entry.status))
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn set(&self, key: String, response: Vec<u8>, status: u16) {
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.entries.iter().next().map(|e| e.key().clone()) {
                self.entries.remove(&oldest);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.entries.insert(
            key,
            HttpCacheEntry {
                response,
                status,
                created_at: Instant::now(),
                ttl: self.default_ttl,
            },
        );
    }

    pub fn set_with_ttl(&self, key: String, response: Vec<u8>, status: u16, ttl: Duration) {
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.entries.iter().next().map(|e| e.key().clone()) {
                self.entries.remove(&oldest);
            }
        }
        self.entries.insert(
            key,
            HttpCacheEntry {
                response,
                status,
                created_at: Instant::now(),
                ttl,
            },
        );
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            layer: CacheLayer::Http,
            entries: self.entries.len(),
            total_bytes: 0,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}
