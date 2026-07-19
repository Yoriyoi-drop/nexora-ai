use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::cache_layer::{CacheLayer, CacheStats};

/// Model cache — cache model weights yang sudah di-load
pub struct ModelCache {
    entries: DashMap<String, ModelCacheEntry>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

struct ModelCacheEntry {
    #[allow(dead_code)]
    model_type: String,
    #[allow(dead_code)]
    load_time_ms: u64,
    size_bytes: u64,
}

impl ModelCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: DashMap::new(),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn is_cached(&self, model_id: &str) -> bool {
        if self.entries.contains_key(model_id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    pub fn register(&self, model_id: String, model_type: String, size_bytes: u64) {
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.entries.iter().next().map(|e| e.key().clone()) {
                self.entries.remove(&oldest);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.entries.insert(
            model_id,
            ModelCacheEntry {
                model_type,
                load_time_ms: 0,
                size_bytes,
            },
        );
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            layer: CacheLayer::Model,
            entries: self.entries.len(),
            total_bytes: self.entries.iter().map(|e| e.size_bytes as usize).sum(),
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}
