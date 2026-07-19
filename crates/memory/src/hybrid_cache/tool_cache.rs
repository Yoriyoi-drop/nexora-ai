use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::cache_layer::{CacheLayer, CacheStats};

/// Tool cache — cache hasil eksekusi tool (idempotent tools)
pub struct ToolCache {
    entries: DashMap<String, serde_json::Value>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ToolCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: DashMap::new(),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        if let Some(entry) = self.entries.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn set(&self, key: String, result: serde_json::Value) {
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.entries.iter().next().map(|e| e.key().clone()) {
                self.entries.remove(&oldest);
            }
        }
        self.entries.insert(key, result);
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            layer: CacheLayer::Tool,
            entries: self.entries.len(),
            total_bytes: 0,
            hits,
            misses,
            evictions: 0,
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}
