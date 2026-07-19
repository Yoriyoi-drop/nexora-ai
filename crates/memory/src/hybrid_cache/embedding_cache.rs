use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::cache_layer::{CacheLayer, CacheStats};

/// Embedding cache — cache hasil embedding vector untuk reuse
pub struct EmbeddingCache {
    entries: DashMap<String, Vec<f32>>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl EmbeddingCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: DashMap::new(),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        if let Some(entry) = self.entries.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn set(&self, key: String, embedding: Vec<f32>) {
        if self.entries.len() >= self.max_entries {
            let oldest = self.entries.iter().next().map(|e| e.key().clone());
            if let Some(k) = oldest {
                self.entries.remove(&k);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.entries.insert(key, embedding);
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            layer: CacheLayer::Embedding,
            entries: self.entries.len(),
            total_bytes: self.entries.len() * 4 * 1536,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}
