use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::cache_layer::{CacheLayer, CacheStats};

/// Token cache — cache mapping token ID ↔ text
pub struct TokenCache {
    id_to_text: DashMap<u32, String>,
    text_to_id: DashMap<String, u32>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl TokenCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            id_to_text: DashMap::new(),
            text_to_id: DashMap::new(),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn get_text(&self, id: u32) -> Option<String> {
        if let Some(entry) = self.id_to_text.get(&id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn get_id(&self, text: &str) -> Option<u32> {
        if let Some(entry) = self.text_to_id.get(text) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(*entry)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn set(&self, id: u32, text: String) {
        if self.id_to_text.len() >= self.max_entries {
            if let Some(oldest) = self.id_to_text.iter().next().map(|e| *e.key()) {
                self.id_to_text.remove(&oldest);
            }
        }
        self.id_to_text.insert(id, text.clone());
        self.text_to_id.insert(text, id);
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            layer: CacheLayer::Token,
            entries: self.id_to_text.len(),
            total_bytes: 0,
            hits,
            misses,
            evictions: 0,
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}
