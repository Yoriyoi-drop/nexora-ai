use dashmap::DashMap;
use std::time::Duration;

use super::cache_layer::{CacheEntry, CacheLayer, CacheStats};
use std::sync::atomic::{AtomicU64, Ordering};

/// Prompt cache — cache hasil tokenisasi prompt untuk reuse
pub struct PromptCache {
    entries: DashMap<String, CacheEntry>,
    max_size: usize,
    current_size: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl PromptCache {
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            entries: DashMap::new(),
            max_size: max_size_mb * 1024 * 1024,
            current_size: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(entry) = self.entries.get(key) {
            if entry.is_expired() {
                drop(entry);
                self.entries.remove(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.value.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn set(&self, key: String, value: Vec<u8>, ttl: Option<Duration>) {
        let size = value.len();
        if size > self.max_size {
            return;
        }

        // Evict if needed
        while self.current_size.load(Ordering::Relaxed) + size as u64 > self.max_size as u64 {
            let oldest_key = self.entries.iter().min_by_key(|e| e.created_at).map(|e| e.key().clone());
            if let Some(k) = oldest_key {
                if let Some((_, entry)) = self.entries.remove(&k) {
                    self.current_size
                        .fetch_sub(entry.size_bytes as u64, Ordering::Relaxed);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }

        let entry = CacheEntry::new(key.clone(), value, CacheLayer::Prompt, ttl);
        self.current_size
            .fetch_add(entry.size_bytes as u64, Ordering::Relaxed);
        self.entries.insert(key, entry);
    }

    pub fn clear(&self) {
        self.entries.clear();
        self.current_size.store(0, Ordering::Relaxed);
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            layer: CacheLayer::Prompt,
            entries: self.entries.len(),
            total_bytes: self.current_size.load(Ordering::Relaxed) as usize,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}
