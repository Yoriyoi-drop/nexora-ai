use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_size: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f32,
}

struct CacheEntry {
    key: Vec<u8>,
    value: Vec<f32>,
    access_count: AtomicUsize,
    created_at: chrono::DateTime<chrono::Utc>,
}

pub struct KVCache {
    entries: RwLock<HashMap<u64, CacheEntry>>,
    max_entries: usize,
    stats_hits: AtomicUsize,
    stats_misses: AtomicUsize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            max_entries: 10_000,
            stats_hits: AtomicUsize::new(0),
            stats_misses: AtomicUsize::new(0),
        }
    }

    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    pub async fn initialize(&mut self) -> Result<(), anyhow::Error> {
        Ok(())
    }

    pub async fn get(&self, key: &[u8]) -> Option<Vec<f32>> {
        let hash = self.hash_key(key);
        let entries = self.entries.read().await;
        if let Some(entry) = entries.get(&hash) {
            if entry.key == key {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                self.stats_hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.value.clone());
            }
        }
        self.stats_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn insert(&self, key: Vec<u8>, value: Vec<f32>) {
        let hash = self.hash_key(&key);
        let mut entries = self.entries.write().await;

        if entries.len() >= self.max_entries {
            self.evict_lru(&mut entries);
        }

        entries.insert(hash, CacheEntry {
            key,
            value,
            access_count: AtomicUsize::new(0),
            created_at: chrono::Utc::now(),
        });
    }

    pub async fn contains(&self, key: &[u8]) -> bool {
        let hash = self.hash_key(key);
        let entries = self.entries.read().await;
        entries.get(&hash).map_or(false, |e| e.key == key)
    }

    pub async fn clear(&self) {
        let mut entries = self.entries.write().await;
        entries.clear();
    }

    pub async fn remove(&self, key: &[u8]) -> bool {
        let hash = self.hash_key(key);
        let mut entries = self.entries.write().await;
        entries.remove(&hash).is_some()
    }

    pub async fn shutdown(&self) -> Result<(), anyhow::Error> {
        self.clear().await;
        Ok(())
    }

    pub fn get_stats(&self) -> CacheStats {
        let hits = self.stats_hits.load(Ordering::Relaxed);
        let misses = self.stats_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            cache_size: 0,
            cache_hits: hits,
            cache_misses: misses,
            hit_rate: if total > 0 { hits as f32 / total as f32 } else { 0.0 },
        }
    }

    fn hash_key(&self, key: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn evict_lru(&self, entries: &mut HashMap<u64, CacheEntry>) {
        let lru_key = entries.iter()
            .min_by_key(|(_, e)| e.access_count.load(Ordering::Relaxed))
            .map(|(k, _)| *k);
        if let Some(key) = lru_key {
            entries.remove(&key);
        }
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}
