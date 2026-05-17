use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::warn;

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_size: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f32,
    pub estimated_memory_bytes: usize,
    pub max_entry_bytes: usize,
    pub max_entries: usize,
    pub evictions: usize,
    pub ttl_evictions: usize,
}

struct CacheEntry {
    key: Vec<u8>,
    value: Vec<f32>,
    access_count: AtomicUsize,
    created_at: Instant,
    last_access: Instant,
}

pub struct KVCache {
    entries: RwLock<HashMap<u64, CacheEntry>>,
    ttl: Duration,
    max_entries: usize,
    max_entry_bytes: usize,
    max_memory_bytes: usize,
    stats_hits: AtomicUsize,
    stats_misses: AtomicUsize,
    stats_evictions: AtomicUsize,
    stats_ttl_evictions: AtomicUsize,
    last_cleanup: RwLock<Instant>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(3600),
            max_entries: 10_000,
            max_entry_bytes: 8_388_608,
            max_memory_bytes: 1_073_741_824,
            stats_hits: AtomicUsize::new(0),
            stats_misses: AtomicUsize::new(0),
            stats_evictions: AtomicUsize::new(0),
            stats_ttl_evictions: AtomicUsize::new(0),
            last_cleanup: RwLock::new(Instant::now()),
        }
    }

    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    pub fn with_max_entry_bytes(mut self, max: usize) -> Self {
        self.max_entry_bytes = max;
        self
    }

    pub fn with_max_memory_bytes(mut self, max: usize) -> Self {
        self.max_memory_bytes = max;
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    pub async fn initialize(&self) -> Result<(), anyhow::Error> {
        Ok(())
    }

    pub async fn get(&self, key: &[u8]) -> Option<Vec<f32>> {
        let hash = self.hash_key(key);
        {
            let entries = self.entries.read().await;
            if let Some(entry) = entries.get(&hash) {
                if entry.key == key {
                    if entry.created_at.elapsed() > self.ttl {
                        self.stats_ttl_evictions.fetch_add(1, Ordering::Relaxed);
                        self.stats_misses.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                    entry.access_count.fetch_add(1, Ordering::Relaxed);
                    self.stats_hits.fetch_add(1, Ordering::Relaxed);
                    return Some(entry.value.clone());
                }
            }
        }
        // On miss, promote to write lock for eviction path (rare)
        self.stats_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn insert(&self, key: Vec<u8>, value: Vec<f32>) {
        let entry_size = value.len() * std::mem::size_of::<f32>() + key.len();
        if entry_size > self.max_entry_bytes {
            warn!("KV insert rejected: {} bytes exceeds limit", entry_size);
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let hash = self.hash_key(&key);
        let mut entries = self.entries.write().await;

        let total_bytes: usize = entries
            .values()
            .map(|e| e.value.len() * std::mem::size_of::<f32>() + e.key.len())
            .sum();

        while entries.len() >= self.max_entries || (total_bytes + entry_size) > self.max_memory_bytes {
            let lru_key = entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| *k);
            match lru_key {
                Some(k) => {
                    entries.remove(&k);
                    self.stats_evictions.fetch_add(1, Ordering::Relaxed);
                }
                None => break,
            }
        }

        entries.insert(
            hash,
            CacheEntry {
                key,
                value,
                access_count: AtomicUsize::new(0),
                created_at: Instant::now(),
                last_access: Instant::now(),
            },
        );

        drop(entries);
        self.maybe_cleanup().await;
    }

    pub async fn contains(&self, key: &[u8]) -> bool {
        let hash = self.hash_key(key);
        let entries = self.entries.read().await;
        entries
            .get(&hash)
            .map_or(false, |e| e.key == key && e.created_at.elapsed() <= self.ttl)
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

    pub async fn evict_expired(&self) -> usize {
        let mut entries = self.entries.write().await;
        let before = entries.len();
        entries.retain(|_, e| e.created_at.elapsed() <= self.ttl);
        let evicted = before - entries.len();
        self.stats_ttl_evictions
            .fetch_add(evicted, Ordering::Relaxed);
        evicted
    }

    pub async fn shutdown(&self) -> Result<(), anyhow::Error> {
        self.clear().await;
        Ok(())
    }

    pub fn get_stats(&self) -> CacheStats {
        let hits = self.stats_hits.load(Ordering::Relaxed);
        let misses = self.stats_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let entries = self.entries.try_read();
        let (cache_size, estimated_memory_bytes) = match entries {
            Ok(guard) => {
                let mem: usize = guard
                    .values()
                    .map(|e| e.value.len() * std::mem::size_of::<f32>() + e.key.len())
                    .sum();
                (guard.len(), mem)
            }
            Err(_) => (0, 0),
        };
        CacheStats {
            cache_size,
            cache_hits: hits,
            cache_misses: misses,
            hit_rate: if total > 0 {
                hits as f32 / total as f32
            } else {
                0.0
            },
            estimated_memory_bytes,
            max_entry_bytes: self.max_entry_bytes,
            max_entries: self.max_entries,
            evictions: self.stats_evictions.load(Ordering::Relaxed),
            ttl_evictions: self.stats_ttl_evictions.load(Ordering::Relaxed),
        }
    }

    async fn maybe_cleanup(&self) {
        let mut last = self.last_cleanup.write().await;
        if last.elapsed() > Duration::from_secs(300) {
            *last = Instant::now();
            drop(last);
            self.evict_expired().await;
        }
    }

    fn hash_key(&self, key: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_insert_and_get() {
        let cache = KVCache::new();
        cache.insert(b"key1".to_vec(), vec![1.0, 2.0, 3.0]).await;
        assert_eq!(cache.get(b"key1").await, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(cache.get(b"nonexistent").await, None);
    }

    #[tokio::test]
    async fn test_ttl_eviction() {
        let cache = KVCache::new().with_ttl(Duration::from_millis(10));
        cache.insert(b"key1".to_vec(), vec![1.0]).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(cache.get(b"key1").await, None);
    }

    #[tokio::test]
    async fn test_max_entries_eviction() {
        let cache = KVCache::new().with_max_entries(2);
        cache.insert(b"k1".to_vec(), vec![1.0]).await;
        cache.insert(b"k2".to_vec(), vec![2.0]).await;
        cache.insert(b"k3".to_vec(), vec![3.0]).await;
        assert!(cache.entries.read().await.len() <= 2);
    }

    #[tokio::test]
    async fn test_oversized_entry_rejected() {
        let cache = KVCache::new().with_max_entry_bytes(10);
        cache
            .insert(b"big".to_vec(), vec![0.0; 100])
            .await;
        let stats = cache.get_stats();
        assert_eq!(stats.cache_size, 0);
    }

    #[tokio::test]
    async fn test_contains() {
        let cache = KVCache::new();
        cache.insert(b"key1".to_vec(), vec![1.0]).await;
        assert!(cache.contains(b"key1").await);
        assert!(!cache.contains(b"key2").await);
    }

    #[tokio::test]
    async fn test_remove() {
        let cache = KVCache::new();
        cache.insert(b"key1".to_vec(), vec![1.0]).await;
        assert!(cache.remove(b"key1").await);
        assert!(!cache.remove(b"key1").await);
    }

    #[tokio::test]
    async fn test_clear() {
        let cache = KVCache::new();
        cache.insert(b"k1".to_vec(), vec![1.0]).await;
        cache.insert(b"k2".to_vec(), vec![2.0]).await;
        cache.clear().await;
        assert_eq!(cache.entries.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let cache = std::sync::Arc::new(KVCache::new());
        let mut handles = Vec::with_capacity(100);
        for i in 0..100 {
            let c = cache.clone();
            handles.push(tokio::spawn(async move {
                c.insert(("k".to_string() + &i.to_string()).into_bytes(), vec![i as f32])
                    .await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(cache.entries.read().await.len(), 100);
    }

    #[test]
    fn test_stats() {
        let cache = KVCache::new();
        let stats = cache.get_stats();
        assert_eq!(stats.cache_size, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }
}
