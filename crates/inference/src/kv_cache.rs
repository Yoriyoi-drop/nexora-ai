use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
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
    value: Arc<Vec<f32>>,
    access_count: AtomicUsize,
    created_at: Instant,
    last_access: AtomicU64,
}

struct CacheStore {
    entries: HashMap<u64, CacheEntry>,
    lru_order: BTreeSet<(u64, u64)>, // (last_access_nanos, hash)
}

fn timestamp_nanos() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos() as u64,
        Err(e) => {
            warn!("System time before UNIX_EPOCH: {}", e);
            0
        }
    }
}

pub struct KVCache {
    store: RwLock<CacheStore>,
    ttl: Duration,
    max_entries: usize,
    max_entry_bytes: usize,
    max_memory_bytes: usize,
    total_memory_used: AtomicUsize,
    stats_hits: AtomicUsize,
    stats_misses: AtomicUsize,
    stats_evictions: AtomicUsize,
    stats_ttl_evictions: AtomicUsize,
    cache_size_count: AtomicUsize,
    estimated_memory: AtomicUsize,
    last_cleanup: RwLock<Instant>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(CacheStore {
                entries: HashMap::new(),
                lru_order: BTreeSet::new(),
            }),
            ttl: Duration::from_secs(
                std::env::var("KV_CACHE_TTL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3600),
            ),
            max_entries: 10_000,
            max_entry_bytes: 8_388_608,
            max_memory_bytes: 1_073_741_824,
            total_memory_used: AtomicUsize::new(0),
            stats_hits: AtomicUsize::new(0),
            stats_misses: AtomicUsize::new(0),
            stats_evictions: AtomicUsize::new(0),
            stats_ttl_evictions: AtomicUsize::new(0),
            cache_size_count: AtomicUsize::new(0),
            estimated_memory: AtomicUsize::new(0),
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
        tracing::info!(
            "KVCache initialized with max_entries={}, ttl={:?}",
            self.max_entries,
            self.ttl
        );
        Ok(())
    }

    pub async fn get(&self, key: &[u8]) -> Option<Vec<f32>> {
        let hash = self.hash_key(key);
        {
            let store = self.store.read().await;
            if !store.entries.get(&hash).map_or(false, |e| e.key == key) {
                self.stats_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
        let mut store = self.store.write().await;
        let mut entry = store.entries.remove(&hash)?;
        if entry.created_at.elapsed() > self.ttl {
            store
                .lru_order
                .remove(&(entry.last_access.load(Ordering::Relaxed), hash));
            self.stats_ttl_evictions.fetch_add(1, Ordering::Relaxed);
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let now = timestamp_nanos();
        let old_ts = entry.last_access.swap(now, Ordering::Relaxed);
        entry.access_count.fetch_add(1, Ordering::Relaxed);
        let value = entry.value.as_ref().clone();
        store.lru_order.remove(&(old_ts, hash));
        store.lru_order.insert((now, hash));
        store.entries.insert(hash, entry);
        self.stats_hits.fetch_add(1, Ordering::Relaxed);
        Some(value)
    }

    pub async fn insert(&self, key: Vec<u8>, value: Vec<f32>) {
        let value = Arc::new(value);
        let entry_size = value.len() * std::mem::size_of::<f32>() + key.len();
        if entry_size > self.max_entry_bytes {
            warn!("KV insert rejected: {} bytes exceeds limit", entry_size);
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let hash = self.hash_key(&key);
        let mut store = self.store.write().await;

        // If key already exists, remove old entry's size first
        if let Some(old) = store.entries.get(&hash) {
            if old.key == key {
                let freed = old.value.len() * std::mem::size_of::<f32>() + old.key.len();
                self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
                self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
                self.cache_size_count.fetch_sub(1, Ordering::Relaxed);
            }
        }

        while store.entries.len() >= self.max_entries
            || (self.total_memory_used.load(Ordering::Relaxed) + entry_size) > self.max_memory_bytes
        {
            let lru_key = store.lru_order.iter().next().copied().map(|(_, h)| h);
            match lru_key {
                Some(k) => {
                    if let Some(removed) = store.entries.remove(&k) {
                        store
                            .lru_order
                            .remove(&(removed.last_access.load(Ordering::Relaxed), k));
                        let freed =
                            removed.value.len() * std::mem::size_of::<f32>() + removed.key.len();
                        self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
                        self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
                    }
                    self.cache_size_count.fetch_sub(1, Ordering::Relaxed);
                    self.stats_evictions.fetch_add(1, Ordering::Relaxed);
                }
                None => break,
            }
        }

        let now = timestamp_nanos();
        store.lru_order.insert((now, hash));
        store.entries.insert(
            hash,
            CacheEntry {
                key,
                value,
                access_count: AtomicUsize::new(0),
                created_at: Instant::now(),
                last_access: AtomicU64::new(now),
            },
        );
        self.cache_size_count.fetch_add(1, Ordering::Relaxed);
        self.total_memory_used
            .fetch_add(entry_size, Ordering::Relaxed);
        self.estimated_memory
            .fetch_add(entry_size, Ordering::Relaxed);

        drop(store);
        self.maybe_cleanup().await;
    }

    pub async fn contains(&self, key: &[u8]) -> bool {
        let hash = self.hash_key(key);
        let store = self.store.read().await;
        store.entries.get(&hash).map_or(false, |e| {
            e.key == key && e.created_at.elapsed() <= self.ttl
        })
    }

    pub async fn clear(&self) {
        let mut store = self.store.write().await;
        store.entries.clear();
        store.lru_order.clear();
        self.cache_size_count.store(0, Ordering::Relaxed);
        self.estimated_memory.store(0, Ordering::Relaxed);
    }

    pub async fn remove(&self, key: &[u8]) -> bool {
        let hash = self.hash_key(key);
        let mut store = self.store.write().await;
        if let Some(removed) = store.entries.remove(&hash) {
            store
                .lru_order
                .remove(&(removed.last_access.load(Ordering::Relaxed), hash));
            let freed = removed.value.len() * std::mem::size_of::<f32>() + removed.key.len();
            self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
            self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
            self.cache_size_count.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub async fn evict_expired(&self) -> usize {
        let mut store = self.store.write().await;
        let before = store.entries.len();
        let evicted_entries: Vec<u64> = store
            .entries
            .iter()
            .filter(|(_, e)| e.created_at.elapsed() > self.ttl)
            .map(|(k, _)| *k)
            .collect();
        for k in &evicted_entries {
            if let Some(removed) = store.entries.remove(k) {
                store
                    .lru_order
                    .remove(&(removed.last_access.load(Ordering::Relaxed), *k));
                let freed = removed.value.len() * std::mem::size_of::<f32>() + removed.key.len();
                self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
                self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
            }
        }
        self.cache_size_count
            .fetch_sub(evicted_entries.len(), Ordering::Relaxed);
        let evicted = evicted_entries.len();
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
        let cache_size = self.cache_size_count.load(Ordering::Relaxed);
        let estimated_memory_bytes = self.estimated_memory.load(Ordering::Relaxed);
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
        // Lock ordering: last_cleanup → store (dropped before evict_expired acquires store)
        let mut last = self.last_cleanup.write().await;
        let eviction_interval = Duration::from_secs(
            std::env::var("KV_CACHE_EVICTION_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
        );
        if last.elapsed() > eviction_interval {
            *last = Instant::now();
            drop(last);
            self.evict_expired().await;
        }
    }

    fn hash_key(&self, key: &[u8]) -> u64 {
        // FNV-1a hash (fast, no crypto overhead — internal cache, not user-exposed)
        let mut hash: u64 = 0xcbf29ce484222325;
        for &byte in key {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
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
        assert!(cache.store.read().await.entries.len() <= 2);
    }

    #[tokio::test]
    async fn test_oversized_entry_rejected() {
        let cache = KVCache::new().with_max_entry_bytes(10);
        cache.insert(b"big".to_vec(), vec![0.0; 100]).await;
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
        assert_eq!(cache.store.read().await.entries.len(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let cache = std::sync::Arc::new(KVCache::new());
        let mut handles = Vec::with_capacity(100);
        for i in 0..100 {
            let c = cache.clone();
            handles.push(tokio::spawn(async move {
                c.insert(
                    ("k".to_string() + &i.to_string()).into_bytes(),
                    vec![i as f32],
                )
                .await;
            }));
        }
        for h in handles {
            match h.await {
                Ok(_) => {}
                Err(e) => tracing::error!("KV cache worker panicked: {:?}", e),
            }
        }
        assert_eq!(cache.store.read().await.entries.len(), 100);
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
