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
    pub shard_count: usize,
    pub eviction_policy: EvictionPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    None,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

struct CacheEntry {
    key: Vec<u8>,
    value: Arc<Vec<f32>>,
    access_count: AtomicUsize,
    created_at: Instant,
    last_access: AtomicU64,
}

struct CacheShard {
    entries: HashMap<u64, CacheEntry>,
    lru_order: BTreeSet<(u64, u64)>,
    size_count: usize,
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

fn hash_key(key: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn shard_index(key: &[u8], shard_count: usize) -> usize {
    let h = hash_key(key);
    (h as usize) % shard_count
}

pub struct KVCache {
    shards: Vec<Arc<RwLock<CacheShard>>>,
    shard_count: usize,
    eviction_policy: EvictionPolicy,
    ttl: Duration,
    max_entries: usize,
    max_entry_bytes: usize,
    max_memory_bytes: usize,
    total_memory_used: AtomicUsize,
    stats_hits: AtomicUsize,
    stats_misses: AtomicUsize,
    stats_evictions: AtomicUsize,
    stats_ttl_evictions: AtomicUsize,
    estimated_memory: AtomicUsize,
    last_cleanup: RwLock<Instant>,
    shutdown: RwLock<bool>,
}

impl KVCache {
    pub fn new() -> Self {
        Self::with_shards(1)
    }

    pub fn with_shards(shard_count: usize) -> Self {
        let shard_count = shard_count.max(1);
        let mut shards = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            shards.push(Arc::new(RwLock::new(CacheShard {
                entries: HashMap::new(),
                lru_order: BTreeSet::new(),
                size_count: 0,
            })));
        }
        Self {
            shards,
            shard_count,
            eviction_policy: EvictionPolicy::default(),
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
            estimated_memory: AtomicUsize::new(0),
            last_cleanup: RwLock::new(Instant::now()),
            shutdown: RwLock::new(false),
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

    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    pub fn with_shard_count(mut self, count: usize) -> Self {
        let count = count.max(1);
        if count != self.shard_count {
            let mut shards = Vec::with_capacity(count);
            for _ in 0..count {
                shards.push(Arc::new(RwLock::new(CacheShard {
                    entries: HashMap::new(),
                    lru_order: BTreeSet::new(),
                    size_count: 0,
                })));
            }
            self.shards = shards;
            self.shard_count = count;
        }
        self
    }

    pub async fn initialize(&self) -> Result<(), anyhow::Error> {
        tracing::info!(
            "KVCache initialized shards={} policy={:?} max_entries={} ttl={:?}",
            self.shard_count,
            self.eviction_policy,
            self.max_entries,
            self.ttl
        );
        Ok(())
    }

    fn select_shard(&self, key: &[u8]) -> usize {
        shard_index(key, self.shard_count)
    }

    pub async fn get(&self, key: &[u8]) -> Option<Vec<f32>> {
        if *self.shutdown.read().await {
            return None;
        }
        let idx = self.select_shard(key);
        let hash = hash_key(key);
        {
            let store = self.shards[idx].read().await;
            if !store.entries.get(&hash).map_or(false, |e| e.key == key) {
                self.stats_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
        let mut store = self.shards[idx].write().await;
        let entry = store.entries.remove(&hash)?;
        if entry.created_at.elapsed() > self.ttl {
            store
                .lru_order
                .remove(&(entry.last_access.load(Ordering::Relaxed), hash));
            store.size_count -= 1;
            let freed = entry.value.len() * std::mem::size_of::<f32>() + entry.key.len();
            self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
            self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
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
        if *self.shutdown.read().await {
            return;
        }
        let value = Arc::new(value);
        let entry_size = value.len() * std::mem::size_of::<f32>() + key.len();
        if entry_size > self.max_entry_bytes {
            warn!("KV insert rejected: {} bytes exceeds limit", entry_size);
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let hash = hash_key(&key);
        let idx = self.select_shard(&key);
        let mut store = self.shards[idx].write().await;

        if let Some(old) = store.entries.get(&hash) {
            if old.key == key {
                let freed = old.value.len() * std::mem::size_of::<f32>() + old.key.len();
                self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
                self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
                store.size_count -= 1;
            }
        }

        while store.entries.len() >= self.max_entries
            || (self.total_memory_used.load(Ordering::Relaxed) + entry_size) > self.max_memory_bytes
        {
            let evict_key = self.select_victim(&store);
            match evict_key {
                Some(k) => {
                    if let Some(removed) = store.entries.remove(&k) {
                        store
                            .lru_order
                            .remove(&(removed.last_access.load(Ordering::Relaxed), k));
                        store.size_count -= 1;
                        let freed =
                            removed.value.len() * std::mem::size_of::<f32>() + removed.key.len();
                        self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
                        self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
                    }
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
        store.size_count += 1;
        self.total_memory_used
            .fetch_add(entry_size, Ordering::Relaxed);
        self.estimated_memory
            .fetch_add(entry_size, Ordering::Relaxed);

        drop(store);
        self.maybe_cleanup().await;
    }

    fn select_victim(&self, store: &CacheShard) -> Option<u64> {
        match self.eviction_policy {
            EvictionPolicy::LRU => {
                store.lru_order.iter().next().copied().map(|(_, h)| h)
            }
            EvictionPolicy::LFU => {
                store
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.access_count.load(Ordering::Relaxed))
                    .map(|(h, _)| *h)
            }
            EvictionPolicy::FIFO => {
                store
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.created_at)
                    .map(|(h, _)| *h)
            }
            EvictionPolicy::Random => {
                let keys: Vec<u64> = store.entries.keys().copied().collect();
                if keys.is_empty() {
                    None
                } else {
                    let idx = (timestamp_nanos() as usize) % keys.len();
                    Some(keys[idx])
                }
            }
            EvictionPolicy::None => None,
        }
    }

    pub async fn contains(&self, key: &[u8]) -> bool {
        let idx = self.select_shard(key);
        let hash = hash_key(key);
        let store = self.shards[idx].read().await;
        store.entries.get(&hash).map_or(false, |e| {
            e.key == key && e.created_at.elapsed() <= self.ttl
        })
    }

    pub async fn clear(&self) {
        for shard in &self.shards {
            let mut store = shard.write().await;
            store.entries.clear();
            store.lru_order.clear();
            store.size_count = 0;
        }
    }

    pub async fn remove(&self, key: &[u8]) -> bool {
        let idx = self.select_shard(key);
        let hash = hash_key(key);
        let mut store = self.shards[idx].write().await;
        if let Some(removed) = store.entries.remove(&hash) {
            store
                .lru_order
                .remove(&(removed.last_access.load(Ordering::Relaxed), hash));
            store.size_count -= 1;
            let freed = removed.value.len() * std::mem::size_of::<f32>() + removed.key.len();
            self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
            self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub async fn evict_expired(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            let mut store = shard.write().await;
            let evicted: Vec<u64> = store
                .entries
                .iter()
                .filter(|(_, e)| e.created_at.elapsed() > self.ttl)
                .map(|(k, _)| *k)
                .collect();
            for k in &evicted {
                if let Some(removed) = store.entries.remove(k) {
                    store
                        .lru_order
                        .remove(&(removed.last_access.load(Ordering::Relaxed), *k));
                    store.size_count -= 1;
                    let freed = removed.value.len() * std::mem::size_of::<f32>() + removed.key.len();
                    self.total_memory_used.fetch_sub(freed, Ordering::Relaxed);
                    self.estimated_memory.fetch_sub(freed, Ordering::Relaxed);
                }
            }
            total += evicted.len();
            self.stats_ttl_evictions
                .fetch_add(evicted.len(), Ordering::Relaxed);
        }
        total
    }

    pub async fn shutdown(&self) -> Result<(), anyhow::Error> {
        *self.shutdown.write().await = true;
        self.clear().await;
        Ok(())
    }

    pub fn get_stats(&self) -> CacheStats {
        let hits = self.stats_hits.load(Ordering::Relaxed);
        let misses = self.stats_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let cache_size: usize = self
            .shards
            .iter()
            .map(|s| s.try_read().map(|g| g.size_count).unwrap_or(0))
            .sum();
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
            shard_count: self.shard_count,
            eviction_policy: self.eviction_policy.clone(),
        }
    }

    async fn maybe_cleanup(&self) {
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
        let total: usize = cache
            .shards
            .iter()
            .map(|s| s.try_read().map(|g| g.entries.len()).unwrap_or(0))
            .sum();
        assert!(total <= 2);
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
        let total: usize = cache
            .shards
            .iter()
            .map(|s| s.try_read().map(|g| g.entries.len()).unwrap_or(0))
            .sum();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let cache = std::sync::Arc::new(KVCache::with_shards(4));
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
        let total: usize = cache
            .shards
            .iter()
            .map(|s| s.try_read().map(|g| g.entries.len()).unwrap_or(0))
            .sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_stats() {
        let cache = KVCache::new();
        let stats = cache.get_stats();
        assert_eq!(stats.cache_size, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[tokio::test]
    async fn test_sharded_insert_and_get() {
        let cache = KVCache::with_shards(8);
        assert_eq!(cache.shard_count, 8);
        cache.insert(b"key_a".to_vec(), vec![1.0]).await;
        cache.insert(b"key_b".to_vec(), vec![2.0]).await;
        cache.insert(b"key_c".to_vec(), vec![3.0]).await;
        assert_eq!(cache.get(b"key_a").await, Some(vec![1.0]));
        assert_eq!(cache.get(b"key_b").await, Some(vec![2.0]));
        assert_eq!(cache.get(b"key_c").await, Some(vec![3.0]));
    }

    #[tokio::test]
    async fn test_eviction_policy_lfu() {
        let cache = KVCache::new()
            .with_max_entries(2)
            .with_eviction_policy(EvictionPolicy::LFU);
        cache.insert(b"a".to_vec(), vec![1.0]).await;
        cache.insert(b"b".to_vec(), vec![2.0]).await;
        // Access 'a' more than 'b'
        cache.get(b"a").await;
        cache.get(b"a").await;
        cache.get(b"b").await;
        // Insert 'c' should evict 'b' (LFU: b accessed once, a accessed twice)
        cache.insert(b"c".to_vec(), vec![3.0]).await;
        assert!(cache.contains(b"a").await);
        assert!(cache.contains(b"c").await);
    }
}
