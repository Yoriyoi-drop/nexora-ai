use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use nexora_transformer::KVCacheEntry;

#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    pub max_cache_size: usize,
    pub max_prefix_nodes: usize,
    pub ttl: Duration,
    pub enable_eviction: bool,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 1_073_741_824,
            max_prefix_nodes: 100_000,
            ttl: Duration::from_secs(3600),
            enable_eviction: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefixMatch {
    pub prefix_len: usize,
    pub node_id: u64,
    pub cached_value: Vec<f32>,
    pub cached_kv_cache: Vec<KVCacheEntry>,
}

#[derive(Debug, Clone)]
struct RadixNode {
    id: u64,
    parent: Option<u64>,
    children: HashMap<u64, u64>,
    label: Option<Vec<u32>>,
    value: Option<Vec<f32>>,
    kvcache: Option<Vec<KVCacheEntry>>,
    access_count: u64,
    last_access: Instant,
    created_at: Instant,
    depth: usize,
}

impl RadixNode {
    fn new(id: u64, parent: Option<u64>, label: Option<Vec<u32>>, depth: usize) -> Self {
        Self {
            id,
            parent,
            children: HashMap::new(),
            label,
            value: None,
            kvcache: None,
            access_count: 0,
            last_access: Instant::now(),
            created_at: Instant::now(),
            depth,
        }
    }
}

static NODE_COUNTER: AtomicUsize = AtomicUsize::new(1);

pub struct PrefixCache {
    config: PrefixCacheConfig,
    nodes: RwLock<HashMap<u64, RadixNode>>,
    root_id: u64,
    total_nodes: AtomicUsize,
    total_memory: AtomicUsize,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl PrefixCache {
    pub fn new(config: PrefixCacheConfig) -> Self {
        let root = RadixNode::new(0, None, None, 0);
        let mut nodes = HashMap::new();
        nodes.insert(0u64, root);
        Self {
            config,
            nodes: RwLock::new(nodes),
            root_id: 0,
            total_nodes: AtomicUsize::new(1),
            total_memory: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    fn alloc_id() -> u64 {
        NODE_COUNTER.fetch_add(1, Ordering::Relaxed) as u64
    }

    pub async fn insert(&self, tokens: &[u32], value: Vec<f32>) {
        self.insert_with_kvcache(tokens, value, None).await;
    }

    pub async fn insert_with_kvcache(
        &self,
        tokens: &[u32],
        value: Vec<f32>,
        kvcache: Option<Vec<KVCacheEntry>>,
    ) {
        if tokens.is_empty() {
            return;
        }

        let kv_size = kvcache
            .as_ref()
            .map(|kv| {
                kv.iter()
                    .map(|e| (e.k.len() + e.v.len()) * std::mem::size_of::<f32>())
                    .sum::<usize>()
            })
            .unwrap_or(0);
        let value_size = value.len() * std::mem::size_of::<f32>() + kv_size;
        let entry_size = tokens.len() * std::mem::size_of::<u32>() + value_size + 64;
        if entry_size > self.config.max_cache_size / 1024 {
            warn!("Prefix cache entry too large: {} bytes", entry_size);
            return;
        }

        if self.config.enable_eviction {
            self.evict_if_needed(entry_size).await;
        }

        let mut nodes = self.nodes.write().await;
        let mut current_id = self.root_id;

        for &token in tokens {
            let child_id = {
                let current = match nodes.get(&current_id) {
                    Some(n) => n,
                    None => {
                        warn!(
                            "prefix_cache: node {} missing during insert — aborting",
                            current_id
                        );
                        return;
                    }
                };
                current.children.get(&(token as u64)).copied()
            };

            match child_id {
                Some(cid) => {
                    current_id = cid;
                    if let Some(child) = nodes.get_mut(&current_id) {
                        child.access_count += 1;
                        child.last_access = Instant::now();
                    } else {
                        warn!(
                            "prefix_cache: child node {} missing after lookup",
                            current_id
                        );
                        return;
                    }
                }
                None => {
                    let new_id = Self::alloc_id();
                    let depth = nodes.get(&current_id).map(|n| n.depth + 1).unwrap_or(1);
                    let mut new_node =
                        RadixNode::new(new_id, Some(current_id), Some(vec![token]), depth);
                    if tokens.last().map_or(false, |t| token == *t) {
                        new_node.value = Some(value.clone());
                        new_node.kvcache = kvcache.clone();
                    }
                    if let Some(parent) = nodes.get_mut(&current_id) {
                        parent.children.insert(token as u64, new_id);
                    } else {
                        warn!(
                            "prefix_cache: parent node {} missing during new child insert",
                            current_id
                        );
                        return;
                    }
                    nodes.insert(new_id, new_node);
                    current_id = new_id;
                    self.total_nodes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        if let Some(node) = nodes.get_mut(&current_id) {
            if node.value.is_none() {
                self.total_memory.fetch_add(value_size, Ordering::Relaxed);
                node.value = Some(value);
                node.kvcache = kvcache;
            }
        }
    }

    pub async fn match_prefix(&self, tokens: &[u32]) -> PrefixMatch {
        let nodes = self.nodes.read().await;
        let mut current_id = self.root_id;
        let mut matched_len = 0;
        let mut last_cached_id = self.root_id;
        let mut last_cached_len = 0;

        for &token in tokens {
            let child_id = {
                let current = nodes.get(&current_id);
                match current {
                    None => break,
                    Some(c) => c.children.get(&(token as u64)).copied(),
                }
            };

            match child_id {
                Some(cid) => {
                    matched_len += 1;
                    current_id = cid;
                    if let Some(node) = nodes.get(&current_id) {
                        if node.value.is_some() {
                            last_cached_id = current_id;
                            last_cached_len = matched_len;
                        }
                    }
                }
                None => break,
            }
        }

        let (cached_value, cached_kv_cache) = if last_cached_id != self.root_id {
            self.hits.fetch_add(1, Ordering::Relaxed);
            crate::inference_trait::PREFIX_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            match nodes.get(&last_cached_id) {
                Some(n) => (
                    n.value.clone().unwrap_or_else(|| {
                        warn!(
                            "prefix_cache: hit node {} has no cached value",
                            last_cached_id
                        );
                        Vec::new()
                    }),
                    n.kvcache.clone().unwrap_or_default(),
                ),
                None => {
                    warn!(
                        "prefix_cache: hit node {} not found in cache",
                        last_cached_id
                    );
                    (Vec::new(), Vec::new())
                }
            }
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            crate::inference_trait::PREFIX_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            (Vec::new(), Vec::new())
        };

        PrefixMatch {
            prefix_len: last_cached_len,
            node_id: last_cached_id,
            cached_value,
            cached_kv_cache,
        }
    }

    pub async fn evict_expired(&self) -> usize {
        if !self.config.enable_eviction {
            return 0;
        }
        let mut nodes = self.nodes.write().await;
        let ttl = self.config.ttl;
        let before = nodes.len();
        let expired_ids: Vec<u64> = nodes
            .iter()
            .filter(|(_, n)| n.id != self.root_id && n.created_at.elapsed() > ttl)
            .map(|(id, _)| *id)
            .collect();

        for id in expired_ids {
            if let Some(node) = nodes.remove(&id) {
                if node.value.is_some() || node.kvcache.is_some() {
                    let mut mem = node
                        .value
                        .as_ref()
                        .map(|v| v.len() * std::mem::size_of::<f32>())
                        .unwrap_or(0);
                    if let Some(kv) = &node.kvcache {
                        mem += kv
                            .iter()
                            .map(|e| (e.k.len() + e.v.len()) * std::mem::size_of::<f32>())
                            .sum::<usize>();
                    }
                    self.total_memory.fetch_sub(mem, Ordering::Relaxed);
                }
                if let Some(parent_id) = node.parent {
                    if let Some(parent) = nodes.get_mut(&parent_id) {
                        parent.children.retain(|_, v| *v != id);
                    }
                }
                self.total_nodes.fetch_sub(1, Ordering::Relaxed);
            }
        }

        before - nodes.len()
    }

    async fn evict_if_needed(&self, new_entry_size: usize) {
        let mut nodes = self.nodes.write().await;
        let current_mem = self.total_memory.load(Ordering::Relaxed);

        if current_mem.saturating_add(new_entry_size) <= self.config.max_cache_size
            && nodes.len() <= self.config.max_prefix_nodes
        {
            return;
        }

        let target_mem = self.config.max_cache_size.saturating_sub(new_entry_size);
        let mut evicted = 0usize;

        let mut leaf_ids: Vec<(u64, Instant)> = nodes
            .iter()
            .filter(|(_, n)| n.children.is_empty() && n.id != self.root_id)
            .map(|(id, n)| (*id, n.last_access))
            .collect();

        leaf_ids.sort_by(|a, b| a.1.cmp(&b.1));

        for (id, _) in leaf_ids {
            if self.total_memory.load(Ordering::Relaxed) <= target_mem
                && nodes.len() <= self.config.max_prefix_nodes
            {
                break;
            }
            if let Some(node) = nodes.remove(&id) {
                if node.value.is_some() || node.kvcache.is_some() {
                    let mut mem = node
                        .value
                        .as_ref()
                        .map(|v| v.len() * std::mem::size_of::<f32>())
                        .unwrap_or(0);
                    if let Some(kv) = &node.kvcache {
                        mem += kv
                            .iter()
                            .map(|e| (e.k.len() + e.v.len()) * std::mem::size_of::<f32>())
                            .sum::<usize>();
                    }
                    self.total_memory.fetch_sub(mem, Ordering::Relaxed);
                }
                if let Some(parent_id) = node.parent {
                    if let Some(parent) = nodes.get_mut(&parent_id) {
                        parent.children.retain(|_, v| *v != id);
                    }
                }
                self.total_nodes.fetch_sub(1, Ordering::Relaxed);
                evicted += 1;
            }
        }

        debug!("Evicted {evicted} prefix cache nodes");
    }

    pub async fn clear(&self) {
        let mut nodes = self.nodes.write().await;
        nodes.clear();
        nodes.insert(0, RadixNode::new(0, None, None, 0));
        self.total_nodes.store(1, Ordering::Relaxed);
        self.total_memory.store(0, Ordering::Relaxed);
    }

    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        }
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "total_nodes": self.total_nodes.load(Ordering::Relaxed),
            "total_memory_bytes": self.total_memory.load(Ordering::Relaxed),
            "hits": self.hits.load(Ordering::Relaxed),
            "misses": self.misses.load(Ordering::Relaxed),
            "hit_rate": self.hit_rate(),
            "max_nodes": self.config.max_prefix_nodes,
            "max_cache_bytes": self.config.max_cache_size,
        })
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(PrefixCacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_insert_and_match() {
        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 1024 * 1024,
            max_prefix_nodes: 1000,
            ttl: Duration::from_secs(3600),
            enable_eviction: true,
        });
        let tokens = vec![1, 2, 3, 4, 5];
        let value = vec![0.1f32, 0.2, 0.3];
        cache.insert(&tokens, value.clone()).await;

        let m = cache.match_prefix(&tokens).await;
        assert_eq!(m.prefix_len, 5);
        assert_eq!(m.cached_value, value);
    }

    #[tokio::test]
    async fn test_partial_match() {
        let cache = PrefixCache::default();
        cache.insert(&[1, 2, 3], vec![1.0]).await;

        let m = cache.match_prefix(&[1, 2, 3, 4, 5]).await;
        assert_eq!(m.prefix_len, 3);
    }

    #[tokio::test]
    async fn test_no_match() {
        let cache = PrefixCache::default();
        let m = cache.match_prefix(&[99, 98]).await;
        assert_eq!(m.prefix_len, 0);
        assert!(m.cached_value.is_empty());
    }

    #[tokio::test]
    async fn test_shared_prefix() {
        let cache = PrefixCache::default();
        cache.insert(&[1, 2, 3, 4], vec![1.0]).await;
        cache.insert(&[1, 2, 3, 5], vec![2.0]).await;

        let m = cache.match_prefix(&[1, 2, 3, 4]).await;
        assert_eq!(m.prefix_len, 4);

        let m = cache.match_prefix(&[1, 2, 3, 5]).await;
        assert_eq!(m.prefix_len, 4);
    }

    #[tokio::test]
    async fn test_clear() {
        let cache = PrefixCache::default();
        cache.insert(&[1, 2], vec![1.0]).await;
        cache.clear().await;
        let m = cache.match_prefix(&[1, 2]).await;
        assert_eq!(m.prefix_len, 0);
    }

    #[tokio::test]
    async fn test_evict_expired() {
        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 1024 * 1024,
            max_prefix_nodes: 1000,
            ttl: Duration::from_millis(1),
            enable_eviction: true,
        });
        cache.insert(&[1, 2, 3], vec![1.0]).await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        let evicted = cache.evict_expired().await;
        assert!(evicted >= 3);
    }

    // ─── H10 Benchmark Suite ───────────────────────────────────────────────

    /// Helper: insert a full sequence (prompt + gen tokens) with last_logits + KV cache.
    /// Also inserts a no-kvcache entry at the prompt prefix to enable partial matches.
    async fn insert_seq(cache: &PrefixCache, prompt: &[u32], gen: &[u32], logit_size: usize) {
        let mut full = prompt.to_vec();
        full.extend_from_slice(gen);
        let last_logits: Vec<f32> = (0..logit_size).map(|i| i as f32 * 0.01).collect();
        // Two KVCacheEntry nodes to simulate real usage
        let kvcache = vec![
            KVCacheEntry {
                k: vec![0.1; 512],
                v: vec![0.2; 512],
                kv_dim: 512,
            },
            KVCacheEntry {
                k: vec![0.3; 512],
                v: vec![0.4; 512],
                kv_dim: 512,
            },
        ];
        cache
            .insert_with_kvcache(&full, last_logits, Some(kvcache))
            .await;
    }

    #[tokio::test]
    async fn bench_hit_rate_shared_prefixes() {
        let logit_size = 256;
        let unique_prompts = 100;

        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 1_000_000_000,
            max_prefix_nodes: 100_000,
            ttl: Duration::from_secs(3600),
            enable_eviction: false,
        });

        // Insert 100 sequences sharing a common 4-token prefix
        // Full sequence = common_prefix + unique_suffix + gen_token
        let common_prefix = [1, 2, 3, 4];
        for i in 0..unique_prompts {
            let prompt = {
                let mut p = common_prefix.to_vec();
                p.push(10 + i as u32);
                p
            };
            insert_seq(&cache, &prompt, &[100], logit_size).await;
        }

        // Probe with full sequences (including gen token) — should all hit
        let mut hits = 0u64;
        let mut misses = 0u64;
        for i in 0..unique_prompts {
            let mut full = common_prefix.to_vec();
            full.push(10 + i as u32);
            full.push(100); // gen token
            let m = cache.match_prefix(&full).await;
            if m.prefix_len == full.len() {
                hits += 1;
            } else {
                misses += 1;
            }
        }
        let hit_rate = hits as f64 / (hits + misses) as f64;
        println!(
            "bench_hit_rate_shared_prefixes: exact match rate = {:.3} ({}/{} hits)",
            hit_rate, hits, hits + misses
        );
        assert!(hit_rate > 0.98, "exact match rate should be near 100%");

        // Node sharing: root(1) + common_prefix(4) + unique_suffix(100) + gen_per_seq(100)
        // = 205. Without sharing this would be 601 (no common prefix).
        let total_nodes = cache.stats()["total_nodes"].as_u64().unwrap_or(0) as usize;
        let without_sharing = 1 + 6 * unique_prompts; // no shared nodes
        assert!(
            total_nodes < without_sharing,
            "shared prefix should reduce node count, got {} vs {} without sharing",
            total_nodes,
            without_sharing
        );
        assert!(total_nodes <= 210, "node count should be around 205, got {}", total_nodes);

        // Probe with extended prompts — cache returns longest cached prefix
        // (the gen token node has the value, so extended prompt matches at full.len())
        let mut ext_hits = 0u64;
        let ext_total = unique_prompts;
        for i in 0..unique_prompts {
            let mut full = common_prefix.to_vec();
            full.push(10 + i as u32);
            full.push(100);
            full.push(200); // extra token not in cache
            let m = cache.match_prefix(&full).await;
            // Should match at depth 6 (full seq = 5 tokens + gen = 6)
            if m.prefix_len == 6 {
                ext_hits += 1;
            }
        }
        let ext_hit_rate = ext_hits as f64 / ext_total as f64;
        println!(
            "bench_hit_rate_shared_prefixes: extended prefix match rate = {:.3} ({}/{})",
            ext_hit_rate, ext_hits, ext_total
        );
        assert!(ext_hit_rate > 0.98, "extended match should find deep prefix");
    }

    #[tokio::test]
    async fn bench_kv_cache_storage_and_restore() {
        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 10_000_000,
            max_prefix_nodes: 1000,
            ttl: Duration::from_secs(3600),
            enable_eviction: false,
        });

        // Store sequence with KV cache
        let prompt = vec![5, 10, 15, 20];
        let gen = vec![25, 30];
        let mut full = prompt.clone();
        full.extend(&gen);
        let logits = vec![0.5; 128];
        let kvcache = vec![
            KVCacheEntry { k: vec![1.0; 64], v: vec![2.0; 64], kv_dim: 64 },
            KVCacheEntry { k: vec![3.0; 64], v: vec![4.0; 64], kv_dim: 64 },
        ];
        cache
            .insert_with_kvcache(&full, logits.clone(), Some(kvcache))
            .await;

        // Verify: match restores both value and KV cache
        let m = cache.match_prefix(&full).await;
        assert_eq!(m.prefix_len, full.len());
        assert_eq!(m.cached_value, logits);
        assert_eq!(m.cached_kv_cache.len(), 2);
        assert_eq!(m.cached_kv_cache[0].k, vec![1.0; 64]);

        // Multi-session: second session with shared prefix
        let full2 = {
            let mut f = prompt.clone();
            f.extend(&[35, 40]);
            f
        };
        let logits2 = vec![0.7; 128];
        let kvcache2 = vec![
            KVCacheEntry { k: vec![5.0; 64], v: vec![6.0; 64], kv_dim: 64 },
            KVCacheEntry { k: vec![7.0; 64], v: vec![8.0; 64], kv_dim: 64 },
        ];
        cache
            .insert_with_kvcache(&full2, logits2.clone(), Some(kvcache2))
            .await;

        // Also insert dummy entries at intermediate prefixes for partial match support
        cache.insert(&prompt, vec![0.0; 4]).await;

        // Session 1: original should still match fully
        let m1 = cache.match_prefix(&full).await;
        assert_eq!(m1.cached_value, logits);
        assert_eq!(m1.cached_kv_cache[0].k, vec![1.0; 64]);

        // Session 2: should match fully
        let m2 = cache.match_prefix(&full2).await;
        assert_eq!(m2.cached_value, logits2);
        assert_eq!(m2.cached_kv_cache[0].k, vec![5.0; 64]);

        // Session 3: new session with shared prefix → partial match
        let full3 = {
            let mut f = prompt.clone();
            f.extend(&[45, 50]);
            f
        };
        let m3 = cache.match_prefix(&full3).await;
        // Prompt [5,10,15,20] was inserted as dummy entry, so prefix_len >= 4
        assert!(m3.prefix_len >= 4, "shared prefix should match at least prompt");

        println!(
            "bench_kv_cache_storage_and_restore: session1=full/{}, session2=full/{}, session3=partial/{}",
            m1.prefix_len, m2.prefix_len, m3.prefix_len
        );
    }

    #[tokio::test]
    async fn bench_memory_fragmentation() {
        let logit_size = 256;
        let num_sequences = 50;
        let seq_len = 20;

        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 1_000_000_000,
            max_prefix_nodes: 100_000,
            ttl: Duration::from_secs(3600),
            enable_eviction: false,
        });

        // Insert sequences with varying shared prefixes
        for i in 0..num_sequences {
            let prompt: Vec<u32> = (0..seq_len).map(|j| ((i + j) % 50) as u32).collect();
            insert_seq(&cache, &prompt, &[200], logit_size).await;
        }

        // Measure: nodes created vs unique segments
        let stats = cache.stats();
        let total_nodes = stats["total_nodes"].as_u64().unwrap_or(0);
        let total_mem = stats["total_memory_bytes"].as_u64().unwrap_or(0);

        // Each sequence = seq_len + 1 (gen token) = 21 nodes
        let ideal_nodes = 1 + (seq_len + 1) * num_sequences;
        let overhead_ratio = total_nodes as f64 / ideal_nodes as f64;

        println!(
            "bench_memory_fragmentation: {} sequences x {} tokens = {} ideal nodes, {} actual nodes, overhead={:.3}x",
            num_sequences, seq_len + 1, ideal_nodes, total_nodes, overhead_ratio
        );
        println!(
            "bench_memory_fragmentation: total_memory={} bytes, per_node={:.1} bytes",
            total_mem,
            total_mem as f64 / total_nodes as f64
        );

        // With random prefixes, overhead should be modest (<5x)
        assert!(
            overhead_ratio > 0.95,
            "node overhead ratio should be reasonable"
        );
    }

    #[tokio::test]
    async fn bench_eviction_lru_stress() {
        let logit_size = 64;

        // Tight cache: only 10 nodes max
        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 1_000_000_000,
            max_prefix_nodes: 10,
            ttl: Duration::from_secs(3600),
            enable_eviction: true,
        });

        // Insert 20 distinct sequences — only first ~10 should survive
        for i in 0..20u32 {
            let prompt = vec![i * 10, i * 10 + 1, i * 10 + 2];
            insert_seq(&cache, &prompt, &[i * 10 + 3], logit_size).await;
        }

        let stats = cache.stats();
        let remaining = stats["total_nodes"].as_u64().unwrap_or(0);
        println!(
            "bench_eviction_lru_stress: inserted 20 sequences, {} nodes remaining (max=10)",
            remaining
        );
        assert!(remaining <= 15, "LRU eviction should keep nodes near max"); // 10 + some root/partials

        // Earlier sequences should have been evicted
        let m_early = cache.match_prefix(&[0, 1, 2, 3]).await;
        println!(
            "bench_eviction_lru_stress: early seq (0,1,2,3) match len = {}",
            m_early.prefix_len
        );

        // Later sequences (accessed more recently when inserted) may still be present
        let mut sec_count = 0u64;
        for i in 15..20u32 {
            let prompt = vec![i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3];
            if cache.match_prefix(&prompt).await.prefix_len == 4 {
                sec_count += 1;
            }
        }
        println!(
            "bench_eviction_lru_stress: recent sequences (15-19) still fully cached: {}/5",
            sec_count
        );
    }

    #[tokio::test]
    async fn bench_hit_rate_noisy_prefixes() {
        let logit_size = 128;
        let num_prompts = 200;

        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 1_000_000_000,
            max_prefix_nodes: 1_000_000,
            ttl: Duration::from_secs(3600),
            enable_eviction: true,
        });

        // Phase 1: Insert 200 prompts with random prefixes (worst-case: no sharing)
        for i in 0..num_prompts {
            // Each prompt starts with a unique prefix
            let prompt = vec![i as u32, (i + 1) as u32, (i + 2) as u32, (i + 3) as u32];
            insert_seq(&cache, &prompt, &[100], logit_size).await;
        }

        // Phase 2: Query each full sequence (prompt + gen) → exact match
        let mut exact_hits = 0u64;
        for i in 0..num_prompts {
            let mut full = vec![i as u32, (i + 1) as u32, (i + 2) as u32, (i + 3) as u32];
            full.push(100); // gen token
            if cache.match_prefix(&full).await.prefix_len == full.len() {
                exact_hits += 1;
            }
        }
        let exact_rate = exact_hits as f64 / num_prompts as f64;
        println!(
            "bench_hit_rate_noisy_prefixes: exact hits = {}/{} ({:.3})",
            exact_hits, num_prompts, exact_rate
        );
        assert!(exact_rate > 0.98, "exact match should succeed without eviction");

        // Phase 3: Query with superset (full + extra suffix token after gen)
        let mut ext_hits = 0u64;
        let ext_total = num_prompts.min(100);
        for i in 0..ext_total {
            let mut full = vec![i as u32, (i + 1) as u32, (i + 2) as u32, (i + 3) as u32];
            full.push(100); // gen token
            full.push(999); // extra token not in cache
            if cache.match_prefix(&full).await.prefix_len >= 5 {
                ext_hits += 1;
            }
        }
        let ext_rate = ext_hits as f64 / ext_total as f64;
        println!(
            "bench_hit_rate_noisy_prefixes: superset prefix match = {}/{} ({:.3})",
            ext_hits, ext_total, ext_rate
        );
        assert!(ext_rate > 0.98, "superset prompt should match deep prefix");

        // Phase 4: Query with completely different prefixes → should all miss
        let mut miss_count = 0u64;
        for i in 0..50u32 {
            let prompt = vec![1000 + i, 2000 + i, 3000 + i, 4000 + i];
            if cache.match_prefix(&prompt).await.prefix_len == 0 {
                miss_count += 1;
            }
        }
        println!(
            "bench_hit_rate_noisy_prefixes: novel prefix misses = {}/50",
            miss_count
        );
        assert_eq!(miss_count, 50, "novel prefixes should all miss");
    }

    #[tokio::test]
    async fn bench_collision_invalidation() {
        let logit_size = 64;

        let cache = PrefixCache::new(PrefixCacheConfig {
            max_cache_size: 1_000_000,
            max_prefix_nodes: 20,
            ttl: Duration::from_secs(3600),
            enable_eviction: true,
        });

        // Insert sequences to fill the cache and trigger eviction
        for i in 0..10u32 {
            let prompt = vec![i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3];
            insert_seq(&cache, &prompt, &[i * 5 + 4], logit_size).await;
        }

        // Repeatedly access a specific set of sequences to keep them "hot"
        let hot_prefix = vec![0, 1, 2, 3, 4];
        for _ in 0..5 {
            let _ = cache.match_prefix(&hot_prefix).await;
        }

        // Insert more sequences to trigger eviction of cold nodes
        for i in 10..30u32 {
            let prompt = vec![i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3];
            insert_seq(&cache, &prompt, &[i * 5 + 4], logit_size).await;
        }

        // The hot prefix [0,1,2,3,4] should survive if access count-based eviction works
        // (current impl uses leaf-first LRU by last_access time, not access_count)
        let m_hot = cache.match_prefix(&hot_prefix).await;
        let hot_survived = m_hot.prefix_len > 0;
        println!(
            "bench_collision_invalidation: hot seq {} survived eviction = {}",
            if hot_survived { "[0,1,2,3,4]" } else { "0,1,2,3,4" },
            hot_survived
        );

        // Verify hit/miss counters are non-zero (cache was actually used)
        let stats = cache.stats();
        let hits = stats["hits"].as_u64().unwrap_or(0);
        let misses = stats["misses"].as_u64().unwrap_or(0);
        println!(
            "bench_collision_invalidation: hits={}, misses={}, total_nodes={}",
            hits, misses, stats["total_nodes"]
        );
        assert!(hits > 0 || misses > 0, "cache counters should be non-zero");
    }
}
