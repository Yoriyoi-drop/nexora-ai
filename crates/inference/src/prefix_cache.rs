use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

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
}

#[derive(Debug, Clone)]
struct RadixNode {
    id: u64,
    parent: Option<u64>,
    children: HashMap<u64, u64>,
    label: Option<Vec<u32>>,
    value: Option<Vec<f32>>,
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
        if tokens.is_empty() {
            return;
        }

        let value_size = value.len() * std::mem::size_of::<f32>();
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
                let current = nodes.get(&current_id)
                    .expect("invariant: node must exist");
                current.children.get(&(token as u64)).copied()
            };

            match child_id {
                Some(cid) => {
                    current_id = cid;
                    let child = nodes.get_mut(&current_id)
                        .expect("invariant: child must exist");
                    child.access_count += 1;
                    child.last_access = Instant::now();
                }
                None => {
                    let new_id = Self::alloc_id();
                    let depth = nodes.get(&current_id)
                        .map(|n| n.depth + 1)
                        .unwrap_or(1);
                    let mut new_node = RadixNode::new(new_id, Some(current_id), Some(vec![token]), depth);
                    if tokens.last().map_or(false, |t| token == *t) {
                        new_node.value = Some(value.clone());
                    }
                    {
                        let parent = nodes.get_mut(&current_id)
                            .expect("invariant: parent must exist");
                        parent.children.insert(token as u64, new_id);
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

        let cached_value = if last_cached_id != self.root_id {
            self.hits.fetch_add(1, Ordering::Relaxed);
            nodes.get(&last_cached_id)
                .and_then(|n| n.value.clone())
                .unwrap_or_default()
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            Vec::new()
        };

        PrefixMatch {
            prefix_len: last_cached_len,
            node_id: last_cached_id,
            cached_value,
        }
    }

    pub async fn evict_expired(&self) -> usize {
        if !self.config.enable_eviction {
            return 0;
        }
        let mut nodes = self.nodes.write().await;
        let ttl = self.config.ttl;
        let before = nodes.len();
        let expired_ids: Vec<u64> = nodes.iter()
            .filter(|(_, n)| n.id != self.root_id && n.created_at.elapsed() > ttl)
            .map(|(id, _)| *id)
            .collect();

        for id in expired_ids {
            if let Some(node) = nodes.remove(&id) {
                if let Some(v) = node.value {
                    self.total_memory.fetch_sub(v.len() * std::mem::size_of::<f32>(), Ordering::Relaxed);
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

        if current_mem + new_entry_size <= self.config.max_cache_size
            && nodes.len() <= self.config.max_prefix_nodes {
            return;
        }

        let target_mem = self.config.max_cache_size.saturating_sub(new_entry_size);
        let mut evicted = 0usize;

        let mut leaf_ids: Vec<(u64, Instant)> = nodes.iter()
            .filter(|(_, n)| n.children.is_empty() && n.id != self.root_id)
            .map(|(id, n)| (*id, n.last_access))
            .collect();

        leaf_ids.sort_by(|a, b| a.1.cmp(&b.1));

        for (id, _) in leaf_ids {
            if self.total_memory.load(Ordering::Relaxed) <= target_mem
                && nodes.len() <= self.config.max_prefix_nodes {
                break;
            }
            if let Some(node) = nodes.remove(&id) {
                if let Some(v) = node.value {
                    self.total_memory.fetch_sub(v.len() * std::mem::size_of::<f32>(), Ordering::Relaxed);
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
        if total == 0 { 0.0 } else { hits as f32 / total as f32 }
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
}
