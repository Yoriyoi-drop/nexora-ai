// ─── PagedKVCacheProvider ──────────────────────────────────────────────────────
// Bridge between PagedKVCache (block-based) and KVCacheProvider (flat per-layer).
// All providers in the same engine share the same underlying PagedKVCache instance,
// enabling cross-sequence block sharing (prefix DAG via refcount + copy-on-write).

use std::sync::{Arc, Mutex};

use nexora_transformer::{KVCacheEntry, KVCacheProvider};

use crate::paged_cache::{PagedCacheConfig, PagedKVCache};

/// Wraps a `PagedKVCache` and presents it as a `KVCacheProvider`.
pub struct PagedKVCacheProvider {
    seq_id: u64,
    cache: Arc<Mutex<PagedKVCache>>,
    num_layers: usize,
    /// Lazy flat mirror for `as_cpu_entries()`.
    flat: Vec<KVCacheEntry>,
    dirty: bool,
    total_tokens: usize,
    /// Number of tokens from a shared prefix that were not computed.
    /// These tokens' KV blocks are shared with another sequence via refcount.
    shared_prefix_tokens: usize,
}

impl PagedKVCacheProvider {
    pub fn new(seq_id: u64, config: PagedCacheConfig) -> Self {
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(seq_id);
        Self {
            seq_id,
            cache: Arc::new(Mutex::new(cache)),
            num_layers: config.num_layers,
            flat: Vec::new(),
            dirty: false,
            total_tokens: 0,
            shared_prefix_tokens: 0,
        }
    }

    pub fn new_shared(
        seq_id: u64,
        cache: Arc<Mutex<PagedKVCache>>,
        num_layers: usize,
    ) -> Self {
        {
            let mut guard = cache.lock().unwrap();
            guard.register_sequence(seq_id);
        }
        Self {
            seq_id,
            cache,
            num_layers,
            flat: Vec::new(),
            dirty: false,
            total_tokens: 0,
            shared_prefix_tokens: 0,
        }
    }

    pub fn seq_id(&self) -> u64 {
        self.seq_id
    }

    pub fn cache_arc(&self) -> &Arc<Mutex<PagedKVCache>> {
        &self.cache
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn shared_prefix_tokens(&self) -> usize {
        self.shared_prefix_tokens
    }

    fn ensure_flat_synced(&mut self) {
        if !self.dirty {
            return;
        }
        let guard = self.cache.lock().unwrap();
        if let Some(entries) = guard.to_flat_cache(self.seq_id) {
            self.flat = entries;
        }
        self.dirty = false;
    }
}

impl KVCacheProvider for PagedKVCacheProvider {
    fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        let token_pos = self.total_tokens;
        let mut guard = self.cache.lock().unwrap();
        guard.append(self.seq_id, layer_idx, token_pos, k, v);
        drop(guard);
        self.total_tokens += 1;
        self.dirty = true;
    }

    fn get_k(&self, _layer_idx: usize) -> &[f32] {
        &[]
    }

    fn get_v(&self, _layer_idx: usize) -> &[f32] {
        &[]
    }

    fn seq_len(&self, _layer_idx: usize) -> usize {
        self.total_tokens
    }

    fn clear(&mut self) {
        let mut guard = self.cache.lock().unwrap();
        guard.remove_sequence(self.seq_id);
        guard.register_sequence(self.seq_id);
        drop(guard);
        self.flat.clear();
        self.total_tokens = 0;
        self.shared_prefix_tokens = 0;
        self.dirty = false;
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn as_cpu_entries(&mut self) -> Option<&mut Vec<KVCacheEntry>> {
        self.ensure_flat_synced();
        Some(&mut self.flat)
    }

    #[cfg(feature = "gpu")]
    fn as_gpu_entries(&mut self) -> Option<&mut Vec<nexora_transformer::GpuKVCacheEntry>> {
        None
    }
}

impl Drop for PagedKVCacheProvider {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.cache.lock() {
            guard.remove_sequence(self.seq_id);
        }
    }
}
