// ─── PagedKVCacheProvider ──────────────────────────────────────────────────────
// Bridge between PagedKVCache (block-based) and KVCacheProvider (flat per-layer).
// All providers in the same engine share the same underlying PagedKVCache instance,
// enabling cross-sequence block sharing (prefix DAG via refcount + copy-on-write).
//
// GPU-native forward path:
//   PagedKVCacheProvider now maintains a `Vec<GpuKVCacheEntry>` GPU mirror that is
//   exposed via `as_gpu_entries()`. This enables the inference engine's GPU-native
//   forward methods (forward_sample_gpu, forward_batched_sample_gpu,
//   forward_gpu_batched_prefill_full) to read/write K/V directly on GPU without
//   going through the CPU paged cache blocks.
//
// Flow:
//   1. append() writes K/V to BOTH paged cache blocks (CPU) AND GPU entries
//   2. as_gpu_entries() returns the GPU entries for GPU-native forward
//   3. GPU forward modifies GPU entries in-place (appends new K/V per token)
//   4. sync_gpu_to_paged() reads back the NEW tokens from GPU entries and
//      appends them to paged cache blocks for block-level consistency

use std::sync::Arc;
use std::sync::RwLock;

use nexora_transformer::{KVCacheEntry, KVCacheProvider};
use tracing::warn;

use crate::paged_cache::{PagedCacheConfig, PagedKVCache};

#[cfg(feature = "gpu")]
use nexora_transformer::GpuKVCacheEntry;

/// Wraps a `PagedKVCache` and presents it as a `KVCacheProvider`.
///
/// Single-write design: when GPU entries are active, `append()` writes ONLY to
/// GPU (never to the CPU paged cache). The CPU paged cache blocks are written
/// only by `sync_gpu_to_paged()` (lazy batched sync). This eliminates redundant
/// CPU-side RwLock acquisition + block allocation on every `append()` call.
pub struct PagedKVCacheProvider {
    seq_id: u64,
    cache: Arc<RwLock<PagedKVCache>>,
    num_layers: usize,
    /// Lazy flat mirror for `as_cpu_entries()`.
    flat: Vec<KVCacheEntry>,
    /// Number of tokens already synced into `flat`. Only tokens at positions >=
    /// `flat_synced_tokens` need to be appended on the next `ensure_flat_synced()`.
    flat_synced_tokens: usize,
    dirty: bool,
    total_tokens: usize,
    /// Number of tokens from a shared prefix that were not computed.
    /// These tokens' KV blocks are shared with another sequence via refcount.
    shared_prefix_tokens: usize,
    /// GPU mirror: GpuKVCacheEntry per layer exposed via as_gpu_entries().
    /// When GPU is available, `append()` writes ONLY to these entries
    /// (single-write). The CPU paged cache is synced lazily via
    /// `sync_gpu_to_paged()`.
    #[cfg(feature = "gpu")]
    gpu_entries: Vec<GpuKVCacheEntry>,
    #[cfg(feature = "gpu")]
    gpu_num_kv_heads: usize,
    #[cfg(feature = "gpu")]
    gpu_head_dim: usize,
    #[cfg(feature = "gpu")]
    gpu_max_seq_len: usize,
    /// When true, GPU entries are the primary storage. `append()` skips the
    /// CPU paged cache write, and `ensure_flat_synced()` reads from GPU entries
    /// instead of from paged cache blocks.
    #[cfg(feature = "gpu")]
    gpu_primary: bool,
}

impl PagedKVCacheProvider {
    pub fn new(seq_id: u64, config: PagedCacheConfig) -> Self {
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(seq_id);
        Self {
            seq_id,
            cache: Arc::new(RwLock::new(cache)),
            num_layers: config.num_layers,
            flat: Vec::new(),
            flat_synced_tokens: 0,
            dirty: false,
            total_tokens: 0,
            shared_prefix_tokens: 0,
            #[cfg(feature = "gpu")]
            gpu_entries: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_num_kv_heads: config.num_kv_heads,
            #[cfg(feature = "gpu")]
            gpu_head_dim: config.head_dim,
            #[cfg(feature = "gpu")]
            gpu_max_seq_len: config.max_seq_len,
            #[cfg(feature = "gpu")]
            gpu_primary: false,
        }
    }

    pub fn new_shared(
        seq_id: u64,
        cache: Arc<RwLock<PagedKVCache>>,
        num_layers: usize,
    ) -> Self {
        // Read dimensions from the underlying paged cache config
        let (num_kv_heads, head_dim, max_seq_len) = tokio::task::block_in_place(|| {
            let guard = match cache.read() {
                Ok(g) => g,
                Err(e) => {
                    warn!("paged cache lock poisoned in new_shared: {}", e);
                    e.into_inner()
                }
            };
            let cfg = guard.config();
            (cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len)
        });
        tokio::task::block_in_place(|| {
            let mut guard = match cache.write() {
                Ok(g) => g,
                Err(e) => {
                    warn!("paged cache lock poisoned in new_shared (register): {}", e);
                    e.into_inner()
                }
            };
            guard.register_sequence(seq_id);
        });
        Self {
            seq_id,
            cache,
            num_layers,
            flat: Vec::new(),
            flat_synced_tokens: 0,
            dirty: false,
            total_tokens: 0,
            shared_prefix_tokens: 0,
            #[cfg(feature = "gpu")]
            gpu_entries: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_num_kv_heads: num_kv_heads,
            #[cfg(feature = "gpu")]
            gpu_head_dim: head_dim,
            #[cfg(feature = "gpu")]
            gpu_max_seq_len: max_seq_len,
            #[cfg(feature = "gpu")]
            gpu_primary: false,
        }
    }

    pub fn seq_id(&self) -> u64 {
        self.seq_id
    }

    pub fn cache_arc(&self) -> &Arc<RwLock<PagedKVCache>> {
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

        // Single-write: when GPU is primary, read flat from GPU entries directly
        #[cfg(feature = "gpu")]
        if self.gpu_primary && !self.gpu_entries.is_empty() && self.total_tokens > 0 {
            self.flat = self.read_flat_from_gpu();
            self.flat_synced_tokens = self.total_tokens;
            self.dirty = false;
            return;
        }

        // Initialize flat cache lazily — build incrementally from paged blocks
        // instead of calling to_flat_cache() (avoids ~2GB allocation spike).
        if self.flat.is_empty() || self.flat.len() != self.num_layers {
            let (kv_elems, num_kv_heads, head_dim) = self.read_cache_dims();
            if kv_elems == 0 {
                return;
            }
            self.flat = (0..self.num_layers)
                .map(|_| KVCacheEntry {
                    k: Vec::with_capacity(self.total_tokens * kv_elems),
                    v: Vec::with_capacity(self.total_tokens * kv_elems),
                    kv_dim: kv_elems,
                    num_kv_heads,
                    head_dim,
                    ..Default::default()
                })
                .collect();
            self.flat_synced_tokens = 0;
        }

        // Incremental: append only new token positions from paged blocks
        if self.flat_synced_tokens < self.total_tokens {
            let guard = match self.cache.read() {
                Ok(g) => g,
                Err(e) => {
                    warn!("paged cache lock poisoned in ensure_flat_synced: {}", e);
                    return;
                }
            };
            let kv_elems = guard.config().num_kv_heads * guard.config().head_dim;
            for pos in self.flat_synced_tokens..self.total_tokens {
                for layer in 0..self.num_layers.min(self.flat.len()) {
                    if let Some((k_row, v_row)) = guard.read(self.seq_id, layer, pos) {
                        self.flat[layer].k.extend(k_row);
                        self.flat[layer].v.extend(v_row);
                        self.flat[layer].kv_dim = kv_elems;
                    }
                }
            }
            self.flat_synced_tokens = self.total_tokens;
        }
        self.dirty = false;
    }

    fn read_cache_dims(&self) -> (usize, usize, usize) {
        match self.cache.read() {
            Ok(g) => {
                let c = g.config();
                (c.num_kv_heads * c.head_dim, c.num_kv_heads, c.head_dim)
            }
            Err(e) => {
                warn!("paged cache lock poisoned in read_cache_dims: {}", e);
                (0, 0, 0)
            }
        }
    }

    /// Read all tokens from GPU entries into a flat Vec<KVCacheEntry>.
    /// Uses bulk GPU readback per layer (1 transfer per layer instead of N per token).
    #[cfg(feature = "gpu")]
    fn read_flat_from_gpu(&self) -> Vec<KVCacheEntry> {
        let kv_elems = self.gpu_num_kv_heads * self.gpu_head_dim;
        let mut entries = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let (mut k, mut v);
            if layer < self.gpu_entries.len() {
                let seq_len = self.total_tokens.min(self.gpu_entries[layer].seq_len);
                if seq_len > 0 {
                    if let Some(tokens) = self.gpu_entries[layer].read_tokens_bulk(0, seq_len) {
                        k = Vec::with_capacity(seq_len * kv_elems);
                        v = Vec::with_capacity(seq_len * kv_elems);
                        for (k_row, v_row) in tokens {
                            k.extend_from_slice(&k_row);
                            v.extend_from_slice(&v_row);
                        }
                    } else {
                        k = Vec::new();
                        v = Vec::new();
                    }
                } else {
                    k = Vec::new();
                    v = Vec::new();
                }
            } else {
                k = Vec::new();
                v = Vec::new();
            }
            entries.push(KVCacheEntry {
                k,
                v,
                kv_dim: kv_elems,
                num_kv_heads: self.gpu_num_kv_heads,
                head_dim: self.gpu_head_dim,
                ..Default::default()
            });
        }
        entries
    }

    /// Initialize GPU entries if they haven't been created yet.
    /// Creates one `GpuKVCacheEntry` per layer with pre-allocated GPU buffers
    /// matching the model dimensions.
    #[cfg(feature = "gpu")]
    fn ensure_gpu_entries(&mut self) {
        if !self.gpu_entries.is_empty() {
            return;
        }
        let ctx = match nexora_deeplearning::autograd::gpu::GpuContext::global() {
            Ok(c) => c,
            _ => return,
        };
        for _ in 0..self.num_layers {
            match GpuKVCacheEntry::new(
                &ctx,
                self.gpu_num_kv_heads,
                self.gpu_head_dim,
                self.gpu_max_seq_len,
                false,
            ) {
                Ok(entry) => self.gpu_entries.push(entry),
                Err(e) => {
                    tracing::warn!("PagedKVCacheProvider: failed to create GPU entry: {e}");
                    self.gpu_entries.clear();
                    return;
                }
            }
        }

        // Sync existing paged cache data into the freshly created GPU entries
        self.sync_paged_to_gpu();

        // Switch to single-write mode: future append() writes to GPU only
        self.gpu_primary = true;
    }

    /// Sync all existing paged cache data INTO GPU entries.
    /// Called after GPU entries are first created to populate them with
    /// any K/V data that was already stored in the paged cache blocks.
    #[cfg(feature = "gpu")]
    fn sync_paged_to_gpu(&mut self) {
        if self.gpu_entries.is_empty() || self.total_tokens == 0 {
            return;
        }
        let ctx = match nexora_deeplearning::autograd::gpu::GpuContext::global() {
            Ok(c) => c,
            _ => return,
        };

        // For each layer, read existing tokens from paged cache and upload to GPU
        let guard = match self.cache.read() {
            Ok(g) => g,
            Err(e) => {
                warn!("paged cache lock poisoned in sync_paged_to_gpu: {}", e);
                return;
            }
        };
        for layer in 0..self.num_layers.min(self.gpu_entries.len()) {
            for pos in 0..self.total_tokens {
                let (k_row, v_row) = match guard.read(self.seq_id, layer, pos) {
                    Some(kv) => kv,
                    None => continue,
                };
                // Upload directly to GPU entry at position pos
                if let (Ok(k_arr), Ok(v_arr)) = (
                    ndarray::ArrayD::from_shape_vec(
                        vec![1, self.gpu_num_kv_heads, self.gpu_head_dim],
                        k_row,
                    ),
                    ndarray::ArrayD::from_shape_vec(
                        vec![1, self.gpu_num_kv_heads, self.gpu_head_dim],
                        v_row,
                    ),
                ) {
                    if let (Ok(k_gpu), Ok(v_gpu)) = (
                        nexora_deeplearning::autograd::gpu::GpuTensor::from_cpu(&k_arr),
                        nexora_deeplearning::autograd::gpu::GpuTensor::from_cpu(&v_arr),
                    ) {
                        let _ = self.gpu_entries[layer].append(&ctx, &k_gpu, &v_gpu);
                    }
                }
            }
        }
        drop(guard);
    }

    /// After a GPU-native forward pass (which modified `gpu_entries` by appending
    /// new K/V), read the newly added tokens' K/V back from GPU and append them to
    /// the paged cache blocks. This keeps the block-level data consistent for
    /// prefix sharing, defragmentation, and CPU fallback.
    ///
    /// Only reads back tokens that haven't been synced yet (i.e., tokens at
    /// positions >= `total_tokens`). Optimized with bulk GPU readback:
    /// reads all new tokens per layer in a single GPU→CPU transfer,
    /// then batch-writes to paged cache with one lock acquisition per layer.
    #[cfg(feature = "gpu")]
    pub fn sync_gpu_to_paged(&mut self) {
        if self.gpu_entries.is_empty() {
            return;
        }
        let gpu_seq_len = self.gpu_entries[0].seq_len;
        if gpu_seq_len <= self.total_tokens {
            return;
        }

        let kv_elems = self.gpu_num_kv_heads * self.gpu_head_dim;
        let start = self.total_tokens;
        for layer in 0..self.gpu_entries.len().min(self.num_layers) {
            let tokens = match self.gpu_entries[layer].read_tokens_bulk(start, gpu_seq_len) {
                Some(t) => t,
                None => continue,
            };
            let mut guard = match self.cache.write() {
                Ok(g) => g,
                Err(e) => {
                    warn!("paged cache lock poisoned in sync_gpu_to_paged: {}", e);
                    continue;
                }
            };
            for (i, (k_vec, v_vec)) in tokens.iter().enumerate() {
                guard.append(self.seq_id, layer, start + i, k_vec, v_vec);
            }
            drop(guard);
        }
        self.total_tokens = gpu_seq_len;
    }

    /// Convenience: sync all paged cache providers in a map.
    /// Called after each generation step when paged cache + GPU is active.
    #[cfg(feature = "gpu")]
    pub fn sync_all_gpu_to_paged(caches: &mut std::collections::HashMap<u64, Box<dyn KVCacheProvider>>) {
        for (_seq_id, provider) in caches.iter_mut() {
            if let Some(paged) = provider.as_any_mut().downcast_mut::<PagedKVCacheProvider>() {
                paged.sync_gpu_to_paged();
            }
        }
    }

    /// Convenience: sync all paged cache providers in a slice of boxed caches.
    /// Used after forward_batched_sample_gpu when the caches are in a Vec.
    #[cfg(feature = "gpu")]
    pub fn sync_all_gpu_to_paged_slice(caches: &mut [Box<dyn KVCacheProvider>]) {
        for cache in caches.iter_mut() {
            if let Some(paged) = cache.as_any_mut().downcast_mut::<PagedKVCacheProvider>() {
                paged.sync_gpu_to_paged();
            }
        }
    }

    /// Import prefix KV cache from flat KVCacheEntry format into paged cache blocks.
    /// This avoids calling to_flat_cache() when restoring a prefix cache hit —
    /// the prefix data is written directly into paged blocks so forward_paged()
    /// can read it without any flat cache allocation.
    pub fn import_prefix_kv_cache(&mut self, prefix_kv: &[KVCacheEntry]) {
        if prefix_kv.is_empty() {
            return;
        }
        let prefix_tokens = prefix_kv[0].k.len() / prefix_kv[0].kv_dim.max(1);
        if prefix_tokens == 0 {
            return;
        }

        let mut guard = match self.cache.write() {
            Ok(g) => g,
            Err(e) => {
                warn!("paged cache lock poisoned in import_prefix_kv_cache: {}", e);
                return;
            }
        };

        for (layer, entry) in prefix_kv.iter().enumerate() {
            if layer >= self.num_layers {
                break;
            }
            let kv_dim = entry.kv_dim;
            if kv_dim == 0 {
                continue;
            }
            for pos in 0..prefix_tokens {
                let start = pos * kv_dim;
                let end = start + kv_dim;
                if end > entry.k.len() || end > entry.v.len() {
                    break;
                }
                let k_row = &entry.k[start..end];
                let v_row = &entry.v[start..end];
                guard.append(self.seq_id, layer, pos, k_row, v_row);
            }
        }
        drop(guard);

        self.total_tokens = self.total_tokens.max(prefix_tokens);
        self.shared_prefix_tokens = prefix_tokens;
        self.dirty = true;
    }
}

impl KVCacheProvider for PagedKVCacheProvider {
    fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        let token_pos = self.total_tokens;

        // Single-write: GPU is primary → write to GPU only, skip CPU paged cache
        #[cfg(feature = "gpu")]
        if self.gpu_primary && !self.gpu_entries.is_empty() && layer_idx < self.gpu_entries.len() {
            let shape = vec![1, self.gpu_num_kv_heads, self.gpu_head_dim];
            if let (Ok(k_gpu), Ok(v_gpu)) = (
                nexora_deeplearning::autograd::gpu::GpuTensor::from_slice(shape.clone(), k),
                nexora_deeplearning::autograd::gpu::GpuTensor::from_slice(shape, v),
            ) {
                if let Ok(ref ctx) = nexora_deeplearning::autograd::gpu::GpuContext::global() {
                    let _ = self.gpu_entries[layer_idx].append(ctx, &k_gpu, &v_gpu);
                }
            }
            self.total_tokens += 1;
            self.dirty = true;
            return;
        }

        // No GPU entries or GPU inactive → write to paged cache blocks (CPU-only mode)
        let mut guard = match self.cache.write() {
            Ok(g) => g,
            Err(e) => {
                warn!("paged cache lock poisoned in append: {}", e);
                return;
            }
        };
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
        #[cfg(feature = "gpu")]
        if let Ok(ref ctx) = nexora_deeplearning::autograd::gpu::GpuContext::global() {
            for entry in &mut self.gpu_entries {
                let _ = entry.clear(ctx);
            }
        }
        #[cfg(feature = "gpu")]
        self.gpu_entries.clear();
        let mut guard = match self.cache.write() {
            Ok(g) => g,
            Err(e) => {
                warn!("paged cache lock poisoned in clear: {}", e);
                return;
            }
        };
        guard.remove_sequence(self.seq_id);
        guard.register_sequence(self.seq_id);
        drop(guard);
        self.flat.clear();
        self.flat_synced_tokens = 0;
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
    fn as_gpu_entries(&mut self) -> Option<&mut Vec<GpuKVCacheEntry>> {
        self.ensure_gpu_entries();
        if self.gpu_entries.is_empty() {
            None
        } else {
            Some(&mut self.gpu_entries)
        }
    }

    /// Downcast to PagedKVCacheProvider for sync_gpu_to_paged.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Drop for PagedKVCacheProvider {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.cache.write() {
            guard.remove_sequence(self.seq_id);
        }
    }
}
