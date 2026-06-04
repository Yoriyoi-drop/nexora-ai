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

use std::sync::{Arc, Mutex};

use nexora_transformer::{KVCacheEntry, KVCacheProvider};
use tracing::warn;

use crate::paged_cache::{PagedCacheConfig, PagedKVCache};

#[cfg(feature = "gpu")]
use nexora_transformer::GpuKVCacheEntry;

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
    /// GPU mirror: GpuKVCacheEntry per layer exposed via as_gpu_entries().
    /// When GPU is available, append() writes to both paged cache blocks AND
    /// these GPU entries. GPU forward methods operate on these entries directly.
    #[cfg(feature = "gpu")]
    gpu_entries: Vec<GpuKVCacheEntry>,
    #[cfg(feature = "gpu")]
    gpu_num_kv_heads: usize,
    #[cfg(feature = "gpu")]
    gpu_head_dim: usize,
    #[cfg(feature = "gpu")]
    gpu_max_seq_len: usize,
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
            #[cfg(feature = "gpu")]
            gpu_entries: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_num_kv_heads: config.num_kv_heads,
            #[cfg(feature = "gpu")]
            gpu_head_dim: config.head_dim,
            #[cfg(feature = "gpu")]
            gpu_max_seq_len: config.max_seq_len,
        }
    }

    pub fn new_shared(
        seq_id: u64,
        cache: Arc<Mutex<PagedKVCache>>,
        num_layers: usize,
    ) -> Self {
        // Read dimensions from the underlying paged cache config
        let (num_kv_heads, head_dim, max_seq_len) = {
            let guard = match cache.lock() {
                Ok(g) => g,
                Err(e) => {
                    warn!("paged cache lock poisoned in new_shared: {}", e);
                    e.into_inner()
                }
            };
            let cfg = guard.config();
            (cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len)
        };
        {
            let mut guard = match cache.lock() {
                Ok(g) => g,
                Err(e) => {
                    warn!("paged cache lock poisoned in new_shared (register): {}", e);
                    e.into_inner()
                }
            };
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
            #[cfg(feature = "gpu")]
            gpu_entries: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_num_kv_heads: num_kv_heads,
            #[cfg(feature = "gpu")]
            gpu_head_dim: head_dim,
            #[cfg(feature = "gpu")]
            gpu_max_seq_len: max_seq_len,
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
        let guard = match self.cache.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("paged cache lock poisoned in ensure_flat_synced: {}", e);
                return;
            }
        };
        if let Some(entries) = guard.to_flat_cache(self.seq_id) {
            self.flat = entries;
        }
        self.dirty = false;
    }

    /// Initialize GPU entries if they haven't been created yet.
    /// Creates one `GpuKVCacheEntry` per layer with pre-allocated GPU buffers
    /// matching the model dimensions.
    #[cfg(feature = "gpu")]
    fn ensure_gpu_entries(&mut self) {
        if !self.gpu_entries.is_empty() {
            return;
        }
        let ctx = match nexora_autograd::gpu::GpuContext::global() {
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
    }

    /// Sync all existing paged cache data INTO GPU entries.
    /// Called after GPU entries are first created to populate them with
    /// any K/V data that was already stored in the paged cache blocks.
    #[cfg(feature = "gpu")]
    fn sync_paged_to_gpu(&mut self) {
        if self.gpu_entries.is_empty() || self.total_tokens == 0 {
            return;
        }
        let ctx = match nexora_autograd::gpu::GpuContext::global() {
            Ok(c) => c,
            _ => return,
        };
        let _kv_elems = self.gpu_num_kv_heads * self.gpu_head_dim;

        // For each layer, read existing tokens from paged cache and upload to GPU
        let guard = match self.cache.lock() {
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
                        nexora_autograd::gpu::GpuTensor::from_cpu(&k_arr),
                        nexora_autograd::gpu::GpuTensor::from_cpu(&v_arr),
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
    /// positions >= `total_tokens`). Typically 1 new token per step, so the
    /// readback is small (~2 × num_layers × kv_heads × head_dim × 4 bytes).
    #[cfg(feature = "gpu")]
    pub fn sync_gpu_to_paged(&mut self) {
        if self.gpu_entries.is_empty() {
            return;
        }
        let gpu_seq_len = self.gpu_entries[0].seq_len;
        if gpu_seq_len <= self.total_tokens {
            return;
        }

        let _kv_elems = self.gpu_num_kv_heads * self.gpu_head_dim;
        for layer in 0..self.gpu_entries.len().min(self.num_layers) {
            for pos in self.total_tokens..gpu_seq_len {
                let (k_vec, v_vec) = match self.gpu_entries[layer].read_token_cpu(pos) {
                    Some(kv) => kv,
                    None => continue,
                };
                let mut guard = match self.cache.lock() {
                    Ok(g) => g,
                    Err(e) => {
                        warn!("paged cache lock poisoned in sync_gpu_to_paged: {}", e);
                        continue;
                    }
                };
                guard.append(self.seq_id, layer, pos, &k_vec, &v_vec);
                drop(guard);
            }
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
}

impl KVCacheProvider for PagedKVCacheProvider {
    fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        let token_pos = self.total_tokens;
        let mut guard = match self.cache.lock() {
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

        // Also append to GPU entries if they exist
        #[cfg(feature = "gpu")]
        if !self.gpu_entries.is_empty() && layer_idx < self.gpu_entries.len() {
            if let Ok(ref ctx) = nexora_autograd::gpu::GpuContext::global() {
                if let (Ok(k_arr), Ok(v_arr)) = (
                    ndarray::ArrayD::from_shape_vec(
                        vec![1, self.gpu_num_kv_heads, self.gpu_head_dim],
                        k.to_vec(),
                    ),
                    ndarray::ArrayD::from_shape_vec(
                        vec![1, self.gpu_num_kv_heads, self.gpu_head_dim],
                        v.to_vec(),
                    ),
                ) {
                    if let (Ok(k_gpu), Ok(v_gpu)) = (
                        nexora_autograd::gpu::GpuTensor::from_cpu(&k_arr),
                        nexora_autograd::gpu::GpuTensor::from_cpu(&v_arr),
                    ) {
                        let _ = self.gpu_entries[layer_idx].append(ctx, &k_gpu, &v_gpu);
                    }
                }
            }
        }
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
        if let Ok(ref ctx) = nexora_autograd::gpu::GpuContext::global() {
            for entry in &mut self.gpu_entries {
                let _ = entry.clear(ctx);
            }
        }
        #[cfg(feature = "gpu")]
        self.gpu_entries.clear();
        let mut guard = match self.cache.lock() {
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
        if let Ok(mut guard) = self.cache.lock() {
            guard.remove_sequence(self.seq_id);
        }
    }
}
