#[cfg(feature = "gpu")]
use super::gpu::GpuKVCacheEntry;

pub trait KVCacheProvider: Send + 'static {
    /// Append a new token's K/V for a given layer.
    /// `k` and `v` are flat slices of length num_kv_heads * head_dim.
    fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]);

    /// Read-only access to the cached K tensor for a layer.
    fn get_k(&self, layer_idx: usize) -> &[f32];

    /// Read-only access to the cached V tensor for a layer.
    fn get_v(&self, layer_idx: usize) -> &[f32];

    /// Number of tokens cached for a given layer.
    fn seq_len(&self, layer_idx: usize) -> usize;

    /// Clear all cached KV data.
    fn clear(&mut self);

    /// Number of layers in this cache.
    fn num_layers(&self) -> usize;

    /// Internal: expose inner `Vec<KVCacheEntry>` for the CPU forward path.
    /// Returns `None` for GPU-backed caches.
    #[doc(hidden)]
    fn as_cpu_entries(&mut self) -> Option<&mut Vec<KVCacheEntry>> {
        None
    }

    /// Internal: expose inner `Vec<GpuKVCacheEntry>` for the GPU forward path.
    /// Returns `None` for CPU-backed caches.
    #[cfg(feature = "gpu")]
    #[doc(hidden)]
    fn as_gpu_entries(&mut self) -> Option<&mut Vec<GpuKVCacheEntry>> {
        None
    }

    /// Downcast to `&mut dyn Any` for type-specific operations.
    /// Required for type-safe downcasting from `Box<dyn KVCacheProvider>`.
    #[doc(hidden)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// CPU implementation of [`KVCacheProvider`].

pub struct CpuKVCache {
    pub entries: Vec<KVCacheEntry>,
}

impl CpuKVCache {
    pub fn new(_num_layers: usize) -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn num_layers(&self) -> usize {
        self.entries.len()
    }
}

impl KVCacheProvider for CpuKVCache {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        let kv_dim = k.len();
        if layer_idx < self.entries.len() {
            let entry = &mut self.entries[layer_idx];
            entry.k.extend_from_slice(k);
            entry.v.extend_from_slice(v);
        } else {
            self.entries.push(KVCacheEntry {
                k: k.to_vec(),
                v: v.to_vec(),
                kv_dim,
                num_kv_heads: 0,
                head_dim: 0,
                k_compressed: Vec::new(),
                v_compressed: Vec::new(),
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                compressed_seq_len: 0,
            });
        }
    }

    fn get_k(&self, layer_idx: usize) -> &[f32] {
        if layer_idx < self.entries.len() {
            &self.entries[layer_idx].k
        } else {
            &[]
        }
    }

    fn get_v(&self, layer_idx: usize) -> &[f32] {
        if layer_idx < self.entries.len() {
            &self.entries[layer_idx].v
        } else {
            &[]
        }
    }

    fn seq_len(&self, layer_idx: usize) -> usize {
        if layer_idx < self.entries.len() {
            self.entries[layer_idx].seq_len()
        } else {
            0
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn num_layers(&self) -> usize {
        self.entries.len()
    }

    fn as_cpu_entries(&mut self) -> Option<&mut Vec<KVCacheEntry>> {
        Some(&mut self.entries)
    }
}

/// GPU implementation of [`KVCacheProvider`].
/// Stores K/V directly on GPU without CPU round-trip.

pub trait PagedCacheReader {
    /// Read KV data for a token position. Returns (k_row, v_row).
    fn read(&self, seq_id: u64, layer: usize, token_pos: usize) -> Option<(Vec<f32>, Vec<f32>)>;
    /// Number of tokens stored for a sequence.
    fn num_tokens(&self, seq_id: u64) -> Option<usize>;
    /// Append a token's KV data (single position).
    fn append(&mut self, seq_id: u64, layer: usize, token_pos: usize, k_row: &[f32], v_row: &[f32]);

    /// Read ALL K/V data for a layer into flat buffers.
    /// Returns `(k_flat, v_flat)` where each is `[num_tokens, kv_dim]` in row-major order.
    /// Default implementation calls `read()` per token — override for efficient batch read.
    fn read_layer_kv(&self, seq_id: u64, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let num_tokens = self.num_tokens(seq_id)?;
        if num_tokens == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        if let Some((first_k, first_v)) = self.read(seq_id, layer, 0) {
            let kv_dim = first_k.len();
            let total = num_tokens * kv_dim;
            let mut k_flat = Vec::with_capacity(total);
            let mut v_flat = Vec::with_capacity(total);
            k_flat.extend_from_slice(&first_k);
            v_flat.extend_from_slice(&first_v);
            for t in 1..num_tokens {
                if let Some((k_row, v_row)) = self.read(seq_id, layer, t) {
                    k_flat.extend_from_slice(&k_row);
                    v_flat.extend_from_slice(&v_row);
                }
            }
            Some((k_flat, v_flat))
        } else {
            None
        }
    }
}


#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub k: Vec<f32>, // flat [seq_len, kv_dim]
    pub v: Vec<f32>, // flat [seq_len, kv_dim]
    pub kv_dim: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,

    // Compressed 4-bit backing (populated by compress())
    pub k_compressed: Vec<u8>,
    pub v_compressed: Vec<u8>,
    pub k_scales: Vec<f32>,
    pub v_scales: Vec<f32>,
    pub compressed_seq_len: usize,
}

impl Default for KVCacheEntry {
    fn default() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            kv_dim: 0,
            num_kv_heads: 0,
            head_dim: 0,
            k_compressed: Vec::new(),
            v_compressed: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            compressed_seq_len: 0,
        }
    }
}

impl KVCacheEntry {
    pub fn seq_len(&self) -> usize {
        if self.kv_dim == 0 {
            0
        } else {
            self.k.len() / self.kv_dim
        }
    }

    /// Compress the current K/V data to 4-bit packed format.
    /// Does NOT clear the f32 data — call `free_f32_memory()` for that.
    pub fn compress(&mut self) {
        if self.k.is_empty() || self.num_kv_heads == 0 || self.head_dim == 0 {
            return;
        }
        let seq_len = self.seq_len();
        let (kp, ks) = crate::kv_cache_compression::quantize_4bit(
            &self.k, self.num_kv_heads, self.head_dim, seq_len,
        );
        let (vp, vs) = crate::kv_cache_compression::quantize_4bit(
            &self.v, self.num_kv_heads, self.head_dim, seq_len,
        );
        self.k_compressed = kp;
        self.v_compressed = vp;
        self.k_scales = ks;
        self.v_scales = vs;
        self.compressed_seq_len = seq_len;
    }

    /// Free f32 data after compression to reclaim memory.
    /// Call `ensure_dequantized()` before reading K/V.
    pub fn free_f32_memory(&mut self) {
        if self.compressed_seq_len > 0 {
            self.k.clear();
            self.v.clear();
            self.k.shrink_to_fit();
            self.v.shrink_to_fit();
        }
    }

    /// Decompress if compressed data exists and f32 data is empty.
    pub fn ensure_dequantized(&mut self) {
        if !self.k.is_empty() {
            return;
        }
        if self.compressed_seq_len == 0 || self.k_scales.is_empty() {
            return;
        }
        let kv_dim = self.kv_dim;
        let seq_len = self.compressed_seq_len;
        self.k = crate::kv_cache_compression::dequantize_4bit(
            &self.k_compressed, &self.k_scales,
            self.num_kv_heads, self.head_dim, seq_len,
        );
        self.v = crate::kv_cache_compression::dequantize_4bit(
            &self.v_compressed, &self.v_scales,
            self.num_kv_heads, self.head_dim, seq_len,
        );
        debug_assert_eq!(self.k.len(), seq_len * kv_dim);
        debug_assert_eq!(self.v.len(), seq_len * kv_dim);
    }
}

