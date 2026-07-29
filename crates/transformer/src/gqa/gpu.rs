use ndarray::{Array1, Array2};
use nexora_deeplearning::autograd::gpu::{GpuContext, GpuDtype, GpuTensor};
use super::kv_cache::{KVCacheProvider, KVCacheEntry};
use super::gqa_cpu::GQA;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
/// Temporary f32 weights upconverted from f16 storage at forward start.
/// Dropped at end of each forward pass — GPU memory reclaimed.
pub(crate) struct GqaGpuTemps {
    pub wq_t: nexora_deeplearning::autograd::gpu::GpuTensor,
    pub wk_t: nexora_deeplearning::autograd::gpu::GpuTensor,
    pub wv_t: nexora_deeplearning::autograd::gpu::GpuTensor,
    pub wo_t: nexora_deeplearning::autograd::gpu::GpuTensor,
}

#[derive(Debug)]
pub(crate) struct GqaGpuWeights {
    /// F32 transposed weights — always present (needed for f32 matmul and readback).
    /// When `use_half_precision` is active, these are None and only f16 packed weights
    /// are stored on GPU (2× less VRAM). The f16→f32 upconversion happens ephemerally
    /// per forward pass via `GqaGpuTemps` (dropped after each forward).
    pub wq_t: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    pub wk_t: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    pub wv_t: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    pub wo_t: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    /// F16 packed weights — present only when `use_half_precision` is active.
    /// Each stores 2× less VRAM than f32.
    pub wq_f16: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    pub wk_f16: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    pub wv_f16: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    pub wo_f16: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    pub wq_shape: Vec<usize>,
    pub wk_shape: Vec<usize>,
    pub wv_shape: Vec<usize>,
    pub wo_shape: Vec<usize>,
    /// Whether this entry stores f16 packed weights (instead of f32).
    pub is_f16: bool,
}

/// GPU-resident KV cache entry.
/// Stores K/V directly on GPU without CPU round-trip.
/// Pre-allocated to max_seq_len with append via GPU buffer copy.
/// Supports F16 packed storage (2 f16 per u32) when f16_storage=true.

#[derive(Debug)]
pub struct GpuKVCacheEntry {
    pub(crate) k: GpuTensor,
    pub(crate) v: GpuTensor,
    pub seq_len: usize,
    pub(crate) capacity: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) f16_storage: bool,
    /// Copy-on-Write reference counter.
    /// When `Some(count)` and count > 1, the GPU buffers are shared.
    /// First mutation (append/clear) triggers real buffer copy.
    pub(crate) cow_count: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
}

#[cfg(feature = "gpu")]
impl GpuKVCacheEntry {
    /// Allocate a new GPU KV cache entry with pre-allocated zero-filled buffers.
    pub fn new(
        ctx: &GpuContext,
        kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        use_f16: bool,
    ) -> Result<Self, nexora_deeplearning::autograd::gpu::GpuError> {
        if use_f16 {
            let num_f16 = max_seq * kv_heads * head_dim;
            let buf_size = ((num_f16 + 1) / 2 * 4) as u64;
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            let k_buf = ctx.alloc_or_create_buffer(buf_size, usage);
            let v_buf = ctx.alloc_or_create_buffer(buf_size, usage);
            let k = GpuTensor::from_raw(vec![max_seq, kv_heads, head_dim], k_buf, GpuDtype::F16);
            let v = GpuTensor::from_raw(vec![max_seq, kv_heads, head_dim], v_buf, GpuDtype::F16);
            ctx.fill_zero_u32(&k)?;
            ctx.fill_zero_u32(&v)?;
            Ok(Self {
                k,
                v,
                seq_len: 0,
                capacity: max_seq,
                kv_heads,
                head_dim,
                f16_storage: true,
                cow_count: None,
            })
        } else {
            Ok(Self {
                k: GpuTensor::zeros_dtype(&[max_seq, kv_heads, head_dim], GpuDtype::F32)?,
                v: GpuTensor::zeros_dtype(&[max_seq, kv_heads, head_dim], GpuDtype::F32)?,
                seq_len: 0,
                capacity: max_seq,
                kv_heads,
                head_dim,
                f16_storage: false,
                cow_count: None,
            })
        }
    }

    /// Append new K and V (each shape [1, kv_heads, head_dim]) to the GPU cache.
    /// Uses batch_dispatch for GPU-side buffer copy — no CPU round-trip.
    /// When f16_storage=true, converts F32→packed F16 before storing.
    pub fn append(
        &mut self,
        ctx: &GpuContext,
        new_k: &GpuTensor,
        new_v: &GpuTensor,
    ) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        if self.seq_len >= self.capacity {
            return Err(nexora_deeplearning::autograd::gpu::GpuError::Unsupported(format!(
                "GpuKVCacheEntry capacity exceeded: {} >= {}",
                self.seq_len, self.capacity
            )));
        }
        // COW: jika buffer shared, copy dulu sebelum write
        self.cow_ensure_unique(ctx)?;

        let kv_elems = self.kv_heads * self.head_dim;

        if self.f16_storage {
            let pk = ctx.f32_to_f16_packed(new_k)?;
            let pv = ctx.f32_to_f16_packed(new_v)?;
            let byte_size = (kv_elems * 2) as u64;
            let offset = (self.seq_len * kv_elems * 2) as u64;
            ctx.batch_dispatch(|enc| {
                enc.copy_buffer_to_buffer(pk.buffer(), 0, self.k.buffer(), offset, byte_size);
                enc.copy_buffer_to_buffer(pv.buffer(), 0, self.v.buffer(), offset, byte_size);
                Ok(())
            })?;
        } else {
            let byte_size = (kv_elems * 4) as u64;
            let offset = (self.seq_len * kv_elems * 4) as u64;
            ctx.batch_dispatch(|enc| {
                enc.copy_buffer_to_buffer(new_k.buffer(), 0, self.k.buffer(), offset, byte_size);
                enc.copy_buffer_to_buffer(new_v.buffer(), 0, self.v.buffer(), offset, byte_size);
                Ok(())
            })?;
        }

        self.seq_len += 1;
        Ok(())
    }

    /// Returns a view of K up to seq_len as [seq_len, kv_heads, head_dim].
    /// Zero-copy: shares the same wgpu::Buffer handle via reshape.
    pub fn k_view(&self) -> GpuTensor {
        self.k
            .view_as(vec![self.seq_len, self.kv_heads, self.head_dim])
    }

    /// Returns a view of V up to seq_len as [seq_len, kv_heads, head_dim].
    /// Zero-copy: shares the same wgpu::Buffer handle via reshape.
    pub fn v_view(&self) -> GpuTensor {
        self.v
            .view_as(vec![self.seq_len, self.kv_heads, self.head_dim])
    }

    /// Get K/V repeated to q_heads for fused_attention.
    /// Returns (k_repeated, v_repeated) as [q_heads, seq_len, head_dim] (head-major).
    /// When f16_storage=true, upconverts packed F16→F32 before repeat.
    pub fn get_repeated_kv(
        &self,
        ctx: &GpuContext,
        q_heads: u32,
    ) -> Result<(GpuTensor, GpuTensor), nexora_deeplearning::autograd::gpu::GpuError> {
        let dim = self.head_dim as u32;
        let kv_heads = self.kv_heads as u32;

        if self.f16_storage {
            let k_f32 = ctx.f16_packed_to_f32(&self.k_view())?;
            let v_f32 = ctx.f16_packed_to_f32(&self.v_view())?;
            let k_repeated = ctx.repeat_heads(&k_f32, kv_heads, q_heads, dim)?;
            let v_repeated = ctx.repeat_heads(&v_f32, kv_heads, q_heads, dim)?;
            Ok((k_repeated, v_repeated))
        } else {
            let k_repeated = ctx.repeat_heads(&self.k_view(), kv_heads, q_heads, dim)?;
            let v_repeated = ctx.repeat_heads(&self.v_view(), kv_heads, q_heads, dim)?;
            Ok((k_repeated, v_repeated))
        }
    }

    /// Bulk-append N tokens of K and V (each shape [N, kv_heads, head_dim]) to the GPU cache.
    /// Uses GPU-side buffer copy — no CPU round-trip.
    /// Supports F16 storage conversion.
    pub fn bulk_append(
        &mut self,
        ctx: &GpuContext,
        new_k: &GpuTensor,
        new_v: &GpuTensor,
        num_tokens: usize,
    ) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        if self.seq_len + num_tokens > self.capacity {
            return Err(nexora_deeplearning::autograd::gpu::GpuError::Unsupported(format!(
                "GpuKVCacheEntry capacity exceeded: {} + {} > {}",
                self.seq_len, num_tokens, self.capacity
            )));
        }
        let kv_elems = self.kv_heads * self.head_dim;

        if self.f16_storage {
            let pk = ctx.f32_to_f16_packed(new_k)?;
            let pv = ctx.f32_to_f16_packed(new_v)?;
            let byte_size = (num_tokens * kv_elems * 2) as u64;
            let offset = (self.seq_len * kv_elems * 2) as u64;
            ctx.batch_dispatch(|enc| {
                enc.copy_buffer_to_buffer(pk.buffer(), 0, self.k.buffer(), offset, byte_size);
                enc.copy_buffer_to_buffer(pv.buffer(), 0, self.v.buffer(), offset, byte_size);
                Ok(())
            })?;
        } else {
            let byte_size = (num_tokens * kv_elems * 4) as u64;
            let offset = (self.seq_len * kv_elems * 4) as u64;
            ctx.batch_dispatch(|enc| {
                enc.copy_buffer_to_buffer(new_k.buffer(), 0, self.k.buffer(), offset, byte_size);
                enc.copy_buffer_to_buffer(new_v.buffer(), 0, self.v.buffer(), offset, byte_size);
                Ok(())
            })?;
        }

        self.seq_len += num_tokens;
        Ok(())
    }

    /// Copy the first `prefix_len` K/V positions from `source` into this entry.
    /// Both entries must have matching dimensions and storage type.
    /// Source must have `seq_len >= prefix_len`.
    /// GPU-side buffer-to-buffer copy — no CPU round-trip.
    /// Sets `self.seq_len = prefix_len`.
    pub fn copy_prefix_from(
        &mut self,
        ctx: &GpuContext,
        source: &GpuKVCacheEntry,
        prefix_len: usize,
    ) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        assert_eq!(
            self.kv_heads, source.kv_heads,
            "copy_prefix_from: kv_heads mismatch {} vs {}",
            self.kv_heads, source.kv_heads
        );
        assert_eq!(
            self.head_dim, source.head_dim,
            "copy_prefix_from: head_dim mismatch {} vs {}",
            self.head_dim, source.head_dim
        );
        assert_eq!(
            self.f16_storage, source.f16_storage,
            "copy_prefix_from: f16_storage mismatch"
        );
        assert!(
            prefix_len <= source.seq_len,
            "copy_prefix_from: source seq_len ({}) < prefix_len ({})",
            source.seq_len,
            prefix_len
        );
        assert!(
            prefix_len <= self.capacity,
            "copy_prefix_from: dest capacity ({}) < prefix_len ({})",
            self.capacity,
            prefix_len
        );

        let kv_elems = self.kv_heads * self.head_dim;
        let elem_size: u64 = if self.f16_storage { 2 } else { 4 };
        let byte_size = (prefix_len as u64) * (kv_elems as u64) * elem_size;

        ctx.batch_dispatch(|enc| {
            enc.copy_buffer_to_buffer(source.k.buffer(), 0, self.k.buffer(), 0, byte_size);
            enc.copy_buffer_to_buffer(source.v.buffer(), 0, self.v.buffer(), 0, byte_size);
            Ok(())
        })?;

        self.seq_len = prefix_len;
        Ok(())
    }

    /// Read back a single token's K/V from this GPU cache entry into CPU f32 slices.
    /// Returns (k_vec, v_vec) each of length `kv_heads * head_dim`.
    /// This is a small readback (typically ~4KB for 8×128) — acceptable for
    /// sync-back after GPU forward to keep the paged cache blocks consistent.
    pub fn read_token_cpu(&self, pos: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if pos >= self.seq_len {
            return None;
        }
        let kv_elems = self.kv_heads * self.head_dim;
        let elem_size: usize = if self.f16_storage { 2 } else { 4 };
        let token_bytes = kv_elems * elem_size;
        let offset = (pos * token_bytes) as u64;

        let k_raw = self.k.to_cpu_raw_bytes_slice(offset, token_bytes as u64).ok()?;
        let v_raw = self.v.to_cpu_raw_bytes_slice(offset, token_bytes as u64).ok()?;

        Some((Self::raw_to_f32(k_raw, self.f16_storage), Self::raw_to_f32(v_raw, self.f16_storage)))
    }

    /// Bulk read tokens [start, end) in a single GPU→CPU transfer.
    /// Returns Vec<(k_vec, v_vec)> — one entry per position.
    /// Eliminates N individual readback calls (each a GPU sync point + allocation).
    pub fn read_tokens_bulk(&self, start: usize, end: usize) -> Option<Vec<(Vec<f32>, Vec<f32>)>> {
        if start >= end || end > self.seq_len {
            return None;
        }
        let kv_elems = self.kv_heads * self.head_dim;
        let elem_size: usize = if self.f16_storage { 2 } else { 4 };
        let token_bytes = kv_elems * elem_size;
        let count = end - start;
        let byte_len = (count * token_bytes) as u64;
        let offset = (start * token_bytes) as u64;

        let k_all = self.k.to_cpu_raw_bytes_slice(offset, byte_len).ok()?;
        let v_all = self.v.to_cpu_raw_bytes_slice(offset, byte_len).ok()?;

        let k_tokens = if self.f16_storage {
            k_all.chunks_exact(token_bytes).map(|chunk| {
                chunk.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect::<Vec<_>>()
            }).collect::<Vec<_>>()
        } else {
            k_all.chunks_exact(token_bytes).map(|chunk| {
                chunk.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()
            }).collect::<Vec<_>>()
        };

        let v_tokens = if self.f16_storage {
            v_all.chunks_exact(token_bytes).map(|chunk| {
                chunk.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect::<Vec<_>>()
            }).collect::<Vec<_>>()
        } else {
            v_all.chunks_exact(token_bytes).map(|chunk| {
                chunk.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()
            }).collect::<Vec<_>>()
        };

        Some(k_tokens.into_iter().zip(v_tokens).collect())
    }

    fn raw_to_f32(raw: Vec<u8>, f16_storage: bool) -> Vec<f32> {
        if f16_storage {
            raw.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        } else {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    }

    /// Clear the cache (reset sequence length).
    /// Does NOT deallocate buffers — memset to zero.
    /// Uses u32 zero fill for F16 storage, f32 zero fill for F32 storage.
    pub fn clear(&mut self, ctx: &GpuContext) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        // COW: jika buffer shared, copy dulu sebelum write
        self.cow_ensure_unique(ctx)?;
        if self.f16_storage {
            ctx.fill_zero_u32(&self.k)?;
            ctx.fill_zero_u32(&self.v)?;
        } else {
            ctx.fill_zero(&self.k)?;
            ctx.fill_zero(&self.v)?;
        }
        self.seq_len = 0;
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl GpuKVCacheEntry {
    /// COW (Copy-on-Write) deep clone.
    /// Instead of immediately copying GPU buffers (expensive), this increments
    /// a shared reference counter. The first `append()` or `clear()` on any
    /// cloned entry triggers the real buffer copy.
    pub fn deep_clone(&self, _ctx: &GpuContext) -> Result<Self, nexora_deeplearning::autograd::gpu::GpuError> {
        let cow_count = self.cow_count.clone().unwrap_or_else(|| {
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1))
        });
        // Increment shared count — this clone shares the buffers
        cow_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let k_buf = unsafe { self.k.buffer().clone() };
        let v_buf = unsafe { self.v.buffer().clone() };
        Ok(Self {
            k: GpuTensor::from_raw(self.k.shape(), k_buf, self.k.dtype()),
            v: GpuTensor::from_raw(self.v.shape(), v_buf, self.v.dtype()),
            seq_len: self.seq_len,
            capacity: self.capacity,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            f16_storage: self.f16_storage,
            cow_count: Some(cow_count),
        })
    }

    /// Ensure unique ownership of GPU buffers.
    /// If COW count > 1, performs actual buffer copy before mutation.
    fn cow_ensure_unique(
        &mut self,
        ctx: &GpuContext,
    ) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        let cow_count = match &self.cow_count {
            Some(c) => c,
            None => return Ok(()),
        };
        let count = cow_count.load(std::sync::atomic::Ordering::Relaxed);

        if count <= 1 {
            return Ok(()); // Already unique owner
        }

        // Decrement shared count
        cow_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

        // Copy GPU buffers
        let k_size = self.k.buffer().size();
        let v_size = self.v.buffer().size();
        let k_new = ctx.alloc_or_create_buffer(k_size, self.k.buffer().usage());
        let v_new = ctx.alloc_or_create_buffer(v_size, self.v.buffer().usage());
        ctx.batch_dispatch(|enc| {
            enc.copy_buffer_to_buffer(self.k.buffer(), 0, &k_new, 0, k_size);
            enc.copy_buffer_to_buffer(self.v.buffer(), 0, &v_new, 0, v_size);
            Ok(())
        })?;

        self.k = GpuTensor::from_raw(self.k.shape(), k_new, self.k.dtype());
        self.v = GpuTensor::from_raw(self.v.shape(), v_new, self.v.dtype());
        self.cow_count = None; // Unique owner now

        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl Clone for GpuKVCacheEntry {
    /// Shallow clone — shares the same wgpu::Buffer handles.
    /// ⚠️ This is intentionally NOT a deep copy. Use `deep_clone()` for that.
    /// The shallow clone is only safe when the original entry is immediately
    /// discarded (e.g., when building temporary mirrors for paged cache sync).
    fn clone(&self) -> Self {
        Self {
            k: self.k.clone(),
            v: self.v.clone(),
            seq_len: self.seq_len,
            capacity: self.capacity,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            f16_storage: self.f16_storage,
            cow_count: self.cow_count.clone(),
        }
    }
}

/// Unified trait for KV cache — supports both CPU (Vec<KVCacheEntry>) and GPU (GpuKVCacheEntry).
/// Enables the inference engine to transparently switch between CPU and GPU KV cache
/// without changing the model forward path.

pub struct GpuKVCache {
    pub entries: Vec<GpuKVCacheEntry>,
}

#[cfg(feature = "gpu")]
impl GpuKVCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let entries = if let Ok(ctx) = nexora_deeplearning::autograd::gpu::GpuContext::global() {
            (0..num_layers)
                .filter_map(|_| {
                    GpuKVCacheEntry::new(&ctx, num_kv_heads, head_dim, max_seq_len, true).ok()
                })
                .collect()
        } else {
            Vec::new()
        };
        Self { entries }
    }

    /// Copy the first `prefix_len` K/V positions from `source` into this cache.
    /// Both caches must have the same number of layers.
    /// Delegates to [`GpuKVCacheEntry::copy_prefix_from`] per layer.
    pub fn copy_prefix_from(
        &mut self,
        source: &GpuKVCache,
        prefix_len: usize,
    ) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        assert_eq!(
            self.entries.len(),
            source.entries.len(),
            "copy_prefix_from: layer count mismatch {} vs {}",
            self.entries.len(),
            source.entries.len()
        );
        let ctx = nexora_deeplearning::autograd::gpu::GpuContext::global().map_err(|_| {
            nexora_deeplearning::autograd::gpu::GpuError::NotInitialized
        })?;
        for (dst_entry, src_entry) in self.entries.iter_mut().zip(source.entries.iter()) {
            dst_entry.copy_prefix_from(&ctx, src_entry, prefix_len)?;
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl KVCacheProvider for GpuKVCache {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        if let Ok(ref ctx) = nexora_deeplearning::autograd::gpu::GpuContext::global() {
            if layer_idx < self.entries.len() {
                use ndarray::ArrayD;
                use nexora_deeplearning::autograd::gpu::GpuTensor;

                let kv_heads = self.entries[layer_idx].kv_heads;
                let head_dim = self.entries[layer_idx].head_dim;
                if let (Ok(k_arr), Ok(v_arr)) = (
                    ArrayD::from_shape_vec(vec![1, kv_heads, head_dim], k.to_vec()),
                    ArrayD::from_shape_vec(vec![1, kv_heads, head_dim], v.to_vec()),
                ) {
                    if let (Ok(k_gpu), Ok(v_gpu)) =
                        (GpuTensor::from_cpu(&k_arr), GpuTensor::from_cpu(&v_arr))
                    {
                        let _ = self.entries[layer_idx].append(ctx, &k_gpu, &v_gpu);
                    }
                }
            }
        }
    }

    fn get_k(&self, _layer_idx: usize) -> &[f32] {
        &[]
    } // GPU-resident, no CPU readback
    fn get_v(&self, _layer_idx: usize) -> &[f32] {
        &[]
    }

    fn seq_len(&self, layer_idx: usize) -> usize {
        if layer_idx < self.entries.len() {
            self.entries[layer_idx].seq_len
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

    fn as_gpu_entries(&mut self) -> Option<&mut Vec<GpuKVCacheEntry>> {
        Some(&mut self.entries)
    }
}

/// Minimal paged cache interface — avoids hard dependency on the inference crate.

impl GQA {
    fn ensure_weights_gpu(&self) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        use nexora_deeplearning::autograd::gpu::{GpuContext, GpuError, GpuTensor};
        if self.gpu_weights.get().is_some() {
            return Ok(());
        }
        let ctx = GpuContext::global()?;
        let mk = |arr: &Array2<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.shape()[0], arr.shape()[1]];
            let data = arr
                .as_slice()
                .ok_or_else(|| GpuError::Unsupported("non-contiguous weight".into()))?
                .to_vec();
            Ok(GpuTensor::from_cpu(
                &ndarray::ArrayD::from_shape_vec(shape, data)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?,
            )?)
        };
        let wq = mk(self.wq.as_ref().ok_or_else(|| {
            GpuError::Unsupported("GQA wq not available for GPU upload".into())
        })?)?;
        let wk = mk(self.wk.as_ref().ok_or_else(|| {
            GpuError::Unsupported("GQA wk not available".into())
        })?)?;
        let wv = mk(self.wv.as_ref().ok_or_else(|| {
            GpuError::Unsupported("GQA wv not available".into())
        })?)?;
        let wo = mk(self.wo.as_ref().ok_or_else(|| {
            GpuError::Unsupported("GQA wo not available".into())
        })?)?;
        let use_f16 = self.use_half_precision;
        let (wq_f16, wk_f16, wv_f16, wo_f16) = if use_f16 {
            let wq_t = ctx.transpose(&wq)?;
            let wk_t = ctx.transpose(&wk)?;
            let wv_t = ctx.transpose(&wv)?;
            let wo_t = ctx.transpose(&wo)?;
            (
                Some(ctx.f32_to_f16_packed(&wq_t)?),
                Some(ctx.f32_to_f16_packed(&wk_t)?),
                Some(ctx.f32_to_f16_packed(&wv_t)?),
                Some(ctx.f32_to_f16_packed(&wo_t)?),
            )
        } else {
            (None, None, None, None)
        };
        // F16 mode: skip f32 upload entirely — f16 packed weights stored only (2× less VRAM).
        // The f16→f32 upconversion happens ephemerally per forward pass (dropped after forward).
        let (wq_t, wk_t, wv_t, wo_t) = if !use_f16 {
            (
                Some(ctx.transpose(&wq)?),
                Some(ctx.transpose(&wk)?),
                Some(ctx.transpose(&wv)?),
                Some(ctx.transpose(&wo)?),
            )
        } else {
            (None, None, None, None)
        };
        let wq_shape = wq.shape().to_vec();
        let wk_shape = wk.shape().to_vec();
        let wv_shape = wv.shape().to_vec();
        let wo_shape = wo.shape().to_vec();
        let _ = self.gpu_weights.set(GqaGpuWeights {
            wq_t,
            wk_t,
            wv_t,
            wo_t,
            wq_f16,
            wk_f16,
            wv_f16,
            wo_f16,
            wq_shape,
            wk_shape,
            wv_shape,
            wo_shape,
            is_f16: use_f16,
        });
        Ok(())
    }

    fn gqa_upconvert_f16(
        cached: &GqaGpuWeights,
        ctx: &nexora_deeplearning::autograd::gpu::GpuContext,
    ) -> Result<GqaGpuTemps, nexora_deeplearning::autograd::gpu::GpuError> {
        use nexora_deeplearning::autograd::gpu::GpuError;
        Ok(GqaGpuTemps {
            wq_t: ctx.f16_packed_to_f32(cached.wq_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("GQA f16 missing wq".into())
            })?)?,
            wk_t: ctx.f16_packed_to_f32(cached.wk_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("GQA f16 missing wk".into())
            })?)?,
            wv_t: ctx.f16_packed_to_f32(cached.wv_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("GQA f16 missing wv".into())
            })?)?,
            wo_t: ctx.f16_packed_to_f32(cached.wo_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("GQA f16 missing wo".into())
            })?)?,
        })
    }

    /// GPU forward with pre-uploaded cos/sin GPU tensors.
    /// cos_gpu / sin_gpu must be shape [1, half] — uploaded once per step, not per-layer.
    /// Uses GPU-resident KV cache — no O(n²) CPU round-trip.
    pub fn forward_gpu_with_rope_gpu(
        &self,
        x_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
        sin_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
    ) -> Result<nexora_deeplearning::autograd::gpu::GpuTensor, nexora_deeplearning::autograd::gpu::GpuError> {
        use nexora_deeplearning::autograd::gpu::{GpuContext, GpuError, GpuTensor};

        let ctx = GpuContext::global()?;
        let shape_v = x_gpu.shape();
        let batch_size = shape_v.first().copied().unwrap_or(1);

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).or_else(|| cached.wq_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wq_t not available on GPU".into())
        })?;
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).or_else(|| cached.wk_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wk_t not available on GPU".into())
        })?;
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).or_else(|| cached.wv_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wv_t not available on GPU".into())
        })?;
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).or_else(|| cached.wo_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wo_t not available on GPU".into())
        })?;

        // 2. QKV projection on GPU
        let q_proj = ctx.matmul(x_gpu, wq_t)?;
        let k_proj = ctx.matmul(x_gpu, wk_t)?;
        let v_proj = ctx.matmul(x_gpu, wv_t)?;

        // 3. Apply RoPE on GPU using PRE-UPLOADED cos/sin
        let q_rotated = ctx.rotary_embedding(&q_proj, cos_gpu, sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_proj, cos_gpu, sin_gpu, self.head_dim as u32)?;

        // 4. Reshape Q to [1, num_heads, 1, head_dim]
        let q_4d = q_rotated.reshape(vec![batch_size, self.num_heads, 1, self.head_dim])?;

        // 5. GPU-side KV cache (no CPU round-trip)
        let (k_repeated, v_repeated) = {
            let mut scratch_guard = self.gpu_scratch.write();
            let cpu_seq_len = if layer_idx < cache.len() {
                cache[layer_idx].seq_len()
            } else {
                0
            };

            // Take existing scratch or create new one (avoids borrow conflicts)
            let mut gpu_entry = match scratch_guard.take() {
                Some(entry) => entry,
                None => GpuKVCacheEntry::new(
                    &ctx,
                    self.num_kv_heads,
                    self.head_dim,
                    4096.max(cpu_seq_len + 4096),
                    false,
                )?,
            };

            // Re-sync from CPU if cache was reset or different generation
            if gpu_entry.seq_len != cpu_seq_len {
                if cpu_seq_len == 0 {
                    gpu_entry.seq_len = 0;
                    ctx.fill_zero(&gpu_entry.k)?;
                    ctx.fill_zero(&gpu_entry.v)?;
                } else {
                    gpu_entry = GpuKVCacheEntry::new(
                        &ctx,
                        self.num_kv_heads,
                        self.head_dim,
                        cpu_seq_len + 4096,
                        false,
                    )?;
                    let ce = &cache[layer_idx];
                    let seq = ce.seq_len();
                    let k_flat: Vec<f32> = ce.k.clone();
                    let v_flat: Vec<f32> = ce.v.clone();
                    let k_cpu = ndarray::ArrayD::from_shape_vec(
                        vec![seq, self.num_kv_heads, self.head_dim],
                        k_flat,
                    )
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?;
                    let v_cpu = ndarray::ArrayD::from_shape_vec(
                        vec![seq, self.num_kv_heads, self.head_dim],
                        v_flat,
                    )
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?;
                    let k_host = GpuTensor::from_cpu(&k_cpu)?;
                    let v_host = GpuTensor::from_cpu(&v_cpu)?;
                    let bytes = (seq * self.num_kv_heads * self.head_dim * 4) as u64;
                    ctx.batch_dispatch(|enc| {
                        enc.copy_buffer_to_buffer(
                            k_host.buffer(),
                            0,
                            gpu_entry.k.buffer(),
                            0,
                            bytes,
                        );
                        enc.copy_buffer_to_buffer(
                            v_host.buffer(),
                            0,
                            gpu_entry.v.buffer(),
                            0,
                            bytes,
                        );
                        Ok(())
                    })?;
                    gpu_entry.seq_len = seq;
                }
            }

            // Append new K/V to GPU cache (GPU-only copy_buffer_to_buffer)
            let k_3d = k_rotated.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
            let v_3d = v_proj.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
            gpu_entry.append(&ctx, &k_3d, &v_3d)?;

            // Get repeated K/V from GPU cache
            let (k_rep, v_rep) = gpu_entry.get_repeated_kv(&ctx, self.num_heads as u32)?;
            let total_seq = gpu_entry.seq_len;

            let k_4d = k_rep.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;
            let v_4d = v_rep.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;

            // Sync back new token to CPU cache (O(1) per step)
            self.sync_back_cpu_cache(&ctx, &gpu_entry, cache, layer_idx)?;

            // Store back for next call
            *scratch_guard = Some(gpu_entry);
            (k_4d, v_4d)
        };

        // 7. Fused attention on GPU
        let attn_output_4d =
            ctx.fused_attention(&q_4d, &k_repeated, &v_repeated, self.head_dim_rs, false)?;

        // 8. Reshape attention output
        let attn_flat = attn_output_4d.reshape(vec![batch_size, self.num_heads * self.head_dim])?;

        // 9. Output projection on GPU
        ctx.matmul(&attn_flat, wo_t)
    }

    /// GPU forward with GPU-resident KV cache AND pre-uploaded cos/sin GPU tensors.
    /// cos_gpu / sin_gpu must be shape [1, half] — uploaded once per step, not per-layer.
    /// No CPU round-trip for K/V or RoPE.
    pub fn forward_gpu_with_cache_precomputed_rope(
        &self,
        x_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
        cache: &mut GpuKVCacheEntry,
        cos_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
        sin_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
    ) -> Result<nexora_deeplearning::autograd::gpu::GpuTensor, nexora_deeplearning::autograd::gpu::GpuError> {
        use nexora_deeplearning::autograd::gpu::GpuContext;

        let ctx = GpuContext::global()?;
        let shape_v = x_gpu.shape();
        let batch_size = shape_v.first().copied().unwrap_or(1);

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).or_else(|| cached.wq_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wq_t not available on GPU".into())
        })?;
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).or_else(|| cached.wk_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wk_t not available on GPU".into())
        })?;
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).or_else(|| cached.wv_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wv_t not available on GPU".into())
        })?;
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).or_else(|| cached.wo_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wo_t not available on GPU".into())
        })?;

        // 2. QKV projection on GPU
        let q_proj = ctx.matmul(x_gpu, wq_t)?;
        let k_proj = ctx.matmul(x_gpu, wk_t)?;
        let v_proj = ctx.matmul(x_gpu, wv_t)?;

        // 3. Reshape Q/K to per-head [total_heads, head_dim] before RoPE
        let q_heads = q_proj.reshape(vec![batch_size * self.num_heads, self.head_dim])?;
        let k_heads = k_proj.reshape(vec![batch_size * self.num_kv_heads, self.head_dim])?;

        // 4. Apply RoPE on GPU using PRE-UPLOADED cos/sin (per-head)
        let q_rotated = ctx.rotary_embedding(&q_heads, cos_gpu, sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_heads, cos_gpu, sin_gpu, self.head_dim as u32)?;

        // 5. Reshape Q to [1, num_heads, 1, head_dim] for fused_attention
        let q_4d = q_rotated.reshape(vec![batch_size, self.num_heads, 1, self.head_dim])?;

        // 6. Append K/V to GPU-resident cache
        let k_3d = k_rotated.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
        let v_3d = v_proj.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
        cache.append(&ctx, &k_3d, &v_3d)?;

        // 6. Get repeated K/V from GPU cache
        let (k_repeated, v_repeated) = cache.get_repeated_kv(&ctx, self.num_heads as u32)?;
        let total_seq = cache.seq_len;

        // 7. Reshape K/V to 4D for fused_attention
        let k_4d = k_repeated.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;
        let v_4d = v_repeated.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;

        // 8. Fused attention on GPU — no CPU round-trip!
        let attn_output_4d = ctx.fused_attention(&q_4d, &k_4d, &v_4d, self.head_dim_rs, false)?;

        // 9. Reshape attention output
        let attn_flat = attn_output_4d.reshape(vec![batch_size, self.num_heads * self.head_dim])?;

        // 10. Output projection on GPU
        ctx.matmul(&attn_flat, wo_t)
    }

    /// GPU forward with GPU-resident KV cache.
    /// No CPU round-trip for K/V cache operations.
    pub fn forward_gpu_with_cache(
        &self,
        x_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
        cache: &mut GpuKVCacheEntry,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Result<nexora_deeplearning::autograd::gpu::GpuTensor, nexora_deeplearning::autograd::gpu::GpuError> {
        use nexora_deeplearning::autograd::gpu::{GpuContext, GpuError, GpuTensor};

        let ctx = GpuContext::global()?;
        let shape_v = x_gpu.shape();
        let batch_size = shape_v.first().copied().unwrap_or(1);

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).or_else(|| cached.wq_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wq_t not available on GPU".into())
        })?;
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).or_else(|| cached.wk_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wk_t not available on GPU".into())
        })?;
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).or_else(|| cached.wv_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wv_t not available on GPU".into())
        })?;
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).or_else(|| cached.wo_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wo_t not available on GPU".into())
        })?;

        // 2. QKV projection on GPU
        let q_proj = ctx.matmul(x_gpu, wq_t)?;
        let k_proj = ctx.matmul(x_gpu, wk_t)?;
        let v_proj = ctx.matmul(x_gpu, wv_t)?;

        // 3. Upload cos/sin for RoPE
        let half = cos.len();
        let cos_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, half], cos.to_vec())
                .map_err(|e| GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, half], sin.to_vec())
                .map_err(|e| GpuError::Unsupported(e.to_string()))?,
        )?;

        // 4. Reshape Q/K to per-head [total_heads, head_dim] before RoPE
        let q_heads = q_proj.reshape(vec![batch_size * self.num_heads, self.head_dim])?;
        let k_heads = k_proj.reshape(vec![batch_size * self.num_kv_heads, self.head_dim])?;

        // Apply RoPE on GPU for Q and K (per-head)
        let q_rotated = ctx.rotary_embedding(&q_heads, &cos_gpu, &sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_heads, &cos_gpu, &sin_gpu, self.head_dim as u32)?;

        // 5. Reshape Q to [1, num_heads, 1, head_dim] for fused_attention
        let q_4d = q_rotated.reshape(vec![batch_size, self.num_heads, 1, self.head_dim])?;

        // 6. Append K/V to GPU-resident cache
        let k_3d = k_rotated.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
        let v_3d = v_proj.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
        cache.append(&ctx, &k_3d, &v_3d)?;

        // 6. Get repeated K/V from GPU cache
        let (k_repeated, v_repeated) = cache.get_repeated_kv(&ctx, self.num_heads as u32)?;
        let total_seq = cache.seq_len;

        // 7. Reshape K/V to 4D for fused_attention
        let k_4d = k_repeated.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;
        let v_4d = v_repeated.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;

        // 8. Fused attention on GPU
        let attn_output_4d = ctx.fused_attention(&q_4d, &k_4d, &v_4d, self.head_dim_rs, false)?;

        // 9. Reshape attention output
        let attn_flat = attn_output_4d.reshape(vec![batch_size, self.num_heads * self.head_dim])?;

        // 10. Output projection on GPU
        ctx.matmul(&attn_flat, wo_t)
    }

    /// GPU forward with CPU-side KV cache (Vec<KVCacheEntry>).
    /// Uses GPU-resident scratch cache internally — no O(n²) CPU round-trip.
    /// Backward-compatible: CPU cache is synced back with O(1) per step.
    pub fn forward_gpu(
        &self,
        x_gpu: &nexora_deeplearning::autograd::gpu::GpuTensor,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Result<nexora_deeplearning::autograd::gpu::GpuTensor, nexora_deeplearning::autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_deeplearning::autograd::gpu::{GpuContext, GpuError, GpuTensor};

        let ctx = GpuContext::global()?;
        let shape_v = x_gpu.shape();
        let batch_size = shape_v.first().copied().unwrap_or(1);

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).or_else(|| cached.wq_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wq_t not available on GPU".into())
        })?;
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).or_else(|| cached.wk_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wk_t not available on GPU".into())
        })?;
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).or_else(|| cached.wv_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wv_t not available on GPU".into())
        })?;
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).or_else(|| cached.wo_t.as_ref()).ok_or_else(|| {
            nexora_deeplearning::autograd::gpu::GpuError::Unsupported("wo_t not available on GPU".into())
        })?;

        // 2. QKV projection on GPU
        let q_proj = ctx.matmul(x_gpu, wq_t)?;
        let k_proj = ctx.matmul(x_gpu, wk_t)?;
        let v_proj = ctx.matmul(x_gpu, wv_t)?;

        // 3. Upload cos/sin for RoPE
        let half = cos.len();
        let cos_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, half], cos.to_vec())
                .map_err(|e| GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, half], sin.to_vec())
                .map_err(|e| GpuError::Unsupported(e.to_string()))?,
        )?;

        // 4. Reshape Q/K to per-head [total_heads, head_dim] before RoPE
        let q_heads = q_proj.reshape(vec![batch_size * self.num_heads, self.head_dim])?;
        let k_heads = k_proj.reshape(vec![batch_size * self.num_kv_heads, self.head_dim])?;

        let q_rotated = ctx.rotary_embedding(&q_heads, &cos_gpu, &sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_heads, &cos_gpu, &sin_gpu, self.head_dim as u32)?;

        // 5. Reshape Q to [1, num_heads, 1, head_dim]
        let q_4d = q_rotated.reshape(vec![batch_size, self.num_heads, 1, self.head_dim])?;

        // 5. GPU-side KV cache (no CPU round-trip)
        let (k_repeated, v_repeated) = {
            let mut scratch_guard = self.gpu_scratch.write();
            let cpu_seq_len = if layer_idx < cache.len() {
                cache[layer_idx].seq_len()
            } else {
                0
            };

            let mut gpu_entry = match scratch_guard.take() {
                Some(entry) => entry,
                None => GpuKVCacheEntry::new(
                    &ctx,
                    self.num_kv_heads,
                    self.head_dim,
                    4096.max(cpu_seq_len + 4096),
                    false,
                )?,
            };

            // Re-sync from CPU if cache was reset or different generation
            if gpu_entry.seq_len != cpu_seq_len {
                if cpu_seq_len == 0 {
                    gpu_entry.seq_len = 0;
                    ctx.fill_zero(&gpu_entry.k)?;
                    ctx.fill_zero(&gpu_entry.v)?;
                } else {
                    gpu_entry = GpuKVCacheEntry::new(
                        &ctx,
                        self.num_kv_heads,
                        self.head_dim,
                        cpu_seq_len + 4096,
                        false,
                    )?;
                    let ce = &cache[layer_idx];
                    let seq = ce.seq_len();
                    let k_flat: Vec<f32> = ce.k.clone();
                    let v_flat: Vec<f32> = ce.v.clone();
                    let k_cpu =
                        ArrayD::from_shape_vec(vec![seq, self.num_kv_heads, self.head_dim], k_flat)
                            .map_err(|e| GpuError::Unsupported(e.to_string()))?;
                    let v_cpu =
                        ArrayD::from_shape_vec(vec![seq, self.num_kv_heads, self.head_dim], v_flat)
                            .map_err(|e| GpuError::Unsupported(e.to_string()))?;
                    let k_host = GpuTensor::from_cpu(&k_cpu)?;
                    let v_host = GpuTensor::from_cpu(&v_cpu)?;
                    let bytes = (seq * self.num_kv_heads * self.head_dim * 4) as u64;
                    ctx.batch_dispatch(|enc| {
                        enc.copy_buffer_to_buffer(
                            k_host.buffer(),
                            0,
                            gpu_entry.k.buffer(),
                            0,
                            bytes,
                        );
                        enc.copy_buffer_to_buffer(
                            v_host.buffer(),
                            0,
                            gpu_entry.v.buffer(),
                            0,
                            bytes,
                        );
                        Ok(())
                    })?;
                    gpu_entry.seq_len = seq;
                }
            }

            // Append new K/V to GPU cache (GPU-only copy_buffer_to_buffer)
            let k_3d = k_rotated.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
            let v_3d = v_proj.reshape(vec![batch_size, self.num_kv_heads, self.head_dim])?;
            gpu_entry.append(&ctx, &k_3d, &v_3d)?;

            // Get repeated K/V from GPU cache
            let (k_rep, v_rep) = gpu_entry.get_repeated_kv(&ctx, self.num_heads as u32)?;
            let total_seq = gpu_entry.seq_len;

            let k_4d = k_rep.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;
            let v_4d = v_rep.reshape(vec![1, self.num_heads, total_seq, self.head_dim])?;

            // Sync back new token to CPU cache (O(1) per step)
            self.sync_back_cpu_cache(&ctx, &gpu_entry, cache, layer_idx)?;

            *scratch_guard = Some(gpu_entry);
            (k_4d, v_4d)
        };

        // 7. Fused attention on GPU
        let attn_output_4d =
            ctx.fused_attention(&q_4d, &k_repeated, &v_repeated, self.head_dim_rs, false)?;

        // 8. Reshape attention output
        let attn_flat = attn_output_4d.reshape(vec![batch_size, self.num_heads * self.head_dim])?;

        // 9. Output projection on GPU
        ctx.matmul(&attn_flat, wo_t)
    }

    /// Sync back only the last token's K/V from GPU cache to CPU cache.
    /// O(1) per step — reads a single (kv_heads * head_dim) element.
    fn sync_back_cpu_cache(
        &self,
        ctx: &nexora_deeplearning::autograd::gpu::GpuContext,
        gpu_entry: &GpuKVCacheEntry,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
    ) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        use nexora_deeplearning::autograd::gpu::GpuError;

        if gpu_entry.seq_len == 0 {
            return Ok(());
        }
        let kv_elems = self.num_kv_heads * self.head_dim;
        let token_bytes = (kv_elems * 4) as u64;
        let last_idx = gpu_entry.seq_len - 1;
        let byte_off = (last_idx * kv_elems * 4) as u64;

        let staging_k = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sync_back_k"),
            size: token_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let staging_v = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sync_back_v"),
            size: token_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        ctx.batch_dispatch(|enc| {
            enc.copy_buffer_to_buffer(gpu_entry.k.buffer(), byte_off, &staging_k, 0, token_bytes);
            enc.copy_buffer_to_buffer(gpu_entry.v.buffer(), byte_off, &staging_v, 0, token_bytes);
            Ok(())
        })?;

        ctx.sync();

        let read_back = |staging: &wgpu::Buffer| -> Result<Vec<f32>, GpuError> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let map_result = loop {
                ctx.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_millis(100)),
                });
                match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(result) => break result,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if std::time::Instant::now() > deadline {
                            break Err(wgpu::BufferAsyncError);
                        }
                        continue;
                    }
                    Err(_) => break Err(wgpu::BufferAsyncError),
                }
            };
            let _map_result =
                map_result.map_err(|e| GpuError::Device(format!("map_async: {e:?}")))?;
            let out: Vec<f32> = {
                let mapped = slice.get_mapped_range();
                let result = mapped
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                result
            };
            staging.unmap();
            Ok(out)
        };

        let new_k = read_back(&staging_k)?;
        let new_v = read_back(&staging_v)?;

        if layer_idx < cache.len() {
            let entry = &mut cache[layer_idx];
            entry.k.extend_from_slice(&new_k);
            entry.v.extend_from_slice(&new_v);
        } else {
            cache.push(KVCacheEntry {
                k: new_k,
                v: new_v,
                kv_dim: kv_elems,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                k_compressed: Vec::new(),
                v_compressed: Vec::new(),
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                compressed_seq_len: 0,
            });
        }

        Ok(())
    }

    /// Pre-upload weights to GPU — call before inference to avoid first-pass latency.
    pub fn preupload_gpu(&self) -> Result<(), nexora_deeplearning::autograd::gpu::GpuError> {
        self.ensure_weights_gpu()
    }
}

