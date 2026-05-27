use std::borrow::Cow;
use std::sync::OnceLock;

use ndarray::{Array1, Array2};
use rand::Rng;
use rayon::prelude::*;

use super::rope::RoPE;

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::{GpuContext, GpuDtype, GpuTensor};

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
/// Temporary f32 weights upconverted from f16 storage at forward start.
/// Dropped at end of each forward pass — GPU memory reclaimed.
pub(crate) struct GqaGpuTemps {
    pub wq_t: nexora_autograd::gpu::GpuTensor,
    pub wk_t: nexora_autograd::gpu::GpuTensor,
    pub wv_t: nexora_autograd::gpu::GpuTensor,
    pub wo_t: nexora_autograd::gpu::GpuTensor,
}

#[derive(Debug)]
pub(crate) struct GqaGpuWeights {
    pub wq_t: nexora_autograd::gpu::GpuTensor,
    pub wk_t: nexora_autograd::gpu::GpuTensor,
    pub wv_t: nexora_autograd::gpu::GpuTensor,
    pub wo_t: nexora_autograd::gpu::GpuTensor,
    pub wq_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub wk_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub wv_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub wo_f16: Option<nexora_autograd::gpu::GpuTensor>,
}

/// GPU-resident KV cache entry.
/// Stores K/V directly on GPU without CPU round-trip.
/// Pre-allocated to max_seq_len with append via GPU buffer copy.
/// Supports F16 packed storage (2 f16 per u32) when f16_storage=true.
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuKVCacheEntry {
    pub k: GpuTensor, // [capacity, kv_heads, head_dim]
    pub v: GpuTensor, // [capacity, kv_heads, head_dim]
    pub seq_len: usize,
    pub capacity: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub f16_storage: bool,
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
    ) -> Result<Self, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuError;
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
            })
        } else {
            Ok(Self {
                k: GpuTensor::zeros(&[max_seq, kv_heads, head_dim])?,
                v: GpuTensor::zeros(&[max_seq, kv_heads, head_dim])?,
                seq_len: 0,
                capacity: max_seq,
                kv_heads,
                head_dim,
                f16_storage: false,
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
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        if self.seq_len >= self.capacity {
            return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                "GpuKVCacheEntry capacity exceeded: {} >= {}",
                self.seq_len, self.capacity
            )));
        }
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
    ) -> Result<(GpuTensor, GpuTensor), nexora_autograd::gpu::GpuError> {
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
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        if self.seq_len + num_tokens > self.capacity {
            return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
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
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
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

    /// Clear the cache (reset sequence length).
    /// Does NOT deallocate buffers — memset to zero.
    /// Uses u32 zero fill for F16 storage, f32 zero fill for F32 storage.
    pub fn clear(&mut self, ctx: &GpuContext) -> Result<(), nexora_autograd::gpu::GpuError> {
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
impl Clone for GpuKVCacheEntry {
    fn clone(&self) -> Self {
        Self {
            k: self.k.clone(),
            v: self.v.clone(),
            seq_len: self.seq_len,
            capacity: self.capacity,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            f16_storage: self.f16_storage,
        }
    }
}

/// Unified trait for KV cache — supports both CPU (Vec<KVCacheEntry>) and GPU (GpuKVCacheEntry).
/// Enables the inference engine to transparently switch between CPU and GPU KV cache
/// without changing the model forward path.
pub trait KVCacheProvider: Send {
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
}

/// CPU implementation of [`KVCacheProvider`].
/// Wraps `Vec<KVCacheEntry>` with flat `Vec<f32>` storage for O(1) amortized append.
pub struct CpuKVCache {
    pub entries: Vec<KVCacheEntry>,
}

impl CpuKVCache {
    pub fn new(_num_layers: usize) -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl KVCacheProvider for CpuKVCache {
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
#[cfg(feature = "gpu")]
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
        let entries = if let Ok(ctx) = nexora_autograd::gpu::GpuContext::global() {
            (0..num_layers)
                .filter_map(|_| {
                    GpuKVCacheEntry::new(&ctx, num_kv_heads, head_dim, max_seq_len, false).ok()
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
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        assert_eq!(
            self.entries.len(),
            source.entries.len(),
            "copy_prefix_from: layer count mismatch {} vs {}",
            self.entries.len(),
            source.entries.len()
        );
        let ctx = nexora_autograd::gpu::GpuContext::global().map_err(|_| {
            nexora_autograd::gpu::GpuError::NotInitialized
        })?;
        for (dst_entry, src_entry) in self.entries.iter_mut().zip(source.entries.iter()) {
            dst_entry.copy_prefix_from(&ctx, src_entry, prefix_len)?;
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl KVCacheProvider for GpuKVCache {
    fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        if let Ok(ref ctx) = nexora_autograd::gpu::GpuContext::global() {
            if layer_idx < self.entries.len() {
                use ndarray::ArrayD;
                use nexora_autograd::gpu::GpuTensor;

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
/// Inference crate's PagedKVCache implements this trait.
pub trait PagedCacheReader {
    /// Read KV data for a token position. Returns (k_row, v_row).
    fn read(&self, seq_id: u64, layer: usize, token_pos: usize) -> Option<(Vec<f32>, Vec<f32>)>;
    /// Number of tokens stored for a sequence.
    fn num_tokens(&self, seq_id: u64) -> Option<usize>;
    /// Append a token's KV data (single position).
    fn append(&mut self, seq_id: u64, layer: usize, token_pos: usize, k_row: &[f32], v_row: &[f32]);
}

#[derive(Debug)]
pub struct GQA {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_groups: usize,
    pub head_dim_rs: f32,
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<GqaGpuWeights>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_scratch: parking_lot::RwLock<Option<GpuKVCacheEntry>>,
    pub(crate) use_half_precision: bool,
}

#[cfg(not(feature = "gpu"))]
impl Clone for GQA {
    fn clone(&self) -> Self {
        Self {
            num_heads: self.num_heads.clone(),
            num_kv_heads: self.num_kv_heads.clone(),
            head_dim: self.head_dim.clone(),
            num_groups: self.num_groups.clone(),
            head_dim_rs: self.head_dim_rs.clone(),
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
            use_half_precision: self.use_half_precision,
        }
    }
}

#[cfg(feature = "gpu")]
impl Clone for GQA {
    fn clone(&self) -> Self {
        Self {
            num_heads: self.num_heads.clone(),
            num_kv_heads: self.num_kv_heads.clone(),
            head_dim: self.head_dim.clone(),
            num_groups: self.num_groups.clone(),
            head_dim_rs: self.head_dim_rs.clone(),
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
            gpu_weights: OnceLock::new(),
            gpu_scratch: parking_lot::RwLock::new(None),
            use_half_precision: self.use_half_precision,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub k: Vec<f32>, // flat [seq_len, kv_dim]
    pub v: Vec<f32>, // flat [seq_len, kv_dim]
    pub kv_dim: usize,
}

impl KVCacheEntry {
    pub fn seq_len(&self) -> usize {
        if self.kv_dim == 0 {
            0
        } else {
            self.k.len() / self.kv_dim
        }
    }
}

impl GQA {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (head_dim as f32).sqrt().recip();

        let wq = Array2::from_shape_fn((num_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let wk = Array2::from_shape_fn((num_kv_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let wv = Array2::from_shape_fn((num_kv_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let wo = Array2::from_shape_fn((hidden_size, num_heads * head_dim), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });

        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            num_groups: num_heads / num_kv_heads,
            head_dim_rs: 1.0 / (head_dim as f32).sqrt(),
            wq,
            wk,
            wv,
            wo,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu_scratch: parking_lot::RwLock::new(None),
            use_half_precision: false,
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        mut cache: Option<&mut Vec<KVCacheEntry>>,
        layer_idx: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Array2<f32> {
        // Pure CPU forward path.
        // GPU-resident execution uses `forward_gpu` (no per-layer readback).
        let (batch_size, _) = x.dim();

        let q_proj = x.dot(&self.wq.t());
        let k_proj = x.dot(&self.wk.t());
        let v_proj = x.dot(&self.wv.t());

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_heads, self.head_dim))
            });
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });

        for mut k_slice in k.axis_iter_mut(ndarray::Axis(0)) {
            let k_row: Vec<f32> = k_slice.iter().copied().collect();
            let rotated_k = RoPE::apply_single(&k_row, cos, sin, self.head_dim, 0);
            let rotated = ndarray::Array2::from_shape_vec(
                (self.num_kv_heads, self.head_dim),
                rotated_k.into_raw_vec(),
            )
            .unwrap_or_else(|_| ndarray::Array2::zeros((self.num_kv_heads, self.head_dim)));
            k_slice.assign(&rotated);
        }

        for mut q_slice in q.axis_iter_mut(ndarray::Axis(0)) {
            let q_row: Vec<f32> = q_slice.iter().copied().collect();
            let rotated_q = RoPE::apply_single(&q_row, cos, sin, self.head_dim, 0);
            let rotated = ndarray::Array2::from_shape_vec(
                (self.num_heads, self.head_dim),
                rotated_q.into_raw_vec(),
            )
            .unwrap_or_else(|_| ndarray::Array2::zeros((self.num_heads, self.head_dim)));
            q_slice.assign(&rotated);
        }

        let kv_dim = self.num_kv_heads * self.head_dim;
        let (k_cached, v_cached, total_seq): (Cow<[f32]>, Cow<[f32]>, usize) = match cache {
            Some(cache) if layer_idx < cache.len() => {
                let entry = &cache[layer_idx];
                let seq = entry.k.len() / (batch_size * kv_dim);
                (Cow::Borrowed(&entry.k), Cow::Borrowed(&entry.v), seq)
            }
            _ => {
                let kf: Vec<f32> = k.iter().copied().collect();
                let vf: Vec<f32> = v.iter().copied().collect();
                (Cow::Owned(kf), Cow::Owned(vf), 1)
            }
        };

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            let results: Vec<(usize, Vec<f32>)> = (0..self.num_heads)
                .into_par_iter()
                .map(|h| {
                    let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                    let kv_off = kv_h * self.head_dim;

                    let q_slice = q.slice(ndarray::s![b, h, ..]);
                    let mut scores = Vec::with_capacity(total_seq);
                    let mut max_score = f32::NEG_INFINITY;
                    for t in 0..total_seq {
                        let k_idx = (b * total_seq + t) * kv_dim + kv_off;
                        let k_view =
                            ndarray::ArrayView1::from(&k_cached[k_idx..k_idx + self.head_dim]);
                        let score = q_slice.dot(&k_view) * self.head_dim_rs;
                        if score > max_score {
                            max_score = score;
                        }
                        scores.push(score);
                    }

                    let mut exp_sum = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }

                    let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                    let mut out_row = vec![0.0; self.head_dim];
                    for d in 0..self.head_dim {
                        let mut weighted = 0.0;
                        for t in 0..total_seq {
                            weighted += scores[t]
                                * inv_exp_sum
                                * v_cached[(b * total_seq + t) * kv_dim + kv_off + d];
                        }
                        out_row[d] = weighted;
                    }
                    (h, out_row)
                })
                .collect();

            for (h, row) in results {
                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    output[[b, out_base + d]] = row[d];
                }
            }
        }

        output.dot(&self.wo.t())
    }

    pub fn forward_with_kv(
        &self,
        x: &Array2<f32>,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Array2<f32> {
        let (batch_size, _) = x.dim();

        let q_proj = x.dot(&self.wq.t());
        let k_proj = x.dot(&self.wk.t());
        let v_proj = x.dot(&self.wv.t());

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_heads, self.head_dim))
            });
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });

        for mut k_slice in k.axis_iter_mut(ndarray::Axis(0)) {
            let k_row: Vec<f32> = k_slice.iter().copied().collect();
            let rotated_k = RoPE::apply_single(&k_row, cos, sin, self.head_dim, 0);
            let rotated = ndarray::Array2::from_shape_vec(
                (self.num_kv_heads, self.head_dim),
                rotated_k.into_raw_vec(),
            )
            .unwrap_or_else(|_| ndarray::Array2::zeros((self.num_kv_heads, self.head_dim)));
            k_slice.assign(&rotated);
        }

        for mut q_slice in q.axis_iter_mut(ndarray::Axis(0)) {
            let q_row: Vec<f32> = q_slice.iter().copied().collect();
            let rotated_q = RoPE::apply_single(&q_row, cos, sin, self.head_dim, 0);
            let rotated = ndarray::Array2::from_shape_vec(
                (self.num_heads, self.head_dim),
                rotated_q.into_raw_vec(),
            )
            .unwrap_or_else(|_| ndarray::Array2::zeros((self.num_heads, self.head_dim)));
            q_slice.assign(&rotated);
        }

        let kv_dim = self.num_kv_heads * self.head_dim;

        if layer_idx < cache.len() {
            let entry = &mut cache[layer_idx];
            entry.k.extend(k.iter().copied());
            entry.v.extend(v.iter().copied());
        } else {
            cache.push(KVCacheEntry {
                k: k.iter().copied().collect(),
                v: v.iter().copied().collect(),
                kv_dim,
            });
        }

        let entry = &cache[layer_idx];
        let k_slice = &entry.k;
        let v_slice = &entry.v;
        let total_seq = entry.seq_len();

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            let results: Vec<(usize, Vec<f32>)> = (0..self.num_heads)
                .into_par_iter()
                .map(|h| {
                    let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                    let kv_off = kv_h * self.head_dim;

                    let q_slice = q.slice(ndarray::s![b, h, ..]);
                    let mut scores = vec![0.0; total_seq];
                    let mut max_score = f32::NEG_INFINITY;
                    for t in 0..total_seq {
                        let k_idx = t * kv_dim + kv_off;
                        let k_view =
                            ndarray::ArrayView1::from(&k_slice[k_idx..k_idx + self.head_dim]);
                        let score = q_slice.dot(&k_view) * self.head_dim_rs;
                        if score > max_score {
                            max_score = score;
                        }
                        scores[t] = score;
                    }

                    let mut exp_sum = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }

                    let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                    let mut out_row = vec![0.0; self.head_dim];
                    for d in 0..self.head_dim {
                        let mut weighted = 0.0;
                        for t in 0..total_seq {
                            weighted += scores[t] * inv_exp_sum * v_slice[t * kv_dim + kv_off + d];
                        }
                        out_row[d] = weighted;
                    }
                    (h, out_row)
                })
                .collect();

            for (h, row) in results {
                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    output[[b, out_base + d]] = row[d];
                }
            }
        }

        output.dot(&self.wo.t())
    }

    /// Forward pass using a paged KV cache (reads K/V from blocks per position).
    /// batch_size is always 1 for autoregressive generation.
    pub fn forward_with_paged(
        &self,
        x: &Array2<f32>,
        cache: &mut dyn PagedCacheReader,
        seq_id: u64,
        layer_idx: usize,
        token_pos: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Array2<f32> {
        let (batch_size, _) = x.dim();
        debug_assert_eq!(
            batch_size, 1,
            "forward_with_paged only supports batch_size=1"
        );

        let q_proj = x.dot(&self.wq.t());
        let k_proj = x.dot(&self.wk.t());
        let v_proj = x.dot(&self.wv.t());

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_heads, self.head_dim))
            });
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });

        // RoPE for K
        for mut k_slice in k.axis_iter_mut(ndarray::Axis(0)) {
            let k_row: Vec<f32> = k_slice.iter().copied().collect();
            let rotated_k = RoPE::apply_single(&k_row, cos, sin, self.head_dim, 0);
            let rotated = ndarray::Array2::from_shape_vec(
                (self.num_kv_heads, self.head_dim),
                rotated_k.into_raw_vec(),
            )
            .unwrap_or_else(|_| ndarray::Array2::zeros((self.num_kv_heads, self.head_dim)));
            k_slice.assign(&rotated);
        }

        // RoPE for Q
        for mut q_slice in q.axis_iter_mut(ndarray::Axis(0)) {
            let q_row: Vec<f32> = q_slice.iter().copied().collect();
            let rotated_q = RoPE::apply_single(&q_row, cos, sin, self.head_dim, 0);
            let rotated = ndarray::Array2::from_shape_vec(
                (self.num_heads, self.head_dim),
                rotated_q.into_raw_vec(),
            )
            .unwrap_or_else(|_| ndarray::Array2::zeros((self.num_heads, self.head_dim)));
            q_slice.assign(&rotated);
        }

        // Append this token's K/V to the paged cache
        let k_flat: Vec<f32> = k.iter().copied().collect();
        let v_flat: Vec<f32> = v.iter().copied().collect();
        cache.append(seq_id, layer_idx, token_pos, &k_flat, &v_flat);

        // Read all cached positions for attention computation
        let num_tokens = cache.num_tokens(seq_id).unwrap_or(0);
        if num_tokens == 0 {
            return Array2::zeros((batch_size, self.num_heads * self.head_dim));
        }

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        let cache_ref: &dyn PagedCacheReader = &*cache;
        for b in 0..batch_size {
            let mut results = Vec::with_capacity(self.num_heads);
            for h in 0..self.num_heads {
                let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                let kv_off = kv_h * self.head_dim;

                let q_slice = q.slice(ndarray::s![b, h, ..]);
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(num_tokens);
                for t in 0..num_tokens {
                    let (k_row, _) = match cache_ref.read(seq_id, layer_idx, t) {
                        Some(r) => r,
                        None => continue,
                    };
                    let k_view = ndarray::ArrayView1::from(&k_row[kv_off..kv_off + self.head_dim]);
                    let score = q_slice.dot(&k_view) * self.head_dim_rs;
                    if score > max_score {
                        max_score = score;
                    }
                    scores.push(score);
                }

                if scores.is_empty() {
                    results.push((h, Vec::new()));
                    continue;
                }

                let mut exp_sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }

                let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                let mut out_row = vec![0.0; self.head_dim];
                for d in 0..self.head_dim {
                    let mut weighted = 0.0;
                    for (t, &s) in scores.iter().enumerate() {
                        let (_, v_row) = match cache_ref.read(seq_id, layer_idx, t) {
                            Some(r) => r,
                            None => continue,
                        };
                        weighted += s * inv_exp_sum * v_row[kv_off + d];
                    }
                    out_row[d] = weighted;
                }
                results.push((h, out_row));
            }

            for (h, row) in results {
                if row.is_empty() {
                    continue;
                }
                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    output[[b, out_base + d]] = row[d];
                }
            }
        }

        output.dot(&self.wo.t())
    }
}

#[cfg(feature = "gpu")]
impl GQA {
    fn ensure_weights_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};
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
        let wq = mk(&self.wq)?;
        let wk = mk(&self.wk)?;
        let wv = mk(&self.wv)?;
        let wo = mk(&self.wo)?;
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
        // F16 mode: f32 transposed QKV+O weights ga dipake —
        // forward_gpu upconvert f16→f32 temp per-call.
        // `zeros(&[1])` DUMMY placeholder (4 bytes per weight, total ~16 bytes),
        // BUKAN 4 matriks perhatian ukuran penuh. VRAM hemat: (4 × hidden × dim) bytes.
        let (wq_t, wk_t, wv_t, wo_t) = if use_f16 {
            let d = GpuTensor::zeros(&[1])?;
            (d.clone(), d.clone(), d.clone(), d)
        } else {
            (
                ctx.transpose(&wq)?,
                ctx.transpose(&wk)?,
                ctx.transpose(&wv)?,
                ctx.transpose(&wo)?,
            )
        };
        let _ = self.gpu_weights.set(GqaGpuWeights {
            wq_t,
            wk_t,
            wv_t,
            wo_t,
            wq_f16,
            wk_f16,
            wv_f16,
            wo_f16,
        });
        Ok(())
    }

    fn gqa_upconvert_f16(
        cached: &GqaGpuWeights,
        ctx: &nexora_autograd::gpu::GpuContext,
    ) -> Result<GqaGpuTemps, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuError;
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
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos_gpu: &nexora_autograd::gpu::GpuTensor,
        sin_gpu: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};

        let ctx = GpuContext::global()?;
        let batch_size = 1;

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).unwrap_or(&cached.wq_t);
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).unwrap_or(&cached.wk_t);
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).unwrap_or(&cached.wv_t);
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).unwrap_or(&cached.wo_t);

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
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut GpuKVCacheEntry,
        cos_gpu: &nexora_autograd::gpu::GpuTensor,
        sin_gpu: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};

        let ctx = GpuContext::global()?;
        let batch_size = 1;

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).unwrap_or(&cached.wq_t);
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).unwrap_or(&cached.wk_t);
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).unwrap_or(&cached.wv_t);
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).unwrap_or(&cached.wo_t);

        // 2. QKV projection on GPU
        let q_proj = ctx.matmul(x_gpu, wq_t)?;
        let k_proj = ctx.matmul(x_gpu, wk_t)?;
        let v_proj = ctx.matmul(x_gpu, wv_t)?;

        // 3. Apply RoPE on GPU using PRE-UPLOADED cos/sin
        let q_rotated = ctx.rotary_embedding(&q_proj, cos_gpu, sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_proj, cos_gpu, sin_gpu, self.head_dim as u32)?;

        // 4. Reshape Q to [1, num_heads, 1, head_dim] for fused_attention
        let q_4d = q_rotated.reshape(vec![batch_size, self.num_heads, 1, self.head_dim])?;

        // 5. Append K/V to GPU-resident cache
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
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut GpuKVCacheEntry,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};

        let ctx = GpuContext::global()?;
        let batch_size = 1;

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).unwrap_or(&cached.wq_t);
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).unwrap_or(&cached.wk_t);
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).unwrap_or(&cached.wv_t);
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).unwrap_or(&cached.wo_t);

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

        // Apply RoPE on GPU for Q and K
        let q_rotated = ctx.rotary_embedding(&q_proj, &cos_gpu, &sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_proj, &cos_gpu, &sin_gpu, self.head_dim as u32)?;

        // 4. Reshape Q to [1, num_heads, 1, head_dim] for fused_attention
        let q_4d = q_rotated.reshape(vec![batch_size, self.num_heads, 1, self.head_dim])?;

        // 5. Append K/V to GPU-resident cache
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
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};

        let ctx = GpuContext::global()?;
        let batch_size = 1;

        // 1. Ensure GPU weights are initialized
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after lazy init".into(),
            )
        })?;

        // 1b. Upconvert f16 → f32 temps if half precision
        let _f16_temps = if self.use_half_precision {
            Some(Self::gqa_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let wq_t = _f16_temps.as_ref().map(|t| &t.wq_t).unwrap_or(&cached.wq_t);
        let wk_t = _f16_temps.as_ref().map(|t| &t.wk_t).unwrap_or(&cached.wk_t);
        let wv_t = _f16_temps.as_ref().map(|t| &t.wv_t).unwrap_or(&cached.wv_t);
        let wo_t = _f16_temps.as_ref().map(|t| &t.wo_t).unwrap_or(&cached.wo_t);

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

        let q_rotated = ctx.rotary_embedding(&q_proj, &cos_gpu, &sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_proj, &cos_gpu, &sin_gpu, self.head_dim as u32)?;

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
        ctx: &nexora_autograd::gpu::GpuContext,
        gpu_entry: &GpuKVCacheEntry,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuError;

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
            let map_result =
                map_result.map_err(|e| GpuError::Device(format!("map_async: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let out: Vec<f32> = mapped
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
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
            });
        }

        Ok(())
    }

    /// Pre-upload weights to GPU — call before inference to avoid first-pass latency.
    pub fn preupload_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        self.ensure_weights_gpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn small_gqa() -> GQA {
        GQA::new(8, 4, 2, 4)
    }

    #[test]
    fn test_gqa_new_shapes() {
        let gqa = small_gqa();
        assert_eq!(gqa.num_heads, 4);
        assert_eq!(gqa.num_kv_heads, 2);
        assert_eq!(gqa.head_dim, 4);
        assert_eq!(gqa.num_groups, 2);
        assert_eq!(gqa.wq.dim(), (16, 8));
        assert_eq!(gqa.wk.dim(), (8, 8));
        assert_eq!(gqa.wv.dim(), (8, 8));
        assert_eq!(gqa.wo.dim(), (8, 16));
    }

    #[test]
    fn test_gqa_forward_shape_no_cache() {
        let gqa = small_gqa();
        let x = array![[1.0; 8]];
        let cos = vec![0.0; 2];
        let sin = vec![1.0; 2];
        let out = gqa.forward(&x, None, 0, &cos, &sin);
        assert_eq!(out.dim(), (1, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gqa_forward_with_kv_shape_and_cache() {
        let gqa = small_gqa();
        let x = array![[1.0; 8]];
        let cos = vec![0.0; 2];
        let sin = vec![1.0; 2];
        let mut cache = vec![KVCacheEntry {
            k: vec![],
            v: vec![],
            kv_dim: 8,
        }];
        let out = gqa.forward_with_kv(&x, &mut cache, 0, &cos, &sin);
        assert_eq!(out.dim(), (1, 8));
        assert!(cache[0].seq_len() > 0);
        assert!(!cache[0].k.is_empty());
    }

    #[test]
    fn test_gqa_forward_batch2() {
        let gqa = small_gqa();
        let x = array![[1.0; 8], [2.0; 8]];
        let cos = vec![0.5; 2];
        let sin = vec![0.5; 2];
        let out = gqa.forward(&x, None, 0, &cos, &sin);
        assert_eq!(out.dim(), (2, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_kv_cache_entry_seq_len() {
        let entry = KVCacheEntry {
            k: vec![1.0; 16],
            v: vec![2.0; 16],
            kv_dim: 8,
        };
        assert_eq!(entry.seq_len(), 2);
        assert_eq!(
            KVCacheEntry {
                k: vec![],
                v: vec![],
                kv_dim: 0
            }
            .seq_len(),
            0
        );
    }

    #[test]
    fn test_cpu_kv_cache_append_read() {
        let mut cache = CpuKVCache::new(0);
        assert_eq!(cache.num_layers(), 0);
        cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.num_layers(), 1);
        assert_eq!(cache.seq_len(0), 1);
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.get_v(0), &[5.0, 6.0, 7.0, 8.0]);
        cache.append(0, &[9.0; 4], &[10.0; 4]);
        assert_eq!(cache.seq_len(0), 2);
        assert_eq!(cache.get_k(0).len(), 8);
        cache.clear();
        assert_eq!(cache.num_layers(), 0);
    }

    #[test]
    fn test_cpu_kv_cache_out_of_range() {
        let cache = CpuKVCache::new(0);
        assert_eq!(cache.get_k(99), &[] as &[f32]);
        assert_eq!(cache.get_v(99), &[] as &[f32]);
        assert_eq!(cache.seq_len(99), 0);
    }
}
