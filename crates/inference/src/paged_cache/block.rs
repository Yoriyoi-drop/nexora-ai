use ndarray::{s, Array2};
use std::time::Instant;

use crate::paged_cache::{MemoryTier, PagedCacheConfig};

// ─── Block Storage ───────────────────────────────────────────────────────────

/// Block storage representation — either f32 or f16 (half-precision).
pub(crate) enum BlockData {
    F32(Array2<f32>),
    F16(Vec<u16>, usize, usize),
    /// 4-bit quantized: packed_u8 + per-head scales + block_size + cols
    Q4(Vec<u8>, Vec<f32>, usize, usize),
}

impl BlockData {
    fn zeroes(f16: bool, block_size: usize, cols: usize) -> Self {
        if f16 {
            BlockData::F16(vec![0u16; block_size * cols], block_size, cols)
        } else {
            BlockData::F32(Array2::zeros((block_size, cols)))
        }
    }

    fn zeroes_quantized(block_size: usize, cols: usize, num_kv_heads: usize) -> Self {
        let num_vals = block_size * cols;
        let packed_len = (num_vals + 1) / 2;
        BlockData::Q4(
            vec![0u8; packed_len],
            vec![1.0f32; num_kv_heads],
            block_size,
            cols,
        )
    }

    fn block_size(&self) -> usize {
        match self {
            BlockData::F32(a) => a.shape()[0],
            BlockData::F16(_, bs, _) => *bs,
            BlockData::Q4(_, _, bs, _) => *bs,
        }
    }

    fn cols(&self) -> usize {
        match self {
            BlockData::F32(a) => a.shape()[1],
            BlockData::F16(_, _, c) => *c,
            BlockData::Q4(_, _, _, c) => *c,
        }
    }

    pub(crate) fn read_row(&self, row: usize) -> Vec<f32> {
        match self {
            BlockData::F32(a) => a.slice(s![row, ..]).to_vec(),
            BlockData::F16(data, _bs, cols) => {
                let start = row * cols;
                data[start..start + cols]
                    .iter()
                    .map(|&b| crate::f16_bits_to_f32(b))
                    .collect()
            }
            BlockData::Q4(packed, scales, _bs, cols) => {
                let kv_dim = *cols;
                let num_kv_heads = scales.len();
                if num_kv_heads == 0 || kv_dim == 0 {
                    return vec![0.0; kv_dim];
                }
                let head_dim = kv_dim / num_kv_heads;
                let mut out = vec![0.0f32; kv_dim];
                for i in 0..kv_dim {
                    let byte_idx = (row * kv_dim + i) / 2;
                    let nibble = if (row * kv_dim + i) % 2 == 0 {
                        (packed[byte_idx] >> 4) & 0x0F
                    } else {
                        packed[byte_idx] & 0x0F
                    };
                    let q_val = nibble as i8 - 8;
                    let h = i / head_dim;
                    let scale = if h < num_kv_heads { scales[h] } else { 1.0 };
                    out[i] = q_val as f32 * scale;
                }
                out
            }
        }
    }

    pub(crate) fn write_row(&mut self, row: usize, values: &[f32]) {
        match self {
            BlockData::F32(a) => {
                for (i, &v) in values.iter().enumerate() {
                    a[[row, i]] = v;
                }
            }
            BlockData::F16(data, _bs, cols) => {
                let cols = *cols;
                let start = row * cols;
                for (i, &v) in values.iter().take(cols).enumerate() {
                    data[start + i] = crate::f32_to_f16_bits(v);
                }
            }
            BlockData::Q4(packed, scales, _bs, cols) => {
                let kv_dim = *cols;
                let num_kv_heads = scales.len();
                if num_kv_heads == 0 || kv_dim == 0 {
                    return;
                }
                let head_dim = kv_dim / num_kv_heads;
                let mut max_abs = vec![0.0f32; num_kv_heads];
                for (i, &v) in values.iter().enumerate() {
                    let h = i / head_dim;
                    if h < num_kv_heads && v.abs() > max_abs[h] {
                        max_abs[h] = v.abs();
                    }
                }
                for h in 0..num_kv_heads {
                    if max_abs[h] > 0.0 && max_abs[h] / 7.0 > scales[h] {
                        scales[h] = max_abs[h] / 7.0;
                    }
                }
                for (i, &v) in values.iter().enumerate() {
                    let h = i / head_dim;
                    let scale = if h < num_kv_heads {
                        scales[h]
                    } else {
                        1.0
                    };
                    let q = ((v / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                    let byte_idx = (row * kv_dim + i) / 2;
                    if (row * kv_dim + i) % 2 == 0 {
                        packed[byte_idx] = (packed[byte_idx] & 0x0F) | (q << 4);
                    } else {
                        packed[byte_idx] = (packed[byte_idx] & 0xF0) | q;
                    }
                }
            }
        }
    }

    /// Like `slice_rows` but extends an existing `Vec<f32>` — avoids
    /// allocating + freeing the intermediate Vec for each block read.
    pub(crate) fn slice_rows_into(&self, row_start: usize, count: usize, out: &mut Vec<f32>) {
        let start_len = out.len();
        match self {
            BlockData::F32(a) => {
                out.reserve(count * a.shape()[1]);
                for r in row_start..row_start + count {
                    if let Some(slice) = a.slice(s![r, ..]).as_slice() {
                        out.extend_from_slice(slice);
                    }
                }
            }
            BlockData::F16(data, _bs, cols) => {
                let start = row_start * cols;
                let end = start + count * cols;
                out.reserve(end - start);
                out.extend(data[start..end].iter().map(|&b| crate::f16_bits_to_f32(b)));
            }
            BlockData::Q4(packed, scales, _bs, cols) => {
                let kv_dim = *cols;
                let num_kv_heads = scales.len();
                let head_dim = if num_kv_heads > 0 { kv_dim / num_kv_heads } else { 1 };
                out.reserve(count * kv_dim);
                for r in row_start..row_start + count {
                    for i in 0..kv_dim {
                        let byte_idx = (r * kv_dim + i) / 2;
                        let nibble = if (r * kv_dim + i) % 2 == 0 {
                            (packed[byte_idx] >> 4) & 0x0F
                        } else {
                            packed[byte_idx] & 0x0F
                        };
                        let q_val = nibble as i8 - 8;
                        let h = i / head_dim;
                        let scale = if h < num_kv_heads { scales[h] } else { 1.0 };
                        out.push(q_val as f32 * scale);
                    }
                }
            }
        }
        debug_assert_eq!(out.len(), start_len + count * match self {
            BlockData::F32(a) => a.shape()[1],
            BlockData::F16(_, _, cols) | BlockData::Q4(_, _, _, cols) => *cols,
        });
    }

    pub(crate) fn slice_rows(&self, row_start: usize, count: usize) -> Vec<f32> {
        match self {
            BlockData::F32(a) => {
                let mut out = Vec::with_capacity(count * a.shape()[1]);
                for r in row_start..row_start + count {
                    if let Some(slice) = a.slice(s![r, ..]).as_slice() {
                        out.extend_from_slice(slice);
                    }
                }
                out
            }
            BlockData::F16(data, _bs, cols) => {
                let start = row_start * cols;
                let end = start + count * cols;
                data[start..end]
                    .iter()
                    .map(|&b| crate::f16_bits_to_f32(b))
                    .collect()
            }
            BlockData::Q4(packed, scales, _bs, cols) => {
                let kv_dim = *cols;
                let num_kv_heads = scales.len();
                let head_dim = if num_kv_heads > 0 {
                    kv_dim / num_kv_heads
                } else {
                    1
                };
                let mut out = Vec::with_capacity(count * kv_dim);
                for r in row_start..row_start + count {
                    for i in 0..kv_dim {
                        let byte_idx = (r * kv_dim + i) / 2;
                        let nibble = if (r * kv_dim + i) % 2 == 0 {
                            (packed[byte_idx] >> 4) & 0x0F
                        } else {
                            packed[byte_idx] & 0x0F
                        };
                        let q_val = nibble as i8 - 8;
                        let h = i / head_dim;
                        let scale = if h < num_kv_heads {
                            scales[h]
                        } else {
                            1.0
                        };
                        out.push(q_val as f32 * scale);
                    }
                }
                out
            }
        }
    }

    pub(crate) fn clone_data(&self) -> Self {
        match self {
            BlockData::F32(a) => BlockData::F32(a.clone()),
            BlockData::F16(d, bs, c) => BlockData::F16(d.clone(), *bs, *c),
            BlockData::Q4(d, s, bs, c) => BlockData::Q4(d.clone(), s.clone(), *bs, *c),
        }
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            BlockData::F32(a) => a.len() * 4,
            BlockData::F16(d, ..) => d.len() * 2,
            BlockData::Q4(d, s, ..) => d.len() + s.len() * 4,
        }
    }

    pub(crate) fn to_f16(&self) -> Self {
        let bs = self.block_size();
        let cols = self.cols();
        let flat: Vec<f32> = (0..bs)
            .flat_map(|r| self.read_row(r))
            .collect();
        BlockData::F16(
            flat.iter().map(|&v| crate::f32_to_f16_bits(v)).collect(),
            bs,
            cols,
        )
    }

    pub(crate) fn to_q4(&self, num_kv_heads: usize) -> Self {
        let bs = self.block_size();
        let cols = self.cols();
        let flat: Vec<f32> = (0..bs)
            .flat_map(|r| self.read_row(r))
            .collect();
        let num_vals = flat.len();
        let mut max_abs = vec![0.0f32; num_kv_heads];
        let head_dim = if num_kv_heads > 0 { cols / num_kv_heads } else { 1 };
        for (i, &v) in flat.iter().enumerate() {
            let h = i / head_dim;
            if h < num_kv_heads && v.abs() > max_abs[h] {
                max_abs[h] = v.abs();
            }
        }
        let scales: Vec<f32> = max_abs.iter().map(|&m| if m > 0.0 { m / 7.0 } else { 1.0 }).collect();
        let packed_len = (num_vals + 1) / 2;
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in flat.iter().enumerate() {
            let h = i / head_dim;
            let scale = if h < num_kv_heads { scales[h] } else { 1.0 };
            let q = ((v / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let byte_idx = i / 2;
            if i % 2 == 0 {
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | (q << 4);
            } else {
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | q;
            }
        }
        BlockData::Q4(packed, scales, bs, cols)
    }

    pub(crate) fn to_f32(&self) -> Self {
        match self {
            BlockData::F32(_) => self.clone_data(),
            BlockData::F16(..) | BlockData::Q4(..) => {
                let bs = self.block_size();
                let cols = self.cols();
                let data: Vec<f32> = (0..bs)
                    .flat_map(|r| self.read_row(r))
                    .collect();
                BlockData::F32(Array2::from_shape_vec((bs, cols), data).expect("read_row returns cols elements per block row, total == bs*cols"))
            }
        }
    }

    fn elem_size(&self) -> usize {
        match self {
            BlockData::F32(_) => 4,
            BlockData::F16(..) => 2,
            BlockData::Q4(..) => 1,
        }
    }

    fn copy_row_to(&self, src_row: usize, dst: &mut Self, dst_row: usize) {
        let row = self.read_row(src_row);
        dst.write_row(dst_row, &row);
    }
}

// ─── Physical Block ──────────────────────────────────────────────────────────

/// Per-sequence metadata untuk eviction dan memory tiering.
pub(crate) struct SeqAccess {
    /// Waktu terakhir sequence ini diakses (read atau append)
    pub(crate) last_access: Instant,
    /// Hitungan total akses (untuk LFU)
    pub(crate) access_count: u64,
    /// Waktu sequence dibuat
    pub(crate) created_at: Instant,
    /// Memory tier saat ini
    pub(crate) tier: MemoryTier,
    /// Waktu terakhir tier berubah (untuk demotion delay tracking)
    pub(crate) tier_changed_at: Instant,
}

impl SeqAccess {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            last_access: now,
            access_count: 0,
            created_at: now,
            tier: MemoryTier::Hot,
            tier_changed_at: now,
        }
    }

    pub(crate) fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    pub(crate) fn idle_secs(&self) -> f64 {
        Instant::now().duration_since(self.last_access).as_secs_f64()
    }

    pub(crate) fn since_tier_change_secs(&self) -> f64 {
        Instant::now().duration_since(self.tier_changed_at).as_secs_f64()
    }

    pub(crate) fn promote(&mut self) {
        let old = self.tier;
        self.tier = match self.tier {
            MemoryTier::Cold => MemoryTier::Warm,
            MemoryTier::Warm => MemoryTier::Hot,
            MemoryTier::Hot => MemoryTier::Hot,
        };
        if self.tier != old {
            self.tier_changed_at = Instant::now();
        }
    }
}

/// A physical block holds KV data for one layer.
/// Shape: `(block_size, num_kv_heads * head_dim)` per K dan V.
pub(crate) struct PhysicalBlock {
    pub(crate) k: BlockData,
    pub(crate) v: BlockData,
    pub(crate) filled: usize,
    pub(crate) ref_count: usize,
}

impl PhysicalBlock {
    pub(crate) fn new(config: &PagedCacheConfig) -> Self {
        let cols = config.num_kv_heads * config.head_dim;
        Self {
            k: if config.q4_storage {
                BlockData::zeroes_quantized(config.block_size, cols, config.num_kv_heads)
            } else {
                BlockData::zeroes(config.f16_storage, config.block_size, cols)
            },
            v: if config.q4_storage {
                BlockData::zeroes_quantized(config.block_size, cols, config.num_kv_heads)
            } else {
                BlockData::zeroes(config.f16_storage, config.block_size, cols)
            },
            filled: 0,
            ref_count: 1,
        }
    }

    pub(crate) fn is_free(&self) -> bool {
        self.ref_count == 0
    }

    pub(crate) fn deep_copy(&self) -> Self {
        Self {
            k: self.k.clone_data(),
            v: self.v.clone_data(),
            filled: self.filled,
            ref_count: 1,
        }
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.k.memory_bytes() + self.v.memory_bytes()
    }
}

// ─── Sequence Block Table ────────────────────────────────────────────────────

/// Mapping dari logical block position ke physical block index per layer.
#[derive(Clone, Debug)]
pub struct BlockTable {
    /// Per layer: logical block index → physical block index
    pub(crate) layers: Vec<Vec<Option<usize>>>,
    /// Total tokens in this sequence
    pub(crate) num_tokens: usize,
    /// Block size (for computing logical index)
    block_size: usize,
}

impl BlockTable {
    pub(crate) fn new(num_layers: usize, block_size: usize) -> Self {
        Self {
            layers: vec![Vec::new(); num_layers],
            num_tokens: 0,
            block_size,
        }
    }

    /// Get physical block index for a given layer and token position.
    pub fn get_block(&self, layer: usize, token_pos: usize) -> Option<usize> {
        let logical = token_pos / self.block_size;
        self.layers.get(layer)?.get(logical).copied()?
    }

    /// Number of logical blocks allocated per layer
    pub fn num_blocks(&self, layer: usize) -> usize {
        self.layers.get(layer).map(|v| v.len()).unwrap_or(0)
    }

    /// Check if a specific block index is fully occupied
    pub fn is_block_full(&self, _layer: usize, block_idx: usize) -> bool {
        let start = block_idx * self.block_size;
        let end = start + self.block_size;
        self.num_tokens >= end
    }

    /// Remap physical block indices in a specific layer for this sequence.
    /// Used after block index compaction (defrag) on that layer.
    pub(crate) fn remap_layer(&mut self, layer: usize, remap: &[usize]) {
        if let Some(entries) = self.layers.get_mut(layer) {
            for entry in entries.iter_mut() {
                if let Some(phys) = entry {
                    if *phys < remap.len() {
                        *phys = remap[*phys];
                    }
                }
            }
        }
    }

    fn num_logical_blocks_needed(&self) -> usize {
        if self.num_tokens == 0 {
            0
        } else {
            (self.num_tokens + self.block_size - 1) / self.block_size
        }
    }
}

// ─── Paged KV Cache ──────────────────────────────────────────────────────────
