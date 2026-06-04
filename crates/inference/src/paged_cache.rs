//! PagedAttention KV Cache — block-based key-value cache manager.
//!
//! Supports conversion to flat cache format via [`to_flat_cache`] for compatibility
//! with the standard forward pass, and direct paged forward via `forward_paged`.
//!
//! Mengelola memory KV cache dalam blok-blok fixed-size untuk:
//! - Menghindari fragmentasi memori (vs growable Vec per sequence)
//! - Pre-allocation contiguous memory
//! - Potensi sharing antar sequence (copy-on-write)
//! - Eviction terprediksi (per-block, bukan per-entry)

use ndarray::{s, Array1, Array2};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use nexora_transformer::KVCacheEntry;
use tracing::{warn, info, debug};
use crate::cold_storage::{ColdStorage, ColdStorageConfig};

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::GpuContext;
#[cfg(feature = "gpu")]
use nexora_autograd::gpu_kv_cache::GpuPageTable;

// ─── Memory Tiering ──────────────────────────────────────────────────────────

/// Memory tier untuk sequence — menentukan presisi penyimpanan K/V cache.
/// Hot = f32 (full precision, akses cepat)
/// Warm = f16 (half precision, 2x hemat)
/// Cold = q4 (4-bit quantized, 8x hemat vs f32)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    Hot,
    Warm,
    Cold,
}

impl MemoryTier {
    pub fn label(&self) -> &'static str {
        match self {
            MemoryTier::Hot => "hot",
            MemoryTier::Warm => "warm",
            MemoryTier::Cold => "cold",
        }
    }
}

// ─── Constants ───────────────────────────────────────────────────────────────

/// Default number of tokens per block.
///
/// 16 is chosen as a reasonable heuristic:
///   - Too small (e.g. 4)  → high overhead from per-block metadata & fragmentation.
///   - Too large (e.g. 64) → more internal fragmentation and wasted memory for
///     short sequences, plus larger GPU memory allocation per block.
///   - 16 balances per-block metadata overhead (hash tables, page tables) against
///     internal fragmentation for typical sequence lengths (512–4096 tokens).
///
/// For optimal performance, compute via [`PagedCacheConfig::suggest_block_size`]
/// based on model dimensions:
///   block_size ≈ head_dim × num_kv_heads / gcd(head_dim, 16)   clamped to [8, 64]
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Default max number of blocks to allocate (reduced from 65536 — grow as needed).
pub const DEFAULT_MAX_BLOCKS: usize = 8192;

/// Default soft memory limit for KV cache in bytes (4 GB).
pub const DEFAULT_MAX_CACHE_MEMORY_BYTES: usize = 4_294_967_296;

/// Default watermark ratio — evict when blocks exceed this fraction of max_blocks.
pub const DEFAULT_EVICTION_WATERMARK: f64 = 0.85;

/// Default idle time before Hot → Warm demotion (seconds).
pub const DEFAULT_HOT_TO_WARM_SECS: f64 = 30.0;

/// Default idle time before Warm → Cold demotion (seconds).
pub const DEFAULT_WARM_TO_COLD_SECS: f64 = 120.0;

/// Default interval for background tier sweep (seconds).
pub const DEFAULT_TIER_SWEEP_INTERVAL_SECS: f64 = 15.0;

// ─── Global Singleton ─────────────────────────────────────────────────────────

/// Global paged KV cache singleton accessible by the inference engine.
/// Initialized via [`init_global_paged_cache`] or `PagedKVCache::new_for_engine`.
pub static GLOBAL_PAGED_CACHE: OnceLock<Mutex<PagedKVCache>> = OnceLock::new();

// ─── Config ──────────────────────────────────────────────────────────────────

/// Eviction policy untuk paged KV cache.
#[derive(Clone, Debug, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used — hapus sequence yang paling lama tidak diakses
    LRU,
    /// Least Frequently Used — hapus sequence dengan akses paling sedikit
    LFU,
    /// Time-To-Live — hapus sequence yang melebihi umur tertentu
    TTL { max_age_secs: f64 },
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

/// Configuration for the paged KV cache
#[derive(Clone, Debug)]
pub struct PagedCacheConfig {
    /// Number of KV tokens per physical block
    pub block_size: usize,
    /// Maximum number of physical blocks (hard cap)
    pub max_blocks: usize,
    /// Soft memory limit in bytes (0 = gunakan max_blocks sebagai limit)
    pub max_memory_bytes: usize,
    /// Eviction policy saat cache penuh
    pub eviction_policy: EvictionPolicy,
    /// Watermark ratio [0..1] — mulai evict saat used_blocks > max_blocks * ratio
    /// Default 0.85: evict ketika 85% kapasitas terpakai
    pub eviction_watermark_ratio: f64,
    /// Minimum sequence age sebelum bisa di-evict (detik)
    pub eviction_min_age_secs: f64,
    /// Jumlah sequence maksimal yang di-evict per siklus
    pub eviction_batch_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of KV heads (after GQA reduction)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Max tokens per sequence (prevents OOM)
    pub max_seq_len: usize,
    /// Store K/V in f16 (half-precision) — 2× memory reduction
    pub f16_storage: bool,
    /// Store K/V in 4-bit quantized — 8× memory reduction (vs f32), 4× (vs f16)
    /// Uses per-head symmetric quantization, wired via BlockData::Q4.
    pub q4_storage: bool,
    // ── Memory Tiering ──────────────────────────────────────────────────────
    /// Enable automatic memory tiering (Hot→Warm→Cold demotion).
    pub enable_memory_tiering: bool,
    /// Idle time before Hot → Warm demotion (seconds).
    pub hot_to_warm_secs: f64,
    /// Idle time before Warm → Cold demotion (seconds).
    pub warm_to_cold_secs: f64,
    /// Background tier sweep interval (seconds). 0 = no background sweep.
    pub tier_sweep_interval_secs: f64,
    /// Compress idle sequences to disk via ColdStorage instead of evicting.
    pub enable_cold_disk_offload: bool,
}

impl Default for PagedCacheConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            max_blocks: DEFAULT_MAX_BLOCKS,
            max_memory_bytes: DEFAULT_MAX_CACHE_MEMORY_BYTES,
            eviction_policy: EvictionPolicy::default(),
            eviction_watermark_ratio: DEFAULT_EVICTION_WATERMARK,
            eviction_min_age_secs: 1.0,
            eviction_batch_size: 4,
            num_layers: 32,
            num_kv_heads: 1,
            head_dim: 128,
            max_seq_len: 4096,
            f16_storage: true,
            q4_storage: true,
            enable_memory_tiering: true,
            hot_to_warm_secs: DEFAULT_HOT_TO_WARM_SECS,
            warm_to_cold_secs: DEFAULT_WARM_TO_COLD_SECS,
            tier_sweep_interval_secs: DEFAULT_TIER_SWEEP_INTERVAL_SECS,
            enable_cold_disk_offload: true,
        }
    }
}

fn elem_size_bytes(config: &PagedCacheConfig) -> usize {
    if config.q4_storage {
        1 // 4-bit → 0.5 byte per value, dibulatkan ke atas
    } else if config.f16_storage {
        2
    } else {
        4
    }
}

impl PagedCacheConfig {
    /// Total memory in bytes for all blocks at max_blocks
    pub fn total_memory_bytes(&self) -> usize {
        let elem = if self.q4_storage {
            (self.block_size * self.num_kv_heads * self.head_dim + 1) / 2 // 4-bit packed
                + self.num_kv_heads * 4 // scales overhead
                + 8 // block_size + cols metadata (2*usize)
        } else if self.f16_storage {
            2
        } else {
            4
        };
        let per_block = self.block_size * self.num_kv_heads * self.head_dim * elem * 2;
        self.max_blocks * per_block * self.num_layers
    }

    /// Memory per block in bytes
    pub fn block_memory_bytes(&self) -> usize {
        let elem = if self.q4_storage {
            (self.block_size * self.num_kv_heads * self.head_dim + 1) / 2
                + self.num_kv_heads * 4
                + 8
        } else if self.f16_storage {
            2
        } else {
            4
        };
        self.block_size * self.num_kv_heads * self.head_dim * elem * 2
    }

    /// Effective max blocks after applying both block cap and memory cap.
    pub fn effective_max_blocks(&self) -> usize {
        let by_block = self.max_blocks;
        if self.max_memory_bytes == 0 {
            return by_block;
        }
        let per_block = self.block_memory_bytes();
        if per_block == 0 {
            return by_block;
        }
        let by_memory = self.max_memory_bytes / per_block.max(1);
        by_block.min(by_memory)
    }

    /// Watermark threshold in blocks — beyond this, eviction kicks in.
    pub fn eviction_threshold(&self) -> usize {
        let effective = self.effective_max_blocks();
        (effective as f64 * self.eviction_watermark_ratio) as usize
    }

    /// Max blocks needed for one sequence of max_seq_len
    pub fn blocks_per_sequence(&self) -> usize {
        (self.max_seq_len + self.block_size - 1) / self.block_size
    }

    /// Suggest a block_size tuned to model dimensions.
    ///
    /// Formula: `min(max(8, head_dim * num_kv_heads / 16), 64)`
    ///
    /// This keeps the block aligned to the key/value vector size while preventing
    /// blocks from growing too large for short sequences.
    pub fn suggest_block_size(head_dim: usize, num_kv_heads: usize) -> usize {
        let suggested = head_dim * num_kv_heads / 16;
        suggested.clamp(8, 64)
    }
}

// ─── Block Storage ───────────────────────────────────────────────────────────

/// Block storage representation — either f32 or f16 (half-precision).
enum BlockData {
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

    fn read_row(&self, row: usize) -> Vec<f32> {
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
                    let q_val = nibble as i8 - 8; // signed [-8, 7]
                    let h = i / head_dim;
                    let scale = if h < num_kv_heads { scales[h] } else { 1.0 };
                    out[i] = q_val as f32 * scale;
                }
                out
            }
        }
    }

    fn write_row(&mut self, row: usize, values: &[f32]) {
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
                // Compute max_abs per head
                let mut max_abs = vec![0.0f32; num_kv_heads];
                for (i, &v) in values.iter().enumerate() {
                    let h = i / head_dim;
                    if h < num_kv_heads && v.abs() > max_abs[h] {
                        max_abs[h] = v.abs();
                    }
                }
                // Update scales
                for h in 0..num_kv_heads {
                    if max_abs[h] > 0.0 && max_abs[h] / 7.0 > scales[h] {
                        scales[h] = max_abs[h] / 7.0;
                    }
                }
                // Quantize and pack
                for (i, &v) in values.iter().enumerate() {
                    let h = i / head_dim;
                    let scale = if h < num_kv_heads {
                        scales[h]
                    } else {
                        1.0
                    };
                    let q = ((v / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8; // signed [-8,7] → u8 [0,15]
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

    fn slice_rows(&self, row_start: usize, count: usize) -> Vec<f32> {
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

    fn copy_row_to(&self, src_row: usize, dst: &mut Self, dst_row: usize) {
        let row = self.read_row(src_row);
        dst.write_row(dst_row, &row);
    }

    fn clone_data(&self) -> Self {
        match self {
            BlockData::F32(a) => BlockData::F32(a.clone()),
            BlockData::F16(d, bs, c) => BlockData::F16(d.clone(), *bs, *c),
            BlockData::Q4(d, s, bs, c) => BlockData::Q4(d.clone(), s.clone(), *bs, *c),
        }
    }

    fn elem_size(&self) -> usize {
        match self {
            BlockData::F32(_) => 4,
            BlockData::F16(..) => 2,
            BlockData::Q4(..) => 1, // 4-bit → 2 values per byte → 0.5 bytes per value
        }
    }

    /// Convert ke f16 (dari format apapun). No-op jika sudah f16.
    fn to_f16(&self) -> Self {
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

    /// Convert ke q4 (dari format apapun). No-op jika sudah q4.
    fn to_q4(&self, num_kv_heads: usize) -> Self {
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

    /// Convert ke f32 (dari format apapun). No-op jika sudah f32.
    fn to_f32(&self) -> Self {
        match self {
            BlockData::F32(_) => self.clone_data(),
            BlockData::F16(..) | BlockData::Q4(..) => {
                let bs = self.block_size();
                let cols = self.cols();
                let data: Vec<f32> = (0..bs)
                    .flat_map(|r| self.read_row(r))
                    .collect();
                BlockData::F32(Array2::from_shape_vec((bs, cols), data).unwrap())
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            BlockData::F32(a) => a.len() * 4,
            BlockData::F16(d, ..) => d.len() * 2,
            BlockData::Q4(d, s, ..) => d.len() + s.len() * 4,
        }
    }
}

// ─── Physical Block ──────────────────────────────────────────────────────────

/// Per-sequence metadata untuk eviction dan memory tiering.
struct SeqAccess {
    /// Waktu terakhir sequence ini diakses (read atau append)
    last_access: Instant,
    /// Hitungan total akses (untuk LFU)
    access_count: u64,
    /// Waktu sequence dibuat
    created_at: Instant,
    /// Memory tier saat ini
    tier: MemoryTier,
    /// Waktu terakhir tier berubah (untuk demotion delay tracking)
    tier_changed_at: Instant,
}

impl SeqAccess {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            last_access: now,
            access_count: 0,
            created_at: now,
            tier: MemoryTier::Hot,
            tier_changed_at: now,
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    /// Hitung idle time sejak akses terakhir.
    fn idle_secs(&self) -> f64 {
        Instant::now().duration_since(self.last_access).as_secs_f64()
    }

    /// Waktu sejak tier terakhir berubah.
    fn since_tier_change_secs(&self) -> f64 {
        Instant::now().duration_since(self.tier_changed_at).as_secs_f64()
    }

    /// Naikkan tier (Cold → Warm → Hot) saat diakses kembali.
    fn promote(&mut self) {
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
struct PhysicalBlock {
    k: BlockData,
    v: BlockData,
    filled: usize,
    ref_count: usize,
}

impl PhysicalBlock {
    fn new(config: &PagedCacheConfig) -> Self {
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

    fn is_free(&self) -> bool {
        self.ref_count == 0
    }

    fn deep_copy(&self) -> Self {
        Self {
            k: self.k.clone_data(),
            v: self.v.clone_data(),
            filled: self.filled,
            ref_count: 1,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.k.memory_bytes() + self.v.memory_bytes()
    }
}

// ─── Sequence Block Table ────────────────────────────────────────────────────

/// Mapping dari logical block position ke physical block index per layer.
#[derive(Clone, Debug)]
pub struct BlockTable {
    /// Per layer: logical block index → physical block index
    layers: Vec<Vec<Option<usize>>>,
    /// Total tokens in this sequence
    num_tokens: usize,
    /// Block size (for computing logical index)
    block_size: usize,
}

impl BlockTable {
    fn new(num_layers: usize, block_size: usize) -> Self {
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
    pub fn is_block_full(&self, layer: usize, block_idx: usize) -> bool {
        let start = block_idx * self.block_size;
        let end = start + self.block_size;
        self.num_tokens >= end
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

/// Block-based KV cache manager dengan LRU eviction + dynamic allocation.
///
/// Alokasi memori dalam fixed-size blocks, bukan growable Vec per sequence.
/// Setiap sequence punya BlockTable yang memetakan logical → physical blocks.
/// Saat mendekati batas memori, sequence yang tidak aktif akan di-evict.
pub struct PagedKVCache {
    config: PagedCacheConfig,
    /// Per-layer physical blocks
    blocks: Vec<Vec<PhysicalBlock>>,
    /// Indices of free (unused) physical blocks per layer
    free_lists: Vec<Vec<usize>>,
    /// Block table per sequence ID
    sequences: HashMap<u64, BlockTable>,
    /// Per-sequence access tracking untuk eviction
    seq_access: HashMap<u64, SeqAccess>,
    /// Total blocks evict (lifetime)
    pub num_evicted: usize,
    /// Stats
    pub num_allocated: usize,
    pub num_freed: usize,
    pub max_used: usize,
    /// GPU page table bridge (enabled with `gpu` feature)
    #[cfg(feature = "gpu")]
    pub gpu_page_table: Option<GpuPageTable>,
    /// Cold storage for disk offload (optional)
    cold_storage: Option<ColdStorage>,
}

impl PagedKVCache {
    /// Access the cache configuration (dimensions, block size, etc.).
    pub fn config(&self) -> &PagedCacheConfig {
        &self.config
    }

    /// Create a new paged KV cache with the given config
    pub fn new(config: PagedCacheConfig) -> Self {
        let num_layers = config.num_layers;
        let effective_max = config.effective_max_blocks();

        let blocks: Vec<Vec<PhysicalBlock>> = (0..num_layers)
            .map(|_| Vec::with_capacity(effective_max.min(4096)))
            .collect();
        let free_lists: Vec<Vec<usize>> = (0..num_layers)
            .map(|_| Vec::with_capacity(effective_max.min(1024)))
            .collect();

        info!(
            "PagedKVCache: {} layers, block_size={}, max_blocks={} (effective={}), f16={}, max_mem={}",
            num_layers,
            config.block_size,
            config.max_blocks,
            effective_max,
            config.f16_storage,
            config.max_memory_bytes,
        );

        let cold_storage = if config.enable_cold_disk_offload {
            Some(ColdStorage::new(ColdStorageConfig::default()))
        } else {
            None
        };

        Self {
            config,
            blocks,
            free_lists,
            sequences: HashMap::new(),
            seq_access: HashMap::new(),
            num_evicted: 0,
            num_allocated: 0,
            num_freed: 0,
            max_used: 0,
            #[cfg(feature = "gpu")]
            gpu_page_table: None,
            cold_storage,
        }
    }

    /// Create a paged cache pre-configured with engine params.
    ///
    /// Shorthand for creating a `PagedCacheConfig` with the given
    /// dimensions and constructing a `PagedKVCache` from it.
    pub fn new_for_engine(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let config = PagedCacheConfig {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Initialize the GPU page table bridge for this cache.
    ///
    /// Creates a [`GpuPageTable`] with dimensions matching this cache's config
    /// and stores it for GPU-side paged attention operations.
    #[cfg(feature = "gpu")]
    pub fn init_gpu_page_table(&mut self) -> Result<(), nexora_autograd::gpu::GpuError> {
        let ctx = GpuContext::global()?;
        let page_table = GpuPageTable::new(
            ctx,
            self.config.max_blocks,
            self.config.block_size,
            self.config.num_kv_heads,
            self.config.head_dim,
        )?;
        self.gpu_page_table = Some(page_table);
        Ok(())
    }

    /// Return a reference to the GPU page table, if initialized.
    #[cfg(feature = "gpu")]
    pub fn gpu_page_table(&self) -> Option<&GpuPageTable> {
        self.gpu_page_table.as_ref()
    }

    /// Return a mutable reference to the GPU page table, if initialized.
    #[cfg(feature = "gpu")]
    pub fn gpu_page_table_mut(&mut self) -> Option<&mut GpuPageTable> {
        self.gpu_page_table.as_mut()
    }

    /// Register a new sequence with access tracking
    pub fn register_sequence(&mut self, seq_id: u64) {
        let table = BlockTable::new(self.config.num_layers, self.config.block_size);
        self.sequences.insert(seq_id, table);
        self.seq_access.insert(seq_id, SeqAccess::new());
    }

    /// Remove a sequence and free all its blocks
    pub fn remove_sequence(&mut self, seq_id: u64) {
        self.seq_access.remove(&seq_id);
        if let Some(table) = self.sequences.remove(&seq_id) {
            for layer in 0..self.config.num_layers {
                for block_idx in 0..table.num_blocks(layer) {
                    if let Some(phys) = table.get_block(layer, block_idx * self.config.block_size) {
                        self.free_block(layer, phys);
                    }
                }
            }
        }
    }

    /// Mark a sequence as recently accessed (update LRU timestamp + promote tier).
    pub fn touch_sequence(&mut self, seq_id: u64) {
        if let Some(access) = self.seq_access.get_mut(&seq_id) {
            let old_tier = access.tier;
            access.touch();
            if self.config.enable_memory_tiering && old_tier != MemoryTier::Hot {
                access.promote();
                if access.tier != old_tier {
                    self.promote_sequence_blocks(seq_id, old_tier);
                }
            }
        }
    }

    /// Convert semua blocks milik sequence ke target tier format.
    fn convert_sequence_blocks(
        &mut self,
        seq_id: u64,
        target: MemoryTier,
    ) {
        let Some(table) = self.sequences.get(&seq_id) else {
            return;
        };
        let num_layers = table.layers.len();
        let num_kv_heads = self.config.num_kv_heads;

        for layer in 0..num_layers {
            for logical in 0..table.layers[layer].len() {
                let Some(phys) = table.layers[layer][logical] else {
                    continue;
                };
                if phys >= self.blocks[layer].len() {
                    continue;
                }
                let block = &self.blocks[layer][phys];
                let new_k = match target {
                    MemoryTier::Hot => block.k.to_f32(),
                    MemoryTier::Warm => block.k.to_f16(),
                    MemoryTier::Cold => block.k.to_q4(num_kv_heads),
                };
                let new_v = match target {
                    MemoryTier::Hot => block.v.to_f32(),
                    MemoryTier::Warm => block.v.to_f16(),
                    MemoryTier::Cold => block.v.to_q4(num_kv_heads),
                };
                let block = &mut self.blocks[layer][phys];
                block.k = new_k;
                block.v = new_v;
            }
        }
    }

    /// Promote sequence blocks dari tier lama ke tier baru (lebih tinggi presisi).
    fn promote_sequence_blocks(&mut self, seq_id: u64, from: MemoryTier) {
        let target = match from {
            MemoryTier::Cold => MemoryTier::Warm,
            MemoryTier::Warm => MemoryTier::Hot,
            MemoryTier::Hot => return,
        };
        info!("Promoting seq {} from {} to {}", seq_id, from.label(), target.label());
        self.convert_sequence_blocks(seq_id, target);
    }

    /// Demote sequence blocks ke tier lebih rendah (lebih hemat memori).
    /// Dipanggil saat sequence idle.
    fn demote_sequence(&mut self, seq_id: u64) -> bool {
        // Clone data dari access sebelum mutable borrow
        let (current, idle) = match self.seq_access.get(&seq_id) {
            Some(acc) => (acc.tier, acc.idle_secs()),
            None => return false,
        };
        let target = match current {
            MemoryTier::Hot => MemoryTier::Warm,
            MemoryTier::Warm => MemoryTier::Cold,
            MemoryTier::Cold => return false,
        };
        info!(
            "Demoting seq {} from {} to {} (idle {:.1}s)",
            seq_id,
            current.label(),
            target.label(),
            idle,
        );
        // Drop immutable borrow sebelum mutable call
        self.convert_sequence_blocks(seq_id, target);
        if let Some(acc) = self.seq_access.get_mut(&seq_id) {
            acc.tier = target;
            acc.tier_changed_at = Instant::now();
        }
        true
    }

    /// Demote satu step untuk sequence paling idle yang eligible.
    /// Returns true jika ada yang di-demote.
    pub fn demote_most_idle(&mut self) -> bool {
        if !self.config.enable_memory_tiering {
            return false;
        }
        let now = Instant::now();
        let mut candidates: Vec<(u64, &SeqAccess)> = self.seq_access.iter()
            .filter(|(_, acc)| {
                acc.tier != MemoryTier::Cold
                    && now.duration_since(acc.tier_changed_at).as_secs_f64() > 1.0
            })
            .map(|(k, v)| (*k, v))
            .collect();
        if candidates.is_empty() {
            return false;
        }
        // Sort by: Cold candidates first (already cold → skip), then by idle time
        candidates.sort_by(|a, b| {
            let a_idle = a.1.idle_secs();
            let b_idle = b.1.idle_secs();
            b_idle.partial_cmp(&a_idle).unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = candidates[0].0;
        // Jangan demote jika masih di bawah threshold
        let idle = candidates[0].1.idle_secs();
        let since_change = candidates[0].1.since_tier_change_secs();
        match candidates[0].1.tier {
            MemoryTier::Hot if idle < self.config.hot_to_warm_secs || since_change < 5.0 => return false,
            MemoryTier::Warm if idle < self.config.warm_to_cold_secs || since_change < 5.0 => return false,
            _ => {}
        }
        self.demote_sequence(best)
    }

    /// Demote semua sequence idle sesuai konfigurasi tier.
    /// Returns jumlah sequence yang di-demote.
    pub fn demote_idle_sequences(&mut self) -> usize {
        if !self.config.enable_memory_tiering {
            return 0;
        }
        let now = Instant::now();
        let seq_ids: Vec<u64> = self.seq_access.iter()
            .filter(|(_, acc)| {
                let idle = now.duration_since(acc.last_access).as_secs_f64();
                let since_tier = now.duration_since(acc.tier_changed_at).as_secs_f64();
                match acc.tier {
                    MemoryTier::Hot => idle >= self.config.hot_to_warm_secs && since_tier >= 5.0,
                    MemoryTier::Warm => idle >= self.config.warm_to_cold_secs && since_tier >= 5.0,
                    MemoryTier::Cold => false,
                }
            })
            .map(|(id, _)| *id)
            .collect();
        let mut count = 0;
        for seq_id in seq_ids {
            if self.demote_sequence(seq_id) {
                count += 1;
            }
        }
        count
    }

    /// Offload Cold sequence Cold → disk (instead of evicting).
    /// Returns 1 jika berhasil, 0 jika tidak ada kandidat.
    pub fn offload_coldest_to_disk(&mut self) -> usize {
        if !self.config.enable_cold_disk_offload {
            return 0;
        }
        let _ = Instant::now();
        // Cari sequence Cold dengan idle terlama
        let mut candidates: Vec<(u64, Instant)> = self.seq_access.iter()
            .filter(|(_, acc)| acc.tier == MemoryTier::Cold)
            .map(|(k, v)| (*k, v.last_access))
            .collect();
        if candidates.is_empty() {
            return 0;
        }
        candidates.sort_by(|a, b| a.1.cmp(&b.1));
        let (seq_id, _) = candidates[0];

        // Convert ke flat KVCacheEntry (compressed)
        let entries = match self.to_flat_cache(seq_id) {
            Some(e) => e,
            None => return 0,
        };
        let token_ids: Vec<u32> = (0..entries.first().map(|e| e.seq_len()).unwrap_or(0))
            .map(|i| i as u32)
            .collect();

        // Save to cold storage (separate scope to avoid borrow conflict)
        let result = self.cold_storage.as_mut().map(|cold| cold.save(&token_ids, &entries));
        match result {
            Some(Ok(())) => {
                info!("Offloaded seq {} to cold disk storage ({} entries)", seq_id, entries.len());
                self.remove_sequence(seq_id);
                1
            }
            Some(Err(e)) => {
                warn!("Failed to offload seq {} to cold disk: {}", seq_id, e);
                0
            }
            None => 0,
        }
    }

    /// Number of allocated blocks across all layers (used, not free).
    pub fn used_block_count(&self) -> usize {
        let total: usize = self.blocks.iter().map(|b| b.len()).sum();
        let free: usize = self.free_lists.iter().map(|f| f.len()).sum();
        total.saturating_sub(free)
    }

    /// Check if a sequence is registered
    pub fn has_sequence(&self, seq_id: u64) -> bool {
        self.sequences.contains_key(&seq_id)
    }

    /// Cek apakah eviction perlu dilakukan berdasarkan used vs threshold.
    fn should_evict(&self) -> bool {
        let used = self.used_block_count();
        let threshold = self.config.eviction_threshold();
        used >= threshold
    }

    /// Evict sequence yang paling jarang diakses.
    ///
    /// Strategi:
    /// - LRU (default): cari sequence dengan `last_access` paling lama
    /// - LFU: cari sequence dengan `access_count` paling rendah
    ///
    /// Returns jumlah sequence yang di-evict.
    pub fn evict_lru(&mut self, count: usize) -> usize {
        if self.seq_access.len() <= 1 || count == 0 {
            return 0;
        }

        let min_age = self.config.eviction_min_age_secs;
        let now = Instant::now();

        // Kumpulkan kandidat eviction: urutkan berdasarkan policy
        let mut candidates: Vec<(u64, &SeqAccess)> = self
            .seq_access
            .iter()
            .filter(|(_, acc)| {
                now.duration_since(acc.created_at).as_secs_f64() >= min_age
            })
            .map(|(k, v)| (*k, v))
            .collect();

        if candidates.is_empty() {
            return 0;
        }

        match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                candidates.sort_by(|a, b| a.1.last_access.cmp(&b.1.last_access));
            }
            EvictionPolicy::LFU => {
                candidates.sort_by(|a, b| a.1.access_count.cmp(&b.1.access_count));
            }
            EvictionPolicy::TTL { max_age_secs } => {
                candidates.retain(|(_, acc)| {
                    now.duration_since(acc.last_access).as_secs_f64() >= max_age_secs
                });
                candidates.sort_by(|a, b| a.1.last_access.cmp(&b.1.last_access));
            }
        }

        let to_evict = count.min(candidates.len());
        let evicted_ids: Vec<u64> = candidates
            .into_iter()
            .take(to_evict)
            .map(|(id, _)| id)
            .collect();

        for &seq_id in &evicted_ids {
            info!(
                "PagedKVCache: evicting seq {} (LRU, {} blocks)",
                seq_id,
                self.sequences.get(&seq_id).map(|t| t.num_tokens).unwrap_or(0)
            );
            self.remove_sequence(seq_id);
            self.num_evicted += 1;
        }

        to_evict
    }

    /// Allocate a new physical block from the free list or create one.
    /// Jika mendekati batas, akan melakukan defrag + tier demotion + eviction.
    fn alloc_block(&mut self, layer: usize) -> Option<usize> {
        let effective_max = self.config.effective_max_blocks();

        // Coba free list dulu
        if let Some(free_idx) = self.free_lists[layer].pop() {
            if free_idx < self.blocks[layer].len() {
                self.blocks[layer][free_idx].filled = 0;
                self.blocks[layer][free_idx].ref_count = 1;
                self.num_allocated += 1;
                return Some(free_idx);
            }
        }

        // Cek apakah perlu eviction
        if self.should_evict() {
            // Step 1: Defrag — mungkin ada blok yang bisa direklamasi
            let reclaimed = self.defragment();
            if reclaimed > 0 {
                if let Some(free_idx) = self.free_lists[layer].pop() {
                    self.blocks[layer][free_idx].filled = 0;
                    self.blocks[layer][free_idx].ref_count = 1;
                    self.num_allocated += 1;
                    return Some(free_idx);
                }
            }

            // Step 2: Demote Hot→Warm atau Warm→Cold untuk sequence idle
            if self.config.enable_memory_tiering {
                let demoted = self.demote_most_idle();
                if demoted {
                    // Demoting frees memory by reducing precision; blocks may still be allocated
                    // but memory pressure is reduced. Try free list again after conversion.
                    if let Some(free_idx) = self.free_lists[layer].pop() {
                        self.blocks[layer][free_idx].filled = 0;
                        self.blocks[layer][free_idx].ref_count = 1;
                        self.num_allocated += 1;
                        return Some(free_idx);
                    }
                }
                // Step 3: Demote 1 Cold sequence ke disk (instead of evict)
                if self.config.enable_cold_disk_offload {
                    let offloaded = self.offload_coldest_to_disk();
                    if offloaded > 0 {
                        if let Some(free_idx) = self.free_lists[layer].pop() {
                            self.blocks[layer][free_idx].filled = 0;
                            self.blocks[layer][free_idx].ref_count = 1;
                            self.num_allocated += 1;
                            return Some(free_idx);
                        }
                    }
                }
            }

            // Step 4: Traditional LRU eviction (last resort)
            let evicted = self.evict_lru(self.config.eviction_batch_size);
            if evicted > 0 {
                if let Some(free_idx) = self.free_lists[layer].pop() {
                    self.blocks[layer][free_idx].filled = 0;
                    self.blocks[layer][free_idx].ref_count = 1;
                    self.num_allocated += 1;
                    return Some(free_idx);
                }
            }
        }

        // Grow jika masih di bawah kapasitas
        if self.blocks[layer].len() < effective_max {
            let idx = self.blocks[layer].len();
            self.blocks[layer].push(PhysicalBlock::new(&self.config));
            self.num_allocated += 1;
            return Some(idx);
        }

        // Fallback: coba free list satu kali lagi
        if let Some(free_idx) = self.free_lists[layer].pop() {
            self.blocks[layer][free_idx].filled = 0;
            self.blocks[layer][free_idx].ref_count = 1;
            self.num_allocated += 1;
            return Some(free_idx);
        }

        warn!(
            "PagedKVCache: max capacity reached ({} blocks), cannot allocate",
            effective_max
        );
        None
    }

    /// Free a physical block (decrease refcount, reclaim if zero)
    fn free_block(&mut self, layer: usize, phys_idx: usize) {
        if phys_idx >= self.blocks[layer].len() {
            return;
        }
        let block = &mut self.blocks[layer][phys_idx];
        if block.ref_count > 0 {
            block.ref_count -= 1;
        }
        if block.is_free() {
            block.filled = 0;
            self.free_lists[layer].push(phys_idx);
            self.num_freed += 1;
        }
    }

    /// Get the next physical block for appending at a given token position.
    /// Allocates a new block if needed. If an existing shared block (ref_count > 1)
    /// is found, performs copy-on-write: deep copies the block, decrements ref_count,
    /// and returns the private copy.
    fn get_or_alloc_block(&mut self, seq_id: u64, layer: usize, token_pos: usize) -> Option<usize> {
        let logical = token_pos / self.config.block_size;

        // Step 1: check existing block with COW (separate borrow on self.sequences)
        {
            let table = self.sequences.get_mut(&seq_id)?;
            while table.layers[layer].len() <= logical {
                table.layers[layer].push(None);
            }
            if let Some(phys) = table.layers[layer][logical] {
                if phys < self.blocks[layer].len() && self.blocks[layer][phys].ref_count > 1 {
                    let copy = self.blocks[layer][phys].deep_copy();
                    self.blocks[layer][phys].ref_count -= 1;
                    let new_idx = self.blocks[layer].len();
                    self.blocks[layer].push(copy);
                    self.num_allocated += 1;
                    table.layers[layer][logical] = Some(new_idx);
                    return Some(new_idx);
                }
                return Some(phys);
            }
        }

        // Step 2: allocate new block (unique borrow)
        let phys = match self.alloc_block(layer) {
            Some(p) => p,
            None => return None,
        };

        // Step 3: update block table (separate borrow again)
        if let Some(table) = self.sequences.get_mut(&seq_id) {
            if logical < table.layers[layer].len() {
                table.layers[layer][logical] = Some(phys);
            }
            Some(phys)
        } else {
            self.free_block(layer, phys);
            None
        }
    }

    /// Append KV data for a token at a given position.
    /// Copies slice into the correct block.
    pub fn append(
        &mut self,
        seq_id: u64,
        layer: usize,
        token_pos: usize,
        k_row: &[f32],
        v_row: &[f32],
    ) {
        let phys = match self.get_or_alloc_block(seq_id, layer, token_pos) {
            Some(p) => p,
            None => return,
        };

        // Touch sequence untuk LRU tracking
        self.touch_sequence(seq_id);

        let offset = token_pos % self.config.block_size;
        let cols = self.config.num_kv_heads * self.config.head_dim;
        let k_len = k_row.len().min(cols);
        let v_len = v_row.len().min(cols);

        // Update sequence token count FIRST (no block borrow yet)
        if let Some(table) = self.sequences.get_mut(&seq_id) {
            if token_pos >= table.num_tokens {
                table.num_tokens = token_pos + 1;
            }
        }

        // Then write KV data (separate borrow for blocks)
        if phys >= self.blocks[layer].len() {
            warn!("phys {} out of bounds for layer {} in append", phys, layer);
            return;
        }
        let block = &mut self.blocks[layer][phys];
        block.k.write_row(offset, &k_row[..k_len]);
        block.v.write_row(offset, &v_row[..v_len]);
        if offset + 1 > block.filled {
            block.filled = offset + 1;
        }
    }

    /// Append complete KV projections (from model forward) into the paged cache.
    /// `k_proj` shape: `(num_kv_heads * head_dim,)` — single token's K projection.
    /// `v_proj` shape: `(num_kv_heads * head_dim,)` — single token's V projection.
    pub fn append_token(
        &mut self,
        seq_id: u64,
        layer: usize,
        token_pos: usize,
        k_proj: &Array2<f32>,
        v_proj: &Array2<f32>,
    ) {
        let k_vec: Vec<f32> = k_proj.iter().copied().collect();
        let v_vec: Vec<f32> = v_proj.iter().copied().collect();
        self.append(seq_id, layer, token_pos, &k_vec, &v_vec);
    }

    /// Read KV data for a token position.
    pub fn read(
        &self,
        seq_id: u64,
        layer: usize,
        token_pos: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let table = self.sequences.get(&seq_id)?;
        let logical = token_pos / self.config.block_size;
        let offset = token_pos % self.config.block_size;
        let phys = table.layers.get(layer)?.get(logical).copied()??;
        let block = self.blocks.get(layer)?.get(phys)?;

        if offset >= block.filled {
            return None;
        }

        let k = block.k.read_row(offset);
        let v = block.v.read_row(offset);
        Some((k, v))
    }

    /// Share prefix blocks between sequences (PrefixDAG).
    ///
    /// Copies the block table entries for the first `prefix_tokens` from `src_seq`
    /// to `dst_seq`, incrementing refcount on shared physical blocks. Subsequent
    /// appends by `dst_seq` that touch a shared block trigger copy-on-write
    /// (deep copy in `get_or_alloc_block` when `ref_count > 1`).
    ///
    /// Returns the number of physical blocks shared.
    pub fn share_prefix_in_blocks(
        &mut self,
        dst_seq: u64,
        src_seq: u64,
        prefix_tokens: usize,
    ) -> usize {
        let Some(src_table) = self.sequences.get(&src_seq) else {
            return 0;
        };
        if src_table.num_tokens < prefix_tokens {
            return 0;
        }
        let num_logical = (prefix_tokens + self.config.block_size - 1) / self.config.block_size;
        if num_logical == 0 {
            return 0;
        }

        // Process layer-by-layer: drain src_table, then borrow dst
        let src_layers_len = src_table.layers.len();
        let mut layer_data: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        for layer_idx in 0..src_layers_len {
            let count = num_logical.min(src_table.layers[layer_idx].len());
            if count == 0 {
                continue;
            }
            let phys_blocks: Vec<(usize, usize)> = src_table.layers[layer_idx][..count]
                .iter()
                .enumerate()
                .filter_map(|(logical, &phys)| phys.map(|p| (logical, p)))
                .collect();
            if !phys_blocks.is_empty() {
                layer_data.push((layer_idx, phys_blocks));
            }
        }

        // Touch dst untuk LRU tracking
        self.touch_sequence(dst_seq);
        if src_seq != dst_seq {
            self.touch_sequence(src_seq);
        }

        // Apply to dst
        let Some(dst_table) = self.sequences.get_mut(&dst_seq) else {
            return 0;
        };
        while dst_table.layers.len() < src_layers_len {
            dst_table.layers.push(Vec::new());
        }
        let mut shared = 0usize;
        for (layer_idx, blocks) in layer_data.iter() {
            for &(logical, phys) in blocks {
                while dst_table.layers[*layer_idx].len() <= logical {
                    dst_table.layers[*layer_idx].push(None);
                }
                dst_table.layers[*layer_idx][logical] = Some(phys);
                dst_table.num_tokens = dst_table.num_tokens.max(prefix_tokens);
                // Increment refcount directly (no deferred Vec needed)
                if *layer_idx < self.blocks.len() && phys < self.blocks[*layer_idx].len() {
                    self.blocks[*layer_idx][phys].ref_count += 1;
                }
                shared += 1;
            }
        }

        shared
    }

    /// Convert paged cache state to the flat `Vec<KVCacheEntry>` format
    /// expected by `CausalLM::forward()`.
    ///
    /// Returns per-layer KVCacheEntry where:
    /// - `k.shape = (num_tokens, num_kv_heads * head_dim)`
    /// - `v.shape = (num_tokens, num_kv_heads * head_dim)`
    pub fn to_flat_cache(&self, seq_id: u64) -> Option<Vec<KVCacheEntry>> {
        let table = self.sequences.get(&seq_id)?;
        let num_tokens = table.num_tokens;
        if num_tokens == 0 {
            return Some(Vec::new());
        }

        let cols = self.config.num_kv_heads * self.config.head_dim;
        let mut entries = Vec::with_capacity(self.config.num_layers);
        let total = num_tokens * cols;

        for layer in 0..self.config.num_layers {
            let mut k_flat = vec![0f32; total];
            let mut v_flat = vec![0f32; total];

            let mut pos = 0;
            while pos < num_tokens {
                let logical = pos / self.config.block_size;
                let offset = pos % self.config.block_size;
                let remaining_in_block = self.config.block_size - offset;
                let tokens_in_block = remaining_in_block.min(num_tokens - pos);

                if let Some(phys) = table.layers[layer].get(logical).copied().flatten() {
                    if let Some(block) = self.blocks[layer].get(phys) {
                        let valid = tokens_in_block.min(block.filled.saturating_sub(offset));
                        if valid > 0 {
                            let k_src = block.k.slice_rows(offset, valid);
                            let v_src = block.v.slice_rows(offset, valid);
                            let dst_start = pos * cols;
                            k_flat[dst_start..dst_start + k_src.len()].copy_from_slice(&k_src);
                            v_flat[dst_start..dst_start + v_src.len()].copy_from_slice(&v_src);
                        }
                    }
                }
                pos += tokens_in_block;
            }

            entries.push(KVCacheEntry {
                k: k_flat,
                v: v_flat,
                kv_dim: cols,
                num_kv_heads: self.config.num_kv_heads,
                head_dim: self.config.head_dim,
                k_compressed: Vec::new(),
                v_compressed: Vec::new(),
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                compressed_seq_len: 0,
            });
        }

        Some(entries)
    }

    /// Get token count for a sequence
    pub fn num_tokens(&self, seq_id: u64) -> Option<usize> {
        self.sequences.get(&seq_id).map(|t| t.num_tokens)
    }

    /// Get block table for a sequence (for debugging/inspection)
    pub fn block_table(&self, seq_id: u64) -> Option<&BlockTable> {
        self.sequences.get(&seq_id)
    }

    /// Total physical blocks allocated across all layers
    pub fn total_blocks(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
    }

    /// Free blocks count across all layers
    pub fn free_blocks(&self) -> usize {
        self.free_lists.iter().map(|f| f.len()).sum()
    }

    /// Clear all sequences and blocks
    pub fn clear(&mut self) {
        self.sequences.clear();
        self.seq_access.clear();
        self.blocks.iter_mut().for_each(|b| b.clear());
        self.free_lists.iter_mut().for_each(|f| f.clear());
    }

    /// Memory usage estimate in bytes — includes data + estimated metadata overhead
    pub fn memory_usage_bytes(&self) -> usize {
        let total = self.total_blocks();
        let freed: usize = self.free_lists.iter().map(|f| f.len()).sum();
        let used = total.saturating_sub(freed);
        if used == 0 {
            return 0;
        }
        // Sum ALL block memory — freed blocks still consume GPU memory until compaction.
        // Iterate all layers and all blocks, not just first `used` blocks (F-8 fix).
        let mut data_bytes = 0usize;
        for layer_blocks in &self.blocks {
            for block in layer_blocks {
                if !block.is_free() {
                    data_bytes += block.memory_bytes();
                }
            }
        }
        // Estimated metadata overhead per block (~56 bytes: ref_count, filled, last_access, padding)
        let block_metadata = total * 56;
        // Sequence block tables: each sequence stores (seq_len / block_size) block IDs per layer
        let block_table_overhead: usize = self.sequences.len()
            * self.blocks.len()
            * std::mem::size_of::<usize>()
            * 4; // 4× estimate for Vec capacity overhead
        // Free lists Vec capacity overhead
        let free_list_overhead: usize = self.free_lists.iter().map(|f| f.capacity() * 8).sum();
        data_bytes + block_metadata + block_table_overhead + free_list_overhead
    }
}

impl nexora_transformer::PagedCacheReader for PagedKVCache {
    fn read(&self, seq_id: u64, layer: usize, token_pos: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        PagedKVCache::read(self, seq_id, layer, token_pos)
    }

    fn num_tokens(&self, seq_id: u64) -> Option<usize> {
        self.sequences.get(&seq_id).map(|t| t.num_tokens)
    }

    fn append(
        &mut self,
        seq_id: u64,
        layer: usize,
        token_pos: usize,
        k_row: &[f32],
        v_row: &[f32],
    ) {
        PagedKVCache::append(self, seq_id, layer, token_pos, k_row, v_row);
    }
}

// ─── Statistics & Fragmentation ──────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct PagedCacheStats {
    pub num_sequences: usize,
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub memory_bytes: usize,
    pub total_tokens: usize,
    /// Fraction of allocated block slots that are empty (internal fragmentation).
    /// `0.0` means zero waste; `0.5` means half the slots are unused.
    pub internal_fragmentation_ratio: f64,
    /// Fraction of free blocks vs total allocated (external fragmentation).
    /// High values indicate many freed-but-unreclaimed blocks scattered across layers.
    pub external_fragmentation_ratio: f64,
    /// Total wasted token slots across all partially-filled blocks
    pub wasted_slots: usize,
    /// Jumlah sequence yang pernah di-evict (lifetime)
    pub sequences_evicted: usize,
    /// Kapasitas efektif (setelah mempertimbangkan max_memory_bytes)
    pub effective_max_blocks: usize,
    /// Watermark threshold untuk eviction
    pub eviction_threshold: usize,
}

impl PagedKVCache {
    /// Compute internal fragmentation: wasted slots in partially-filled blocks.
    ///
    /// Each block can hold `block_size` tokens. If a block has only `filled` tokens,
    /// the remaining `block_size - filled` slots are wasted. This sums across all
    /// allocated blocks (both in-use and free).
    fn compute_internal_fragmentation(&self) -> (f64, usize) {
        let bs = self.config.block_size;
        if bs == 0 {
            return (0.0, 0);
        }
        let (total_slots, filled_slots) = self.blocks.iter().flatten().fold(
            (0usize, 0usize),
            |(total, filled), block| {
                let cap = bs;
                (total + cap, filled + block.filled)
            },
        );
        if total_slots == 0 {
            return (0.0, 0);
        }
        let wasted = total_slots.saturating_sub(filled_slots);
        (wasted as f64 / total_slots as f64, wasted)
    }

    /// Compute external fragmentation: fraction of blocks that are free vs total.
    fn compute_external_fragmentation(&self) -> f64 {
        let total = self.total_blocks();
        if total == 0 {
            return 0.0;
        }
        let free = self.free_blocks();
        free as f64 / total as f64
    }

    /// Defragment by compacting sparse blocks within each layer.
    ///
    /// Two-phase compaction:
    ///   1. Free blocks (ref_count == 0): merge data from sparser blocks into fuller ones.
    ///      Freed blocks returned to free list. No remap needed.
    ///   2. Active blocks (ref_count > 0): merge consecutive partially-filled blocks
    ///      belonging to the same sequence. Block tables are updated to reflect the new
    ///      physical block mapping.
    ///
    /// Returns number of blocks reclaimed.
    pub fn defragment(&mut self) -> usize {
        let bs = self.config.block_size;
        if bs == 0 || self.blocks.is_empty() {
            return 0;
        }
        let mut total_reclaimed = 0usize;
        let num_layers = self.blocks.len();

        for layer in 0..num_layers {
            // ── Phase 1: Compact free blocks (ref_count == 0) ──
            total_reclaimed += self.defrag_free_blocks(layer, bs);

            // ── Phase 2: Compact active blocks (ref_count > 0) ──
            // Build reverse map for this layer to update block tables
            total_reclaimed += self.defrag_active_blocks(layer, bs);
        }

        total_reclaimed
    }

    /// Phase 1: Compact free blocks (ref_count == 0).
    /// Safe — no sequences reference these blocks, so no remap needed.
    fn defrag_free_blocks(&mut self, layer: usize, bs: usize) -> usize {
        let mut partial_indices: Vec<usize> = (0..self.blocks[layer].len())
            .filter(|&i| {
                let b = &self.blocks[layer][i];
                b.ref_count == 0 && b.filled > 0 && b.filled < bs
            })
            .collect();

        if partial_indices.len() <= 1 {
            return 0;
        }

        // Sort by filled ascending (sparsest first)
        partial_indices.sort_by(|&a, &b| {
            self.blocks[layer][a]
                .filled
                .cmp(&self.blocks[layer][b].filled)
        });

        let mut drain_targets: Vec<usize> = Vec::new();
        let mut j = 0;
        let mut reclaimed = 0usize;

        while j + 1 < partial_indices.len() {
            let src = partial_indices[j];
            let dst = partial_indices[j + 1];
            let src_filled = self.blocks[layer][src].filled;
            let dst_filled = self.blocks[layer][dst].filled;
            let dst_capacity = bs - dst_filled;
            let move_count = src_filled.min(dst_capacity);

            if move_count > 0 {
                for row in 0..move_count {
                    let k_row = self.blocks[layer][src].k.read_row(row);
                    let v_row = self.blocks[layer][src].v.read_row(row);
                    let dst_row_idx = dst_filled + row;
                    self.blocks[layer][dst].k.write_row(dst_row_idx, &k_row);
                    self.blocks[layer][dst].v.write_row(dst_row_idx, &v_row);
                }
                self.blocks[layer][dst].filled = dst_filled + move_count;
                self.blocks[layer][src].filled = src_filled - move_count;
            }

            if self.blocks[layer][src].filled == 0 {
                drain_targets.push(src);
                j += 2;
            } else {
                // Not fully drained, advance by one (keep src, skip dst)
                partial_indices[j + 1] = src; // swap: keep the still-partial src
                j += 1;
            }
        }

        // Free fully-drained blocks (no remap needed — ref_count == 0)
        for &phys in &drain_targets {
            self.blocks[layer][phys].filled = 0;
            self.blocks[layer][phys].ref_count = 0;
            self.free_lists[layer].push(phys);
            self.num_freed += 1;
            reclaimed += 1;
        }

        reclaimed
    }

    /// Phase 2: Compact active blocks (ref_count > 0).
    ///
    /// SAFETY NOTE (BF44): Active block defrag is fundamentally unsafe without
    /// per-logical-block row offset tracking. Each logical block reads from row 0
    /// of its mapped physical block. Moving data between physical blocks and
    /// remapping logical→physical entries causes silent data corruption because:
    ///   - The moved data's logical position changes (it now resides at row N of
    ///     a different physical block, NOT row 0 where the logical block reads).
    ///   - Shared blocks (ref_count > 1) affect ALL referencing sequences.
    ///
    /// A correct implementation would require per-sequence, per-logical-block
    /// row_offset tracking (where does this logical block's data start within
    /// its physical block?). Phase 1 (free block defrag) reclaims fragmented
    /// blocks that are not live, which is the primary source of reclaimable space.
    /// Memory tiering (Hot→Warm→Cold demotion) provides additional memory savings
    /// by reducing precision of idle sequences.
    fn defrag_active_blocks(&mut self, _layer: usize, _bs: usize) -> usize {
        0
    }

    pub fn stats(&self) -> PagedCacheStats {
        let total = self.total_blocks();
        let free = self.free_blocks();
        let used = total.saturating_sub(free);
        let total_tokens: usize = self.sequences.values().map(|t| t.num_tokens).sum();
        let (int_frag, wasted_slots) = self.compute_internal_fragmentation();
        let ext_frag = self.compute_external_fragmentation();

        // Update global observability atomics
        let mem_bytes = self.memory_usage_bytes();
        crate::inference_trait::KV_CACHE_TOTAL_BLOCKS.store(total as u64, std::sync::atomic::Ordering::Relaxed);
        crate::inference_trait::KV_CACHE_USED_BLOCKS.store(used as u64, std::sync::atomic::Ordering::Relaxed);
        crate::inference_trait::KV_CACHE_MEMORY_BYTES.store(mem_bytes as u64, std::sync::atomic::Ordering::Relaxed);
        crate::inference_trait::KV_CACHE_INTERNAL_FRAG.store(
            (int_frag * 1_000_000.0) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        crate::inference_trait::KV_CACHE_EXTERNAL_FRAG.store(
            (ext_frag * 1_000_000.0) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        PagedCacheStats {
            num_sequences: self.sequences.len(),
            total_blocks: total,
            free_blocks: free,
            used_blocks: used,
            memory_bytes: self.memory_usage_bytes(),
            total_tokens,
            internal_fragmentation_ratio: int_frag,
            external_fragmentation_ratio: ext_frag,
            wasted_slots,
            sequences_evicted: self.num_evicted,
            effective_max_blocks: self.config.effective_max_blocks(),
            eviction_threshold: self.config.eviction_threshold(),
        }
    }
}

// ─── Global Initializer ──────────────────────────────────────────────────────

/// Initialize [`GLOBAL_PAGED_CACHE`] with engine parameters.
///
/// Creates a `PagedKVCache` via [`PagedKVCache::new_for_engine`] and stores it
/// in the global singleton. When the `gpu` feature is enabled, also attempts
/// to initialize the GPU page table bridge.
///
/// Returns a reference to the initialized global mutex. Safe to call multiple
/// times — subsequent calls return the existing instance.
pub fn init_global_paged_cache(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> &'static Mutex<PagedKVCache> {
    GLOBAL_PAGED_CACHE.get_or_init(|| {
        let mut cache =
            PagedKVCache::new_for_engine(num_layers, num_kv_heads, head_dim, max_seq_len);
        #[cfg(feature = "gpu")]
        if let Err(e) = cache.init_gpu_page_table() {
            tracing::warn!("Failed to init GPU page table for global paged cache: {e}");
        }
        Mutex::new(cache)
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PagedCacheConfig {
        PagedCacheConfig {
            block_size: 4,
            max_blocks: 64,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 64,
            f16_storage: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_register_and_remove_sequence() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        assert!(cache.has_sequence(1));
        cache.remove_sequence(1);
        assert!(!cache.has_sequence(1));
    }

    #[test]
    fn test_append_and_read_single_block() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(42);

        let k_row = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let v_row = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7];

        cache.append(42, 0, 0, &k_row, &v_row);

        let (k_read, v_read) = cache.read(42, 0, 0).unwrap();
        assert_eq!(k_read.len(), 8);
        assert!((k_read[0] - 0.1).abs() < 1e-6);
        assert!((v_read[0] - 1.0).abs() < 1e-6);

        assert_eq!(cache.num_tokens(42), Some(1));
    }

    #[test]
    fn test_get_or_alloc_block_sequential() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(99);

        // pos=0: should allocate block 0
        let b0 = cache.get_or_alloc_block(99, 0, 0);
        assert_eq!(b0, Some(0));

        // pos=1: should reuse block 0
        let b1 = cache.get_or_alloc_block(99, 0, 1);
        assert_eq!(b1, Some(0));

        // pos=4: should allocate block 1
        let b4 = cache.get_or_alloc_block(99, 0, 4);
        assert_eq!(b4, Some(1));

        // pos=5: should reuse block 1
        let b5 = cache.get_or_alloc_block(99, 0, 5);
        assert_eq!(b5, Some(1));
    }

    #[test]
    fn test_append_multiple_blocks() {
        let config = test_config();
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(7);

        // Fill 5 tokens (= 2 blocks: first block 4 tokens, second block 1 token)
        for pos in 0..5 {
            // Each position writes k_row where k_row[0] = f32::from(pos) / 10.0
            let k_row: Vec<f32> = (0..8).map(|i| (pos * 8 + i) as f32 / 10.0).collect();
            let v_row: Vec<f32> = (0..8).map(|i| (pos * 8 + i) as f32).collect();
            cache.append(7, 0, pos, &k_row, &v_row);
        }

        assert_eq!(cache.num_tokens(7), Some(5));

        // Verify all tokens readable
        for pos in 0..5 {
            let (k, v) = cache.read(7, 0, pos).unwrap();
            let expected_k0 = (pos * 8) as f32 / 10.0;
            let expected_v0 = (pos * 8) as f32;
            assert!(
                (k[0] - expected_k0).abs() < 1e-6,
                "pos={}: expected k[0]={}, got k[0]={}",
                pos,
                expected_k0,
                k[0]
            );
            assert!(
                (v[0] - expected_v0).abs() < 1e-6,
                "pos={}: expected v[0]={}, got v[0]={}",
                pos,
                expected_v0,
                v[0]
            );
        }
    }

    #[test]
    fn test_to_flat_cache() {
        let config = test_config();
        let mut cache = PagedKVCache::new(config);
        cache.register_sequence(99);

        let k_row: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..8).map(|i| i as f32 * 10.0).collect();

        for pos in 0..3 {
            cache.append(99, 0, pos, &k_row, &v_row);
            cache.append(99, 1, pos, &k_row, &v_row);
        }

        let flat = cache.to_flat_cache(99).unwrap();
        assert_eq!(flat.len(), 2); // 2 layers
        assert_eq!(flat[0].k.len(), 3 * 8); // 3 tokens * 8 cols
        assert_eq!(flat[0].v.len(), 3 * 8);
    }

    #[test]
    fn test_remove_sequence_frees_blocks() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        cache.register_sequence(2);

        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];

        for pos in 0..8 {
            cache.append(1, 0, pos, &k_row, &v_row);
            cache.append(2, 0, pos, &k_row, &v_row);
        }

        let stats_before = cache.stats();
        assert_eq!(stats_before.num_sequences, 2);

        cache.remove_sequence(1);

        let stats_after = cache.stats();
        assert_eq!(stats_after.num_sequences, 1);
        assert_eq!(stats_after.total_blocks, stats_before.total_blocks); // blocks still allocated
    }

    #[test]
    fn test_clear_all() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        cache.register_sequence(2);

        let k_row = vec![1.0; 8];
        let v_row = vec![2.0; 8];
        cache.append(1, 0, 0, &k_row, &v_row);
        cache.append(2, 0, 0, &k_row, &v_row);

        cache.clear();
        assert_eq!(cache.stats().num_sequences, 0);
        assert_eq!(cache.total_blocks(), 0);
    }

    #[test]
    fn test_memory_usage() {
        let mut config = test_config();
        config.f16_storage = false; // use f32 for predictable sizing in this test
        let cache = PagedKVCache::new(config.clone());
        let per_block = config.block_size * config.num_kv_heads * config.head_dim * 4 * 2;
        assert!(cache.memory_usage_bytes() == 0);

        // After allocating blocks, memory should increase
        let mut cache2 = PagedKVCache::new(config);
        cache2.register_sequence(1);
        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];
        for pos in 0..4 {
            cache2.append(1, 0, pos, &k_row, &v_row);
            cache2.append(1, 1, pos, &k_row, &v_row);
        }
        // One block per layer (2 layers) = 2 blocks total
        assert_eq!(cache2.total_blocks(), 2);
        assert!(cache2.memory_usage_bytes() >= 2 * per_block);
    }

    #[test]
    fn test_f16_memory_half() {
        let mut config = test_config();
        let per_block = config.block_size * config.num_kv_heads * config.head_dim * 4 * 2;
        config.f16_storage = true;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(1);
        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];
        for pos in 0..4 {
            cache.append(1, 0, pos, &k_row, &v_row);
            cache.append(1, 1, pos, &k_row, &v_row);
        }

        let f16_bytes = cache.memory_usage_bytes();

        // Compare with f32 version
        let mut f32_config = test_config();
        f32_config.f16_storage = false;
        let mut f32_cache = PagedKVCache::new(f32_config);
        f32_cache.register_sequence(1);
        for pos in 0..4 {
            f32_cache.append(1, 0, pos, &k_row, &v_row);
            f32_cache.append(1, 1, pos, &k_row, &v_row);
        }
        let f32_bytes = f32_cache.memory_usage_bytes();

        assert!(
            f16_bytes < f32_bytes,
            "f16 memory ({f16_bytes}) should be less than f32 ({f32_bytes})"
        );
        let data_savings = f32_bytes - f16_bytes;
        assert!(
            data_savings >= per_block / 2,
            "f16 data savings ({data_savings}) should be at least half a block ({})",
            per_block / 2
        );
    }

    #[test]
    fn test_append_token_from_array() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);

        let k_arr = Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32).collect()).unwrap();
        let v_arr =
            Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32 * 2.0).collect()).unwrap();

        cache.append_token(1, 0, 0, &k_arr, &v_arr);

        let (k, v) = cache.read(1, 0, 0).unwrap();
        assert!((k[3] - 3.0).abs() < 1e-6);
        assert!((v[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_unregistered_sequence() {
        let cache = PagedKVCache::new(test_config());
        assert!(!cache.has_sequence(999));
        assert!(cache.to_flat_cache(999).is_none());
    }

    #[test]
    fn test_to_flat_cache_empty() {
        let cache = PagedKVCache::new(test_config());
        // No sequences registered
        assert!(cache.to_flat_cache(1).is_none());
    }

    #[test]
    fn test_alloc_stats() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        assert_eq!(cache.num_allocated, 0);

        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];
        cache.append(1, 0, 0, &k_row, &v_row);

        // First append allocates one block per layer
        assert!(cache.num_allocated > 0);
    }

    #[test]
    fn test_forward_paged_integration() {
        use ndarray::Array1;
        use nexora_transformer::{CausalLM, TransformerConfig};

        let cfg = PagedCacheConfig {
            block_size: 4,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_blocks: 1024,
            max_seq_len: 512,
            f16_storage: false,
            ..Default::default()
        };

        let mc = TransformerConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 2,
            max_seq_len: 512,
            vocab_size: 256,
            ..Default::default()
        };

        let model = CausalLM::new(mc);
        let mut paged_cache = PagedKVCache::new(cfg);
        paged_cache.register_sequence(1);

        let input = [1u32, 2u32, 3u32];
        for (i, &token) in input.iter().enumerate() {
            let logits = model
                .forward_paged(&[token], &mut paged_cache, 1)
                .expect("forward_paged should succeed in test");
            assert_eq!(logits.len(), 256);
            if i > 0 {
                let got = paged_cache.num_tokens(1).unwrap_or(0);
                assert_eq!(
                    got,
                    i + 1,
                    "after token {i}, cache should have {} entries",
                    i + 1
                );
            }
        }

        assert_eq!(paged_cache.num_tokens(1), Some(3));
        assert!(paged_cache.total_blocks() > 0);
    }

    // ─── PrefixDAG sharing tests ─────────────────────────────────────────────

    #[test]
    fn test_prefix_sharing_basic() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 10.0).collect();

        // Fill seq 10 with 4 tokens (1 block)
        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        let total_before = cache.total_blocks();
        let refcount_before = cache.blocks[0][0].ref_count;

        // Share prefix (4 tokens = 1 full block per layer) from 10 → 20
        let shared = cache.share_prefix_in_blocks(20, 10, 4);
        assert_eq!(shared, 2, "should share 1 block per layer (2 layers)");

        // No new blocks allocated — just ref_count bump
        assert_eq!(cache.total_blocks(), total_before);
        assert_eq!(
            cache.blocks[0][0].ref_count,
            refcount_before + 1,
            "ref_count should be incremented"
        );
        assert!(
            cache.has_sequence(20),
            "dst sequence should exist after share"
        );

        // Both sequences can read the same data from shared blocks
        for pos in 0..4 {
            let (k10, v10) = cache.read(10, 0, pos).unwrap();
            let (k20, v20) = cache.read(20, 0, pos).unwrap();
            for i in 0..kv_dim {
                assert!((k10[i] - k20[i]).abs() < 1e-6, "K mismatch at pos {pos}");
                assert!((v10[i] - v20[i]).abs() < 1e-6, "V mismatch at pos {pos}");
            }
        }
    }

    #[test]
    fn test_prefix_sharing_cow_on_write() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;

        // Fill seq 10 with distinct per-position data
        for pos in 0..4 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * kv_dim + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * kv_dim + i + 100) as f32).collect();
            cache.append(10, 0, pos, &k, &v);
            cache.append(10, 1, pos, &k, &v);
        }

        let shared = cache.share_prefix_in_blocks(20, 10, 4);
        assert_eq!(shared, 2, "should share 1 block per layer");

        let blocks_before_cow = cache.total_blocks();
        assert_eq!(cache.blocks[0][0].ref_count, 2);

        // Seq 20 writes to position 0 — triggers COW (block ref_count=2)
        let cow_k: Vec<f32> = (200..200 + kv_dim).map(|i| i as f32).collect();
        let cow_v: Vec<f32> = (300..300 + kv_dim).map(|i| i as f32).collect();
        cache.append(20, 0, 0, &cow_k.as_slice(), &cow_v.as_slice());

        // After COW: new block allocated, original ref_count decremented
        assert_eq!(
            cache.total_blocks(),
            blocks_before_cow + 1,
            "COW should allocate one new block"
        );
        assert_eq!(
            cache.blocks[0][0].ref_count,
            1,
            "original block ref_count decremented after COW"
        );

        // Seq 10 data should be unchanged
        let (k10, v10) = cache.read(10, 0, 0).unwrap();
        let expected_k: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let expected_v: Vec<f32> = (0..kv_dim).map(|i| (i + 100) as f32).collect();
        for i in 0..kv_dim {
            assert!(
                (k10[i] - expected_k[i]).abs() < 1e-6,
                "seq 10 K[{i}] changed after COW: expected {}, got {}",
                expected_k[i],
                k10[i]
            );
            assert!(
                (v10[i] - expected_v[i]).abs() < 1e-6,
                "seq 10 V[{i}] changed after COW"
            );
        }

        // Seq 20 data should be the new value
        let (k20, v20) = cache.read(20, 0, 0).unwrap();
        for i in 0..kv_dim {
            assert!(
                (k20[i] - cow_k[i]).abs() < 1e-6,
                "seq 20 K[{i}] should be cow value after COW"
            );
            assert!(
                (v20[i] - cow_v[i]).abs() < 1e-6,
                "seq 20 V[{i}] should be cow value after COW"
            );
        }
    }

    #[test]
    fn test_prefix_sharing_cow_data_integrity() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;

        // Fill seq 10 with 8 tokens (2 blocks)
        for pos in 0..8 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i + 1000) as f32).collect();
            cache.append(10, 0, pos, &k, &v);
            cache.append(10, 1, pos, &k, &v);
        }

        // Share first 4 tokens (1 block) from 10 to 20
        cache.share_prefix_in_blocks(20, 10, 4);

        // Seq 20 writes to position 0 and 4 (position 4 falls in NEW block, no COW)
        let cow_k: Vec<f32> = (0..kv_dim).map(|i| 999.0 + i as f32).collect();
        let cow_v: Vec<f32> = (0..kv_dim).map(|i| 888.0 + i as f32).collect();
        cache.append(20, 0, 0, &cow_k, &cow_v);

        let new_k: Vec<f32> = (0..kv_dim).map(|i| 777.0 + i as f32).collect();
        let new_v: Vec<f32> = (0..kv_dim).map(|i| 666.0 + i as f32).collect();
        cache.append(20, 0, 4, &new_k, &new_v);

        // Verify seq 10 unchanged at position 0
        let (k10_p0, v10_p0) = cache.read(10, 0, 0).unwrap();
        for i in 0..kv_dim {
            assert!((k10_p0[i] - i as f32).abs() < 1e-6, "seq 10 K[0][{i}] corrupted");
            assert!((v10_p0[i] - (i + 1000) as f32).abs() < 1e-6, "seq 10 V[0][{i}] corrupted");
        }

        // Verify seq 10 unchanged at position 4
        let (k10_p4, v10_p4) = cache.read(10, 0, 4).unwrap();
        for i in 0..kv_dim {
            assert!((k10_p4[i] - (4 * 10 + i) as f32).abs() < 1e-6, "seq 10 K[4][{i}] corrupted");
        }

        // Verify seq 20 has cow value at position 0
        let (k20_p0, v20_p0) = cache.read(20, 0, 0).unwrap();
        for i in 0..kv_dim {
            assert!((k20_p0[i] - (999.0 + i as f32)).abs() < 1e-6, "seq 20 K[0][{i}] wrong after COW");
        }

        // Verify seq 20 has new value at position 4 (new block, not shared)
        let (k20_p4, v20_p4) = cache.read(20, 0, 4).unwrap();
        for i in 0..kv_dim {
            assert!((k20_p4[i] - (777.0 + i as f32)).abs() < 1e-6, "seq 20 K[4][{i}] wrong");
        }

        // Verify seq 20 still sees shared data at positions 1-3 (no COW on those)
        for pos in 1..4 {
            let (k20, v20) = cache.read(20, 0, pos).unwrap();
            let (k10, v10) = cache.read(10, 0, pos).unwrap();
            for i in 0..kv_dim {
                assert!((k20[i] - k10[i]).abs() < 1e-6, "seq 20 K[{pos}][{i}] diverged from shared");
                assert!((v20[i] - v10[i]).abs() < 1e-6, "seq 20 V[{pos}][{i}] diverged from shared");
            }
        }
    }

    #[test]
    fn test_prefix_sharing_multiple_sequences() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);
        cache.register_sequence(30);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();

        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        // Share from 10 to both 20 and 30
        cache.share_prefix_in_blocks(20, 10, 4);
        cache.share_prefix_in_blocks(30, 10, 4);

        // All three sequences share the same physical block
        assert_eq!(cache.blocks[0][0].ref_count, 3, "3-way sharing ref_count");

        // COW on seq 20 — decrements one ref_count
        let cow_k: Vec<f32> = (0..kv_dim).map(|i| 555.0 + i as f32).collect();
        let cow_v: Vec<f32> = (0..kv_dim).map(|i| 555.0 + i as f32).collect();
        cache.append(20, 0, 0, &cow_k, &cow_v);

        // Original block now has ref_count=2 (shared by 10 and 30)
        assert_eq!(cache.blocks[0][0].ref_count, 2, "after COW on 20, ref_count drops to 2");

        // COW on seq 30 — decrements another ref_count
        cache.append(30, 0, 0, &cow_k, &cow_v);
        assert_eq!(cache.blocks[0][0].ref_count, 1, "after COW on 30, ref_count drops to 1");
    }

    #[test]
    fn test_prefix_sharing_noop_invalid() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();

        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        // Invalid dst seq
        assert_eq!(cache.share_prefix_in_blocks(999, 10, 4), 0, "invalid dst");
        // Invalid src seq
        assert_eq!(cache.share_prefix_in_blocks(10, 999, 4), 0, "invalid src");
        // prefix_tokens = 0
        assert_eq!(cache.share_prefix_in_blocks(10, 10, 0), 0, "zero prefix");
    }

    #[test]
    fn test_prefix_sharing_free_after_share() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();

        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        cache.share_prefix_in_blocks(20, 10, 4);
        assert_eq!(cache.blocks[0][0].ref_count, 2);

        // Remove seq 20 — ref_count should drop
        cache.remove_sequence(20);
        assert_eq!(cache.blocks[0][0].ref_count, 1, "ref_count decremented after removing shared seq");
    }

    #[test]
    fn test_prefix_sharing_append_after_share() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;

        // Seq 10: 4 tokens
        for pos in 0..4 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i + 100) as f32).collect();
            cache.append(10, 0, pos, &k, &v);
            cache.append(10, 1, pos, &k, &v);
        }

        // Share prefix + seq 20 appends 2 more tokens
        cache.share_prefix_in_blocks(20, 10, 4);

        // Append position 4 and 5 to seq 20 (new block)
        for pos in 4..6 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i + 100) as f32).collect();
            cache.append(20, 0, pos, &k, &v);
            cache.append(20, 1, pos, &k, &v);
        }

        assert_eq!(cache.num_tokens(20), Some(6), "seq 20 should have 6 tokens after append");

        // Seq 10 still has 4 tokens
        assert_eq!(cache.num_tokens(10), Some(4), "seq 10 should have 4 tokens");

        // Verify shared prefix data still matches
        for pos in 0..4 {
            let (k10, v10) = cache.read(10, 0, pos).unwrap();
            let (k20, v20) = cache.read(20, 0, pos).unwrap();
            for i in 0..kv_dim {
                assert!((k10[i] - k20[i]).abs() < 1e-6, "shared prefix K mismatch at pos {pos}");
                assert!((v10[i] - v20[i]).abs() < 1e-6, "shared prefix V mismatch at pos {pos}");
            }
        }

        // Verify new tokens in seq 20
        for pos in 4..6 {
            let (k20, _) = cache.read(20, 0, pos).unwrap();
            assert!((k20[0] - (pos * 10) as f32).abs() < 1e-6, "appended K wrong at pos {pos}");
        }
    }

    // ─── PrefixTrie tests ────────────────────────────────────────────────────

    #[test]
    fn test_prefix_trie_insert_and_find_shared() {
        use crate::continuous_batching::PrefixTrie;

        let mut trie = PrefixTrie::new();

        // Insert seq 1 with prompt [1,2,3,4,5]
        trie.insert(1, &[1, 2, 3, 4, 5]);

        // Insert seq 2 with prompt [1,2,3,6,7,8] — shared prefix [1,2,3]
        trie.insert(2, &[1, 2, 3, 6, 7, 8]);

        // Find shared prefix for seq 2
        let result = trie.find_shared_prefix(&[1, 2, 3, 6, 7, 8], 2);
        assert!(result.is_some(), "should find shared prefix");
        let (len, src) = result.unwrap();
        assert_eq!(len, 3, "should find 3 shared tokens");
        assert_eq!(src, 1, "should find seq 1 as source");

        // Find shared prefix for seq 1
        let result = trie.find_shared_prefix(&[1, 2, 3, 4, 5], 1);
        assert!(result.is_some(), "should find shared prefix for seq 1");
        let (len, src) = result.unwrap();
        assert_eq!(len, 3, "should find 3 shared tokens for seq 1 too");
        assert_eq!(src, 2, "should find seq 2 as source");

        // No match
        let result = trie.find_shared_prefix(&[9, 10, 11], 3);
        assert!(result.is_none(), "no match for unknown prefix");
    }

    #[test]
    fn test_prefix_trie_exact_match() {
        use crate::continuous_batching::PrefixTrie;

        let mut trie = PrefixTrie::new();
        trie.insert(1, &[1, 2, 3, 4]);
        trie.insert(2, &[1, 2, 3, 4]);

        let result = trie.find_shared_prefix(&[1, 2, 3, 4], 3);
        assert!(result.is_some(), "exact match should return Some");
        let (len, src) = result.unwrap();
        assert_eq!(len, 4, "exact match should return full length");
        assert!(src == 1 || src == 2, "exact match should find any matching seq");
    }

    #[test]
    fn test_prefix_trie_empty_prompt() {
        use crate::continuous_batching::PrefixTrie;

        let mut trie = PrefixTrie::new();
        trie.insert(1, &[1, 2, 3]);

        let result = trie.find_shared_prefix(&[], 2);
        assert!(result.is_none(), "empty prompt should match nothing");
    }

    #[test]
    fn test_paged_vs_flat_cache_parity() {
        use ndarray::Array1;
        use nexora_transformer::{CausalLM, KVCacheEntry, TransformerConfig};

        let cfg = PagedCacheConfig {
            block_size: 4,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_blocks: 1024,
            max_seq_len: 512,
            f16_storage: false,
            ..Default::default()
        };

        let mc = TransformerConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 2,
            max_seq_len: 512,
            vocab_size: 256,
            ..Default::default()
        };

        let model = CausalLM::new(mc);
        let mut flat_cache = model.reset_cache();
        let mut paged_cache = PagedKVCache::new(cfg);
        paged_cache.register_sequence(1);

        let tokens = [5u32, 10u32, 15u32];
        for (i, &token) in tokens.iter().enumerate() {
            let logits_flat = model
                .forward(&[token], &mut flat_cache)
                .expect("forward should succeed in test");
            let logits_paged = model
                .forward_paged(&[token], &mut paged_cache, 1)
                .expect("forward_paged should succeed in test");

            for j in 0..logits_flat.len() {
                let diff = (logits_flat[j] - logits_paged[j]).abs();
                assert!(
                    diff < 1e-4,
                    "mismatch at token {i}, logit {j}: flat={} paged={}",
                    logits_flat[j],
                    logits_paged[j]
                );
            }
        }
    }
}
