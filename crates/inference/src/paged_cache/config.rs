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

use std::sync::{Mutex, OnceLock};

use crate::paged_cache::PagedKVCache;

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
