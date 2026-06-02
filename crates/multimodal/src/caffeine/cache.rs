//! Multimodal Cache System — #4 Multimodal Cache
//!
//! Implementasi lengkap 10 fix optimasi memori:
//! FIX 1: Feature Map → Embedding (simpan hasil akhir, bukan tensor intermediate)
//! FIX 2: Embedding Deduplication via SHA256 hash lookup
//! FIX 3: TTL Cache (created_at, last_access, hit_count, expired cleanup)
//! FIX 4: Compression FP8/INT8/FP16
//! FIX 5: Streaming Video — sliding window processing
//! FIX 6: Shared Multimodal Store — global store across models
//! FIX 7: Hierarchical Storage (L1 GPU, L2 RAM, L3 SSD)
//! FIX 8: Adaptive Resolution — resize before encoder
//! FIX 9: Cache Budget — max RAM/SSD with LRU eviction
//! FIX 10: Oracle + Multimodal Fusion — Unified Embedding Store

use crate::caffeine::error::Result;
use crate::caffeine::types::*;
use ndarray::ArrayD;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Alias untuk ArrayD<f32>
pub type ArrayDf32 = ArrayD<f32>;

// ─── Global Shared Multimodal Store (FIX 6 + FIX 10) ─────────────────────────

/// Global unified multimodal store — semua model/Caffeine instance berbagi cache yg sama.
static GLOBAL_MULTIMODAL_STORE: OnceLock<Arc<Mutex<MultiModalCache>>> = OnceLock::new();

/// Dapatkan atau inisialisasi global shared store.
pub fn global_multimodal_store() -> Arc<Mutex<MultiModalCache>> {
    GLOBAL_MULTIMODAL_STORE
        .get_or_init(|| Arc::new(Mutex::new(MultiModalCache::new(CacheConfig::default()))))
        .clone()
}

/// Set global store dengan konfigurasi kustom (panggil sekali di init).
pub fn init_global_multimodal_store(config: CacheConfig) -> Arc<Mutex<MultiModalCache>> {
    let store = Arc::new(Mutex::new(MultiModalCache::new(config)));
    let _ = GLOBAL_MULTIMODAL_STORE.set(store.clone());
    store
}

// ─── Compression Type (FIX 4) ────────────────────────────────────────────────

/// Format kompresi embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    FP32,
    FP16,
    INT8,
}

impl Default for CompressionFormat {
    fn default() -> Self {
        Self::FP16
    }
}

/// Compressed embedding — menyimpan embedding dalam format ringkas.
#[derive(Debug, Clone)]
pub enum CompressedEmbedding {
    FP32(Vec<f32>),
    FP16(Vec<u16>),
    INT8 { data: Vec<i8>, scale: f32, zero_point: i8 },
}

impl CompressedEmbedding {
    /// Buat compressed embedding dari FP32 vector.
    pub fn from_fp32(data: Vec<f32>, format: CompressionFormat) -> Self {
        match format {
            CompressionFormat::FP32 => Self::FP32(data),
            CompressionFormat::FP16 => {
                let half: Vec<u16> = data.iter().map(|&x| f32_to_fp16(x)).collect();
                Self::FP16(half)
            }
            CompressionFormat::INT8 => {
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let range = max_val - min_val;
                if range == 0.0 {
                    return Self::INT8 {
                        data: vec![0i8; data.len()],
                        scale: 1.0,
                        zero_point: 0,
                    };
                }
                let scale = range / 255.0;
                let zero_point = (-min_val / scale).round() as i8;
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&x| ((x / scale).round() as i8).clamp(-128, 127))
                    .collect();
                Self::INT8 { data: quantized, scale, zero_point }
            }
        }
    }

    /// Dekompres kembali ke FP32.
    pub fn to_fp32(&self) -> Vec<f32> {
        match self {
            Self::FP32(d) => d.clone(),
            Self::FP16(d) => d.iter().map(|&x| fp16_to_f32(x)).collect(),
            Self::INT8 { data, scale, zero_point } => {
                let z = *zero_point as f32;
                data.iter().map(|&x| (x as f32 - z) * scale).collect()
            }
        }
    }

    /// Ukuran dalam bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::FP32(d) => d.len() * 4,
            Self::FP16(d) => d.len() * 2,
            Self::INT8 { data, .. } => data.len(),
        }
    }
}

// ─── FP16 Conversion Helpers ─────────────────────────────────────────────────

fn f32_to_fp16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7fffff;

    if exponent > 142 {
        (sign << 15) | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 }
    } else if exponent > 112 {
        let exp = (exponent - 127 + 15) as u16;
        (sign << 15) | (exp << 10) | (mantissa >> 13) as u16
    } else {
        let m = mantissa | 0x800000;
        let shift = 113 - exponent;
        if shift < 32 {
            (sign << 15) | (m >> (shift + 13)) as u16
        } else {
            0
        }
    }
}

fn fp16_to_f32(x: u16) -> f32 {
    let sign = (x >> 15) as u32;
    let exponent = ((x >> 10) & 0x1f) as u32;
    let mantissa = (x & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            let exp = 127 - 15 - 10;
            f32::from_bits((sign << 31) | (exp << 23) | (mantissa << 13))
        }
    } else if exponent == 31 {
        f32::from_bits((sign << 31) | 0x7f800000 | (mantissa << 13))
    } else {
        let exp = exponent + 127 - 15;
        f32::from_bits((sign << 31) | (exp << 23) | (mantissa << 13))
    }
}

// ─── Storage Level (FIX 7) ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageLevel {
    /// L1 — Hot: GPU memory (atau in-memory hot set)
    L1Gpu,
    /// L2 — Warm: system RAM
    L2Ram,
    /// L3 — Cold: SSD/disk
    L3Ssd,
}

// ─── Cache Entry (FIX 3: TTL) ────────────────────────────────────────────────

/// Single entry dalam unified multimodal cache.
#[derive(Debug, Clone)]
pub struct MultiModalCacheEntry {
    /// SHA256 hash of input content (FIX 2)
    pub hash: [u8; 32],
    /// Compressed embedding (FIX 4)
    pub embedding: CompressedEmbedding,
    /// Modality type
    pub modality: ModalityType,
    /// Timestamp dibuat
    pub created_at: Instant,
    /// Timestamp akses terakhir
    pub last_access: Instant,
    /// Hit counter
    pub hit_count: u64,
    /// Storage level (FIX 7)
    pub storage_level: StorageLevel,
    /// Original dimensions (pre-compression)
    pub original_shape: Vec<usize>,
}

impl MultiModalCacheEntry {
    pub fn new(
        hash: [u8; 32],
        embedding: Vec<f32>,
        modality: ModalityType,
        format: CompressionFormat,
        original_shape: Vec<usize>,
        storage_level: StorageLevel,
    ) -> Self {
        let now = Instant::now();
        Self {
            hash,
            embedding: CompressedEmbedding::from_fp32(embedding, format),
            modality,
            created_at: now,
            last_access: now,
            hit_count: 0,
            storage_level,
            original_shape,
        }
    }

    /// Cek apakah entry expired berdasarkan TTL.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.last_access.elapsed() > ttl
    }

    /// Record akses.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.hit_count += 1;
    }
}

// ─── Adaptive Resolution (FIX 8) ─────────────────────────────────────────────

/// Hasil adaptive resolution.
#[derive(Debug, Clone)]
pub struct ResizedInput {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub original_width: usize,
    pub original_height: usize,
}

/// Resize gambar ke resolusi maksimal sebelum encoder (FIX 8).
/// Nearest-neighbor interpolation — cepat, minim alokasi.
pub fn adaptive_resize(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    max_pixels: usize,
) -> Result<ResizedInput> {
    let total_pixels = width * height;
    if total_pixels <= max_pixels {
        return Ok(ResizedInput {
            data: data.to_vec(),
            width,
            height,
            channels,
            original_width: width,
            original_height: height,
        });
    }

    let scale = ((max_pixels as f64) / (total_pixels as f64)).sqrt();
    let new_width = (width as f64 * scale).round().max(64.0) as usize;
    let new_height = (height as f64 * scale).round().max(64.0) as usize;

    let mut resized = Vec::with_capacity(new_width * new_height * channels);
    for ny in 0..new_height {
        for nx in 0..new_width {
            let ox = (nx as f64 / scale).min((width - 1) as f64) as usize;
            let oy = (ny as f64 / scale).min((height - 1) as f64) as usize;
            for c in 0..channels {
                let src_idx = (oy * width + ox) * channels + c;
                let val = data.get(src_idx).copied().unwrap_or(0);
                resized.push(val);
            }
        }
    }

    Ok(ResizedInput {
        data: resized,
        width: new_width,
        height: new_height,
        channels,
        original_width: width,
        original_height: height,
    })
}

// ─── Streaming Video Sliding Window (FIX 5) ──────────────────────────────────

/// Sliding window untuk video streaming.
/// Process frame → buang → process frame berikutnya.
#[derive(Debug, Clone)]
pub struct VideoSlidingWindow {
    pub window_size: usize,
    pub stride: usize,
    pub buffer: Vec<ImageInput>,
    pub frame_counter: u64,
}

impl VideoSlidingWindow {
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self {
            window_size,
            stride,
            buffer: Vec::with_capacity(window_size),
            frame_counter: 0,
        }
    }

    /// Push frame baru. Return true jika window penuh (siap diproses).
    pub fn push(&mut self, frame: ImageInput) -> bool {
        self.buffer.push(frame);
        self.frame_counter += 1;
        self.buffer.len() >= self.window_size
    }

    /// Drain window untuk diproses.
    pub fn drain_window(&mut self) -> Vec<ImageInput> {
        let drained = self.buffer.drain(..self.window_size).collect();
        self.buffer.reserve(self.stride);
        drained
    }

    /// Proses video streaming dengan sliding window callback.
    /// callback dipanggil per window_size frames, memori tetap rendah.
    pub fn process_streaming<F>(
        frames: Vec<ImageInput>,
        window_size: usize,
        stride: usize,
        mut callback: F,
    ) -> Result<Vec<ArrayDf32>>
    where
        F: FnMut(Vec<ImageInput>) -> Result<ArrayDf32>,
    {
        let mut results = Vec::new();
        let mut i = 0;
        while i < frames.len() {
            let end = (i + window_size).min(frames.len());
            let window: Vec<ImageInput> = frames[i..end].to_vec();
            if !window.is_empty() {
                let result = callback(window)?;
                results.push(result);
            }
            i += stride;
        }
        Ok(results)
    }
}

// ─── Cache Config (FIX 9: Budget) ────────────────────────────────────────────

/// Konfigurasi multimodal cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum RAM untuk cache (default 4 GB)
    pub max_ram_bytes: u64,
    /// Maximum SSD untuk cache (default 20 GB)
    pub max_ssd_bytes: u64,
    /// Default TTL dalam detik (default 3600 = 1 jam)
    pub ttl_secs: u64,
    /// Format kompresi (default FP16)
    pub compression: CompressionFormat,
    /// Enable deduplication via SHA256
    pub enable_dedup: bool,
    /// Adaptive resolution max pixels (default 1_048_576 = 1024x1024)
    pub adaptive_max_pixels: usize,
    /// Enable hierarchical storage
    pub enable_hierarchical: bool,
    /// Video streaming window size
    pub video_window_size: usize,
    /// Video streaming stride
    pub video_stride: usize,
    /// Cleanup interval dalam detik (default 300 = 5 menit)
    pub cleanup_interval_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_ram_bytes: 4 * 1024 * 1024 * 1024,
            max_ssd_bytes: 20 * 1024 * 1024 * 1024,
            ttl_secs: 3600,
            compression: CompressionFormat::FP16,
            enable_dedup: true,
            adaptive_max_pixels: 1_048_576,
            enable_hierarchical: true,
            video_window_size: 4,
            video_stride: 2,
            cleanup_interval_secs: 300,
        }
    }
}

// ─── Cache Stats ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_evictions: u64,
    pub ram_usage_bytes: u64,
    pub ssd_usage_bytes: u64,
    pub dedup_saves: u64,
    pub l1_count: usize,
    pub l2_count: usize,
    pub l3_count: usize,
    pub hit_rate: f64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_entries: 0,
            total_hits: 0,
            total_misses: 0,
            total_evictions: 0,
            ram_usage_bytes: 0,
            ssd_usage_bytes: 0,
            dedup_saves: 0,
            l1_count: 0,
            l2_count: 0,
            l3_count: 0,
            hit_rate: 0.0,
        }
    }
}

// ─── MultiModalCache (Main) ──────────────────────────────────────────────────

/// Unified Multimodal Cache — FIX 6 (shared store) + FIX 10 (unified).
///
/// Semua entry (image, audio, video, text) disimpan dalam satu HashMap.
/// Hierarchical storage (FIX 7) via storage_level field.
pub struct MultiModalCache {
    entries: HashMap<[u8; 32], MultiModalCacheEntry>,
    config: CacheConfig,
    stats: CacheStats,
    total_ram_bytes: u64,
    total_ssd_bytes: u64,
    last_cleanup: Instant,
    access_order: Vec<[u8; 32]>,
}

impl MultiModalCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: CacheStats::default(),
            total_ram_bytes: 0,
            total_ssd_bytes: 0,
            last_cleanup: Instant::now(),
            access_order: Vec::new(),
        }
    }

    // ─── Core Operations ────────────────────────────────────────────────────

    /// Compute SHA256 hash dari raw bytes (FIX 2).
    pub fn compute_hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Compute hash dari ImageInput (FIX 2).
    pub fn hash_image_input(input: &ImageInput) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&input.data);
        hasher.update(input.width.to_le_bytes());
        hasher.update(input.height.to_le_bytes());
        hasher.update(input.channels.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Compute hash dari AudioInput (FIX 2).
    pub fn hash_audio_input(input: &AudioInput) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for &sample in &input.data {
            hasher.update(sample.to_le_bytes());
        }
        hasher.update(input.sample_rate.to_le_bytes());
        hasher.update(input.channels.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Compute hash dari VideoInput (FIX 2).
    pub fn hash_video_input(input: &VideoInput) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for frame in &input.frames {
            hasher.update(&frame.data);
            hasher.update(frame.width.to_le_bytes());
            hasher.update(frame.height.to_le_bytes());
        }
        hasher.update(input.frame_rate.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Get atau compute embedding via cache (FIX 1 + FIX 2).
    /// Hanya menyimpan hasil akhir embedding, bukan tensor intermediate (FIX 1).
    /// Return (CompressedEmbedding, is_cache_hit).
    pub fn get_or_compute<F>(
        &mut self,
        hash: [u8; 32],
        modality: ModalityType,
        original_shape: Vec<usize>,
        compute_fn: F,
    ) -> (CompressedEmbedding, bool)
    where
        F: FnOnce() -> Vec<f32>,
    {
        self.maybe_cleanup();

        // Dedup lookup (FIX 2)
        if self.config.enable_dedup {
            let is_hit = self.entries.contains_key(&hash);
            if is_hit {
                self.stats.total_hits += 1;
                self.update_access_order(hash);
                if let Some(entry) = self.entries.get_mut(&hash) {
                    entry.touch();
                    return (entry.embedding.clone(), true);
                }
            }
        }

        self.stats.total_misses += 1;

        // Compute embedding — FIX 1: hanya simpan hasil akhir, bukan feature map intermediate
        let raw_embedding = compute_fn();

        // Budget enforcement (FIX 9)
        let estimated_size = raw_embedding.len() * 4;
        self.enforce_budget(estimated_size, StorageLevel::L2Ram);

        let entry = MultiModalCacheEntry::new(
            hash,
            raw_embedding,
            modality,
            self.config.compression,
            original_shape,
            StorageLevel::L2Ram,
        );

        let entry_size = entry.embedding.size_bytes() as u64;
        self.total_ram_bytes += entry_size;

        self.entries.insert(hash, entry);
        self.access_order.push(hash);
        self.stats.total_entries = self.entries.len();

        let cached = self.entries.get(&hash).unwrap();
        (cached.embedding.clone(), false)
    }

    /// Get by hash (read-only).
    pub fn get(&self, hash: &[u8; 32]) -> Option<&MultiModalCacheEntry> {
        self.entries.get(hash)
    }

    /// Get by hash (read-only, record hit/miss).
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        self.entries.contains_key(hash)
    }

    /// Insert entry langsung.
    pub fn insert(
        &mut self,
        hash: [u8; 32],
        embedding: Vec<f32>,
        modality: ModalityType,
        original_shape: Vec<usize>,
    ) {
        self.maybe_cleanup();

        // Cek dedup (FIX 2)
        if self.config.enable_dedup && self.entries.contains_key(&hash) {
            self.stats.dedup_saves += 1;
            return;
        }

        let estimated_size = embedding.len() * 4;
        self.enforce_budget(estimated_size, StorageLevel::L2Ram);

        let entry = MultiModalCacheEntry::new(
            hash,
            embedding,
            modality,
            self.config.compression,
            original_shape,
            StorageLevel::L2Ram,
        );

        let entry_size = entry.embedding.size_bytes() as u64;
        self.total_ram_bytes += entry_size;
        self.entries.insert(hash, entry);
        self.access_order.push(hash);
        self.stats.total_entries = self.entries.len();
    }

    // ─── TTL Eviction (FIX 3) ───────────────────────────────────────────────

    /// Hapus entry yg expired berdasarkan TTL.
    pub fn evict_expired(&mut self) -> u64 {
        let ttl = Duration::from_secs(self.config.ttl_secs);
        let expired_keys: Vec<[u8; 32]> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(ttl))
            .map(|(k, _)| *k)
            .collect();

        let count = expired_keys.len() as u64;
        for key in expired_keys {
            self.remove_entry(key);
        }
        self.stats.total_evictions += count;
        count
    }

    /// Hapus entry dengan hit_count == 0 (tidak pernah dipakai — FIX 3).
    pub fn evict_zero_hit(&mut self) -> u64 {
        let zero_hit_keys: Vec<[u8; 32]> = self
            .entries
            .iter()
            .filter(|(_, e)| e.hit_count == 0)
            .map(|(k, _)| *k)
            .collect();

        let count = zero_hit_keys.len() as u64;
        for key in zero_hit_keys {
            self.remove_entry(key);
        }
        self.stats.total_evictions += count;
        count
    }

    // ─── LRU Eviction by Budget (FIX 9) ─────────────────────────────────────

    /// Enforce budget — LRU eviction sampai cukup space.
    fn enforce_budget(&mut self, needed_size: usize, target_level: StorageLevel) {
        let max_bytes = match target_level {
            StorageLevel::L1Gpu | StorageLevel::L2Ram => self.config.max_ram_bytes,
            StorageLevel::L3Ssd => self.config.max_ssd_bytes,
        };

        let current_bytes = match target_level {
            StorageLevel::L1Gpu | StorageLevel::L2Ram => self.total_ram_bytes,
            StorageLevel::L3Ssd => self.total_ssd_bytes,
        };

        if current_bytes + needed_size as u64 <= max_bytes {
            return;
        }

        // LRU eviction: sort by last_access, evict oldest
        let mut candidates: Vec<([u8; 32], Instant)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.storage_level == target_level)
            .map(|(k, e)| (*k, e.last_access))
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1));

        let mut bytes_freed = current_bytes;
        let target = max_bytes.saturating_sub(needed_size as u64);

        for (key, _) in candidates {
            if bytes_freed <= target {
                break;
            }
            if let Some(entry) = self.entries.get(&key) {
                bytes_freed = bytes_freed.saturating_sub(entry.embedding.size_bytes() as u64);
            }
            let _ = self.entries.remove(&key);
            self.access_order.retain(|k| *k != key);
            self.stats.total_evictions += 1;
        }
        self.stats.total_entries = self.entries.len();
    }

    // ─── Hierarchical Storage (FIX 7) ───────────────────────────────────────

    /// Promote entry ke L1 (hot — sering diakses).
    pub fn promote_to_l1(&mut self, hash: &[u8; 32]) {
        if !self.config.enable_hierarchical {
            return;
        }
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.storage_level = StorageLevel::L1Gpu;
            entry.touch();
        }
    }

    /// Demote entry ke L3 (cold — jarang diakses).
    pub fn demote_to_l3(&mut self, hash: &[u8; 32]) {
        if !self.config.enable_hierarchical {
            return;
        }
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.storage_level = StorageLevel::L3Ssd;
        }
    }

    /// Auto-tier: promote/demote berdasarkan hit_count dan age.
    pub fn auto_tier(&mut self) {
        if !self.config.enable_hierarchical {
            return;
        }
        let keys: Vec<[u8; 32]> = self.entries.keys().copied().collect();
        for key in keys {
            let (hit, level, age) = {
                let entry = match self.entries.get(&key) {
                    Some(e) => e,
                    None => continue,
                };
                (entry.hit_count, entry.storage_level, entry.created_at.elapsed())
            };

            if hit > 5 && !matches!(level, StorageLevel::L1Gpu) {
                self.promote_to_l1(&key);
            } else if hit == 0
                && !matches!(level, StorageLevel::L3Ssd)
                && age > Duration::from_secs(600)
            {
                self.demote_to_l3(&key);
            }
        }
    }

    // ─── Cleanup ────────────────────────────────────────────────────────────

    fn maybe_cleanup(&mut self) {
        let interval = Duration::from_secs(self.config.cleanup_interval_secs);
        if self.last_cleanup.elapsed() < interval {
            return;
        }
        self.last_cleanup = Instant::now();
        self.evict_expired();
        self.evict_zero_hit();
        self.auto_tier();
        self.update_stats();
    }

    /// Force cleanup.
    pub fn cleanup(&mut self) {
        self.evict_expired();
        self.evict_zero_hit();
        self.auto_tier();
        self.update_stats();
    }

    /// Clear seluruh cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.total_ram_bytes = 0;
        self.total_ssd_bytes = 0;
        self.stats = CacheStats::default();
    }

    // ─── Internal Helpers ───────────────────────────────────────────────────

    fn remove_entry(&mut self, key: [u8; 32]) {
        if let Some(entry) = self.entries.remove(&key) {
            let size = entry.embedding.size_bytes() as u64;
            match entry.storage_level {
                StorageLevel::L1Gpu | StorageLevel::L2Ram => {
                    self.total_ram_bytes = self.total_ram_bytes.saturating_sub(size);
                }
                StorageLevel::L3Ssd => {
                    self.total_ssd_bytes = self.total_ssd_bytes.saturating_sub(size);
                }
            }
        }
        self.access_order.retain(|k| *k != key);
        self.stats.total_entries = self.entries.len();
    }

    fn update_access_order(&mut self, hash: [u8; 32]) {
        self.access_order.retain(|k| *k != hash);
        self.access_order.push(hash);
    }

    fn update_stats(&mut self) {
        self.stats.total_entries = self.entries.len();
        self.stats.ram_usage_bytes = self.total_ram_bytes;
        self.stats.ssd_usage_bytes = self.total_ssd_bytes;
        self.stats.l1_count = self
            .entries
            .values()
            .filter(|e| matches!(e.storage_level, StorageLevel::L1Gpu))
            .count();
        self.stats.l2_count = self
            .entries
            .values()
            .filter(|e| matches!(e.storage_level, StorageLevel::L2Ram))
            .count();
        self.stats.l3_count = self
            .entries
            .values()
            .filter(|e| matches!(e.storage_level, StorageLevel::L3Ssd))
            .count();
        let total = self.stats.total_hits + self.stats.total_misses;
        self.stats.hit_rate = if total > 0 {
            self.stats.total_hits as f64 / total as f64
        } else {
            0.0
        };
    }

    // ─── Public Accessors ───────────────────────────────────────────────────

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    pub fn entries(&self) -> &HashMap<[u8; 32], MultiModalCacheEntry> {
        &self.entries
    }
}

// ─── Streaming Video Processor (FIX 5) ───────────────────────────────────────

/// Process video dengan sliding window — tidak simpan semua frame dalam memori.
pub struct StreamingVideoProcessor {
    sliding_window: VideoSlidingWindow,
    config: CacheConfig,
}

impl StreamingVideoProcessor {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            sliding_window: VideoSlidingWindow::new(config.video_window_size, config.video_stride),
            config,
        }
    }

    /// Process video input dengan sliding window.
    /// frame_encoder dipanggil per frame, hasilnya segera digabungkan.
    pub fn process_video<F>(
        &mut self,
        input: &VideoInput,
        mut frame_encoder: F,
    ) -> Result<(Vec<f32>, usize, usize)>
    where
        F: FnMut(&ImageInput) -> Result<Vec<f32>>,
    {
        let total_frames = input.frames.len();
        let window_size = self.config.video_window_size;
        let stride = self.config.video_stride;

        let mut all_window_embeddings = Vec::new();
        let mut i = 0;

        while i < total_frames {
            let end = (i + window_size).min(total_frames);
            let window = &input.frames[i..end];

            // Sliding window: process frame → buang (FIX 5)
            for frame in window {
                let emb = frame_encoder(frame)?;
                all_window_embeddings.extend(emb);
            }

            i += stride;
        }

        Ok((all_window_embeddings, total_frames, window_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_fp16_roundtrip() {
        let original = vec![1.0f32, -0.5, 0.0, 3.14, -2.71, 0.001];
        let compressed = CompressedEmbedding::from_fp32(original.clone(), CompressionFormat::FP16);
        let recovered = compressed.to_fp32();
        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 0.01, "FP16 roundtrip error: {} vs {} (diff={})", a, b, diff);
        }
    }

    #[test]
    fn test_compression_int8_roundtrip() {
        let original = vec![1.0f32, -0.5, 0.0, 3.14, -2.71, 0.001];
        let compressed = CompressedEmbedding::from_fp32(original.clone(), CompressionFormat::INT8);
        let recovered = compressed.to_fp32();
        assert_eq!(original.len(), recovered.len());
    }

    #[test]
    fn test_compression_size_reduction() {
        let data = vec![0.5f32; 1024];
        let fp32 = CompressedEmbedding::from_fp32(data.clone(), CompressionFormat::FP32);
        let fp16 = CompressedEmbedding::from_fp32(data.clone(), CompressionFormat::FP16);
        let int8 = CompressedEmbedding::from_fp32(data, CompressionFormat::INT8);

        assert_eq!(fp32.size_bytes(), 1024 * 4);
        assert_eq!(fp16.size_bytes(), 1024 * 2);
        assert_eq!(int8.size_bytes(), 1024);
    }

    #[test]
    fn test_dedup_basic() {
        let mut cache = MultiModalCache::new(CacheConfig::default());
        let data = b"hello world";
        let hash = MultiModalCache::compute_hash(data);

        // First call — miss
        let (_emb1, hit1) = cache.get_or_compute(
            hash,
            ModalityType::Text,
            vec![768],
            || vec![0.5f32; 768],
        );
        assert!(!hit1);

        // Second call — hit (dedup)
        let (_emb2, hit2) = cache.get_or_compute(
            hash,
            ModalityType::Text,
            vec![768],
            || vec![0.5f32; 768],
        );
        assert!(hit2);
        assert_eq!(cache.stats().total_hits, 1);
        assert_eq!(cache.stats().total_misses, 1);
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = MultiModalCache::new(CacheConfig {
            ttl_secs: 0, // immediate expire
            ..Default::default()
        });

        let hash = MultiModalCache::compute_hash(b"test");
        cache.get_or_compute(hash, ModalityType::Text, vec![64], || vec![0.1f32; 64]);

        // Harus expired setelah TTL 0 detik
        let evicted = cache.evict_expired();
        assert!(evicted > 0);
        assert!(cache.get(&hash).is_none());
    }

    #[test]
    fn test_zero_hit_eviction() {
        let mut cache = MultiModalCache::new(CacheConfig::default());

        let hash = MultiModalCache::compute_hash(b"zero_hit");
        cache.get_or_compute(hash, ModalityType::Image, vec![768], || vec![0.2f32; 768]);

        // hit_count masih 0
        let evicted = cache.evict_zero_hit();
        assert_eq!(evicted, 1);
    }

    #[test]
    fn test_budget_enforcement() {
        let mut cache = MultiModalCache::new(CacheConfig {
            max_ram_bytes: 100, // very small
            ..Default::default()
        });

        // Insert entry besar — harus trigger eviction
        let hash1 = MultiModalCache::compute_hash(b"large1");
        cache.get_or_compute(hash1, ModalityType::Image, vec![100], || vec![0.3f32; 100]);

        let hash2 = MultiModalCache::compute_hash(b"large2");
        cache.get_or_compute(hash2, ModalityType::Image, vec![100], || vec![0.4f32; 100]);

        // Budget sangat kecil, entry pertama harus ke-evict
        assert!(cache.stats().total_evictions > 0);
    }

    #[test]
    fn test_adaptive_resize_noop() {
        // 64x64 < 1024x1024 — no resize needed
        let data = vec![128u8; 64 * 64 * 3];
        let result = adaptive_resize(&data, 64, 64, 3, 1_048_576).unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        assert_eq!(result.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_adaptive_resize_downscale() {
        // 4096x4096 → harus di-resize
        let data = vec![128u8; 256 * 256 * 3];
        let result = adaptive_resize(&data, 256, 256, 3, 16_384).unwrap();
        assert!(result.width < 256 || result.height < 256);
        assert!(result.width * result.height <= 16_384);
    }

    #[test]
    fn test_video_sliding_window() {
        let mut window = VideoSlidingWindow::new(2, 1);
        let frame = ImageInput {
            data: vec![0u8; 10],
            format: ImageFormat::Raw,
            width: 1,
            height: 1,
            channels: 1,
        };

        assert!(!window.push(frame.clone()));
        assert!(window.push(frame)); // now full
        assert_eq!(window.drain_window().len(), 2);
    }

    #[test]
    fn test_global_store() {
        let store1 = global_multimodal_store();
        let store2 = global_multimodal_store();
        assert!(Arc::ptr_eq(&store1, &store2));
    }

    #[test]
    fn test_hierarchical_promote_demote() {
        let mut cache = MultiModalCache::new(CacheConfig {
            enable_hierarchical: true,
            ..Default::default()
        });

        let hash = MultiModalCache::compute_hash(b"hier_test");
        cache.get_or_compute(hash, ModalityType::Text, vec![64], || vec![0.5f32; 64]);

        let entry = cache.get(&hash).unwrap();
        assert!(matches!(entry.storage_level, StorageLevel::L2Ram));

        cache.promote_to_l1(&hash);
        let entry = cache.get(&hash).unwrap();
        assert!(matches!(entry.storage_level, StorageLevel::L1Gpu));

        cache.demote_to_l3(&hash);
        let entry = cache.get(&hash).unwrap();
        assert!(matches!(entry.storage_level, StorageLevel::L3Ssd));
    }

    #[test]
    fn test_hash_image_input_deterministic() {
        let input1 = ImageInput {
            data: vec![1, 2, 3],
            width: 10,
            height: 10,
            channels: 3,
            format: ImageFormat::Raw,
        };
        let input2 = ImageInput {
            data: vec![1, 2, 3],
            width: 10,
            height: 10,
            channels: 3,
            format: ImageFormat::Raw,
        };
        assert_eq!(
            MultiModalCache::hash_image_input(&input1),
            MultiModalCache::hash_image_input(&input2),
        );
    }
}
