use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array2;
use nexora_transformer::KVCacheEntry;
use tracing::{debug, info, warn};

use crate::cold_storage::{ColdStorage, ColdStorageConfig};
use crate::paged_cache::{
    BlockTable, EvictionPolicy, MemoryTier, PagedCacheConfig, PhysicalBlock, SeqAccess,
};
#[cfg(feature = "gpu")]
use nexora_autograd::gpu::GpuContext;
#[cfg(feature = "gpu")]
use nexora_autograd::gpu_kv_cache::GpuPageTable;

pub struct PagedKVCache {
    pub(crate) config: PagedCacheConfig,
    /// Per-layer physical blocks
    pub(crate) blocks: Vec<Vec<PhysicalBlock>>,
    /// Indices of free (unused) physical blocks per layer
    pub(crate) free_lists: Vec<Vec<usize>>,
    /// Block table per sequence ID
    pub(crate) sequences: HashMap<u64, BlockTable>,
    /// Per-sequence access tracking untuk eviction
    pub(crate) seq_access: HashMap<u64, SeqAccess>,
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
    pub(crate) cold_storage: Option<ColdStorage>,
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

        // Convert ke flat KVCacheEntry via read_layer_kv() (avoids to_flat_cache() 2GB alloc)
        let num_layers = self.config.num_layers;
        let mut entries = Vec::with_capacity(num_layers);
        let token_ids: Vec<u32> = (0..self.sequences.get(&seq_id).map(|t| t.num_tokens).unwrap_or(0))
            .map(|i| i as u32)
            .collect();
        let cols = self.config.num_kv_heads * self.config.head_dim;
        for layer in 0..num_layers {
            let (k, v) = match self.read_layer_kv(seq_id, layer) {
                Some(kv) => kv,
                None => {
                    entries.push(KVCacheEntry {
                        k: Vec::new(), v: Vec::new(), kv_dim: cols,
                        num_kv_heads: self.config.num_kv_heads, head_dim: self.config.head_dim,
                        k_compressed: Vec::new(), v_compressed: Vec::new(),
                        k_scales: Vec::new(), v_scales: Vec::new(), compressed_seq_len: 0,
                    });
                    continue;
                }
            };
            entries.push(KVCacheEntry {
                k, v,
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

        // Save to cold storage
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

    /// Proactive eviction — panggil saat VRAM forecast mengindikasikan OOM.
    /// Menggunakan `predicted_pressure()` logic: jika rasio alokasi tinggi,
    /// evict sequence yang paling idle sebelum OOM benar-benar terjadi.
    /// Returns jumlah sequence yang di-evict.
    pub fn evict_proactive(&mut self, target_free_blocks: usize) -> usize {
        let current_used = self.used_block_count();
        if current_used == 0 {
            return 0;
        }
        let effective_max = self.config.effective_max_blocks();
        let free_needed = target_free_blocks.min(effective_max / 4);
        let current_free = effective_max.saturating_sub(current_used);
        if current_free >= free_needed {
            return 0; // Already enough free space
        }
        let need_to_free = free_needed.saturating_sub(current_free);
        let blocks_per_seq = self.used_block_count() / self.seq_access.len().max(1);
        let seqs_to_evict = (need_to_free / blocks_per_seq.max(1)).max(1);

        // Prioritaskan: Cold → Warm → Hot berdasarkan idle time
        let now = Instant::now();
        struct Candidate {
            seq_id: u64,
            tier_label: &'static str,
            last_access: Instant,
        }
        let mut candidates: Vec<Candidate> = self.seq_access.iter()
            .filter(|(_, acc)| {
                now.duration_since(acc.created_at).as_secs_f64()
                    >= self.config.eviction_min_age_secs
            })
            .map(|(k, v)| Candidate {
                seq_id: *k,
                tier_label: v.tier.label(),
                last_access: v.last_access,
            })
            .collect();

        // Sort by: Cold first (tier), then LRU within tier
        candidates.sort_by(|a, b| {
            let tier_cmp = a.tier_label.cmp(b.tier_label).reverse(); // Cold first
            if tier_cmp != std::cmp::Ordering::Equal {
                return tier_cmp;
            }
            a.last_access.cmp(&b.last_access) // then LRU
        });

        let mut evicted = 0usize;
        for c in candidates.iter().take(seqs_to_evict) {
            tracing::info!(
                "Proactive eviction: seq {} (tier={:?}, blocks={})",
                c.seq_id,
                self.seq_access.get(&c.seq_id).map(|a| a.tier),
                self.sequences.get(&c.seq_id).map(|t| t.num_tokens).unwrap_or(0)
            );
            // Coba offload Cold ke disk dulu
            if self.config.enable_cold_disk_offload {
                let offloaded = self.offload_coldest_to_disk();
                if offloaded > 0 {
                    evicted += 1;
                    continue;
                }
            }
            self.remove_sequence(c.seq_id);
            self.num_evicted += 1;
            evicted += 1;
        }
        evicted
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
                    let filled_rows = self.blocks[layer][phys].filled;
                    let copy = self.blocks[layer][phys].deep_copy(filled_rows);
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

    /// Read ALL K/V data for a layer into flat buffers using block-level batch reads.
    /// Uses `slice_rows_into` for zero-intermediate-alloc bulk reads per block.
    /// This replaces per-token `read()` calls with O(1) allocations per layer.
    pub fn read_layer_kv(&self, seq_id: u64, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let table = self.sequences.get(&seq_id)?;
        let num_tokens = table.num_tokens;
        if num_tokens == 0 {
            return Some((Vec::new(), Vec::new()));
        }

        let cols = self.config.num_kv_heads * self.config.head_dim;
        let total = num_tokens * cols;
        let mut k_flat = Vec::with_capacity(total);
        let mut v_flat = Vec::with_capacity(total);

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
                        block.k.slice_rows_into(offset, valid, &mut k_flat);
                        block.v.slice_rows_into(offset, valid, &mut v_flat);
                    }
                }
            }
            pos += tokens_in_block;
        }

        Some((k_flat, v_flat))
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
            let mut k_flat = Vec::with_capacity(total);
            let mut v_flat = Vec::with_capacity(total);

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
                            block.k.slice_rows_into(offset, valid, &mut k_flat);
                            block.v.slice_rows_into(offset, valid, &mut v_flat);
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

    fn read_layer_kv(&self, seq_id: u64, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        PagedKVCache::read_layer_kv(self, seq_id, layer)
    }
}

// ─── Statistics & Fragmentation ──────────────────────────────────────────────

