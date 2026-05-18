//! PagedAttention KV Cache — block-based key-value cache manager.
//!
//! Mengelola memory KV cache dalam blok-blok fixed-size untuk:
//! - Menghindari fragmentasi memori (vs growable Vec per sequence)
//! - Pre-allocation contiguous memory
//! - Potensi sharing antar sequence (copy-on-write)
//! - Eviction terprediksi (per-block, bukan per-entry)

use std::collections::HashMap;
use ndarray::{Array2, s};

use nexora_foundation::models::transformer::KVCacheEntry;
use tracing::warn;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Default number of tokens per block
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Default max number of blocks to allocate
pub const DEFAULT_MAX_BLOCKS: usize = 65536;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the paged KV cache
#[derive(Clone, Debug)]
pub struct PagedCacheConfig {
    /// Number of KV tokens per physical block
    pub block_size: usize,
    /// Maximum number of physical blocks
    pub max_blocks: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of KV heads (after GQA reduction)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Max tokens per sequence (prevents OOM)
    pub max_seq_len: usize,
}

impl Default for PagedCacheConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            max_blocks: DEFAULT_MAX_BLOCKS,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
        }
    }
}

impl PagedCacheConfig {
    /// Total memory in bytes for all blocks (f32 = 4 bytes)
    pub fn total_memory_bytes(&self) -> usize {
        let per_block = self.block_size * self.num_kv_heads * self.head_dim * 4 * 2; // K + V
        self.max_blocks * per_block * self.num_layers
    }

    /// Memory per block in bytes
    pub fn block_memory_bytes(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim * 4 * 2
    }

    /// Max blocks needed for one sequence of max_seq_len
    pub fn blocks_per_sequence(&self) -> usize {
        (self.max_seq_len + self.block_size - 1) / self.block_size
    }
}

// ─── Physical Block ──────────────────────────────────────────────────────────

/// A physical block holds KV data for one layer.
/// Shape: `(block_size, num_kv_heads * head_dim)` per K dan V.
struct PhysicalBlock {
    /// Key data: [block_size, kv_heads * head_dim]
    k: Array2<f32>,
    /// Value data: [block_size, kv_heads * head_dim]
    v: Array2<f32>,
    /// Number of valid tokens in this block (last block may be partial)
    filled: usize,
    /// Reference count for copy-on-write sharing
    ref_count: usize,
}

impl PhysicalBlock {
    fn new(config: &PagedCacheConfig) -> Self {
        let cols = config.num_kv_heads * config.head_dim;
        Self {
            k: Array2::zeros((config.block_size, cols)),
            v: Array2::zeros((config.block_size, cols)),
            filled: 0,
            ref_count: 1,
        }
    }

    fn is_free(&self) -> bool {
        self.ref_count == 0
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

/// Block-based KV cache manager.
///
/// Alokasi memori dalam fixed-size blocks, bukan growable Vec per sequence.
/// Setiap sequence punya BlockTable yang memetakan logical → physical blocks.
pub struct PagedKVCache {
    config: PagedCacheConfig,
    /// Per-layer physical blocks
    blocks: Vec<Vec<PhysicalBlock>>,
    /// Indices of free (unused) physical blocks per layer
    free_lists: Vec<Vec<usize>>,
    /// Block table per sequence ID
    sequences: HashMap<u64, BlockTable>,
    /// Stats
    pub num_allocated: usize,
    pub num_freed: usize,
    pub max_used: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache with the given config
    pub fn new(config: PagedCacheConfig) -> Self {
        let num_layers = config.num_layers;
        let max_blocks = config.max_blocks;

        let blocks: Vec<Vec<PhysicalBlock>> = (0..num_layers)
            .map(|_| Vec::with_capacity(max_blocks))
            .collect();
        let free_lists: Vec<Vec<usize>> = (0..num_layers).map(|_| Vec::with_capacity(max_blocks / 4)).collect();

        Self {
            config,
            blocks,
            free_lists,
            sequences: HashMap::new(),
            num_allocated: 0,
            num_freed: 0,
            max_used: 0,
        }
    }

    /// Register a new sequence and return its block table
    pub fn register_sequence(&mut self, seq_id: u64) {
        let table = BlockTable::new(self.config.num_layers, self.config.block_size);
        self.sequences.insert(seq_id, table);
    }

    /// Remove a sequence and free all its blocks
    pub fn remove_sequence(&mut self, seq_id: u64) {
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

    /// Check if a sequence is registered
    pub fn has_sequence(&self, seq_id: u64) -> bool {
        self.sequences.contains_key(&seq_id)
    }

    /// Allocate a new physical block from the free list or create one
    fn alloc_block(&mut self, layer: usize) -> Option<usize> {
        if let Some(free_idx) = self.free_lists[layer].pop() {
            self.blocks[layer][free_idx].filled = 0;
            self.blocks[layer][free_idx].ref_count = 1;
            self.num_allocated += 1;
            return Some(free_idx);
        }

        let idx = self.blocks[layer].len();
        if idx >= self.config.max_blocks {
            warn!("PagedKVCache: max_blocks ({}) reached, cannot allocate", self.config.max_blocks);
            return None;
        }
        self.blocks[layer].push(PhysicalBlock::new(&self.config));
        self.num_allocated += 1;
        Some(idx)
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
    /// Allocates a new block if needed.
    fn get_or_alloc_block(
        &mut self,
        seq_id: u64,
        layer: usize,
        token_pos: usize,
    ) -> Option<usize> {
        let logical = token_pos / self.config.block_size;

        // Step 1: ensure block table has space (separate borrow)
        {
            let table = self.sequences.get_mut(&seq_id)?;
            while table.layers[layer].len() <= logical {
                table.layers[layer].push(None);
            }
            if let Some(phys) = table.layers[layer][logical] {
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
            // Sequence was removed between steps — free the block
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
        let block = &mut self.blocks[layer][phys];
        for c in 0..k_len {
            block.k[[offset, c]] = k_row[c];
        }
        for c in 0..v_len {
            block.v[[offset, c]] = v_row[c];
        }
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
        let k_slice: &[f32] = k_proj.as_slice().unwrap_or_else(|| {
            // Fallback: alokasi hanya jika tidak contiguous (jarang)
            let v: Vec<f32> = k_proj.iter().copied().collect();
            Box::leak(v.into_boxed_slice())
        });
        let v_slice: &[f32] = v_proj.as_slice().unwrap_or_else(|| {
            let v: Vec<f32> = v_proj.iter().copied().collect();
            Box::leak(v.into_boxed_slice())
        });
        self.append(seq_id, layer, token_pos, k_slice, v_slice);
    }

    /// Read KV data for a token position.
    pub fn read(&self, seq_id: u64, layer: usize, token_pos: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let table = self.sequences.get(&seq_id)?;
        let logical = token_pos / self.config.block_size;
        let offset = token_pos % self.config.block_size;
        let phys = table.layers.get(layer)?.get(logical).copied()??;
        let block = self.blocks.get(layer)?.get(phys)?;

        if offset >= block.filled {
            return None;
        }

        let cols = self.config.num_kv_heads * self.config.head_dim;
        let k = block.k.slice(s![offset, ..]).to_vec();
        let v = block.v.slice(s![offset, ..]).to_vec();
        Some((k, v))
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

        for layer in 0..self.config.num_layers {
            let mut k_flat = Array2::zeros((num_tokens, cols));
            let mut v_flat = Array2::zeros((num_tokens, cols));

            // Group by block and use slice assignment for contiguous regions
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
                            let k_rows = block.k.slice(s![offset..offset + valid, ..]);
                            k_flat.slice_mut(s![pos..pos + valid, ..]).assign(&k_rows);
                            let v_rows = block.v.slice(s![offset..offset + valid, ..]);
                            v_flat.slice_mut(s![pos..pos + valid, ..]).assign(&v_rows);
                        }
                    }
                }
                pos += tokens_in_block;
            }

            entries.push(KVCacheEntry {
                k: k_flat,
                v: v_flat,
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
        self.blocks.iter_mut().for_each(|b| b.clear());
        self.free_lists.iter_mut().for_each(|f| f.clear());
    }

    /// Memory usage estimate in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let per_block = self.config.block_size * self.config.num_kv_heads * self.config.head_dim * 4 * 2;
        self.total_blocks() * per_block
    }
}

impl nexora_foundation::models::transformer::PagedCacheReader for PagedKVCache {
    fn read(&self, seq_id: u64, layer: usize, token_pos: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        PagedKVCache::read(self, seq_id, layer, token_pos)
    }

    fn num_tokens(&self, seq_id: u64) -> Option<usize> {
        self.sequences.get(&seq_id).map(|t| t.num_tokens)
    }

    fn append(&mut self, seq_id: u64, layer: usize, token_pos: usize, k_row: &[f32], v_row: &[f32]) {
        PagedKVCache::append(self, seq_id, layer, token_pos, k_row, v_row);
    }
}

// ─── Statistics ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct PagedCacheStats {
    pub num_sequences: usize,
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub memory_bytes: usize,
    pub total_tokens: usize,
}

impl PagedKVCache {
    pub fn stats(&self) -> PagedCacheStats {
        let total = self.total_blocks();
        let free = self.free_blocks();
        let total_tokens: usize = self.sequences.values().map(|t| t.num_tokens).sum();
        PagedCacheStats {
            num_sequences: self.sequences.len(),
            total_blocks: total,
            free_blocks: free,
            used_blocks: total.saturating_sub(free),
            memory_bytes: self.memory_usage_bytes(),
            total_tokens,
        }
    }
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
                pos, expected_k0, k[0]
            );
            assert!(
                (v[0] - expected_v0).abs() < 1e-6,
                "pos={}: expected v[0]={}, got v[0]={}",
                pos, expected_v0, v[0]
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
        assert_eq!(flat[0].k.shape(), &[3, 8]); // 3 tokens, 8 cols
        assert_eq!(flat[0].v.shape(), &[3, 8]);
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
        let config = test_config();
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
        assert_eq!(cache2.memory_usage_bytes(), 2 * per_block);
    }

    #[test]
    fn test_append_token_from_array() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);

        let k_arr = Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32).collect()).unwrap();
        let v_arr = Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32 * 2.0).collect()).unwrap();

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
        use nexora_foundation::models::transformer::{CausalLM, TransformerConfig};

        let cfg = PagedCacheConfig {
            block_size: 4,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_blocks: 1024,
            max_seq_len: 512,
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
            let logits = model.forward_paged(&[token], &mut paged_cache, 1);
            assert_eq!(logits.len(), 256);
            if i > 0 {
                let got = paged_cache.num_tokens(1).unwrap_or(0);
                assert_eq!(got, i + 1, "after token {i}, cache should have {} entries", i + 1);
            }
        }

        assert_eq!(paged_cache.num_tokens(1), Some(3));
        assert!(paged_cache.total_blocks() > 0);
    }

    #[test]
    fn test_paged_vs_flat_cache_parity() {
        use ndarray::Array1;
        use nexora_foundation::models::transformer::{CausalLM, KVCacheEntry, TransformerConfig};

        let cfg = PagedCacheConfig {
            block_size: 4,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_blocks: 1024,
            max_seq_len: 512,
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
        let mut flat_cache: Vec<KVCacheEntry> = model.reset_cache();
        let mut paged_cache = PagedKVCache::new(cfg);
        paged_cache.register_sequence(1);

        let tokens = [5u32, 10u32, 15u32];
        for (i, &token) in tokens.iter().enumerate() {
            let logits_flat = model.forward(&[token], &mut flat_cache);
            let logits_paged = model.forward_paged(&[token], &mut paged_cache, 1);

            for j in 0..logits_flat.len() {
                let diff = (logits_flat[j] - logits_paged[j]).abs();
                assert!(
                    diff < 1e-4,
                    "mismatch at token {i}, logit {j}: flat={} paged={}",
                    logits_flat[j], logits_paged[j]
                );
            }
        }
    }
}
