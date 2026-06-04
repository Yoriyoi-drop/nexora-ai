use std::sync::Mutex;

use crate::paged_cache::PagedKVCache;
use crate::GLOBAL_PAGED_CACHE;

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
    #[test]
    fn test_dummy() {
        // Placeholder for paged_cache tests
    }
}
