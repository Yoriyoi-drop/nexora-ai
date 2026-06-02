# Batch 33 Plan — Remaining Critical Issues

**Tanggal**: 2 Juni 2026
**Base branch**: main
**Target**: 6 remaining critical issues dari audit

---

## Issues

### M-3: KV Cache Defrag No Remap (CRITICAL)
**File**: `crates/inference/src/paged_cache.rs:1128-1211`
**Root Cause**: `defragment()` moves data between physical blocks but never updates sequence `BlockTable.layers` entries. Blocks with `ref_count == 0` are selected but after compaction, sequences still reference the old physical block index. Additionally, line 1185-1186 has a logic bug where both `src` and `dst` overwrite `partial_indices[i]`.
**Fix**: 
- After moving data, iterate all sequences' block tables per layer and remap physical indices
- Correct `ref_count` on affected blocks after remap
- Fix the `partial_indices` overlapping assignment bug
**Effort**: Medium (~4-6 jam)
**Files**: `crates/inference/src/paged_cache.rs`

### M-5: Causal Mask CPU GQA Forward (CRITICAL)
**File**: `crates/transformer/src/gqa.rs:922-1074, 1076-1212`
**Root Cause**: Both CPU forward paths (`forward()` and `forward_with_kv()`) compute attention scores over ALL KV cache positions with no causal mask. Future positions are not masked out, breaking causality during autoregressive generation.
**Fix**: 
- Add `current_pos` parameter to `forward()` signature
- Before softmax, set scores for positions `t > current_pos` to `f32::NEG_INFINITY`
- In `forward_with_kv()`, derive current position from `total_seq` and mask accordingly
**Effort**: Small (~1-2 jam)
**Files**: `crates/transformer/src/gqa.rs`

### M-7: All Encoders Placeholder (CRITICAL)
**Files**: `crates/multimodal/src/caffeine/encoders/{image,audio,video,text,regional}*.rs`
**Root Cause**: None of the 5 encoders contain real neural networks. All use hand-written heuristics, fixed sine/cosine math, or element-wise operations with zero trained parameters. `load_model()` just sets `model_loaded = true`.
**Fix**: Implement real shallow MLP projection for each encoder:
- Image: Conv1x1 patch projection + 2-layer MLP (ViT patch embed analog)
- Audio: Mel filterbank + 2-layer MLP 
- Text: Learned token embedding table + 2-layer transformer block
- Video: Per-frame image encoder + temporal mean pool + 1-layer MLP
- Regional alignment: Real scaled dot-product attention + projection
**Effort**: Large (but can be scoped to shallow MLPs — not full CLIP/Whisper) (~1-2 hari)
**Files**: All 5 files under `crates/multimodal/src/caffeine/encoders/`

### M-10: SharedOracleMemory Singleton (CRITICAL)
**File**: `crates/oracle/src/shared_memory.rs:171-188, 226`
**Root Cause**: Global `OnceLock<Mutex<SharedOracleMemory>>` singleton is shared process-wide. Tests that use it interfere with each other. No `reset()` function exists. Test `test_stats` has assertion bug (`total_misses == 0` after miss).
**Fix**:
- Add `global_reset()` function using `OnceLock` internal mutation
- Fix `test_stats` assertion
- OR replace with dependency injection pattern
**Effort**: Small (~1-2 jam)
**Files**: `crates/oracle/src/shared_memory.rs`, callers of `global_oracle_memory()`

### M-11: MLA Concatenate Heads Shape Mismatch (CRITICAL)
**File**: `crates/oracle/src/backbone.rs:396-422, 444, 472`
**Root Cause**: `LatentAttentionHead::forward()` outputs `(batch, seq, latent_dim=512)` at line 472, but `concatenate_heads()` assumes each head outputs `(batch, seq, head_dim=128)`. Creates buffer of wrong size; slice assignment will panic.
**Fix**: Change `out_proj` output dim to `head_dim` (not `latent_dim`), and fix downstream reshape to use `head_dim * n_heads` (which equals `d_model`).
**Effort**: Small (~1 jam)
**Files**: `crates/oracle/src/backbone.rs`

### M-13: All Format Loaders Vec OOM (CRITICAL)
**File**: `crates/datastream/src/format_loader.rs`
**Root Cause**: Every loader returns `Vec<DataSample>` after loading entire file into memory. No streaming/chunking. For large datasets, RAM usage is 2-3x file size.
**Fix**: 
- Add `load_dataset_streaming()` returning `Box<dyn Iterator<Item=Result<DataSample>>>` 
- Refactor CSV/JSONL to line-by-line iteration
- Keep existing `load_dataset()` for backward compat
**Effort**: Large (~1-2 hari)
**Files**: `crates/datastream/src/format_loader.rs`, callers

---

## Scope

### Must Have (BF33)
- M-5: Causal mask CPU GQA forward
- M-10: SharedOracleMemory singleton
- M-11: MLA concatenate heads shape

### Should Have
- M-3: KV Cache defrag remap (higher risk of regression)

### Could Have (if time permits)
- M-7: Shallow MLP encoders
- M-13: Streaming format loaders

### Verification
- `cargo check` zero errors
- `cargo clippy` clean
- `cargo nextest run` for affected crates

---

## Risk Assessment
- M-3: HIGH — touches core paged cache, potential regression in KV cache correctness
- M-5: LOW — well-understood pattern (add mask before softmax)
- M-7: LOW — encoders are already unused/fake, any improvement is additive
- M-10: LOW — tests only, production path unchanged
- M-11: LOW — shape fix, currently panics so any change is improvement
- M-13: MEDIUM — API change may affect callers
