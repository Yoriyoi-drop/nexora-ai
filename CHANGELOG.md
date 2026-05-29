# Changelog

## 0.2.0

### Version Bump

- Upgraded entire workspace from 0.1.0 to 0.2.0
- All 40 Cargo.toml files updated (workspace + 39 crates)

### Phase 4 — Native Specialized Systems (Complete)

All 10 model crates wired to real subsystems:

| Crate | Wiring |
|-------|--------|
| Omnis | MoE gating (`has-moe-ffn` Router) |
| Aether | CaffeineProcessor text pipeline + emotion classifier |
| Axiom | Full SACA 6-phase reasoning pipeline |
| Spectra | CaffeineProcessor multimodal pipeline |
| Vortex | Oracle CodeVerifierManager (4 rule-based verifiers) |
| Cipher | Oracle security verifier integration |
| Kronos | SACA temporal reasoning |
| Swift | MoE Router for latency-aware dispatch |
| Genesis | Multi-iteration quality refinement with SACA feedback |
| Nexum | Oracle verifier + SACA reasoning for task decomposition |

### Phase 5a — Memory Architecture (Paged Cache + Prefix DAG)

- `PagedKVCacheProvider` wrapping shared block-based paged cache
- Block-level prefix sharing with copy-on-write
- Continuous batching engine integration
- Config toggle: `use_paged_cache`, `paged_block_size`, `paged_max_blocks`

### Phase 5b — GPU Backend Auto-Detection (CUDA + wgpu)

- `GpuBackend::Wgpu` / `GpuBackend::Cuda` auto-detection at init
- CUDA: 18+ ops (cuBLAS matmul, NVRTC softmax, broadcast add, transpose, gelu_inplace)
- MoE Router and Expert with CUDA forward paths (fallback chain: CUDA → wgpu → CPU)
- `cuda` feature flag (separate from `gpu` / wgpu)

## 0.1.0

### Panic Safety (Biggest change)

Eliminated all production `.unwrap()` calls across the entire workspace (~500+ points):

- **memory** (161): lock poison, HashMap access, tensor reshape, option unwrap
- **foundation** (262): tensor reshape in checkpoint loading, Regex compilation, HashMap access, slice bounds
- **core** (43): mutex poison, HashMap access, runtime creation
- **deeplearning** (130+): tensor contiguity (`as_slice`/`as_slice_mut`), shape validation (`from_shape_vec`, `into_dimensionality`), mutex poison
- **datastream** (37): Regex compilation safety
- **infrastructure** (25): Regex, float comparison (`partial_cmp`), system time, mutex poison
- **api** (6): route build, middleware
- **isolation** (6): HashMap access, slice access
- **database** (3): timestamp conversion, connection pool
- **data** (2): system time, float comparison
- **tokenizer** (1): trie access

### Features

- `Serialize`/`Deserialize` on `InferenceRequest`, `InferenceResponse`, `GeneratedToken`, `FinishReason`
- API integration tests (8 tests: health, metrics, echo, status, 404, CORS, concurrent, response time)
- rust-toolchain.toml (channel = "stable")
- Dockerfile (multi-stage build with cargo-chef, debian runtime)
- ARCHITECTURE.md

### Fixes

- Removed `#![allow(deprecated)]` from foundation, fixed `uninitialized` → `uninit()` and `scalar_sum` → `sum`
- Fixed 14 clippy warnings in deeplearning crate
- Fixed ambiguous glob re-exports (`MemoryEntry` → `EmrMemoryEntry`, `utils` → `core_utils`)
- Fixed `static mut GLOBAL_BLAS` → `OnceLock` for Rust 2024 compatibility
- Removed excessive `#![allow()]` from memory crate
- Fixed dropping_references warnings in memory/cache.rs and lru_memory.rs
- Fixed mixed_script_confusables warning in memory/coherence.rs (Greek θ → theta)

### Infrastructure

- `Cargo.lock` in `.gitignore` (not committed)
- `deny.toml`, `clippy.toml`, `rustfmt.toml`, `Makefile` added
