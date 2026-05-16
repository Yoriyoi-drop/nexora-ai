# Changelog

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
