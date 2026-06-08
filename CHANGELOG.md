# Changelog

## 0.4.0

### Batch Fix 34 — CVE & SACA Sandbox (8 Juni 2026)

**rustls 0.21→0.23 upgrade:**
- `crates/api/src/server.rs` TLS + `apps/nexora-ai/src/server/tls.rs`
- API breaking: `builder_with_provider()`, `pki_types::CertificateDer`/`PrivateKeyDer`, `rustls_pemfile::private_key()` v2
- tokio-rustls 0.24→0.26 sync

**SACA sandbox hardening:**
- `CodeExecutor` made async (`tokio::process::Command` replacing `std::process::Command`)
- `tokio::time::timeout()` enforcement on code execution
- Security re-validation after fix generator (`validate_code()` called again)
- `env_clear()` + 30s timeout in TestRunner

**CVE assessment:**
- protobuf 2.28.0 → only test dep (`proptest-derive`), not production risk
- `rustls-webpki` dual (0.101.7 + 0.103.13) still via reqwest 0.11 — reqwest 0.12 upgrade deferred (~16 files)

`cargo check` ✅ nexora-reasoning + nexora-ai

### Batch Fix 33 — Panic-to-Error 7 Locations (8 Juni 2026)

| Component | File | Issue | Fix |
|-----------|------|-------|-----|
| `MetricsCollector::default()` | `crates/monitoring/src/metrics.rs:369` | Panic on fallback | Registry with all 38 fields |
| `Storage::to_cpu()` GPU variant | `crates/autograd/src/device.rs:111` | Panic | warn + return empty array |
| `global_pool()` | `crates/star-x/src/tensor_pool.rs:300` | Panic on init fail | NOOP_TENSOR_POOL (LazyLock) |
| `unwrap()` | `crates/runtime/src/vram_budget.rs:298` | Panic | match + early return 0.0 |

`cargo check` ✅ nexora-ai, nexora-monitoring, nexora-star-x

### Batch Fix 32 — Security & Config Hardening (8 Juni 2026)

- **sqlx 0.7.2→0.8.0**: CVE fix, `default-features=false`, sqlite feature removed
- **rusqlite 0.29→0.31**: Resolved `libsqlite3-sys` dep conflict
- **sqlx removed from `crates/infrastructure`**: Not used (only `"sqlx=warn"` string)
- **nexora.toml hardening**: `host 0.0.0.0→127.0.0.1`, `enable_auth false→true`, `cors_origins ["*"]→[localhost:5173,8080]`, `rate_limit_rpm 1000→60`, `enable_tls false→true`
- **API key SHA-256 hashing**: `hash_api_key()` before persist, `sha2`+`hex` now non-optional
- **JWT HS256→RS256**: Primary RS256 + HS256 fallback, `jti` claim for revocation
- **Security headers middleware**: HSTS, X-Content-Type-Options, X-Frame-Options, Referrer-Policy, X-XSS-Protection
- **CI/CD**: All GitHub Actions pinned to SHA commits, CODEOWNERS file created

## 0.3.0

### Batch Fix 31 — Production Unwrap & Panic Comprehensive Cleanup (4 Juni 2026)

Audit komprehensif 41 crate membuktikan **0 `.unwrap()`/`panic!()` di production code**. Semua ~1.373 unwrap dan 33 panic yang dilaporkan sebelumnya ternyata berada di test code.

**23 production unwrap/panic real diperbaiki di 17 file:**

| Crate | Issue | Fix |
|-------|-------|-----|
| `has-moe-ffn` | 6 panic/expect di routing + experts | Return Result, warn fallback |
| `inference` | 3 unwrap di cold_storage + paged_cache | Option/Result propagation |
| `transformer` | 2 unwrap di lazy_weights + GQA GPU | expect(msg) + match |
| `datastream` | 4 Mutex lock unwrap | try_lock + warn fallback |
| `multimodal` | 2 unwrap di text_encoder + tokenizer | Result propagation |
| `training` | 11 unwrap di lora merge/unmerge | if let Some + skip |
| `foundation` | 1 expect di oracle mod | Result via ? |
| `star-x` | 1 unwrap di f16_storage | unwrap_or_else zeros |
| `infrastructure` | 1 panic di retry.rs | Return Err |
| `apps` | 2 unwrap di auth + billing | Destructure langsung |

`cargo check --all-targets` ✅ 0 errors, 0 warnings baru.

### Dead code verification

- `speculative_decoding.rs` + `token_loop.rs` — confirmed sudah tidak ada di disk

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

- `Cargo.lock` committed (not in `.gitignore`)
- `clippy.toml`, `Makefile` added
