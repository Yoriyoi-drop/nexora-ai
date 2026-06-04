# Plan Split File `.rs` Besar

## Ringkasan

Dari 15 file `.rs` terbesar (total ~43.000 baris), 11 file **layak dipisah**, 4 file **tidak layak dipisah** sekarang.

| # | File | Baris | Layak Split? | Target File |
|---|------|-------|-------------|-------------|
| 1 | `autograd/src/gpu/gpu_context.rs` | 7.418 | ✅ | 5 file baru |
| 2 | `transformer/src/model.rs` | 3.983 | ✅ | 4 file baru |
| 3-6 | `vortex/agents/*.rs` (4 file) | 12.645 | ❌ | Shared types + macro |
| 7 | `apps/nexora-ai/src/cli/training.rs` | 2.683 | ✅ | 5 file |
| 8 | `inference/src/paged_cache.rs` | 2.568 | ✅ | 4 file |
| 9 | `transformer/src/gqa.rs` | 2.378 | ✅ | 3 file |
| 10 | `inference/src/continuous_batching.rs` | 2.152 | ✅ | 4 file |
| 11 | `models/src/axiom/config.rs` | 2.086 | ✅ | 8 file |
| 12 | `inference/src/engine.rs` | 2.053 | ✅ | 5 file |
| 13 | `autograd/src/gpu_sedc.rs` | 1.982 | ✅ | 5 file |
| 14 | `nexum/agents/resource_optimizer.rs` | 1.876 | ✅ | 2 file |
| 15 | `autograd/src/ops/nn.rs` | 1.617 | ❌ | Utuh |

---

## Prioritas Eksekusi

### Iterasi 1 — Risiko Rendah, Manfaat Tinggi

#### 1A. `autograd/src/ops/nn.rs` — TETAP UTUH
**Alasan**: 11 fungsi dengan pola identik (GPU path → CPU path → grad), interdependent (softmax dipanggil oleh log_softmax, causal_softmax, causal_attention, cross_entropy_loss). Split akan tambah boilerplate tanpa kurangi kompleksitas.

#### 1B. `autograd/src/gpu/gpu_context.rs` (7.418 → ~1.200 baris)
**Strategi**: Extension trait pattern. Method dipindah ke file terpisah, dipanggil via `use GpuContextMatmulExt` agar `ctx.matmul(...)` tetap work.

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `gpu/wgsl_shaders.rs` | ~2.300 | 30+ `const FOO_WGSL: &str` — paling aman, 0 dependensi |
| `gpu/ops_matmul.rs` | ~900 | `matmul`, `matmul_f16`, `matmul_int8`, `matmul_int8_weight`, `matmul_int4_weight` + backward |
| `gpu/ops_norm_attention.rs` | ~1.600 | softmax, rms_norm, layer_norm, cross_entropy, embedding, transpose, fused_attention + backward |
| `gpu/ops_util.rs` | ~1.800 | fill_zero, scale, l2_norm, causal_softmax, gradient_clip, gradient_allreduce, moe_scatter_add, elementwise_unary/binary, reduce, dropout, temperature/top-k/top-p, multinomial, rotary_embedding, repeat_heads |
| `gpu/dispatch_profiling.rs` | ~800 | dispatch, dispatch_profiled, dispatch_batch, timestamp queries, batch_dispatch |
| `gpu/gpu_context.rs` (sisa) | ~1.200 | Init, pipeline compilation, cache, singleton, recovery, encoder management |

**Langkah**: `wgsl_shaders.rs` dulu (0 breakage) → `dispatch_profiling.rs` (0 deps ke ops) → `ops_matmul.rs` → `ops_norm_attention.rs` → `ops_util.rs`.

#### 1C. `inference/src/paged_cache.rs` (2.568 → ~900 baris)
**Strategi**: Sub-direktori `paged_cache/` dengan mod.rs.

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `paged_cache/config.rs` | ~160 | `PagedCacheConfig`, `EvictionPolicy`, `MemoryTier` |
| `paged_cache/block.rs` | ~430 | `BlockData`, `PhysicalBlock`, `SeqAccess`, `BlockTable` |
| `paged_cache/cache.rs` | ~900 | `PagedKVCache` struct + core operations |
| `paged_cache/fragmentation.rs` | ~200 | `PagedCacheStats`, defragment, `init_global_paged_cache` |
| `paged_cache/mod.rs` | ~10 | Re-export |

---

### Iterasi 2 — Risiko Sedang

#### 2A. `transformer/src/gqa.rs` (2.378 → ~800 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `transformer/src/kv_cache.rs` | ~660 | `KVCacheProvider` trait, `CpuKVCache`, `GpuKVCache`, `KVCacheEntry`, `GpuKVCacheEntry`, `PagedCacheReader` |
| `transformer/src/gqa_gpu.rs` | ~710 | GPU forward variants (cfg-gated) |
| `transformer/src/gqa.rs` (sisa) | ~800 | `GQA` struct, CPU forward, init, weight mgmt |

**Catatan**: `kv_cache.rs` independen — bisa diekstrak duluan.

#### 2B. `transformer/src/model.rs` (3.983 → ~700 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `transformer/src/gpu_forward.rs` | ~1.500 | Semua `forward_gpu_*`, `forward_gpu_batched_*` (cfg-gated) |
| `transformer/src/generation.rs` | ~400 | `generate*`, `sample_token*` |
| `transformer/src/checkpoint.rs` | ~500 | `from_checkpoint*`, `save_checkpoint`, `readback_weights` |
| `transformer/src/compression.rs` | ~120 | `collect_weights_for_sedc`, `compress_sedc_*` |
| `transformer/src/model.rs` (sisa) | ~700 | `CausalLM` struct, `new`, CPU forward, utilities |

#### 2C. `inference/src/continuous_batching.rs` (2.152 → ~1.075 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `batching/config.rs` | ~115 | `ContinuousBatchingConfig`, `SchedulingPolicy` |
| `batching/prefix_trie.rs` | ~60 | `PrefixTrie` struct + impl |
| `batching/step_result.rs` | ~35 | `StepResult` struct |
| `batching/engine.rs` | ~1.075 | `ContinuousBatchingEngine` struct + semua method |
| `batching/mod.rs` | ~10 | Re-export |

---

### Iterasi 3 — Risiko Tertinggi (Public API)

#### 3A. `inference/src/engine.rs` (2.053 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `engine/config.rs` | ~100 | `InferenceConfig` |
| `engine/state.rs` | ~30 | `EngineState`, `RequestStatus`, `EngineStats` |
| `engine/handle.rs` | ~230 | `InferenceEngineHandle` |
| `engine/inference.rs` | ~1.050 | `submit_request`, `submit_streaming_request`, `generate_internal`, `generate_continuous_batched` |
| `engine/lifecycle.rs` | ~170 | `cancel`, `session`, `shutdown`, `stats` |
| `engine/mod.rs` | ~200 | `InferenceEngine` struct, `new`, `with_model`, `initialize` |

#### 3B. `apps/nexora-ai/src/cli/training.rs` (2.683 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `cli/training/checkpoint.rs` | ~150 | `CheckpointManager` + helpers |
| `cli/training/metrics.rs` | ~100 | `MetricsAccumulator` + reporting |
| `cli/training/infra.rs` | ~100 | `init_gpu`, ANSI helpers, thread config |
| `cli/training/train.rs` | ~1.250 | `run_train()` |
| `cli/training/evaluate.rs` | ~1.030 | `run_evaluate()` |
| `cli/training/mod.rs` | ~15 | Re-export |

#### 3C. `autograd/src/gpu_sedc.rs` (1.982 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `sedc/config.rs` | ~120 | `SedcError`, `SedcConfig`, report types |
| `sedc/vet.rs` | ~80 | CPU algorithms: spectral_entropy, vet, egss |
| `sedc/shaders.rs` | ~260 | 8 WGSL shader constants |
| `sedc/gpu_ops.rs` | ~835 | `impl GpuContext` SEDC methods |
| `sedc/compressor.rs` | ~480 | `SedcCompressor`, `CompressedWeight`, high-level API |
| `sedc/mod.rs` | ~160 | Re-export + tests |

#### 3D. `models/src/axiom/config.rs` (2.086 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `config/mod.rs` | ~200 | `AxiomConfig`, `ConfigurationSummary`, `PerformanceFeedback` |
| `config/logical.rs` | ~300 | `LogicalReasoningConfig` + 8 enum turunan |
| `config/math.rs` | ~250 | `MathematicalReasoningConfig` + 7 enum |
| `config/proof.rs` | ~250 | `ProofGenerationConfig` + `ProofVerificationConfig` |
| `config/inference.rs` | ~200 | `InferenceEngineConfig`, `MemoryManagementConfig` |
| `config/knowledge.rs` | ~200 | `KnowledgeBaseConfig`, `KnowledgeSource` |
| `config/performance.rs` | ~200 | `PerformanceConfig`, `ParallelProcessing` |
| `config/resources.rs` | ~200 | `ResourceConfig`, `MemoryConfig`, `CPUConfig` |

#### 3E. `nexum/agents/resource_optimizer.rs` (1.876 baris)

| File Baru | Baris | Isi |
|-----------|-------|-----|
| `resource_optimizer/types.rs` | ~1.260 | Semua struct/enum + `impl Default` |
| `resource_optimizer/agent.rs` | ~500 | `ResourceOptimizerAgent` + `BaseAgent` impl |
| `resource_optimizer/mod.rs` | ~120 | Re-export + tests |

---

## Tidak Layak Dipisah

### Vortex Agents (4 file, ~12.645 baris total)
- **75% boilerplate type definitions** (struct/enum + impl Default)
- Pola identik: 1 struct utama → 4 config struct → `BaseAgent` impl
- Shared type names `SeverityLevel`, `DetailedFinding` muncul di 2-3 file
- `arch_weaver` import dari `code_sentinel::SeverityLevel` dan `debug_phantom::DetailedFinding`
- **Rekomendasi**: Buat `vortex_shared_types.rs` untuk shared types, buat macro `impl_default_nested!` untuk boilerplate Default. Tapi split jenis agent ke sub-direktori akan TAMBAH kompleksitas tanpa kurangi LOC.

### `autograd/src/ops/nn.rs` (1.617 baris)
- 11 fungsi dengan pola identik (GPU path → CPU path → grad)
- Interdependent: softmax dipanggil oleh 4 fungsi lain
- Setiap fungsi ~147 baris — tidak ekstrem

---

## Ringkasan Dampak

| Metrik | Sebelum | Sesudah |
|--------|---------|---------|
| File terbesar | 7.418 baris | ~1.200 baris |
| File >2.000 baris | 10 file | 3 file (vortex agents) |
| Total LOC | ~43.000 | ~44.500 (+3.5% boilerplate) |
| Jumlah file baru | — | ~40 file |
| Risiko breakage | — | Rendah-Sedang (extension trait pattern) |

## Urutan Eksekusi yang Direkomendasikan

```
Minggu 1:
  ├── 1A. gpu_context.rs → wgsl_shaders.rs (hari 1, 0 breakage)
  ├── 1B. paged_cache.rs → sub-direktori (hari 1-2)
  └── 1C. gpu_context.rs → dispatch + ops (hari 2-3)

Minggu 2:
  ├── 2A. gqa.rs → kv_cache.rs (hari 1)
  ├── 2B. gqa.rs → gqa_gpu.rs (hari 1-2)
  ├── 2C. model.rs → gpu_forward.rs (hari 2-3)
  └── 2D. model.rs → generation + checkpoint (hari 3-4)

Minggu 3:
  ├── 3A. continuous_batching.rs → sub-direktori (hari 1-2)
  ├── 3B. gpu_sedc.rs → sub-direktori (hari 2-3)
  └── 3C. resource_optimizer.rs → types + agent (hari 3)

Minggu 4:
  ├── 4A. axiom/config.rs → 8 domain file (hari 1-2)
  ├── 4B. engine.rs → sub-direktori (hari 2-3)
  └── 4C. training.rs → sub-direktori (hari 3-4)
```

Setiap langkah harus diakhiri dengan `cargo check` dan `cargo test` untuk memastikan 0 breakage sebelum lanjut.
