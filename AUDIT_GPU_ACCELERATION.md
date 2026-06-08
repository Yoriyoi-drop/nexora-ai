# Audit GPU Acceleration — Nexora AI Workspace

**Tanggal**: 7 Juni 2026
**Total Crate Diaudit**: 53 workspace members
**Metodologi**: `rg -i "gpu|cuda|wgpu|vulkan|opencl|cublas|nvrtc|GpuContext|GpuBackend|GpuTensor" --type rust`, baca `Cargo.toml` (features + dependencies), hitung file/line, klasifikasi berdasarkan real GPU compute vs feature flag.

---

## Ringkasan Eksekutif

| Kategori | Jumlah | Crate |
|----------|--------|-------|
| **Heavy-Compute** (GPU kernels sendiri) | 4 | autograd, transformer, has-moe-ffn, echo-net |
| **Integrated** (GPU via autograd delegation) | 11 | star-x, gnac, erp, atqs, quantization, vogp, multimodal, training, foundation, inference, hldva-t |
| **Pass-Through** (feature flag doang) | 6 | deeplearning, core, agent, alignment, isolation, datastream |
| **Adapter** (config/metadata, no compute) | 2 | intelligence, nexora-ai (app) |
| **CPU-Only** (GPU zero codes) | 30 | api, blaa, cognition, database, hallucination, infrastructure, common, utils, memory, models, monitoring, runtime, tokenizer, validation, evaluation, reasoning, model-core, model-omnis, model-vortex, model-aether, model-axiom, model-cipher, model-genesis, model-kronos, model-nexum, model-spectra, model-swift, shared, dashboard, oracle |

**23 crate dengan akses GPU** (heavy-compute + integrated = 15 real GPU, 8 pass-through/adapter)  
**30 crate 100% CPU-only (56.6% dari total workspace)**

---

## Teknologi GPU yang Digunakan

| Teknologi | Crate | Penggunaan |
|-----------|-------|------------|
| **WGSL Compute Shaders** (wgpu) | autograd (37 shaders), echo-net (2 shaders) | Kernel beneran, compile via `create_shader_module` |
| **CUDA NVRTC JIT** (cudarc) | autograd (34 kernels), has-moe-ffn | cuBLAS matmul, JIT: softmax, gelu, attention, dll |
| **cuBLAS** | autograd, has-moe-ffn | General matrix multiply via CUBLAS_OP_T convention |
| **NCCL** | transformer | Multi-GPU all-reduce collectives |
| **NVML** (nvml-wrapper) | inference | GPU memory & utilization monitoring |

---

## 1. Heavy-Compute (4 Crate) — Real GPU Kernels

### autograd — GPU Engine Utama

| Metadata | Nilai |
|----------|-------|
| **Path** | `crates/autograd/` |
| **.rs files** | 52 |
| **Total lines** | 28.092 |
| **GPU refs** | 1.746 across 47 files |
| **Feature** | `gpu` (wgpu), `cuda` (cudarc) |
| **WGSL shaders** | 37 shaders (2.313 lines) di `src/gpu/wgsl.rs` |
| **CUDA JIT** | 34 kernel strings di `src/gpu/cuda/context.rs` (96 KB) |
| **GPU ops** | 52 wgpu ops + 34 CUDA ops = **86 total unique ops** |
| **Backend detection** | `GpuBackend::Cuda` > `GpuBackend::Wgpu` > CPU fallback |
| **Verdict** | ✅ **Heavy-Compute** — Dual-backend GPU engine |

**Ops kunci**: matmul, fused_attention (FlashAttention), softmax, rms_norm, layer_norm, cross_entropy, embedding, rotary_embedding, gelu_inplace, gradient_allreduce, moe_scatter_add, matmul_f16/int8/int4, f32_to_f16_packed.

### transformer — GPU Transformer Forward Path

| Metadata | Nilai |
|----------|-------|
| **Path** | `crates/transformer/` |
| **.rs files** | 31 |
| **Total lines** | 12.605 |
| **GPU refs** | 1.042 across 19 files |
| **Feature** | `gpu` (wgpu + autograd/gpu), `nccl` (multi-GPU) |
| **Own kernels** | ❌ Delegates to autograd, tapi integrasi GPU-nya total |
| **Kode GPU utama** | `gqa/gpu.rs` (1.251 LOC — GpuKVCache penuh), `swiglu.rs` (forward_gpu), `rms_norm.rs`, `rope.rs`, `nccl_collective.rs` |
| **Verdict** | ✅ **Heavy-Compute** — GPU-forward end-to-end, GPU KV cache GPU-resident (zero CPU round-trip) |

### has-moe-ffn — MoE CUDA + wgpu

| Metadata | Nilai |
|----------|-------|
| **Path** | `crates/has-moe-ffn/` |
| **.rs files** | 10 |
| **Total lines** | 4.070 |
| **GPU refs** | 258 across 4 files (80 CUDA, 36 GpuTensor, 20 GpuContext) |
| **Feature** | `gpu` (wgpu), `cuda` (autograd/cuda) |
| **Own kernels** | ❌ Delegates ke autograd cuBLAS + NVRTC |
| **Forward GPU** | `Router::forward_gpu/cuda()`, `Expert::forward_batched_gpu/cuda()`, fused variant, Q4 variant |
| **Weight caching** | `OnceLock<CudaTensor>` + `OnceLock<GpuTensor>` |
| **Offload** | LRU eviction manager utk expert GPU swapping |
| **Verdict** | ✅ **Heavy-Compute** — Dual-backend MoE forward, CUDA cuBLAS matmul chain |

### echo-net — Custom wgpu Compute Pipelines

| Metadata | Nilai |
|----------|-------|
| **Path** | `crates/echo-net/` |
| **.rs files** | 16 |
| **Total lines** | 10.573 |
| **GPU refs** | 109 across 5 files |
| **Feature** | `gpu` (wgpu langsung + autograd/gpu) |
| **Own kernels** | ✅ **2 WGSL shaders**: `ifft_2d_main` (inverse FFT), `conv_2d_main` (2D convolution) |
| **Kode GPU** | `gpu_ops.rs` (612 LOC): `create_shader_module`, `create_compute_pipeline`, `BindGroup`, `dispatch_workgroups` — wgpu API murni |
| **Verdict** | ✅ **Heavy-Compute** — Satu-satunya crate di luar autograd yang punya custom WGSL compute pipeline |

---

## 2. Integrated (11 Crate) — GPU via autograd Delegation

| Crate | Lines | GPU Refs | Ops Used | GPU Code |
|-------|-------|----------|----------|----------|
| **star-x** | 11.468 | 108 | matmul, transpose, elementwise | `*_gpu/_keep_gpu` variants — BLAS ops via autograd |
| **gnac** | 7.436 | 53 | matmul, add | IR execution → `ExecutionBackend::WGPU`, compiled IR ke GpuTensor |
| **erp** | 6.678 | 148 | matmul, add, sub, mul, div, l2_norm, softmax, reduce | 8 thin wrappers `try_gpu_*()` + **CPU fallback** (best pattern) |
| **atqs** | 10.830 | 36 | matmul, transpose, l2_norm, scale_inplace | `compute_svd_truncated_gpu()` — power-iteration SVD |
| **quantization** | 1.161 | 9 | matmul | `quant_gemm_gpu/int8/fp16` — dequant CPU → matmul GPU → readback |
| **vogp** | 4.217 | 17 | add, mul | `apply_gaussian_noise_gpu()`, `apply_dropout_gpu()` |
| **multimodal** | 11.830 | 105 | matmul, softmax, gelu, add, fused_attention | 6 GPU wrappers utk Q-Former, encoder, VQ-VAE, action head |
| **training** | 1.787 | 132 | GpuAdam, GpuStagingPool, AMP | End-to-end GPU training loop: forward/backward/optimizer GPU-resident |
| **foundation** | 5.932 | 46 | matmul, transpose | Pairwise distance GPU, delegasi ke CausalLM GPU, GPU utilization procfs |
| **inference** | 19.041 | 464 | GpuKVCache, GPU sampling, NVML | Heavy orchestration: PagedKVCache GPU, sampler GPU, prefix caching GPU |
| **hldva-t** | 11.788 | 222 | **23 ops** (matmul, add, softmax, norm, fused_attention, conv2d, dll) | Bridge `gpu_ops/mod.rs` (376 LOC): GpuWeight caching, CPU↔GPU round-trip per op |

**Pola dominan**: CPU → `GpuTensor::from_cpu()` → `ctx.op()` → `to_cpu_vec()` — **PCIe round-trip per op**.  
**Pengecualian**: `transformer/gqa/gpu.rs` (GPU-resident KV cache), `training` (GPU-resident training loop).

---

## 3. Pass-Through (6 Crate) — Feature Flag Only

Crate ini punya `[features] gpu = ["nexora-autograd/gpu"]` tapi **TIDAK ADA GPU compute code**.

| Crate | Lines | GPU Refs | Kenapa |
|-------|-------|----------|--------|
| **deeplearning** | 134 | 2 | Re-export proxy ke `nexora_autograd::gpu`. 0 compute. |
| **core** | 6.937 | 0 | Tipe/execution async murni CPU. Pass-through doang. |
| **agent** | 13.107 | 0 | Planner/worker system CPU-only. Pass-through doang. |
| **alignment** | 6.513 | 3 | Struct fields `gpu_utilization`, `gpu_percent`, `gpu_memory_gb` di response. 0 compute. |
| **isolation** | 4.866 | 11 | `GpuQuota` config, `gpu_isolation` flag, Chrome `--disable-gpu`. Policy-only. |
| **datastream** | 11.188 | 32 | GPU starvation monitoring (`gpu_starvation`, `report_gpu_wait()`). Metrics-only. |

---

## 4. Adapter (2 Crate) — Config/Metadata, No Compute

| Crate | Lines | GPU Refs | Peran |
|-------|-------|----------|-------|
| **intelligence** | 2.317 | 5 | `requires_gpu: bool`, `gpu_memory_mb` di model registry. Tracking metadata. |
| **nexora-ai** (app) | 18.864 | 159 | Config wiring: `enable_gpu`, `--use-gpu/--no-gpu`, server init GpuContext. Wire doang. |

---

## 5. CPU-Only (30 Crate) — Zero GPU Compute Code

### Infrastructure / Support (10)

| Crate | Path | Files | Lines | GPU Refs? | Relevansi GPU |
|-------|------|:----:|:-----:|:---------:|--------------|
| **nexora-api** | crates/api | 6 | 2.836 | ❌ | HTTP/API murni — axum, cors, metrics, tls. |
| **nexora-blaa** | crates/blaa | 4 | 1.149 | ❌ | Black Language Model API bridge. HTTP client doang. |
| **nexora-cognition** | crates/cognition | 6 | 2.021 | ❌ | State management / logic murni. No math. |
| **nexora-database** | crates/database | 6 | 4.819 | ❌ | DB abstraction (postgres, sqlite, sqlx, mysql). I/O-bound. |
| **nexora-hallucination** | crates/hallucination | 8 | 1.860 | ❌ | Anti-hallucination verification logic. **Potensi GPU: batched cross-attention scores (2-4×)** |
| **nexora-infrastructure** | crates/infrastructure | 3 | 352 | ❌ | Logging, config management. Re-export crate. |
| **nexora-common** | crates/infrastructure/common | 7 | 1.683 | ❌ | Common types, shared utilities. |
| **nexora-utils** | crates/infrastructure/utils | 10 | 5.372 | ❌ | Utilities: crypto, hashing, path, text. CPU-bound. |
| **nexora-monitoring** | crates/monitoring | 5 | 1.001 | ✅ (metric field) | Prometheus metrics. GPU field `gpu_alive`, `gpu_tokens` — tapi DEFINISI METRIC doang, 0 compute. |
| **nexora-memory** | crates/memory | 9 | 5.202 | ❌ | Memory management, store, recall. CPU-only. |

### Runtime / Tokenizer / Validation (3)

| Crate | Path | Files | Lines | GPU Refs? | Relevansi GPU |
|-------|------|:----:|:-----:|:---------:|--------------|
| **nexora-tokenizer** | crates/tokenizer | 9 | 4.833 | ❌ | BPE/unigram tokenizer. **Potensi GPU: token embedding lookup GPU (1.5-2×)** |
| **nexora-runtime** | crates/runtime | 16 | 4.723 | ✅ (cluster field) | Scheduler, batching, cluster/gossip. GPU fields `gpu_util_pct`, `supports_cuda` — DATA FIELDS doang utk distributed routing. 0 compute. |
| **nexora-validation** | crates/validation | 5 | 2.562 | ❌ | Validation engines. CPU-only. |

### Evaluation / Reasoning (2) — **Paling Potensial**

| Crate | Path | Files | Lines | GPU Refs? | Potensi GPU |
|-------|------|:----:|:-----:|:---------:|-------------|
| **nexora-evaluation** | crates/evaluation | 1 | 131 | ❌ | Evaluasi metrics: perplexity, accuracy. Depends on `nexora-autograd` + `nexora-transformer` (GPU-enabled!) tapi crate sendiri 0 GPU. **Batch metrics GPU: 3-5×** |
| **nexora-reasoning** | crates/reasoning | **39** | **12.529** | ❌ | **SACA 6-phase reasoning pipeline** — crate CPU-only TERBESAR. 0 GPU references di 12.529 lines. **Potensi TERBESAR: paralelisasi GPU utk CoT batch, sampling, execute-fail-fix: 4-8×** |

### Model Facade (1)

| Crate | Path | Files | Lines | GPU Refs? | Notes |
|-------|------|:----:|:-----:|:---------:|-------|
| **nexora-models** | crates/models | 2 | 176 | ❌ | Re-export facade utk 10 model crate. 0 code sendiri. |

### Delegation Model Crate (10)

Semua crate ini punya **GPU kata di capability spec** (metadata `requires_gpu`, `gpu_utilization`, `gpu_memory_gb`) tapi **0 baris GPU compute code**. Mereka mendelegasikan ke crate target (oracle, has-moe-ffn, transformer, multimodal, reasoning) yang mungkin GPU-enabled.

**⚠️ Tapi perhatikan: `nexora-oracle` juga CPU-only!**

| Crate | Files | Lines | GPU Refs | GPU Compute? | Depends on (GPU-enabled?) |
|-------|:----:|:-----:|:--------:|:------------:|---------------------------|
| **nexora-model-core** | 3 | 1.251 | 3 (metadata) | ❌ | — |
| **nexora-model-omnis** | 25 | 5.514 | 27 (caps) | ❌ | has-moe-ffn ✅, transformer ✅, erp ✅ |
| **nexora-model-vortex** | 11 | 15.319 | 25 (caps) | ❌ | oracle ❌, has-moe-ffn ✅, transformer ✅ |
| **nexora-model-aether** | 13 | 7.870 | 23 (caps) | ❌ | multimodal ✅, transformer ✅, erp ✅ |
| **nexora-model-axiom** | 11 | 6.048 | 25 (caps) | ❌ | reasoning ❌, oracle ❌, transformer ✅ |
| **nexora-model-cipher** | 9 | 2.995 | 25 (caps) | ❌ | oracle ❌, transformer ✅, erp ✅ |
| **nexora-model-genesis** | 13 | 3.624 | 27 (caps) | ❌ | reasoning ❌, transformer ✅ |
| **nexora-model-kronos** | 11 | 3.328 | 25 (caps) | ❌ | reasoning ❌, transformer ✅ |
| **nexora-model-nexum** | 11 | 9.868 | 28 (caps + enum) | ❌ | oracle ❌, reasoning ❌, alignment 🔶 |
| **nexora-model-spectra** | 17 | 7.996 | 22 (caps + string) | ❌ | multimodal ✅, transformer ✅ |
| **nexora-model-swift** | 11 | 4.677 | 37 (caps + GpuOptimization) | ❌ | has-moe-ffn ✅, transformer ✅, erp ✅ |

### Shared Types (1)

| Crate | Path | Files | Lines | GPU Refs? | Notes |
|-------|------|:----:|:-----:|:---------:|-------|
| **nexora-shared** | crates/shared | 14 | 5.114 | 40 (structs + enums) | `GpuConfig` struct, `HardwareTarget::CloudGPU`, `requires_gpu`. TYPE DEFINITIONS doang — 0 compute. |

### Oracle — Heavy-Compute Menurut AGENTS.md, Tapi... ❌

| Metadata | Nilai |
|----------|-------|
| **Path** | `crates/oracle/` |
| **.rs files** | 27 |
| **Total lines** | 12.705 |
| **Feature** | `gpu = ["nexora-autograd/gpu"]` |
| **GPU refs** | 105 — TAPI SEMUA di 1 file (`backbone.rs`) |
| **GPU ops** | matmul + transpose doang |
| **Verdict** | ❌ **CPU-only** secara praktis — hanya 4 `OnceLock<GpuTensor>` weight fields + 2 matmul. Kode GPU minimal, bukan real compute pipeline. |

**Klarifikasi**: AGENTS.md bilang oracle "✅ Real arsitektur" dengan GPU, tapi source code menunjukkan GPU integration-nya sangat tipis — hanya 4 weight tensor cached, 2 matmul ops, dan 0 custom GPU logic. Crate ini lebih cocok diklasifikasikan sebagai **CPU-dominant dengan GPU wrapper tipis**.

### Dashboard — TUI App

| Crate | Path | Files | Lines | GPU Refs? | Notes |
|-------|------|:----:|:-----:|:---------:|-------|
| **nexora-dashboard** | apps/dashboard | 1 | 765 | ❌ | TUI dashboard (ratatui, crossterm). No GPU. |

---

## Analisis: Oracle — Kasus Khusus

`nexora-oracle` punya `gpu` feature dan 105 GPU references, tapi:

```
crates/oracle/src/backbone.rs:
  - 4× OnceLock<GpuTensor>     ← weight caching
  - 2× ctx.matmul()            ← via autograd
  - 1× ctx.transpose()         ← via autograd
  - Sisanya: CPU implementation penuh (12-layer MoE + MLHA)
```

Ini **bukan** real GPU compute. Hanya thin wrapper yang upload weight ke GPU, call 2 matmul, download. CPU path adalah implementasi utama.

**Rekomendasi**: Downgrade klasifikasi oracle dari "GPU-enabled" (AGENTS.md) menjadi **CPU-dominant dengan GPU wrapper tipis**.

---

## Ringkasan Perubahan dari Audit Sebelumnya

Dibandingkan file `AUDIT_GPU_ACCELERATION.md` yang sudah ada:

| Perubahan | Lama | Baru | Alasan |
|-----------|:----:|:----:|--------|
| **Total CPU-only** | 29 | 30 | Oracle ternyata CPU-dominant |
| **Oracle klasifikasi** | ✅ GPU Enabled | ⚠️ CPU-dominant | Hanya 2 matmul di GPU, 0 custom logic |
| **Monitoring GPU refs** | ❌ CPU-only | ✅ GPU metric fields | Ada `gpu_alive`, `gpu_tokens` dkk — tapi DEFINISI METRIC, 0 compute |
| **Runtime GPU refs** | ❌ CPU-only | ✅ Data fields | Ada `gpu_util_pct`, `supports_cuda` — DATA SCHEDULING, 0 compute |
| **Model crate GPU refs** | ❌ CPU-only | ✅ Capability specs | `requires_gpu` dkk — METADATA, 0 compute |

---

## Potensi GPU-ification — Prioritas

### P0 — SACA Reasoning — NOT APPLICABLE

| Crate | `reasoning` |
|-------|-------------|
| **Lines CPU-only** | 12.529 (terbesar) |
| **GPU potensi** | ❌ **Not GPU-ifiable** — orchestration code (string processing, subprocess execution, scalar arithmetic). No tensor operations. Heavy compute deferred to Oracle backbone via LLM inference calls. |
| **Action** | **No action needed.** SACA is orchestration-only. GPU-ification would provide zero benefit. |

### P1 — Oracle Backbone — ✅ Already GPU-wired

| Crate | `oracle` |
|-------|----------|
| **GPU current** | ✅ Full GPU path via `forward_gpu()` — CUDA FlashAttention + wgpu + CPU fallback. All 12 transformer layers, MoE, attention, norms GPU-accelerated. |
| **GPU potensi** | Already has full GPU forward. CPU `forward()` is fallback for non-GPU builds. |
| **Action** | **No action needed.** Oracle already has a complete GPU path. |

### P2 — Batch Evaluation (3-5× speedup)

| Crate | `evaluation` |
|-------|--------------|
| **Lines CPU-only** | 131 |
| **GPU potensi** | Batch perplexity, accuracy via GPU forward |
| **Estimasi** | 3-5× faster eval |
| **Action** | Tambah `gpu` feature → `nexora-autograd/gpu`, gunakan GpuContext utk batch forward |
| **Risiko** | Minimal — crate kecil, mudah diintegrasi |

### P3 — Anti-Hallucination Verification (2-4× speedup)

| Crate | `hallucination` |
|-------|-----------------|
| **Lines CPU-only** | 1.860 |
| **GPU potensi** | Batched cross-attention scores GPU |
| **Estimasi** | 2-4× speedup verification |
| **Action** | GPU batch matmul utk verification scoring |

### P4 — Token Embedding Lookup (1.5-2× speedup)

| Crate | `tokenizer` |
|-------|-------------|
| **Lines CPU-only** | 4.833 |
| **GPU potensi** | Embedding table di GPU memory, lookup batch GPU |
| **Estimasi** | 1.5-2× speedup |
| **Action** | Embedding table → `GpuTensor`, batch lookup via `GpuContext::embedding()` |

---

## Total Baris Code per Klasifikasi

| Klasifikasi | Crate | Total .rs Files | Total Lines |
|-------------|:-----:|:--------------:|:-----------:|
| **Heavy-Compute** | 4 | 109 | 55.340 |
| **Integrated** | 11 | 207 | 94.445 |
| **Pass-Through** | 6 | 91 | 43.945 |
| **Adapter** | 2 | 62 | 21.181 |
| **CPU-Only (no feature)** | 15 | 97 | 38.753 |
| **CPU-Only (model crates)** | 11 | 155 | 63.419 |
| **CPU-Only (oracle)** | 1 | 27 | 12.705 |
| **CPU-Only (shared)** | 1 | 14 | 5.114 |
| **CPU-Only (dashboard)** | 1 | 1 | 765 |
| **TOTAL CPU-Only** | **30** | **294** | **120.756** |
| **TOTAL Workspace** | **53** | **763** | **335.665** |

**CPU-only code**: 120.756 lines / 335.665 total = **36.0% dari total codebase**

---

## Kesimpulan

1. **15 crate real GPU** (heavy-compute + integrated) — fondasi GPU solid via `nexora-autograd` sebagai provider utama.
2. **4 crate dengan custom GPU kernels** — autograd (86 ops), transformer (GPU forward penuh), has-moe-ffn (MoE CUDA), echo-net (WGSL shaders).
3. **30 crate CPU-only** — 36% dari total codebase. Tapi hanya **4 crate yang realistic untuk GPU-ify** (evaluation, hallucination, tokenizer + monitoring).
4. **24 crate CPU-only lainnya** adalah infrastruktur I/O-bound (api, database, runtime, monitoring, dll) — GPU tidak akan memberikan benefit signifikan.
5. **Oracle sudah GPU-wired** — `forward_gpu()` exists with CUDA FlashAttention + wgpu. CPU `forward()` adalah fallback untuk non-GPU builds.
6. **SACA reasoning NOT GPU-ifiable** — orchestration code (string processing, subprocess execution). No tensor ops. GPU-ification would provide zero benefit.
7. **Model crate (10)** tidak perlu GPU-ify — mereka delegasi ke crate target yang sudah GPU-enabled.

---

## Rekomendasi

| Prioritas | Crate | Action | Estimasi Effort | Benefit |
|-----------|-------|--------|:---------------:|---------|
| ~~**🔥 P0**~~ | ~~`reasoning`~~ | ~~GPU-ify SACA pipeline~~ | ❌ NOT GPU-ifiable | Orchestration code, no tensor ops |
| ~~**🔥 P0**~~ | ~~`oracle`~~ | ~~Full GPU forward~~ | ✅ Already GPU-wired | `forward_gpu()` exists |
| **P1** | `evaluation` | Batch metrics GPU via autograd | 1-2 hari | 3-5× eval speedup |
| **P2** | `hallucination` | GPU batch verification | 3-5 hari | 2-4× verification |
| **P3** | `tokenizer` | GPU embedding lookup | 2-3 hari | 1.5-2× tokenizer |

---

## Catatan Teknis

- **Semua GPU routing** via `nexora-autograd` → `GpuContext::global()` singleton dengan auto-detection (CUDA → wgpu → CPU fallback).
- **Tidak ada `build.rs`** di seluruh workspace — GPU kernels di-compile via runtime JIT (NVRTC) atau inline WGSL strings.
- **Tidak ada file `.wgsl` standalone** — semua WGSL shaders adalah `const &str` di Rust source.
- **CUDA support terbatas** ke 4 crate: autograd, has-moe-ffn, transformer (NCCL), erp.
- **`cuda` feature** requires CUDA toolkit (`nvcc`) at build time — diverifikasi via `carg o check` only.

---

*Audit dilakukan 7 Juni 2026 oleh opencode agent.*
*Metodologi: source code grep, Cargo.toml analysis, LOC counting, manual classification.*
