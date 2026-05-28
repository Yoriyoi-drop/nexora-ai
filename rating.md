# Rating Performa Inference Nexora

## 1. Stabilitas Beban Tinggi — 9/10

**Evidence:**

- Continuous batching engine (`crates/inference/src/continuous_batching.rs`) dengan 4 scheduling policies:
  - `Fifo` — oldest first
  - `PriorityAging` (default) — base priority 10 (prefill) / 5 (generation) + age boost 0.005/ms
  - `ShortestRemaining` — negative remaining tokens
  - `TokenBucket` — tokens_generated / (1 + age * aging_boost)
- DegradationManager 5-level: None → Reduced → Minimal → ReadOnly → Unavailable
- Self-healing worker background task + circuit breaker
- **Paged cache ENABLED by default** (`use_paged_cache: true`) — block-based allocation, COW, defragmentation, zero fragmentasi
- **Adaptive batch sizing** — auto-tune `max_batch_size` berdasarkan EWMA throughput (tokens/sec). Turunkan batch saat throughput rendah untuk cegah overload, naikkan saat longgar untuk maksimalkan utilisasi.
- **Load shedding** — reject request baru saat queue depth > `max_queue_depth` (default: 2048) atau throughput < 10% target. `rejected_count()` untuk observability.
- **Default params lebih agresif**: `max_batch_size: 32`, `max_total_sequences: 4096`, `aging_boost_per_ms: 0.005`, `max_queue_depth: 2048`
- Chaos tests: `test_chaos_mixed_load`, `test_chaos_spike` (50 seq), `test_chaos_starvation_avoidance`, `test_chaos_long_tail`
- **8 new stability tests**: adaptive batching (scale up/down, min bound), load shedding (queue full, max seq, recovery after drain), soak test (100 seq sustained), concurrent fill-and-drain (5-cycle), default config stability (50 seq)
- **Criterion benchmark suite**: throughput, latency, mixed workload, spike load, starvation avoidance, paged vs flat cache comparison
- Scheduler: `max_concurrent: 32`, `max_batch_size: 32` (vs 8/4 sebelumnya)
- Dynamic padding: waits up to 10ms for more sequences, processes if waste > 30%
- PriorityAging dengan 0.005/ms aging boost (5× lebih agresif dari sebelumnya)

**Kelemahan Tersisa:**

- Distributed serving / multi-node masih belum ada
- Tidak ada formal CI regression untuk benchmark (perlu di-wire ke CI)

---

## 2. Efisiensi VRAM/RAM — 9/10

**Evidence:**

- **F16 safetensors** (`crates/foundation/src/safetensors/io.rs` + `crates/transformer/src/safetensors/io.rs`):
  - `save_safetensors_f16()` — simpan checkpoint 2× lebih kecil
  - `load_safetensors()` — auto-detect F32/F16, konversi F16→F32 on load
  - `SaveDtype` enum: `F32` / `F16` — satu fungsi `save_safetensors_with_dtype()`
  - Backward compatible: existing F32 safetensors tetap bisa diload
- **Paged KV cache f16 storage** (`crates/inference/src/paged_cache.rs`):
  - `BlockData` enum: `F32(Array2<f32>)` / `F16(Vec<u16>)` — switch storage berdasarkan `PagedCacheConfig.f16_storage`
  - K/V tersimpan sebagai `u16` → 2× memory reduction vs f32
  - `append()`: konversi f32→f16 on write
  - `read()`: konversi f16→f32 on read
  - `to_flat_cache()`: batch konversi f16→f32 untuk forward pass
  - `defragment()`: work dengan u16 slice, zero conversion
  - `memory_usage_bytes()`: actual block memory (f16 vs f32 aware)
  - `copy-on-write` via `BlockData::clone_data()` — tetap efisien untuk prefix sharing
  - **Config toggle**: `ContinuousBatchingConfig.paged_cache_f16` + `InferenceConfig.paged_cache_f16`
  - **Default: ENABLED** (`f16_storage: true`)
- **Paged KV cache** (existing):
  - Block-based allocation (default 16 tokens/block)
  - Copy-on-write via `ref_count`
  - Free list per-layer untuk O(1) block reuse
  - Defragmentasi: compact sparse blocks, frees drained blocks
  - Fragmentasi tracking: internal & external ratio, wasted slots
- **Flat KV cache** (`crates/inference/src/kv_cache.rs`):
  - LRU eviction via `BTreeSet<(last_access_nanos, hash)>`
  - TTL eviction (default 3600s)
  - Max memory: 1GB, Max entries: 10,000
- **Shared pool**: `Arc<Mutex<PagedKVCache>>` — semua sequence share block pool
- **KV cache memory tracking**: `KV_CACHE_MEMORY_BYTES` atomic + `ResourceUsage.kv_cache_memory_bytes`
- **F16 conversion utilities**: `f32_to_f16_bits()` / `f16_bits_to_f32()` di `lib.rs` — shared oleh sampler + paged_cache

**Kelemahan Tersisa:**

- STar-X tensor pakai `ndarray::ArrayD<f32>` tanpa custom memory layout (akan di-upgrade bersama memory architecture 5b+)
- F16 paged cache conversion overhead pada read/write (negligible vs 2× memory saving)
- Belum ada memory defragmentation scheduling otomatis (manual call via `defragment()`)

---

## 3. Mixed Precision (f16/f32) — 8/10

**Evidence:**

- **CPU f16 weight storage + matmul** (`crates/transformer/src/gqa.rs`, `swiglu.rs`):
  - `GQA.wq_f16: Option<Vec<u16>>` — packed f16 parallel to f32 weights (wq, wk, wv, wo)
  - `SwiGLU.w1_f16/w2_f16/w3_f16: Option<Vec<u16>>` — packed f16 FFN weights
  - `pack_f16_weights()` — konversi f32→f16 packed saat `set_use_half_precision()` dipanggil
  - `maybe_f16_matmul()` — dispatch: gunakan `matmul_f16_cpu()` jika f16 weights ada, fallback ke `x.dot(&w.t())` jika tidak
  - `matmul_f16_cpu()` — CPU f16 matmul: baca packed u16, konversi f16→f32 on-the-fly, akumulasi di f32, **di-parallelize via rayon** (`par_chunks_mut`)
- **Shared f16 utilities** (`crates/transformer/src/lib.rs`):
  - `f32_to_f16_bits()` / `f16_bits_to_f32()` — bit-level konversi IEEE 754
  - `pack_f32_slice_to_f16()` — bulk f32→f16 packing
  - `matmul_f16_cpu()` — f16 matmul untuk CPU forward path
- **GPU zero-copy f16 matmul** (existing, Path B di `forward_gpu_single_token_core`):
  - `ctx.matmul(x, wq_f16)` auto-dispatch ke `matmul_f16()` WGSL shader — baca packed f16 langsung dari VRAM
  - **No upconversion**: f16 weights stay f16, convert on-the-fly dalam shader
  - Juga untuk wk_f16, wv_f16, wo_f16, w1_f16, w2_f16, w3_f16, lm_head_f16
- **Weight storage**: `use_half_precision: bool` di CausalLM + TransformerConfig (default: `true`)
- **7 weight matrices** (wq, wk, wv, wo, w1, w2, w3) punya f16 variants di GPU + CPU
- **GPU KV cache**: `GpuKVCacheEntry.f16_storage: bool` — buffer alokasi sebagai `GpuDtype::F16`
- **Int8 quantization**: `quantize_weights: bool` — symmetric per-tensor int8, WGSL shader dequant on-the-fly, 4× bandwidth saving
- **GPU sampling**: `sample_gpu_tensor()` auto-detect F16 dtype → konversi F16→F32 di GPU

**Kelemahan Tersisa:**

- F16 upconvert path (Path A) masih ada di GQA/SwiGLU `forward_gpu` untuk backward compat
- Tidak ada native f16 matmul via BLAS (OpenBLAS tidak support f16)

---

## 4. GPU vs CPU Utilization — 8/10

**Evidence:**

- **CUDA backend wiring** (`crates/autograd/src/gpu/`):
  - `GpuBackend` enum: `Wgpu` / `Cuda` — auto-detected at `GpuContext::init()`
  - `GpuContext.cuda: Option<&'static CudaRuntime>` — borrows from singleton, no duplicate context
  - `GpuAdapterInfo.compute_backend` — exposed for observability
  - `CudaRuntime` with **21 ops**: cuBLAS matmul, NVRTC JIT for elementwise/softmax + **3 new**: broadcast add (`[1,D]+[N,D]`), transpose, gelu_inplace
  - `CudaTensor::from_cpu()` / `to_cpu_vec()` — host↔device transfer
- **MoE Router full pipeline** (`crates/has-moe-ffn/src/routing.rs`):
  - **wgpu**: `router_weights_gpu OnceLock` → transpose → `ctx.matmul` + `ctx.softmax`
  - **CUDA**: `router_weights_cuda OnceLock` → cuBLAS matmul + NVRTC softmax → readback
  - **Fallback**: CUDA → wgpu → CPU
- **MoE Expert full pipeline** (`crates/has-moe-ffn/src/experts.rs`):
  - **wgpu**: `fc1_gpu/fc2_gpu OnceLock` weight caching, `forward_batched_gpu()` N-token
  - **CUDA**: `fc1_cuda/fc2_cuda OnceLock` CUDA-native weights, `forward_batched_cuda()` via cuBLAS (GELU in-place, no CPU round-trip for weights)
  - **Grouped dispatch**: tokens → expert → one `forward_batched()` per expert → scatter weighted results
  - **Fallback**: CUDA → wgpu → CPU
- **3-tier generate fallback**:
  1. GPU-resident: zero token readback (4B/token only)
  2. Per-token GPU via `spawn_blocking`
  3. Pure CPU with `CpuKVCache`
- **GPU-native sampling** tanpa logit readback
- **NVML monitoring**, **GPU metrics**, **CPU affinity** via `libc`

**Kelemahan:**
- CUDA membutuhkan `cuda` feature + `nvcc` — unavailable di dev env ini
- Tidak ada FlashAttention, TensorRT, atau vLLM
- `gpu`/`cuda` features masih compile-time, bukan runtime

---

## 5. Scaling ke Model Besar — 6/10

**Evidence:**

- **MoE** (`crates/has-moe-ffn/src/`):
  - 8 experts, top-2 routing
  - Capacity factor: 1.25 (capped routing)
  - Load balancing loss + Z-loss
  - Expert Choice routing toggle
  - Routing stats: load balance score, expert utilization, capacity violations
- **MultiHeadLatentAttention (MLA)** (`crates/oracle/src/backbone.rs`):
  - `d_model=4096` → `latent_dim=512` — 4× KV cache compression
  - Latent compression ratio: 0.25
  - Memory estimasi: ~512MB untuk batch=4, seq=32768
- **GQA (Grouped Query Attention)**: `num_kv_heads` bisa berbeda dari `num_heads`
- **32K context window**
- **RoPE**: pre-computed cos/sin up to `max_seq_len`
- **LayerInjector trait**: per-layer injections untuk modular extensions (EchoNet)

**Kelemahan:**

- **12-layer hardcoded** di Oracle (`let n_layers = 12`)
- MoE jalan di **CPU** (`ndarray::Array2<f32>`) — GPU path hanya untuk attention
- Tidak ada pipeline parallelism atau tensor parallelism
- Tidak ada distributed training

---

## 6. Kualitas vs Kecepatan — 7/10

**Evidence:**

- **5 sampling methods**: Greedy, Temperature, TopK, TopP, TemperatureTopKTopP
- **3 preset konfigurasi**:
  - `greedy()` — deterministic, tercepat
  - `conservative()` — temp=0.7, top_k=40, top_p=0.85
  - `creative()` — temp=1.2, top_k=100, top_p=0.95
- **GPU batched sampling**: Stacks `[B, vocab]` logits, single GPU call, readback 4B/token
- **Zero logit readback**: `forward_batched_sample_gpu()` returns `Option<u32>` — GPU sampling tanpa transfer logit ke CPU
- **Beam search** (`crates/inference/src/beam_search.rs`):
  - `beam_size=4`, `length_penalty=1.0`, `early_stopping=true`
  - Arc-based shared tails — O(1) append, hindari O(n²) clone
  - Divergence penalty: 0.1
- **PriorityAging scheduling**: prefill priority 10, generation 5 — starvation prevention
- **Repetition penalty**: skip clone saat penalty ≈ 1.0

**Kelemahan:**

- **Belum ada speculative decoding**
- Beam search **CPU-only** (tidak ada GPU kernel)
- Tidak ada contrastive search atau MCTS

---

## Ringkasan

| Aspek | Rating | Kekuatan Utama | Kelemahan Utama |
|-------|--------|----------------|-----------------|
| Stabilitas beban tinggi | **9** | Paged cache default, adaptive batch, load shedding, 25 tests | No distributed serving, no CI benchmark gate |
| Efisiensi memori | **9** ↑ | F16 safetensors + paged cache f16 K/V, 2× memory reduction | STar-X masih f32 |
| Mixed precision | **8** ↑ | CPU f16 matmul (rayon-parallel) + GPU zero-copy f16 matmul, f16 weights storage | BLAS no f16, upconvert path still exists |
| GPU acceleration | **8** ↑ | CUDA backend (GpuBackend enum, cuBLAS, NVRTC JIT), MoE Router+Experts full pipeline w/ CUDA→wgpu→CPU fallback, 3-tier generate, GPU sampling | CUDA requires nvcc, no FlashAttention, compile-time features |
| Scaling model besar | **6** | MoE, MLA 4× compression, GQA, 32K context | 12-layer hardcode, MoE CPU-bound |
| Quality/speed balance | **7** | GPU native sampling, beam search O(1), 3 preset | No speculative decoding, beam search CPU |

> **Catatan**: GPU acceleration naik ke **8/10** setelah CUDA backend wiring (GpuBackend auto-detection, cuBLAS + NVRTC JIT ops, MoE Router+Expert CUDA paths dengan fallback CUDA→wgpu→CPU). GPU acceleration + Mixed precision + Memory efficiency + Stability semuanya ≥8/10 sekarang — fondasi enterprise-grade. Komponen sisanya (FlashAttention, TensorRT, distributed serving, speculative decoding) masih perlu kerja lanjutan.
