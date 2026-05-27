# Rating Performa Inference Nexora

## 1. Stabilitas Beban Tinggi — 7/10

**Evidence:**

- Continuous batching engine (`crates/inference/src/continuous_batching.rs`) dengan 4 scheduling policies:
  - `Fifo` — oldest first
  - `PriorityAging` (default) — base priority 10 (prefill) / 5 (generation) + age boost 0.001/ms
  - `ShortestRemaining` — negative remaining tokens
  - `TokenBucket` — tokens_generated / (1 + age * aging_boost)
- DegradationManager 5-level: None → Reduced → Minimal → ReadOnly → Unavailable
- Self-healing worker background task
- Circuit breaker via `Sampler::allow_gpu_fallback`
- Chaos tests: `test_chaos_mixed_load` (5 short + 2 long), `test_chaos_spike` (50 sequences), `test_chaos_starvation_avoidance`, `test_chaos_long_tail`
- Max sequences: 1024, Max batch: 8, Semaphore concurrency: 4
- Dynamic padding: waits up to 10ms for more sequences, processes if waste > 30%

**Kelemahan:**

- Paged cache **disabled by default** (`use_paged_cache: false`)
- Tidak ada distributed serving / multi-node
- Tidak ada benchmark suite / CI performance regression

---

## 2. Efisiensi VRAM/RAM — 6/10

**Evidence:**

- **Paged KV cache** (`crates/inference/src/paged_cache.rs`):
  - Block-based allocation (default 16 tokens/block)
  - Copy-on-write via `ref_count` — `get_or_alloc_block()` deep copy saat ref_count > 1
  - Free list per-layer untuk O(1) block reuse
  - Defragmentasi: compact sparse blocks, frees drained blocks
  - Fragmentasi tracking: internal & external ratio, wasted slots
  - Memory calculation: `block_size * num_kv_heads * head_dim * 4 * 2`
- **Flat KV cache** (`crates/inference/src/kv_cache.rs`):
  - LRU eviction via `BTreeSet<(last_access_nanos, hash)>`
  - TTL eviction (default 3600s)
  - Max memory: 1GB, Max entries: 10,000
- **Shared pool**: `Arc<Mutex<PagedKVCache>>` — semua sequence share block pool

**Kelemahan:**

- Safetensors **f32-only** (`dtype: "F32"`) — no f16 saving
- STar-X tensor pakai `ndarray::ArrayD<f32>` tanpa custom memory layout
- Paged cache disabled by default
- CPU path tetap f32 untuk semuanya

---

## 3. Mixed Precision (f16/f32) — 5/10

**Evidence:**

- **Weight storage**: `use_half_precision: bool` di CausalLM — "store weights as packed F16 (2 f16 per u32, 2× memory savings)"
- **7 weight matrices** (wq, wk, wv, wo, w1, w2, w3) punya `_f16: Option<GpuTensor>` variants
- **GPU KV cache**: `GpuKVCacheEntry.f16_storage: bool` — buffer alokasi sebagai `GpuDtype::F16`
- **Int8 quantization**: `quantize_weights: bool` — symmetric per-tensor int8, WGSL shader dequant on-the-fly, 4× bandwidth saving
- **GPU sampling**: `sample_gpu_tensor()` auto-detect F16 dtype → konversi F16→F32 di GPU
- **F16 CPU fallback**: manual unpacking `u32` → two `u16` → `f16_bits_to_f32()` — tanpa `half` crate

**Kelemahan:**

- **CPU path tetap f32** — f16 hanya untuk GPU weight storage
- F16 diupconvert ke F32 tiap forward pass
- `use_half_precision: false` by default
- Tidak ada native f16 matmul

---

## 4. GPU vs CPU Utilization — 6/10

**Evidence:**

- **3-tier fallback chain** di `generate_internal()`:
  1. GPU-resident: `model.generate_with_gpu_keep_gpu()` — entire loop di GPU, 4 byte/token readback
  2. Per-token GPU: `run_generation_loop()` via `spawn_blocking` — tetap pakai GPU via `forward()`
  3. Pure CPU: `model.forward()` dengan `CpuKVCache`
- **GPU-native sampling** dengan zero logit readback
- **NVML monitoring**: `read_gpu_memory()` via `nvml_wrapper::Nvml`
- **GPU metrics**: `GPU_RESIDENT_SUCCESSES`, `GPU_RESIDENT_FALLBACKS`, `GPU_FORWARD_ERRORS`, `GPU_BUSY_NS`, `PCIE_READ_BYTES`
- **CPU affinity**: Linux `sched_setaffinity` via `libc`
- **BLAS**: ndarray `"blas"` feature untuk 5-10× CPU matmul (via OpenBLAS)

**Kelemahan:**

- Backend **WebGPU (`wgpu`)** bukan CUDA native — ada overhead translasi
- Tidak ada FlashAttention, TensorRT, atau vLLM integration
- GPU feature di-gate `#[cfg(feature = "gpu")]` — compile-time, bukan runtime

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
| Stabilitas beban tinggi | **7** | Continuous batching, PriorityAging, degradation mgmt | Paged cache non-default, no distributed serving |
| Efisiensi memori | **6** | Paged cache COW, defrag, LRU eviction | f32 di mana-mana, safetensors f32-only |
| Mixed precision | **5** | f16 weight storage, int8 quant, GPU KV cache f16 | CPU f32-only, upconvert tiap forward pass |
| GPU acceleration | **6** | 3-tier fallback, zero logit readback, NVML | WebGPU bukan CUDA, no FlashAttention |
| Scaling model besar | **6** | MoE, MLA 4× compression, GQA, 32K context | 12-layer hardcode, MoE CPU-bound |
| Quality/speed balance | **7** | GPU native sampling, beam search O(1), 3 preset | No speculative decoding, beam search CPU |

> **Catatan**: Nexora adalah **research-grade codebase dengan arsitektur modern**. Komponen individu (paged cache, MoE, MLA, continuous batching, GPU-native sampling) dirancang dengan benar secara konseptual. Namun integrasi masih parsial — banyak fitur non-default, GPU path terbatas WebGPU, dan CPU path tidak teroptimasi untuk f16. Potensi arsitektural tinggi, tapi eksekusi saat ini masih di bawah framework mature seperti vLLM atau TensorRT-LLM.
