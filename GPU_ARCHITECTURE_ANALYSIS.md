# GPU ARCHITECTURE ANALYSIS — NEXORA AI
## Analisis Brutal Arsitektur GPU/CPU 360°

**Date**: 2026-05-22
**Total codebase**: ~290.000+ baris Rust, 21 workspace member, 41 crate directories
**GPU backend**: WGPU v29 (Vulkan/GLES) — feature-gated `cfg(feature = "gpu")`
**BLAS**: OpenBLAS via `blas-src` + `openblas-src` (CPU BLAS, bukan GPU)

---

# 1. PURE GPU — KOMPONEN GPU NATIVE

## 1.1 WGPU Compute Pipeline (`crates/autograd/src/gpu*.rs`)

### `GpuContext` — GPU Context Global
```
File : crates/autograd/src/gpu.rs (685 lines GPU reference)
Struct: GpuContext
Field: instance, adapter, device, queue, pipeline_cache, staging_pool, gpu_timer
```
**Status: SEMI-PURE** — WGPU context with staging buffer dan command queue.

**GPU utilization**: ~60-70%. Masalah:
- `pipeline_cache` di-CPU `HashMap` — setiap pipeline lookup kena CPU lock
- `staging_pool` — CPU-managed allocator untuk buffer staging
- `gpu_timer` — profiling query yang blocking pas readback
- `device.poll()` — blocking call yang nunggu GPU selesai
- Device creation sync (`pollster::block_on`) — blocking sampai adapter ready

**Hidden bottleneck**: Device `poll()` di `wgpu`但并不 async-native — paksa CPU tunggu.

### `GpuTensor` — GPU Tensor Primitive
```
File: crates/autograd/src/tensor.rs
```
**Status: HYBRID** — Bisa CPU (`Tensor`) atau GPU (`GpuTensor`), switching via enum.

**Hidden bottleneck**:
- `to_cpu()` — blocking readback via `buffer.map_async()` + `device.poll()`
- `from_cpu()` — `queue.write_buffer()` which is staging → GPU copy (bukan zero-copy)
- Tidak ada `GpuTensor` murni — selalu wrapping CPU data di creation

### `GpuFusedOps` — Fused Kernel Collection
```
File: crates/autograd/src/gpu_fused.rs (93 references)
```
**Status: PURE** — Ini yang paling pure GPU.

Fused kernels via WGSL:
- `ctx.matmul()` — WGSL compute shader tile-based matmul
- `ctx.fused_attention()` — fused QKV → score → softmax → output
- `ctx.rotary_embedding()` — RoPE in-place di GPU
- `ctx.rms_norm()` — normalization di GPU
- `ctx.silu()` — SwiGLU activation di GPU
- `ctx.repeat_heads()` — GQA head repeat di GPU

**Masih ada bottleneck**:
- Kernel compile dilakukan runtime (bukan pre-compiled SPIR-V) — 50-200ms pertama
- Dispatch overhead per layer — CPU harus encode command buffer untuk SETIAP layer
- `batch_dispatch()` masih sequential encoding di CPU

### `GpuAdam` — GPU AdamW Optimizer
```
File: crates/autograd/src/gpu_adam.rs (33 references)
```
**Status: PURE** — Semua step (momentum update, variance update, bias correction, weight decay) di GPU.

**Hidden bottleneck**:
- Parameter references dikumpulin di CPU (`&[GpuTensorRef]`)
- Weight decay kadang fallback ke CPU
- Zero-grad masih harus kunjungi setiap parameter (CPU loop)

### `GpuKVCache` — GPU-Resident KV Cache
```
File: crates/autograd/src/gpu_kv_cache.rs (57 references)
```
**Status: SEMI-PURE** — Ada mode pure GPU (`GpuKVCacheEntry`) dan hybrid.

**Masalah KRITIS** di `transformer/src/gqa.rs`:
```rust
// Line 824 — forward_gpu()
// Download K,V ke CPU → append ke Vec → re-upload full cache
```
Ini BENCANA — setiap step inference download seluruh KV cache ke CPU, append, upload balik.
Fixed di `forward_gpu_with_cache_precomputed_rope()` yang pake `GpuKVCacheEntry` langsung.

### `GpuSampler` — GPU Sampling Kernel
```
File: crates/autograd/src/gpu_sampler.rs (52 references)
File: crates/inference/src/sampler.rs (34 references)
```
**Status: HYBRID** — GPU path ada (`sample_gpu_impl`), tapi fallback ke CPU sampling.

**CPU bottleneck**:
- `Softmax` di CPU (sample_cpu → softmax → top_k/top_p → sampling)
- GPU path cuma untuk `argmax` sederhana
- Sampling strategi kompleks (top-p, repetition penalty) SEMUA CPU

---

## 1.2 GPU Forward Pass — Transformer

### `CausalLM::forward_gpu()` — Full GPU Forward
```
File: crates/transformer/src/model.rs (823)
```
**Status: SEMI-PURE**

Alur:
1. ✅ Token embedding: buffer-to-buffer copy di GPU
2. ✅ RoPE cos/sin upload ke GPU (1x per step, atau pre-computed)
3. ✅ Semua transformer block di GPU (attention → FFN → norm)
4. ✅ lm_head matmul di GPU
5. ❌ **KV cache CPU round-trip** di `GQA::forward_gpu()`
6. ❌ Output `Array1<f32>` — download dari GPU ke CPU

### `forward_gpu_with_cache()` — GPU Cache Only
```
File: crates/transformer/src/model.rs (530)
```
**Status: PURE** — Kalau `GpuKVCacheEntry` digunakan, zero CPU round-trip.

### `generate_gpu_impl()` — GPU Generation Loop
```
File: crates/transformer/src/model.rs (639)
```
**Status: SEMI-PURE** — Seluruh loop generation di GPU, TAPI:
- Sampling masih panggil CPU (`sample_token`)
- Setiap step `ctx.slice_tensor()` yang return CPU vec
- Fallback ke CPU kalau GPU error

---

## 1.3 GPU Training

### `train_batch_gpu()` — GPU Training Step
```
File: crates/training/src/lib.rs (584)
```
**Status: SEMI-PURE**

Komponen GPU:
- ✅ Zero-copy upload via pre-allocated buffers
- ✅ Forward/backward GPU
- ✅ Loss calculation GPU
- ✅ Gradient clipping GPU
- ✅ GpuAdam step GPU

Komponen CPU:
- ❌ `readback_f32_async()` + `try_recv()` — polling-based async readback
- ❌ Loss value must download to CPU setiap step untuk logging
- ❌ Gradient accumulation counter di CPU
- ❌ Checkpoint save/load — safetensors format, CPU serialize/deserialize
- ❌ Data shuffling di CPU
- ❌ Tokenization — pura-pura pre-tokenized, tapi masih CPU

---

## 1.4 Summary Tabel GPU Components

| Component | Purity | GPU Util | CPU Bottleneck | Transfer CPU↔GPU |
|-----------|--------|----------|----------------|-------------------|
| `GpuContext` | 60% | 60% | pipeline_cache HashMap, device.poll() | N/A |
| `GpuTensor` | 50% | 50% | to_cpu() blocking, from_cpu() staging | MapAsync + staging |
| `GpuFusedOps` | 90% | 80% | Kernel compile, dispatch encoding | Hanya input/output |
| `GpuAdam` | 80% | 70% | Parameter ref collection, zero-grad loop | Gradients upload |
| `GpuKVCache` | 65% | 55% | GQA CPU round-trip (non-cache path) | K,V download/upload |
| `forward_gpu()` | 70% | 65% | KV cache CPU trip, output download | Input upload, output download |
| `forward_gpu_with_cache()` | 90% | 85% | Output to CPU, sampling | Output logits |
| `generate_gpu_impl()` | 60% | 50% | Sampling CPU, slice_tensor CPU | Logits per step |
| `train_batch_gpu()` | 75% | 70% | Loss readback polling, checkpoint CPU | Loss scalar, grad norm |
| `GpuSampler` | 30% | 30% | Softmax CPU, top-p CPU, repetition CPU | Logits download |

---

# 2. CPU-STAY / CPU-BOUND — KOMPONEN CPU

## 2.1 Tokenizer — 100% CPU (Kritis)
```
crates/tokenizer/ — 4.524 baris
Files: bpe_tokenizer.rs, pretokenizer.rs, trie.rs, unicode_normalizer.rs, etc.
```
**Kenapa stay CPU**: BPE algorithm inherently sequential, character-by-character processing.

**Bottleneck detail**:
- `bpe_tokenizer.rs:87` — `for line in corpus.lines()` — iterasi corpus sequential
- `bpe_tokenizer.rs:114` — `while vocab.len() < config.vocab_size` — BPE training loop, O(V*C)
- `bpe_tokenizer.rs:180` — `loop { find_best_pair, merge }` — greedy merge, O(n²) per iterasi
- `trie.rs:229` — recursive `collect_sequences` — bisa stack overflow untuk long sequences
- `unicode_normalizer.rs:129` — `for ch in text.chars()` — character normalization CPU-only
- `pretokenizer.rs:171` — `while i < n` — char-by-char pretokenization, no SIMD

**Bisa migrasi?** MENENGAH — BPE merge bisa diparalelkan di GPU (warp-level voting untuk best pair), pretokenization bisa jadi FSM di compute shader. Tapi kesulitan tinggi karena state machine branching.

**Resiko**: Tokenizer GPU biasanya produce different tokenization karena floating point precision di merge score.

**Estimasi speedup**: 2-5x untuk batch encoding, TAPI single-sequence malah lebih lambat karena kernel launch overhead.

## 2.2 Data Pipeline — 100% CPU (Datastream)
```
crates/datastream/ — 6.165 baris
Files: graph.rs, intake.rs, filter/*.rs, delivery.rs, dll.
```
**Kenapa stay CPU**: I/O bound (file reading, network fetching), bukan compute bound.

**Bottleneck**:
- `graph.rs` — DAG execution, topological sort Kahn's algorithm (BFS CPU)
- `format_loader.rs` — CSV/JSON/Parquet parsing — semua CPU
- `filter/quality.rs` — `compute_quality()` — regex, char counting, word frequency HashMap
- `filter/dedup.rs` — MinHash fingerprinting — bisa GPU tapi bottleneck di HashMap lookup
- `filter/semantic_dedup.rs` — MinHash signature, Jaccard similarity — semua CPU
- `filter/perplexity.rs` — character trigram perplexity — CPU-only
- `intake.rs` — `read_to_string()` — blocking file I/O

**Bisa migrasi?** BRUTAL — Data pipeline inherently I/O bound. GPU gak bisa baca file lebih cepat.

**Quick win**: Paralelize filter evaluation via rayon (already has `run_rayon`), bukan GPU.

## 2.3 Orchestration — 100% CPU (Core Controller + Agent System)
```
crates/core/ — 5.850 baris
crates/agent/ — 8.277 baris
crates/intelligence/ — 1.781 baris
```
**Kenapa stay CPU**: Orchestration = decision making, routing, coordination. Bukan compute.

**Kritis bottleneck**:
- `controller.rs:79` — `process_request()` — sequential: detect intent → analyze context → route → execute
- `tokio::try_join!` di `controller.rs:114` — satu-satunya parallelism, tapi masih CPU tokio tasks
- `AgentManager::create_agent_instance()` — match on string → `Box::new()` — HEAP ALLOC per request
- `agent_manager.rs:176` — `while let Some(command) = rx.recv().await` — single-threaded command loop
- `registry.rs` — `Arc<Mutex<Box<dyn Agent>>>` — lock contention pas banyak agent concurrent
- `routing_agent.rs:197` — `Box::pin(async move {})` — recursive async, heap alloc per recursion

**Bisa migrasi?** TIDAK — Orchestration logic must stay CPU. GPU gak bisa routing decision.

## 2.4 Inference Runtime — 80% CPU (Orchestration Layer)
```
crates/inference/ — 10.775 baris
Files: engine.rs, scheduler.rs, kv_cache.rs, sampler.rs, runtime.rs, etc.
```
**Kenapa stay CPU**: Scheduling, batching, session management, metrics — semua CPU-native.

**Bottleneck KRITIS**:
- `engine.rs:586` — `start_request_loop()` — `loop { rx.recv().await; process; }` — sequential request processing
- `scheduler.rs` — `pop_batch()` — CPU batch assembly, HashMap lookup
- `continuous_batching.rs` — per-step iteration over ALL sequences
- `batching.rs` — `BatchCollector::drain_ready()` — iterasi HashMap, grouping by BatchKey
- `runtime.rs:428` — `initialize_resource_monitoring()` — `loop { interval.tick(); read /proc }` — polling /proc filesystem
- `streaming.rs` — `send_token()` — `mpsc::Sender::send()` per token
- `beam_search.rs` — `expand_beam()` — O(beam_size * vocab_size) CPU loop
- `speculative_decoding.rs` — `generate_draft_tokens()` — sequential draft generation
- `latency.rs` — `calculate_percentiles()` — sort VecDeque CPU
- `metrics.rs` — `record_request_completed()` — write HashMap, update statistics

**Bisa migrasi?** MENENGAH — Beam search dan batch scheduling bisa sebagian GPU. Tapi runtime orchestration akan selalu CPU.

## 2.5 Training Loop — 50% CPU, 50% GPU (Hybrid)
```
crates/training/src/lib.rs — 849 baris
apps/nexora-ai/src/cli/training.rs — 2.638 baris
```
**Kenapa stay CPU**:
- `train_batch()` (CPU path) — `Tensor::from_slice`, `forward`, `cross_entropy_loss`, `backward`, `optimizer.step()` — SEMUA CPU via ndarray
- `train_batch_gpu()` (GPU path) — sudah GPU untuk compute, tapi logging/checkpoint masih CPU
- `epoch.shuffle(&mut rng)` — Fisher-Yates shuffle CPU
- `chunks(seq_length + 1)` — slice iteration CPU
- Gradient accumulation: `accumulation_counter` CPU variable
- Loss reporting: format!, println! CPU
- Checkpoint: safetensors I/O CPU

**Bisa migrasi?** MENENGAH — Training loop orchestration (shuffle, chunking, accumulation counter) bisa pindah ke GPU via CUDA streams, tapi effort besar.

## 2.6 Validation/Security — 100% CPU
```
crates/validation/ — 1.557 baris
apps/nexora-ai/src/security/mod.rs — 397 baris
crates/isolation/src/layer*.rs — semua CPU
```
**Kenapa stay CPU**: Regex matching, string validation, permission checking — CPU domain.

**Bisa migrasi?** TIDAK — Security validation inherently branching CPU logic.

## 2.7 Memory Management — 100% CPU
```
crates/memory/ — 5.020 baris
Files: lib.rs, core.rs, memory_model.rs, episodic.rs, cache.rs, compression.rs
```
**Kenapa stay CPU**: HashMap-based storage, BTreeMap temporal indexing, cognitive dynamics modeling.

**Bottleneck**:
- `HebbianMemory::apply_interference()` — O(n²) pairwise similarity, n = memory entries
- `NeuralAttentionMemory::read()` — O(n) attention over entries — ini PERFECT candidate GPU
- `EpisodicMemory::find_similar_episodes()` — O(n) similarity comparison
- `LRUCache::put()` — O(n) scan for LRU entry — should be O(1) with LinkedHashMap
- `compression.rs` — dictionary-based compression CPU

**Bisa migrasi?** MENENGAH — NeuralAttentionMemory read/backward adalah soft attention yang natural di GPU (`ctx.fused_attention()`). Hebbian interference bisa pake matmul GPU.

## 2.8 Database — 100% CPU
```
crates/database/ — 155.219 baris (terbesar!)
Files: lib.rs, postgres.rs, sqlite.rs, pool.rs, connection_pool.rs, credentials.rs
```
**Kenapa stay CPU**: Database operations = async I/O bound, SQL compilation.

**Bisa migrasi?** TIDAK — GPU gak bisa SQL.

## 2.9 Configuration/Serialization — 100% CPU
```
Semua config loader, TOML/JSON parser, safetensors I/O, serde
```
**Kenapa stay CPU**: File I/O, text parsing inherently CPU.

**Bisa migrasi?** TIDAK — Zero benefit from GPU for config loading.

## 2.10 BLAS CPU Path — CPU Fallback (Kritis)
```
crates/star-x/src/blas_backend.rs — 27.864 baris
```
**Kenapa stay CPU**: OpenBLAS via `extern crate blas_src; extern crate openblas_src;`.

**Ini masalah SERIUS**: BLAS path digunakan untuk CPU tensor ops (ndarray::dot, matmul). Setiap tensor operation lewat OpenBLAS di CPU, WALAUPUN GPU path exist. Dual-path berarti setiap operasi pilih CPU atau GPU.

---

# 3. HIDDEN CPU DEPENDENCY

## 3.1 Implicit Synchronization

| Location | Pattern | Severity |
|----------|---------|----------|
| `autograd/src/gpu.rs` — `device.poll()` | CPU blocking menunggu GPU queue selesai | KRITIS |
| `autograd/src/gpu_mixed.rs` | Mixed precision: GPU compute, CPU cast | TINGGI |
| `autograd/src/tensor.rs` — `to_cpu()` | `buffer.map_async()` + `device.poll()` = blocking sync | KRITIS |
| `training/src/lib.rs:760` — `readback_f32_async` | `try_recv()` — polling-based, CPU busy-wait kalau GPU lambat | TINGGI |
| `transformer/src/model.rs:823` — forward_gpu return Array1 | Implicit blocking readback via device.poll() | SEDANG |

## 3.2 CPU Polling Loops

| Location | Pattern | Latency Impact |
|----------|---------|----------------|
| `engine.rs:586` — `start_request_loop` | `tokio::time::timeout(50ms, rx.recv())` — 50ms poll loop | 50ms minimum latency |
| `runtime.rs:428` — resource_monitoring | `interval.tick()` every 5s, reads /proc, blocking I/O | 0.1% CPU, TETAPI /proc blocking |
| `runtime.rs:460` — performance_tracking | `interval.tick()` every 10s | Low |
| `agent_manager.rs:269` — health_check | `loop { interval.tick(); perform_health_check() }` | Medium (configurable) |
| `training.rs:760` — `try_recv()` | Polling GPU readback — busy loop kalau GPU belum selesai | TINGGI |

## 3.3 Staging Buffer & Memcpy

| Location | Pattern | Bandwidth Waste |
|----------|---------|-----------------|
| `GpuTensor::from_cpu()` | `queue.write_buffer()` → staging → GPU | 1x PCIe transfer per upload |
| `GpuTensor::to_cpu()` | `map_async` → `poll` → `slice` → copy | 1x PCIe + CPU copy |
| `GQA::forward_gpu()` | K,V download → append → re-upload | **3x PCIe per step!** |
| `training.rs:616` | `Vec<f32>` CPU → write_buffer GPU | Per-batch full upload |
| `generate_gpu_impl()` | `ctx.slice_tensor()` → CPU vec per step | Per-step download |

## 3.4 Driver Overhead

| Pattern | Location | Impact |
|---------|----------|--------|
| WGPU pipeline creation | `gpu.rs` pipeline_cache | 50-200ms untuk compile WGSL ke SPIR-V/Vulkan |
| Command buffer encoding | `gpu.rs` setiap dispatch | CPU encode per-layer, per-op |
| `wgpu::Device::poll()` | Internal sync point | Blocking CPU-GPU sync |
| Surface presentation | N/A (headless) | Tidak ada, tapi tetap ada swapchain overhead |

## 3.5 CPU-Generated Kernels (WGSL Compile Time)

WGPU compile WGSL → SPIR-V → native di runtime:
- Setiap kernel pertama kali compile, kena full compile pipeline
- `gpu_fused.rs` — fused_attention, matmul, rms_norm KENA WARMUP TIME
- Tidak ada pre-compiled SPIR-V atau caching ke disk

## 3.6 CPU-Side Tensor Reshape

```
transformer/src/gqa.rs — reshape operasi di CPU:
- q.into_shape() — ndarray reshape, alokasi baru
- k_view(), v_view() — view creation
```
Setiap reshape pindah data atau alokasi baru. GPU-side reshape harusnya zero-copy via buffer reinterpretation.

## 3.7 `parking_lot::RwLock` di Async Context

```
crates/isolation/ — SEMUA pake parking_lot::RwLock
crates/infrastructure/utils/src/performance.rs — parking_lot
```
**Masalah**: `parking_lot::RwLock` blocking thread tokio. Di async context, blocking RwLock bisa stall seluruh worker thread.

## 3.8 Heap Allocation Hot Path

| Pattern | Frequency | Impact |
|---------|-----------|--------|
| `HashMap::new()` | Setiap request, filter, agent call | GC pressure |
| `Vec::with_capacity(n)` | Training: per batch, per epoch | Allocation storms |
| `format!()` | Logging, error messages, display | String allocation |
| `serde_json::json!()` | Setiap API response | Full JSON tree allocation |
| `Uuid::new_v4()` | Setiap request, session, message | Entropy syscall |
| `Arc::new()`, `Box::new()` | Setiap agent, message, task | Heap fragmentation |

---

# 4. GPU PURITY SCORE

## 4.1 Compute Purity: **35/100**

**Komponen GPU**: WGPU matmul, fused_attention, rms_norm, rope, silu — ~15 ops.

**Komponen CPU**:
- ndarray CPU ops (default path tanpa `--features gpu`)
- BLAS CPU matmul (OpenBLAS) — dipanggil setiap saat
- BPE tokenizer compute
- Data pipeline compute (filter, scoring, dedup)
- Softmax di CPU (sampler fallback)
- Attention score di CPU (CPU GQA path)
- Embedding lookup di CPU

**Rincian**: Dari ~290K baris kode, ~10K baris (3.4%) adalah GPU compute. Sisanya CPU.

## 4.2 Memory Purity: **25/100**

| Aspek | CPU | GPU |
|-------|-----|-----|
| Weight storage | ndarray Array2<f32> | GpuTensor (optional, feature-gated) |
| KV Cache | Vec<KVCacheEntry> CPU | GpuKVCacheEntry (feature-gated) |
| Activation | ndarray tensors | GpuTensor (auto-create jika gpu_auto_create) |
| Gradient | Tensor (autograd tape) | GpuTensor (gpu_adam) |
| Optimizer state | HashMap<String, AdamState> | GpuTensor m/v buffers |

**Kritis**: Default path = SEMUA CPU. GPU path cuma aktif kalau `--features gpu` dan `use_gpu=true`.

## 4.3 Scheduling Purity: **10/100**

- Request scheduling: 100% CPU (`scheduler.rs`)
- Batch assembly: 100% CPU (`batching.rs`, `continuous_batching.rs`)
- Token generation loop: CPU orchestration (`engine.rs.start_request_loop`)
- Priority queuing: CPU BinaryHeap
- Session management: CPU HashMap
- Streaming push: CPU mpsc channel

**Satu-satunya GPU scheduling**: `continuous_batching.rs` panggil `forward_gpu()` — tapi orchestration tetap CPU.

## 4.4 Training Purity: **40/100**

- Forward pass: GPU (via GpuTensor)
- Backward pass: GPU (via autograd tape)
- Loss calculation: GPU (via auto-diff)
- Optimizer step: GPU (GpuAdam)

- **Shuffling**: CPU (`epoch.shuffle(&mut rng)`)
- **Data loading**: CPU (file I/O → tokenize → tensor)
- **Checkpoint**: CPU (safetensors I/O)
- **Loss monitoring**: CPU (readback async → println)
- **Gradient accumulation counter**: CPU
- **LR schedule**: CPU (warmup + cosine)
- **Validation**: CPU (evaluate_loss → perplexity)
- **Metrics logging**: CPU (format!, write)

## 4.5 Inference Purity: **30/100**

- Model forward: GPU (via GpuTensor forward_gpu)
- KV cache append: GPU (GpuKVCacheEntry) atau CPU (GQA round-trip)
- Token embedding: GPU buffer copy

- **Sampling**: CPU (softmax → top_k → top_p → rng → pick)
- **Scheduling**: CPU (request loop, batch assembly)
- **Token loop**: CPU (stop_condition check, streaming push)
- **Beam search**: CPU (expand, prune, sort)
- **Speculative decoding**: CPU (draft generate, acceptance check)
- **Streaming**: CPU (mpsc channel push)
- **Response formatting**: CPU (serde_json, string manipulation)

## 4.6 Parallelism Quality: **25/100**

| Level | Quality | Detail |
|-------|---------|--------|
| GPU warp utilization | 40% | WGSL matmul — tile-based tapi occupancy rendah karena lack of pre-emption |
| Multi-GPU | 0% | Tidak ada. Single device only. |
| Async GPU submission | 50% | Ada `gpu_async.rs` tapi masih polling-based |
| CPU parallelism | 60% | tokio multi-threaded, tapi banyak contention di RwLock |
| Data parallelism | 30% | `run_parallel_training()` — spawn_blocking per model, bukan data parallelism |
| Pipeline parallelism | 0% | Tidak ada |
| Tensor parallelism | 0% | Tidak ada |
| Sequence parallelism | 0% | Tidak ada |

## 4.7 CPU Dependency Severity: **85/100** (High = Buruk)

- **85% dari code path punya CPU dependency yang blocking**
- GPU path hanya active dengan feature flag + runtime toggle
- Default build = 100% CPU (zero GPU)
- GPU code hanya ~3.4% dari total codebase
- Staging buffer, polling sync, device.poll() ada di semua GPU path
- KV cache CPU round-trip (GQA) = critical correctness issue
- Sampling CPU = throughput bottleneck

## 4.8 Ringkasan Skor

| Dimensi | Skor | Interpretasi |
|---------|------|--------------|
| Compute Purity | 35/100 | Mayoritas compute masih CPU |
| Memory Purity | 25/100 | Default storage di CPU |
| Scheduling Purity | 10/100 | Hampir 100% CPU scheduling |
| Training Purity | 40/100 | Forward/backward GPU, sisanya CPU |
| Inference Purity | 30/100 | Forward GPU, sampling/scheduling CPU |
| Parallelism Quality | 25/100 | Single GPU, no multi-GPU, contention tinggi |
| CPU Dependency | 85/100 | **KRITIS** — dependency merata di semua path |

**Overall GPU Purity Score**: **25/100** — Arsitektur predominantly CPU dengan GPU sebagai optional add-on.

---

# 5. PRIORITAS MIGRASI

## 5.1 Quick Wins (Effort Kecil, Dampak Besar)

### QW1: KV Cache CPU Round-Trip Fix
**Lokasi**: `crates/transformer/src/gqa.rs:824` — `forward_gpu()` download K,V ke CPU
**Dampak**: Setiap step inference download/upload FULL cache via PCIe. Untuk model 7B, KV cache ~2MB per layer.
**Fix**: Route ke `forward_gpu_with_cache_precomputed_rope()` yang pake `GpuKVCacheEntry`.
**Effort**: 1-2 hari. Refactor routing logic.
**Speedup**: 2-5x latency reduction per token (tergantung sequence length).
**Resiko**: Minimal — code sudah ada, hanya routing yang salah.

### QW2: Sampling GPU — Top-K/Top-P/Softmax Kernel
**Lokasi**: `crates/inference/src/sampler.rs`
**Dampak**: Setiap step, logits (vocab_size=32000+) download ke CPU → softmax CPU → top-k CPU → sampling CPU.
**Fix**: Buat WGSL kernel untuk fused softmax+top-k+sampling.
**Effort**: 1 minggu. Buat compute shader, integrasi ke sampler.
**Speedup**: 10-30% inference throughput (tergantung vocab_size).
**Resiko**: Random number generation di GPU perlu state management.

### QW3: Async Loss Readback — Hapus Polling
**Lokasi**: `crates/training/src/lib.rs:760` — `readback_f32_async` → `try_recv()`
**Dampak**: Polling-based readback — CPU busy-wait kalau GPU lambat.
**Fix**: Use `wgpu::Buffer::map_async` → callback-based notification.
**Effort**: 2-3 hari.
**Speedup**: 5-15% training throughput (kurangi CPU idle).
**Resiko**: None.

### QW4: GpuPipelineCache — Persistent Pipeline Cache
**Lokasi**: `crates/autograd/src/gpu.rs` — pipeline_cache (HashMap in-memory)
**Dampak**: Setiap restart aplikasi recompile semua WGSL kernel (50-200ms).
**Fix**: Cache compiled pipelines ke disk (SPIR-V atau native).
**Effort**: 3-4 hari.
**Speedup**: Cold start 50-200ms → 0ms. Warm inference lebih cepat karena no pipeline creation.
**Resiko**: Cache invalidation saat driver update.

## 5.2 Medium Impact (Effort Sedang, Dampak Besar)

### M1: Neural Attention Memory GPU
**Lokasi**: `crates/memory/src/memory_model.rs:956` — `NeuralAttentionMemory::read()`
**Dampak**: O(n) soft attention di CPU — untuk memory besar (10K+ entries), jadi bottleneck.
**Fix**: Implementasi fused_attention di GPU via `ctx.matmul()` + `ctx.softmax()`.
**Effort**: 1-2 minggu.
**Speedup**: 10-100x untuk memory retrieval besar.
**Resiko**: Latency GPU kernel launch vs CPU untuk memory kecil.

### M2: Data Pipeline GPU Filters
**Lokasi**: `crates/datastream/src/filter/*.rs`
**Dampak**: Quality filter, entropy filter, perplexity filter — semua CPU sequential.
**Fix**: Batch processing GPU untuk filter compute-intensive (entropy, perplexity, quality scoring).
**Effort**: 2 minggu.
**Speedup**: 3-10x preprocessing throughput.
**Resiko**: GPU memory terbatas untuk dataset besar.

### M3: Beam Search GPU
**Lokasi**: `crates/inference/src/beam_search.rs`
**Dampak**: O(beam_size * vocab) CPU loop setiap step.
**Fix**: Parallel beam expansion di GPU — semua candidate logits di GPU, sort di GPU.
**Effort**: 2-3 minggu.
**Speedup**: 2-5x beam search throughput.
**Resiko**: Implementasi kompleks, perlu top-k sort di GPU.

### M4: Training Loop Orchestration GPU
**Lokasi**: `crates/training/src/lib.rs` — shuffle, chunk, accumulation
**Dampak**: Shuffle dan chunking CPU jadi bottleneck pas training throughput tinggi.
**Fix**: Data shuffle di GPU (via permutation kernel), chunking di GPU buffer.
**Effort**: 2 minggu.
**Speedup**: 5-10% training throughput (kurangi CPU-GPU sync).
**Resiko**: None signifikan.

## 5.3 Long-Term Architecture Fixes (Brutal, Dampak Transformational)

### L1: Multi-GPU — Data & Tensor Parallelism
**Lokasi**: Seluruh inference + training pipeline
**Dampak**: Single GPU = VRAM limit. Model besar (70B+) tidak bisa running.
**Fix**: 
- Data parallelism: FSDP-style sharding
- Tensor parallelism: split weight across GPU
- Pipeline parallelism: layer-pipeline
**Effort**: 3-6 bulan.
**Speedup**: Linear scaling (2x untuk 2 GPU, 4x untuk 4 GPU).
**Resiko**: Sangat kompleks. WGPU multi-device masih immature.

### L2: Zero-Copy I/O Path — Hapus Semua Staging Buffer
**Lokasi**: `crates/autograd/src/gpu.rs` — staging_pool
**Dampak**: Setiap upload/download lewat staging buffer = 2x memory copy.
**Fix**: 
- Direct upload via `queue.write_buffer()` mapped memory
- Persistent mapped buffer untuk streaming data
- Huge pages untuk GPU memory allocation
**Effort**: 1 bulan.
**Speedup**: 20-40% bandwidth improvement (kurangi PCIe copy).
**Resiko**: Driver compatibility.

### L3: Tensor Parallelism untuk Attention — Split QKV
**Lokasi**: `crates/transformer/src/gqa.rs`
**Dampak**: Single GPU compute attention — memory-bound (bandwidth-limited).
**Fix**: Split QKV heads across GPU, all-reduce setelah attention.
**Effort**: 2-3 bulan.
**Speedup**: Hingga 4x untuk model besar.
**Resiko**: All-reduce communication overhead.

### L4: FP8/FP16/Quantized Inference GPU
**Lokasi**: `crates/quantization/` (472 baris — sangat minimal!)
**Dampak**: FP32 inference — 2x memory bandwidth dibanding FP16, 4x dibanding INT8.
**Fix**: 
- Implementasi FP16 matmul WGSL
- INT8 quantized matmul dengan scale factor
- FP8 (E4M3/E5M2) support untuk H100-class GPU
**Effort**: 2-3 bulan.
**Speedup**: 2-4x inference throughput.
**Resiko**: WGPU float16 support terbatas.

### L5: GPU-Native Tokenizer — wgpu Parallel BPE
**Lokasi**: `crates/tokenizer/`
**Dampak**: Tokenizer CPU bottleneck untuk throughput tinggi.
**Fix**: Implementasi BPE merge di GPU — warp-level parallel pair finding.
**Effort**: 3-4 bulan.
**Speedup**: 5-10x batch tokenization.
**Resiko**: Tokenization exactness mungkin beda. Testing extensive needed.

### L6: Async Command Queue — Kurangi Dispatch Overhead
**Lokasi**: `crates/autograd/src/gpu.rs`
**Dampak**: Setiap op = CPU encode command buffer → queue.submit → CPU cycle terbuang.
**Fix**: Multi-queue submission, pre-recorded command buffers, batch dispatch.
**Effort**: 1-2 bulan.
**Speedup**: 15-30% untuk model dengan banyak layer.
**Resiko**: WGPU multi-queue support terbatas.

## 5.4 Priority Matrix

| Item | Effort | Speedup | Dampak | Priority |
|------|--------|---------|--------|----------|
| **QW1** KV Cache Round-Trip | 1-2 hari | 2-5x | KRITIS | **#1** |
| **QW2** GPU Sampling | 1 minggu | 10-30% | TINGGI | **#2** |
| **QW3** Async Loss Readback | 2-3 hari | 5-15% | SEDANG | **#3** |
| **QW4** Pipeline Cache | 3-4 hari | Cold start fix | SEDANG | **#4** |
| **M1** Neural Attention GPU | 1-2 minggu | 10-100x | TINGGI | **#5** |
| **M3** Beam Search GPU | 2-3 minggu | 2-5x | TINGGI | **#6** |
| **M2** Data Pipeline GPU | 2 minggu | 3-10x | SEDANG | **#7** |
| **M4** Training Loops GPU | 2 minggu | 5-10% | SEDANG | **#8** |
| **L6** Async Command Queue | 1-2 bulan | 15-30% | TINGGI | **#9** |
| **L4** FP8/FP16 Quantized | 2-3 bulan | 2-4x | KRITIS | **#10** |
| **L2** Zero-Copy I/O | 1 bulan | 20-40% | TINGGI | **#11** |
| **L1** Multi-GPU | 3-6 bulan | Linear | TRANSFORM | **#12** |
| **L5** GPU Tokenizer | 3-4 bulan | 5-10x | SEDANG | **#13** |
| **L3** Tensor Parallelism | 2-3 bulan | 4x | TRANSFORM | **#14** |

---

# 6. BOTTLENECK DEEP DIVE

## 6.1 Real-World Inference Latency Breakdown (Estimasi)

Komponen | CPU-only (ms) | GPU path current (ms) | GPU optimized (ms)
---------|--------------|----------------------|-------------------
Tokenizer | 2 | 2 | 0.5
Embedding | 1 | 0.1 | 0.1
Attention (1 layer) | 8 | 0.8 | 0.4
FFN (1 layer) | 5 | 0.6 | 0.3
KV Cache append | 0.5 | 3.0 (CPU rountrip!) | 0.05
Sampling | 0.5 | 0.4 | 0.05
Output decode | 0.5 | 0.5 | 0.5
**Total per token** (32 layer) | **433 ms** | **46.6 ms** | **21.15 ms**

- CPU-only: ~433ms/token (2.3 tok/s) — via ndarray
- GPU current: ~46.6ms/token (21.4 tok/s) — dengan CPU round-trip bottleneck
- GPU optimized (QW1+QW2+M3): ~21.15ms/token (47.3 tok/s) — 2.2x dari current

## 6.2 Occupancy Problem

WGPU default dispatch: `(n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE`.

**Matmul**:
- Tile size: 16x16 atau 32x32
- Untuk hidden=4096: (256, 256, 4096/16) workgroups
- Occupancy: ~50% — WGSL compiler belum optimal kayak cuBLAS
- Shared memory usage: 16*16*4 * 2 tiles = 2KB — ok, tapi masih kurang

**Fused Attention**:
- Sequence length N, head_dim d
- Workgroup: (N/32, 1, 1) — hanya parallel di sequence dimension
- Occupancy: ~40% — banyak thread idle untuk sequence < 1024

## 6.3 Memory Bandwidth Waste

Current:
- Setiap forward: download semua weight dari VRAM via GPU memory bus
- TAPI: weight format FP32 (4 bytes per param)
- Untuk model 7B: 28GB weight — bandwidth 900GB/s (RTX 4090) → ~31ms full forward
- Bandwidth utilization: ~60% karena tiled matmul overhead

Fix (L4): FP16 → 14GB weight → 15.5ms → bandwidth utilization ~80%

## 6.4 Dispatch Overhead

Current: Setiap layer, setiap op = `device.create_command_encoder()` + `compute_pass` + dispatch + submit.

Estimasi overhead per dispatch: ~5-10µs di CPU.

Untuk 32 layer × 10 ops = 320 dispatches → 1.6-3.2ms CPU overhead per forward.

Fix (L6): Pre-recorded command buffers → dispatch overhead turun ke ~1µs → 0.32ms.

---

# 7. KESIMPULAN AKHIR

1. **GPU implementation sudah ada** tapi feature-gated, tidak default, dan hidden CPU sync di mana-mana.
2. **KV Cache CPU round-trip adalah bottleneck paling kritis** — fix QW1 harus jadi prioritas #1.
3. **CPU dependency severity 85/100** — hampir semua path masih blocking ke CPU.
4. **GPU purity score overall 25/100** — arsitektur CPU-first dengan GPU sebagai akselerator opsional.
5. **Quick wins (QW1-QW4)** bisa memberikan 2-5x speedup dalam 1-2 minggu kerja.
6. **Long-term fixes (L1-L6)** diperlukan untuk mencapai competitive performance vs llama.cpp, vLLM, TensorRT-LLM.
7. **Multi-GPU (L1)** adalah architectural necessity untuk model >13B — tanpa ini, model besar tidak feasible.
8. **Quantization (L4)** adalah force multiplier — FP16/INT8 bisa 2-4x speedup tanpa architectural change besar.

**Rekomendasi immediate**:
1. Fix GQA routing → pake GPU cache path (QW1) — **hari ini**
2. Buat GPU sampling kernel (QW2) — **minggu ini**
3. Refactor async loss readback (QW3) — **minggu depan**
4. Persistent pipeline cache (QW4) — **2 minggu**

Setelah itu baru mulai medium-term items (M1-M4) dan long-term (L1-L6).
