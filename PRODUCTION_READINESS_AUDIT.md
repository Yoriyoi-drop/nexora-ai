# Nexora AI — Production Readiness Audit

**Codebase**: 346,335 LOC | 808 `.rs` files | 38 crates + 2 apps
**Audit date**: 2026-05-25
**Auditor**: Deep architecture + pattern analysis

---

## Estimasi Readiness: 35-40%

Sistem memiliki fondasi yang solid (wgpu GPU backend asli, autograd engine lengkap, arsitektur crate terstruktur), tapi sebagian besar fitur *high-level* adalah scaffolding / fake completion. Pipeline inference memiliki batching palsu, GPU fallback diam-diam di mana-mana, dan ada duplikasi arsitektur massif.

---

# CRITICAL

## 1. Batching GPU adalah Palsu — Sequential Loop per-Sequence

**File**: `crates/transformer/src/model.rs:1238`
**Function**: `forward_gpu_batched()`
**Issue**: Method bernama `forward_gpu_batched` memproses setiap sequence SATU PER SATU dalam for loop:

```rust
for seq_idx in 0..batch_size {
    let token_id = batch_tokens[seq_idx];
    // embedding, RoPE, forward through ALL blocks, download logits — for ONE sequence
    for (layer_idx, block) in self.blocks.iter().enumerate() {
        h = block.forward_gpu_with_cache(&h, &mut gpu_caches[seq_idx], ...);
    }
    let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
    all_logits.push(Array1::from_vec(logits_flat));
}
```

**Kenapa berbahaya**: Tidak ada batch matmul. Tidak ada tensor-level parallelism. Setiap sequence mengalokasikan GPU tensor sendiri, menjalankan forward pass sendiri, dan melakukan readback sendiri. Throughput tidak lebih baik daripada sequential processing — malah lebih buruk karena overhead setup per-sequence.

**Impact**: Nama `batched` menyesatkan. Di load tinggi (32+ concurrent requests), system tidak akan scale. Setiap request tetap dapat full serial latency.

**Saran**: Implementasi true batched matmul dengan concatenation across batch dimension, atau gunakan CUDA Graph / wgpu indirect dispatch. Sampai itu terjadi, hapus kata "batched" dari nama fungsi.

---

## 2. 50+ Silent GPU→CPU Fallback Path — Tidak Ada Auto-Disable

**Files**:
- `crates/inference/src/sampler.rs:71,153-167,194-208` — fallback GPU→CPU sampling
- `crates/inference/src/engine.rs:299,359` — "falling back to CPU"
- `crates/inference/src/inference_trait.rs:140` — "GPU forward failed, falling back to CPU"
- `crates/transformer/src/model.rs:299-300,308` — "falling back to CPU"
- `crates/inference/src/continuous_batching.rs:206` — "falling back to argmax"
- `crates/inference/src/decoding.rs:285,346-368` — 3 CPU fallback paths

**Issue**: Ada ~52 fallback path dari GPU ke CPU di seluruh inference pipeline. `Sampler.allow_gpu_fallback` default `true`. Bahkan jika 100% request fallback ke CPU, system **tetap mencoba GPU** setiap kali — tidak pernah disable otomatis.

**Impact**: Di production dengan GPU bermasalah, setiap request membayar overhead GPU attempt + CPU fallback. Latensi naik 2-5x tanpa user sadar.

**Saran**: Implementasi circuit breaker: jika fallback rate > threshold (e.g., 20% dalam 1 menit), disable GPU otomatis dan kirim alert. Hapus `allow_gpu_fallback: true` sebagai default.

---

## 3. `CausalLM` Clone = Duplikasi Semua Weight Matrix (GB-scale)

**File**: `crates/transformer/src/model.rs:175-210`
**Function**: `CausalLM::clone()`

```rust
fn clone(&self) -> Self {
    Self {
        token_embedding: self.token_embedding.clone(),  // [vocab_size, hidden_size] f32
        blocks: self.blocks.clone(),                     // semua transformer block
        lm_head: self.lm_head.clone(),                   // [vocab_size, hidden_size] f32
        precomputed_cos: self.precomputed_cos.clone(),
        precomputed_sin: self.precomputed_sin.clone(),
        ...
    }
}
```

**Kenapa berbahaya**: `Arc::new(model.clone())` di engine.rs:103 mengkopi SEMUA weight matrix. Untuk model 7B param = 28GB RAM per clone. Di engine, model di-clone multiple times: di `with_model()`, di `spawn_batch_processor()`, di setiap `process_batch()`.

**Saran**: Gunakan `Arc<CausalLM>` saja, jangan clone. Atau implementasi copy-on-write. Hapus implementasi Clone yang mengkopi weights.

---

## 4. `GpuKVCache` Sync Back ke CPU Setiap Token — O(n²) Data Transfer

**File**: `crates/transformer/src/model.rs:908-984`
**Function**: `forward_gpu_with_cache_provider()`

Setelah GPU forward, system melakukan sync back **setiap layer** untuk mengkopi K/V terbaru dari GPU buffer ke CPU cache — via staging buffer + `map_async` + polling loop.

**Impact**: Setiap token menghasilkan N_layers × 2 × (n_kv_heads × head_dim × 4) bytes transfer GPU→CPU. Untuk model 32 layer, 8 KV heads, 128 head_dim = 256KB per token sync. Di 2048 token = 512MB transfer.

**Saran**: Jika menggunakan GPU cache, jangan sync back ke CPU. Biarkan K/V tetap di GPU. CPU cache hanya perlu untuk CPU inference path.

---

# HIGH PRIORITY

## 5. Dua Scheduler Redundan — Tidak Ada yang Kanonik

**Files**:
- `crates/inference/src/scheduler.rs` — 576 lines, FIFO + BatchCollector
- `crates/runtime/src/scheduler.rs` — 708 lines, 5 strategi (FIFO/Priority/SJF/RoundRobin/Fair)

**Issue**: Dua struct `RequestScheduler` terpisah dengan logika duplikatif. RoundRobin dan Fair didefinisikan di enum `SchedulingStrategy` tapi `insert_into_queue()` hanya implementasi Priority. Dua strategi lain tidak punya implementasi.

**Impact**: Jika user memilih RoundRobin atau Fair, system fallback ke Priority tanpa warning. Kebingungan di operasional.

**Saran**: Hapus salah satu scheduler, pilih satu sebagai canonical. Implementasi RoundRobin/Fair atau hapus dari enum.

---

## 6. Tiga KV Cache Berbeda — Tidak Jelas yang Aktif

**Files**:
1. `crates/inference/src/kv_cache.rs` — LRU cache (398 lines, digunakan oleh engine.rs)
2. `crates/inference/src/paged_cache.rs` — PagedAttention-style (990 lines, global singleton, TIDAK dipakai)
3. `crates/runtime/src/kv_cache.rs` — Sharded cache (666 lines)
4. `crates/autograd/src/gpu_kv_cache.rs` — GPU page table (132 lines)
5. `crates/star-x/src/kv_cache.rs` — Star-X cache (50 lines)

**Issue**: Engine menggunakan simple LRU cache (`kv_cache.rs`) sementara `paged_cache.rs` yang jauh lebih sophisticated (PagedAttention-style) ada sebagai global `OnceLock` singleton yang tidak dipanggil oleh path manapun.

**Saran**: Integrasikan PagedAttention cache ke engine path, atau hapus. Jangan biarkan kode mati 990 lines.

---

## 7. Mixed Precision Infrastructure Ada Tapi Tidak Dipakai Inference

**File**: `crates/autograd/src/mixed_precision.rs` — 449 lines

`DType` enum: `F32`, `F16`, `BF16` — defined. `LossScaler`, `GpuDtype` — defined. Tapi:
- `inference` crate: **zero references** ke fp16/bf16/mixed_precision/AMP
- `gpu.rs:1176`: `GpuDtype::F32 | GpuDtype::F16 | GpuDtype::Bf16 => 4, // all 4-byte for now` — fp16/bf16 di-padded ke 4 byte = fp32

**Impact**: Inference berjalan di fp32 penuh. 2x VRAM usage, 2x bandwidth dibanding fp16. Model 7B yang seharusnya muat di 14GB VRAM malah 28GB.

**Saran**: Wire fp16 ke inference path. Implementasi true half-precision compute shaders di wgpu.

---

## 8. Monolithic `gpu.rs` — 4839 Lines Satu File

**File**: `crates/autograd/src/gpu.rs`

GPU context, tensor operations, matrix multiply, element-wise ops, reductions, softmax, sampling, memory pooling, shader compilation, pipeline cache, profiling — semua dalam SATU file.

**Saran**: Split menjadi submodule: `context.rs`, `matmul.rs`, `elementwise.rs`, `reductions.rs`, `sampling.rs`, `memory.rs`, `profiling.rs`.

---

## 9. `forward_keep_gpu` adalah Passthrough Identik

**File**: `crates/transformer/src/model.rs:1162-1168`

```rust
pub fn forward_keep_gpu(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> ... {
    self.forward_gpu(input_ids, kv_cache)
}
```

**Issue**: Method dengan nama berbeda (`forward_keep_gpu`) memanggil method yang sama persis (`forward_gpu`). Docstring mengklaim "preferred method when keep_on_gpu is set" tapi tidak ada perbedaan behavior. `keep_on_gpu` flag di-reset ke `false` saat checkpoint load (line 616) — tidak dipakai.

---

## 10. Speculative Decoding & Beam Search — Unwired

**Files**:
- `crates/inference/src/speculative_decoding.rs` — 386 lines (tidak dipanggil dari engine)
- `crates/inference/src/beam_search.rs` — 661 lines (tidak dipanggil dari engine)

**Issue**: Kedua fitur punya implementasi lengkap (config, strat, loop) tapi **tidak terhubung** ke `InferenceEngine`. Speculative decoding punya masalah: draft & target model harus TIPE SAMA karena single generic parameter.

**Saran**: Integrasikan ke engine path atau tandai explicit sebagai experimental/unused.

---

## 11. 1500+ `unwrap()` Panggilan di Production Code

**Critical unwraps**:
- `crates/training/src/data_parallel.rs:35,58,63,123,142,168,176` — `self.master.lock().unwrap()` di hot training loop. Poison = panic.
- `crates/echo-net/src/gpu_ops.rs:115,142` — `PIPELINE_CACHE.lock().unwrap()` di GPU ops.
- `crates/autograd/src/training_pipeline.rs:559-560` — `self.params.as_ref().unwrap()` — panics if params not set.
- `apps/nexora-ai/src/core/processing.rs:755-815` — `result.unwrap()` di processing pipeline.

**Saran**: Ganti dengan `?` operator atau `unwrap_or_else` dengan error handling yang proper. Prioritaskan yang di hot path dan yang bisa kena poison.

---

## 12. `parking_lot::Mutex` Global Singleton di Async Context

**File**: `crates/inference/src/paged_cache.rs:45`
```rust
pub static GLOBAL_PAGED_CACHE: OnceLock<Mutex<PagedKVCache>>
```

**Issue**: `parking_lot::Mutex` tidak aman di-hold across `.await`. Global singleton dengan `parking_lot::Mutex` bisa menyebabkan deadlock async jika ada code path yang hold lock sambil await.

**Files with same issue**:
- `crates/star-x/src/tensor_pool.rs:280` — `LazyLock<TensorPool>` dengan `parking_lot::Mutex`
- `crates/echo-net/src/gpu_ops.rs:105` — `LazyLock<Mutex<HashMap>>`

---

## 13. Scalar Triple-Nested Loops di CPU Attention — Extremely Slow

**File**: `crates/transformer/src/gqa.rs:458-534, 606-649, 688-759`

CPU GQA attention menggunakan explicit scalar `for` loops untuk RoPE dan score computation:

```rust
for b in 0..batch_size {
    for h in 0..self.num_heads {
        for d in 0..self.head_dim {
            // scalar assignment
        }
    }
}
```

**Impact**: Tanpa vektorisasi, CPU inference lambat ~100-1000x dibanding GPU. Untuk model dengan 32 heads, head_dim=128, seq_len=2048 = 8 juta iterasi per layer per token.

**Saran**: Gunakan ndarray axis operations atau implementasi dengan `rayon` parallel iterators.

---

## 14. Full Vocab `to_vec()` di Setiap Token Generation

**Files**:
- `crates/inference/src/engine.rs:878,887` — `logits.as_slice().unwrap_or(&[]).to_vec()` di generation loop
- `crates/inference/src/sampler.rs:127,315,387,451,483,534,558` — `.to_vec()` di setiap sampling method
- `crates/transformer/src/model.rs:341-350, 1123-1134` — `.to_owned()` di RoPE slice per forward

**Issue**: Setiap token generation step mengkopi full `[vocab_size]` vector. Untuk vocab 50k, itu 200KB per token. Di 2048 token = 400MB copy hanya untuk logits.

**Saran**: Gunakan view/slice reference, jangan copy. Sampling bisa dilakukan in-place.

---

## 15. `batch_size = 1` Hardcoded di CPU Forward

**File**: `crates/transformer/src/model.rs:324`
```rust
let batch_size = 1;
```

**Issue**: CPU forward tidak support batched input. Variable-length batch inference tidak mungkin di CPU. Setiap request harus forward satu-per-satu.

---

## 16. Global Singleton GPU Context — Multi-GPU Hanya di Type System

**File**: `crates/autograd/src/gpu.rs:65`
```rust
static GPU_CTX: OnceCell<GpuContext>;
```

`Device::Gpu(usize)` menerima device_id, tapi `OnceCell` hanya bisa di-initialize sekali dengan device 0. Multi-GPU tidak mungkin tanpa refactor total.

---

# MEDIUM PRIORITY

## 17. 88k LOC Models Crate — Sebagian Besar Metadata/Identity, Bukan Logika

**File**: `crates/models/` — 88,204 LOC, 150 files

10 model families (aether, axiom, cipher, genesis, kronos, nexum, omnis, spectra, swift, vortex), masing-masing dengan:
- `mod.rs`, `architecture.rs`, `config.rs`, `identity.rs`, `capabilities.rs`
- Sub-direktori `agents/` dengan 5-15 agent files per model

**Issue**: Perbedaan antar model FAMILIES sebagian besar adalah metadata (nama, capabilities list, deskripsi). Tidak ada model-specific arsitektur. Semua model menggunakan `CausalLM` yang sama dari `nexora-transformer`. Agents adalah prompt templates / capability lists, bukan kode executable yang berbeda secara fundamental.

**Indikasi fake completion**: Surface area besar (88k LOC, 150 files) tapi actual behavior diversity rendah. Ini ciri khas codebase yang di-"pump" agar terlihat besar.

---

## 18. GNAC — 60 File, 7k LOC, Sebagian Besar Scaffolding

**File**: `crates/gnac/src/` — 60 files

Submodules: canvas, collaboration, distillation, elastic, execution, experiment, intervention, lensing, logic, rce, sandbox, scheduler, smart_tensor, swarm.

**Issue**: Banyak submodule yang tampaknya baru kerangka:
- `execution/backend/cpu.rs` — ada, tapi `backend/gpu.rs` tidak ada
- `sandbox/security.rs`, `sandbox/verification.rs` — implementasi?
- `swarm/` — objective, pruning, search — apa benar dipakai?
- `collaboration/branching.rs`, `live_editing.rs` — fitur kolaborasi real-time di framework ML?

---

## 19. Quantization: Dequantize → Compute di fp32 → Tidak Ada Manfaat

**File**: `crates/transformer/src/quantized.rs` — 174 lines

Semua weight didequantize ke fp32 sebelum komputasi. Tidak ada quantized compute kernel. Tidak ada INT8/FP16 matmul.

**Kenapa**: Quantization tanpa quantized compute = hanya hemat penyimpanan, bukan komputasi. Kecepatan inference tetap fp32.

---

## 20. Global `OnceLock` State Bertebaran

Global singletons di seluruh codebase:
- `GLOBAL_PAGED_CACHE` — paged_cache.rs:45
- `GPU_CTX` — gpu.rs:65
- `GLOBAL_TOKENIZER` — decoding.rs:703
- `GLOBAL_BLAS` — blas_backend.rs:1097
- `GLOBAL_TENSOR_POOL` — tensor_pool.rs:280
- `GLOBAL_RECOVERY` — error_recovery.rs:697
- `GLOBAL_REGISTRY` — model_registry.rs:525
- `GLOBAL_SAFETY` — safety_gate.rs:335
- `MODEL_AGENTS` — model_agent_manager.rs:79
- `METRICS` — handlers.rs:10
- `ERROR_HANDLER` — error.rs:875
- `PIPELINE_CACHE` — gpu_ops.rs:105

**Issue**: Global state = testing nightmare, isolation impossible, hidden coupling.

---

## 21. `eprintln!` di Production Tensor Ops

**File**: `crates/autograd/src/tensor.rs:296`
```rust
eprintln!("[tensor::randn] shape mismatch (unexpected): {e}");
```

Debug print yang tertinggal di production code. Akan muncul di stderr setiap shape mismatch.

---

## 22. Regex Compilation di Setiap Call (Hot Path)

**Files** (tanpa static `Lazy<Regex>`):
- `crates/multimodal/src/caffeine/action_head/execution.rs:665-746` — 6+ regex per call
- `apps/nexora-ai/src/security/mod.rs:24-72` — per security check
- `crates/hallucination/src/pre_generation.rs:48-64` — 8 regex per check
- `crates/hallucination/src/in_generation.rs:30-40` — 5 regex per token

**Impact**: Regex compilation = NFA/DFA construction = expensive. Di hot path (per token), ini menambah latensi signifikan.

**Saran**: Gunakan `std::sync::LazyLock<Regex>` atau `once_cell::sync::Lazy<Regex>`.

---

## 23. `println!` di Training Library

**Files**:
- `crates/oracle/src/trainer.rs:210-669` — puluhan `println!` di training loop
- `crates/alignment/src/sparo/trainer.rs:404-419` — `println!` di training

**Impact**: Synchronous I/O blocking. Gunakan `tracing::info!` sebagai gantinya.

---

## 24. `Debug!`/`Trace!` di Per-Token Hot Path

**Files**:
- `crates/inference/src/sampler.rs:288` — trace per sampling
- `crates/inference/src/decoding.rs:117,259,553,582` — debug per sampling method
- `crates/inference/src/beam_search.rs:253,309,396` — debug per expansion

**Issue**: Walaupun `tracing` memiliki overhead rendah di level `info`, di level `debug`/`trace`, event tetap dievaluasi. Di hot path per-token, ini tetap punya cost.

---

# LOW PRIORITY

## 25. `is_cuda()` Adalah Legacy Alias

**File**: `crates/autograd/src/tensor.rs:250-253`
```rust
pub fn is_cuda(&self) -> bool {
    self.is_gpu()  // routes to wgpu, not CUDA
}
```

## 26. `std::thread::sleep` di Potentially Async Context

**File**: `crates/core/src/error_recovery.rs:704`
```rust
std::thread::sleep(std::time::Duration::from_millis(200));
```
Blocking call di async context = thread pool starvation.

## 27. Integer Cast Risky: `id as u8`, `m as i32`, `k as i32`

**Files**:
- `crates/foundation/src/causal_lm_model.rs:76` — `char::from(id as u8)` — truncation jika id > 255
- `crates/star-x/src/blas_backend.rs:454-456,479-481` — `k as i32`, `m as i32` — overflow risk di matrix besar

## 28. `#[allow(dead_code)]` pada Error Variants

**Files**:
- `crates/transformer/src/lib.rs:37` — `NotImplemented` error variant
- `crates/foundation/src/lib.rs:82` — `NotImplemented` error variant

## 29. Empty Match Arms di Op Dispatch

**Files**:
- `crates/autograd/src/ops/matmul.rs:109` — `_ => {}`
- `crates/autograd/src/ops/math.rs:67,170,252,313` — `_ => {}`

Unrecognized operations silently drop without error.

## 30. `block_on` di Production GPU Init

**File**: `crates/autograd/src/gpu.rs:458`
```rust
pollster::block_on(Self::new())  // blocking in async runtime
```

---

# Daftar Placeholder Tersembunyi

| Temuan | File | Baris | Kategori |
|--------|------|-------|----------|
| "GPU forward pass failed, falling back to CPU" | model.rs | 308 | Silent fallback |
| "Sampler failed ... falling back to argmax" | engine.rs | 359 | Degraded mode |
| "CPU fallback" | decoding.rs | 285 | Hidden fallback |
| "all 4-byte for now" (fp16/bf16 = fp32) | gpu.rs | 1176 | Fake mixed precision |
| "Scheduler not ready" | scheduler.rs | 133 | State placeholder |
| "Processor not ready" | processor.rs | 118 | State placeholder |
| "Engine not ready" | engine.rs | 245 | State placeholder |
| "Cache not ready" | kv_cache.rs | 256, 346 | State placeholder |
| `allow_gpu_fallback: true` (default) | sampler.rs | 71 | Bypass permanen |
| `forward_keep_gpu` = `forward_gpu` | model.rs | 1162-1168 | Identical redirect |
| RoundRobin/Fair unimplemented | scheduler.rs | 12-18 | Dead enum variants |
| PagedCache ada tapi tidak dipakai | paged_cache.rs | global | Dead code (990 lines) |
| Batch size = 1 di CPU | model.rs | 324 | Hardcoded limitation |
| `use_gpu` flag diabaikan | engine.rs | 55 | Compile-time gate vs runtime flag |

---

# Fake Completion Indicators

1. **Batching palsu**: `forward_gpu_batched` → sequential for loop
2. **Mixed precision palsu**: F16/BF16 di-padded ke 4 byte = fp32
3. **Speculative decoding unwired**: 386 lines, tidak dipanggil
4. **Beam search unwired**: 661 lines, tidak dipanggil
5. **PagedAttention unwired**: 990 lines, global singleton mati
6. **Quantization tanpa benefit**: Dequantize → fp32 compute
7. **forward_keep_gpu passthrough**: Nama berbeda, logika sama
8. **Scheduling strategies palsu**: 5 enum, 3 implementasi
9. **88k LOC models = metadata shell**: 10 model families, 1 backend
10. **GNAC 60 files = scaffolding**: Submodule depth tanpa substansi

---

# GPU-Native Namanya, CPU-Heavy Kenyataannya

| Bagian | Kesan | Realita |
|--------|-------|---------|
| `forward_gpu_batched` | Batched GPU | Sequential CPU-like loop |
| `GpuKVCache` | Full GPU cache | Sync back ke CPU setiap step |
| `MixedPrecision` | F16/BF16 | Padded ke 4 byte = FP32 |
| `PagedCache` | PagedAttention | Tidak dipakai |
| `sample_token_gpu` | GPU sampling | Fallback ke CPU di setiap error |
| `gpu` feature gate | GPU required | Compile-time only, runtime fallback diam-diam |
| `CUDA` reference | CUDA support | Wrapper of wgpu, not CUDA |

---

# Terlihat Parallel, Tapi Sebenarnya Serial

| Bagian | Ilusi | Realita |
|--------|------|---------|
| Batched forward | Batch parallelism | Sequential for loop |
| `tokio::spawn` per request | True parallelism | Green threads, same GPU context |
| `process_batch` | Batch processing | Individual `spawn_blocking` per request |
| `continuous_batching` | Continuous batch | Sequential prefill → generate per request |
| Rayon di GQA | Parallel attention | Hanya di satu variant (forward_with_kv) |
| `batch_dispatch` | Coalesced GPU ops | Still per-sequence, per-layer submit |

---

# Ringkasan Per Crate

| Crate | LOC | Readiness | Masalah Utama |
|-------|-----|-----------|--------------|
| `crates/autograd` | 20,739 | 70% | gpu.rs monolithic (4839 lines), mixed precision palsu |
| `crates/models` | 88,204 | 20% | 88k LOC metadata shell, fake agent variety |
| `crates/inference` | 12,698 | 40% | Batching palsu, fallback di mana-mana, unwired features |
| `crates/transformer` | 4,486 | 50% | CPU attention 3x nested loop, clone berat |
| `crates/gnac` | 7,322 | 15% | 60 file scaffolding, gpu backend tidak ada |
| `crates/foundation` | 3,607 | 60% | NotImplemented variants, dead code |
| `crates/reasoning` | 11,601 | 40% | SACA kompleks tapi apakah benar dipakai? |
| `crates/multimodal` | 9,298 | 50% | CAFFEINE pipeline, regex compile berulang |
| `crates/alignment` | 6,451 | 50% | SPARO/DPO/KTO/IPO infrastructure ada |
| `crates/runtime` | 3,690 | 40% | Scheduler duplikat, strategi palsu |
| `crates/core` | 6,128 | 60% | thread::sleep di async |
| `crates/datastream` | 7,010 | 70% | Paling solid, DAG pipeline real |
| `crates/star-x` | 8,640 | 50% | BLAS backend, KV cache |
| `crates/training` | 2,070 | 40% | unwrap di data_parallel hot path |
| `apps/nexora-ai` | 11,454 | 50% | Security regex compile per call, unwrap |

---

## Kesimpulan

Codebase ini memiliki **pondasi teknis yang solid** (wgpu GPU backend, autograd, transformer blocks, tokenizer, DAG data pipeline) tapi **lapisan aplikasi di atasnya sebagian besar adalah scaffolding**.

Polanya jelas: developer menghabiskan energi membuat struktur yang terlihat besar dan kompleks (88k LOC models crate, 60-file GNAC, 5 scheduler strategies, 3 KV caches, speculative decoding, beam search) tapi **lupa menyelesaikan koneksi antar komponen**.

Yang paling berbahaya untuk production:
1. **Batching tidak memberikan speedup** — orang akan deploy dengan ekspektasi throughput tinggi
2. **GPU fallback silent** — degradation tidak terdeteksi
3. **Memory explosion dari clone** — OOM di production tidak terhindarkan
4. **Deadlock potential dari parking_lot::Mutex di async** — random crash di load tinggi

### Rekomendasi 6 Bulan ke Production

1. **Refactor engine**: Hapus fake batching, implementasi true batched matmul
2. **Circuit breaker GPU**: Auto-disable GPU jika fallback rate tinggi
3. **Hapus semua `Arc<RwLock<HashMap>>` engine state**: Ganti dengan channel-based architecture
4. **Eliminate semua global singleton**: Testability dan isolation
5. **Ganti semua `unwrap()` di production**: Minimal di hot path
6. **Hapus dead code**: PagedCache, speculative decoding, beam search dari engine — atau integrasikan beneran
7. **Selesaikan mixed precision**: F16/bf16 di inference
8. **Split gpu.rs**: 4839 lines tidak maintainable

Setelah itu, baru codebase ini siap untuk staging load test, apalagi production.
