# Audit Produksi Readiness — Nexora AI

**Tanggal:** 25 Mei 2026
**Total LOC:** ~315.382 baris Rust
**Crates:** 43 workspace members
**Metodologi:** Static analysis + arsitektur deep-dive (bukan sekadar grep keyword)

---

## Estimasi Readiness Production: **~35%**

Codebase ini secara arsitektur sangat ambisius — tapi sebagian besar adalah **scaffolding yang kelihatan selesai**.
Banyak modul yang secara *struktur* sudah ada, tapi secara *behavior* masih sequential, fallback ke CPU,
atau bahkan tidak pernah dipanggil. Ini adalah "software yang dicat rumahnya tapi pondasinya lumpur."

---

# CRITICAL

Issue yang akan menyebabkan sistem **collapse atau silently wrong** di production.

---

## C1. Quantization: 4 Implementasi, 0 Digunakan

**File:** `crates/quantization/src/lib.rs` (481 LOC), `crates/atqs/src/awq.rs` (329 LOC), `crates/star-x/src/quantization.rs` (682 LOC), `crates/transformer/src/quantized.rs` (174 LOC)

**Deskripsi:** Ada **empat** implementasi kuantisasi terpisah:
1. `nexora-quantization` — menyatakan `QUANTIZATION_IS_STORAGE_ONLY: bool = true` di doc-nya.
2. ATQS AWQ — quantize/dequantize hanya dipakai di test sendiri.
3. Star-X QuantizationEngine — GPTQ, AWQ, mixed precision. Nol call site eksternal.
4. `transformer/src/quantized.rs` — **module tidak pernah dideklarasikan di lib.rs.** Dead code total.

**Semua implementasi hanya dequantize → compute in fp32.** Tidak ada quantized matmul kernel.
Zero performa benefit. Berat storage tetap sama di runtime.

**Kenapa berbahaya:** Setiap model yang mengklaim "quantized inference" sebenarnya jalan di fp32.
Inference lebih lambat dari yang seharusnya karena dequantize overhead + fp32 compute.

**Impact ke production:** Memory tidak berkurang, throughput inference tidak naik.
Kuantisasi adalah **cosmetic feature** yang tampak selesai tapi tidak berguna.

**Saran:** Pilih SATU implementasi, buang 3 lainnya. Implementasikan quantized matmul kernel
(INT8/INT4) yang benar-benar dipakai inference. Atau hapus klaim quantized dari dokumentasi.

---

## C2. GPU Mixed Precision: WGSL Shaders Ada, Inference Jalan di FP32

**File:** `crates/autograd/src/gpu_mixed.rs` (baris 1-7)

```
WARNING: This module provides F16/BF16 conversion primitives but is NOT yet
integrated into the inference pipeline. The inference engine currently uses
fp32 throughout.
```

**Deskripsi:** Ada 6 WGSL compute shader untuk F32↔F16/BF16 conversion, scale, unscaling.
Semua dikompilasi. Tapi **tidak ada satu jalur inference pun yang menggunakannya.**
Inference engine, KV cache, sampler, semua fp32.

**Kenapa berbahaya:** Ini adalah **half-implemented feature.** Kelihatannya mixed precision
support sudah ada (ada GPU pipeline, ada LossScaler, ada GpuDType), tapi kenyataannya:
- `AmpOptimizer::cast_model_to_compute_dtype()` adalah **stub no-op** (mixed_precision.rs:328)
- Tidak ada kode yang mengkonversi weight ke F16 saat GPU upload
- KV cache tidak punya F16 storage path
- Sampler tidak handle F16 logits

**Impact ke production:** GPU VRAM usage 2x dari yang seharusnya. Throughput inference GPU
setengah dari potensi. Sistem kelihatan sudah modern tapi jalan di fp32.

**Saran:** Selesaikan 3 integration point yang disebut di warning, atau hapus gpu_mixed module
sampai siap diintegrasikan.

---

## C3. GNAC ExecutionBackend: 4 dari 5 Backend Palsu

**File:** `crates/gnac/src/execution/mod.rs` (baris 42-43), `crates/gnac/src/execution/compiled.rs` (baris 51-57)

```rust
pub enum ExecutionBackend {
    CUDA,
    Vulkan,   // ❌ error: "not available"
    TPU,      // ❌ error: "not available"
    WebGPU,   // ❌ error: "not available"
    CPU,      // ✅ satu-satunya yang jalan
}
```

**Deskripsi:** Empat dari lima varian `ExecutionBackend` tidak memiliki implementasi. Vulkan, TPU,
dan WebGPU langsung return error. CUDA "ada" tapi implementasinya (`execute_cuda`) menggunakan
`nexora_autograd::gpu::GpuContext` yang adalah **wgpu** — bukan CUDA runtime sungguhan.
Tidak ada `cuda_runtime`, `cublas`, atau `cudnn` call.

**Kenapa berbahaya:** `ExecutionBackend` enum memberi ilusi bahwa GNAC dapat menjalankan
graph di berbagai hardware. Realitanya, hanya CPU yang berfungsi. Siapa pun yang memilih CUDA
akan mendapat wgpu fallback — yang notabene lebih lambat dari CPU-native untuk banyak operasi.

**Impact ke production:** Semua GNAC graph execution berjalan di CPU. Fitur "multi-backend"
adalah UI decoration. User tidak bisa memanfaatkan GPU accelerators.

**Saran:** Hapus varian yang tidak diimplementasikan dari enum publik. Atau implementasikan
satu backend non-CPU yang benar (misal WGSL via wgpu yang sudah ada).

---

## C4. Agent Coordinator: 7 dari 10 Strategi Koordinasi Palsu

**File:** `crates/shared/src/agent_coordinator.rs` (baris 281-294)

```rust
CoordinationStrategy::Adaptive => Box::new(SequentialCoordinator),        // ❌
CoordinationStrategy::ConsensusBased { .. } => Box::new(SequentialCoordinator), // ❌
CoordinationStrategy::PriorityBased => Box::new(SequentialCoordinator),    // ❌
CoordinationStrategy::EmpathyDriven => Box::new(SequentialCoordinator),    // ❌
CoordinationStrategy::Consensus => Box::new(SequentialCoordinator),        // ❌
```

**Deskripsi:** Dari 10 strategi koordinasi, hanya 3 yang benar-benar diimplementasikan
(Sequential, Parallel, Hierarchical). 7 sisanya — termasuk Adaptive, ConsensusBased,
PriorityBased, EmpathyDriven — semuanya **silently fallback ke SequentialCoordinator.**
Tidak ada warning, tidak ada log. Pengguna yang memilih Adaptive strategy akan mendapat
sequential execution tanpa tahu.

**Kenapa berbahaya:** Ini adalah **silent correctness bug.** Sistem mengaku melakukan
adaptive/consensus/priority coordination tapi realitanya sequential. Untuk multi-agent
workflow, bottleneck akan parah dan pengguna tidak akan tahu kenapa.

**Impact ke production:** Semua multi-agent orchestration kecuali Parallel dan Hierarchical
jalan sequential. Throughput multi-agent tidak scalable.

**Saran:** Buang strategi yang tidak punya implementasi nyata dari enum publik. Atau
implementasikan dengan benar.

---

## C5. DataParallel: 2 Implementasi, 0 Call Site

**File:** `crates/autograd/src/data_parallel.rs` (428 LOC), `crates/training/src/data_parallel.rs` (183 LOC)

**Deskripsi:** Dua implementasi data parallelism:
1. `autograd::DataParallel` — gradient accumulator, all reduce (CPU-only, ndarray-based)
2. `training::DataParallelTrainer` — multi-worker via `std::thread::scope`, CPU-only

**Keduanya tidak dipanggil oleh kode production mana pun di seluruh workspace.**
Zero import, zero instantiation, zero usage.

**Kenapa berbahaya:** Ini memberikan ilusi bahwa training sudah distributed/multi-GPU.
Tidak ada yang benar-benar menggunakan data parallelism. Semua training jalan single-device.

**Impact ke production:** Training tidak bisa scale ke multi-GPU. Klaim "data parallel training"
adalah kosmetik.

**Saran:** Hapus sampai benar-benar dibutuhkan, atau integrasikan dengan NCCL/cuda-aware MPI.

---

## C6. Continuous Batching: Sequential, Deprecated

**File:** `crates/inference/src/continuous_batching.rs` (baris 14, 35, 161)

```rust
#[deprecated(note = "Use InferenceEngine with sampler-based decoding instead")]
pub struct SequentialBatchingEngine<M> {
```

**Deskripsi:** File bernama `continuous_batching.rs` mengandung struct bernama
`SequentialBatchingEngine` yang sudah `#[deprecated]`. Fase prefill-nya memproses
satu sequence per satu dalam for loop (baris 181). Fase generation memang manggil
`forward_batched`, tapi implementasi default `forward_batched` di trait (inference_trait.rs:36-46)
adalah **sequential loop** — iterasi satu per satu:

```rust
input_ids.iter().zip(kv_caches.iter_mut())
    .map(|(&id, cache)| self.forward(&[id], cache))
    .collect()
```

**Kenapa berbahaya:** Sistem mengklaim support continuous batching (fitur vital untuk
production LLM serving seperti vLLM atau TensorRT-LLM). Realitanya, batching adalah
sequential loop dengan GPU upload/download setiap step. Token throughput tidak naik
dengan batch size.

**Impact ke production:** LLM serving throughput tidak scalable. Pada load tinggi,
setiap request tambahan akan memperlambat semua request (karena sequential).

**Saran:** Implementasi true continuous batching dengan padded batched matmul untuk
prefill dan generation. Atau hapus klaim continuous batching dari dokumentasi.

---

## C7. BLAA Bridge: Experimental / Deprecated

**File:** `crates/blaa/src/lib.rs`, `crates/inference/src/blaa_integration.rs` (baris 25)

```rust
#[deprecated(note = "BLAA bridge is experimental and may be removed in future releases")]
```

**Deskripsi:** BLAA (Black Language Model API) bridge ditandai deprecated di level trait.
Ini adalah external model API bridge — bagian vital dari sistem inference jika dipakai
untuk multi-model serving. Jika deprecated, seluruh jalur inference yang bergantung
padanya akan patah.

**Kenapa berbahaya:** Deprecated API di production adalah time bomb. Suatu hari akan
dihapus dan semua kode yang bergantung padanya akan compile break.

**Impact ke production:** External model integration tidak reliable. Tidak ada garansi
stability.

**Saran:** Jika mau dipakai, copot deprecated. Jika tidak, hapus.

---

## C8. GPU Blocking Spin Loop di Async Context

**File:** `crates/autograd/src/gpu_async.rs` (baris 25-44)

```rust
pub fn recv(&self) -> Result<T, GpuError> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match self.receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(val) => return Ok(val),
            Err(RecvTimeoutError::Timeout) => {
                // ...
                std::thread::sleep(std::time::Duration::from_millis(1)); // 🔴
                continue;
            }
```

Juga di `crates/autograd/src/gpu.rs` (baris 1526-1547, 4696-4740):
- `loop { device.poll(Wait, timeout=1ms); try_recv(); continue; }`
- `readback_inner()`: `loop { device.poll(Wait, 100ms); rx.recv_timeout(100ms); }` — 30s deadline

Juga: `device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })` —
**blocking infinite wait** — di `sync()` (baris 1086), `wait_device()` (baris 1098),
`dispatch_profiled_detailed()` (baris 1353).

**Deskripsi:** Kode GPU async memiliki blocking loops yang:
1. Tidak dibungkus `spawn_blocking` — memblokir async runtime thread
2. Polling dengan sleep 1ms — CPU-bound busy wait
3. Infinite blocking `poll(Wait, None)` — bisa hang selamanya jika GPU crash

**Kenapa berbahaya:** Jika dipanggil dari async context (misal inference engine),
akan memblokir tokio worker thread. Satu thread terblokir → semua task di worker
thread itu antri. Bisa menyebabkan **total throughput collapse** di bawah load.

**Impact ke production:** Resource starvation di async runtime. Latency spikes tidak
terprediksi. GPU hang → sistem freeze total.

**Saran:** Bungkus semua blocking wait dengan `tokio::task::spawn_blocking`.
Gunakan `device.poll(Wait, timeout=Some(duration))` dengan timeout wajar.
Ganti spin loop dengan channel-based notification.

---

# HIGH PRIORITY

## H1. Ratusan `.unwrap()` di Non-Test Code

**File:** Tersebar di ~30+ file production (transformer, inference, foundation, autograd, dll)

**Contoh:**
- `crates/transformer/src/trainable.rs`: 453-536 (checkpoint loading — 6 `.unwrap()`)
- `crates/transformer/src/model.rs`: 1985, 1995 (inference — 2 `.unwrap()`)
- `crates/inference/src/paged_cache.rs`: 729-889 (cache read — 6 `.unwrap()`)
- `crates/foundation/src/causal_lm_model.rs`: 1251-1322 (model lifecycle — 7 `.unwrap()`)
- `crates/quantization/src/lib.rs`: 381, 432, 466 (quantization — 3 `.unwrap()`)
- `crates/intelligence/src/serving/unified_api.rs`: 637-700 (model serving — 13 `.unwrap()`)
- `crates/core/src/async_executor.rs`: 640-658 (executor — 3 `.unwrap()`)
- `crates/models/src/swift/agents/fast_cache.rs`: 724, 732 (cache — 2 `.unwrap()`)
- `crates/autograd/src/training_pipeline.rs`: 102, 132, 135, 310 (training — `.expect()`)

**Deskripsi:** Production code dipenuhi `.unwrap()` dan `.expect()` yang akan panic
jika ada error. Checkpoint loading, model inference, cache operation — semua
menggunakan `.unwrap()` yang akan crash seluruh proses jika gagal.

**Kenapa berbahaya:** Satu GPU OOM, satu file corrupt, satu network timeout → process crash.
Tidak ada graceful degradation.

**Impact ke production:** Zero fault tolerance. Setiap error minor menjadi outage total.

**Saran:** Ganti semua `.unwrap()`/`.expect()` di non-test code dengan `?`, `.context()`,
atau `unwrap_or_else` dengan logging dan recovery.

---

## H2. Clone Berlebihan di Hot Path Inference

**File:** `crates/inference/src/engine.rs` (baris 278-314, 341, 522, 557, 634, 665, 696-703, 829-837, 1044-1052)
`crates/transformer/src/model.rs` (baris 957, 962, 1466, 1471)
`crates/transformer/src/gqa.rs` (baris 134-135, 922-923, 1257-1258)
`crates/inference/src/token_loop.rs` (baris 252, 492)
`crates/autograd/src/ops/math.rs` (baris 253-258, 293-294, 365-366)

**Contoh konkret:**
- `engine.rs:341`: `let mut all_ids = prompt_ids.clone()` — clone seluruh prompt per request
- `engine.rs:522-523`: `tokenizer.clone()`, `prompt_ids_for_loop = prompt_ids.clone()`
- `engine.rs:492`: `let mut stats = self.stats.read().await.clone()` — clone full stats di setiap read
- `token_loop.rs:252-253`: Clone prompt IDs 2x per request
- `math.rs:293-294`: **Double clone grad** — `grad.clone() * b_bc` dan `grad.clone() * a_bc`
- `model.rs:957,962`: KV cache clone per layer di forward

**Kenapa berbahaya:** Setiap clone di hot path inference = alokasi memori + copy data.
Untuk sequence panjang (4K+ token), clone prompt_ids berulang = O(n) alloc per iterasi.
Di backward pass, clone grad berarti allocation pressure tinggi.

**Impact ke production:** Memory allocation jadi bottleneck. GC/allocator pressure
meningkatkan latency p50 dan p99. Throughput turun drastis pada sequence panjang.

**Saran:** Gunakan Arc/Cow untuk sharing data. Refactor cloning pattern di backward ops
(cukup satu referensi, bukan clone). Gunakan bump allocator atau arena untuk temporary.

---

## H3. `std::sync::Mutex` di Async Context

**File:** `crates/runtime/src/batching/processor.rs` (baris 5), `crates/database/src/sqlite.rs` (baris 10),
`crates/hallucination/src/monitoring.rs` (baris 4), `crates/training/src/data_parallel.rs` (baris 1)

**Deskripsi:** `std::sync::Mutex` digunakan di async code. Ketika `lock()` dipanggil,
akan memblokir thread tokio worker. Jika lock contention tinggi, worker thread
terblokir dan semua task di thread itu antri.

**Kenapa berbahaya:** Di async runtime, blocking mutex dapat menyebabkan deadlock
dan thread pool starvation. Tokio worker threads terbatas — 1 blocked thread = loss
dari 1/N capacity.

**Impact ke production:** Latency instability, request timeout, dan potensi
deadlock di kondisi tertentu.

**Saran:** Ganti dengan `tokio::sync::Mutex` untuk critical section yang pendek,
atau `tokio::sync::RwLock` untuk read-heavy. Atau pindahkan blocking code ke
`spawn_blocking`.

---

## H4. EchoNet ISC: O(N⁴) CPU Fallback untuk Inverse FFT

**File:** `crates/echo-net/src/isc.rs` (baris 410-435)

```rust
// Simple inverse FFT implementation (for demonstration)  ← "for demonstration"
for i in 0..context.nrows() {        // N
    for j in 0..context.ncols() {    // N
        for ki in 0..context.nrows() {// N
            for kj in 0..context.ncols() { // N
```

**Deskripsi:** CPU fallback untuk Inverse IFFT adalah implementasi O(N⁴) naive dengan
4 nested loop. Komentar menyebut "for demonstration". Untuk matriks 256x256,
ini adalah 256⁴ = ~4 miliar iterasi.

**Kenapa berbahaya:** GPU path mungkin fallback, dan CPU fallback akan **menggantung
thread selama menit.** Tidak ada timeout atau progress check.

**Impact ke production:** Jika GPU gagal, inference akan stuck sangat lama. Bisa
menyebabkan request timeout di semua load.

**Saran:** Gunakan FFT library (rustfft) untuk CPU fallback. Atau set timeout
dan graceful error, bukan demo code.

---

## H5. Dead Code: transformer/src/quantized.rs Tidak Pernah Dikompilasi

**File:** `crates/transformer/src/quantized.rs` (174 LOC)

**Deskripsi:** Module `pub mod quantized;` tidak ada di `crates/transformer/src/lib.rs` (baris 10).
File ini berisi `QuantizedWeights`, `DequantizedWeights`, `BlockQuantizedWeights` —
tapi tidak pernah masuk ke kompilasi. Rust compiler akan melewatkannya.

**Kenapa berbahaya:** Memberi ilusi bahwa quantization sudah terintegrasi di
transformer layer. Ini adalah **fake completion** yang paling jelas:
struktur data sudah ada, fungsi sudah ditulis, tapi tidak pernah di-compile.

**Impact ke production:** Zero. Tapi membingungkan developer baru.

**Saran:** Hapus file, atau wire ke lib.rs dan integrasikan dengan inference.

---

## H6. GPU Sampler Silent Degradation

**File:** `crates/inference/src/sampler.rs` (baris 54-147)

**Deskripsi:** GPU sampler memiliki circuit breaker yang melacak `gpu_fallback_count`.
Jika GPU sampling gagal, secara diam-diam fallback ke CPU. Ada `is_gpu_degraded()`
dan `gpu_fallback_ratio()`, tapi tidak ada alarm atau metric exposure yang jelas.

**Kenapa berbahaya:** GPU bisa silent degrade ke CPU tanpa notifikasi.
Operator tidak tahu bahwa inference kini berjalan di CPU (jauh lebih lambat).

**Impact ke production:** Degradasi performa tanpa visibility. On-call tidak
ter-trigger karena tidak ada error — hanya lambat.

**Saran:** Expose GPU degradation sebagai metric Prometheus + health check alert.
Jangan silent fallback — setidaknya warn di log dengan rate limiting.

---

## H7. `dummy_cos_sin()` di Production Code

**File:** `crates/transformer/src/block.rs` (baris 228)

```rust
fn dummy_cos_sin() -> (Vec<f32>, Vec<f32>) {
```

**Deskripsi:** Fungsi bernama `dummy_` ada di production path transformer block.
Digunakan sebagai fallback RoPE cos/sin jika precompute kosong.

**Kenapa berbahaya:** Nama `dummy_` menunjukkan ini placeholder. Jika dipanggil
di production (karena precompute gagal), RoPE akan pakai nilai random/dummy.
Semua token akan mendapat positional encoding acak.

**Impact ke production:** Output model jadi nonsense tanpa error jelas.

**Saran:** Hapus dummy fallback. Lebih baik panic/error jelas daripada output rusak.

---

# MEDIUM PRIORITY

## M1. Excessive GPU→CPU Transfer (`to_cpu()`)

**File:** ~50+ call site di autograd, transformer, star-x, echo-net, atqs, hldva-t, has-moe-ffn

**Deskripsi:** Hampir setiap GPU operation diakhiri dengan `to_cpu()` call untuk
membaca hasil kembali ke CPU. Pattern: upload → compute GPU → download CPU.
Ini menghilangkan benefit GPU karena bottleneck di PCIe transfer.

**Contoh:**
- `transformer/src/model.rs:785`: `logits.to_cpu()?.iter().copied().collect()`
- `autograd/src/ops/nn.rs:22,178,583`: Setiap backward pass download ke CPU
- `star-x/src/blas_backend.rs:732,796`: BLAS ops download ke CPU
- `hldva-t/src/dit/attention.rs:263,447`: Attention download ke CPU

**Kenapa berbahaya:** GPU utilization rendah karena constantly waiting for CPU readback.
Setiap operasi kecil (misal activation function) tetap trigger full sync.

**Impact ke production:** Throughput GPU inference 2-5x lebih rendah dari potensi.
**Saran:** Batch readback menggunakan `gpu_batch.rs` + `readback_f32_async`.
Tahan operasi di GPU selama mungkin. Hanya download hasil final, bukan intermediate.

---

## M2. Streaming Hot Path: Intermediate Vec Allocation

**File:** `crates/inference/src/engine.rs` (baris 298-302)

```rust
let context: String = results.iter().take(3)
    .map(|r| format!("- {}", r.value))
    .collect::<Vec<_>>()  // 🔴 intermediate Vec alloc
    .join("\n");
```

**File:** `crates/multimodal/src/caffeine/action_head/execution.rs` (baris 680-699)

**Deskripsi:** Pattern `collect::<Vec<_>>().join()` mengalokasikan Vec intermediate
hanya untuk join. Di streaming hot path, ini alokasi memory yang tidak perlu.

**Impact ke production:** Memory churn, GC pressure di setiap request.
**Saran:** Gunakan `Iterator::fold` atau tulis langsung ke String dengan `write!`.

---

## M3. Shader Template Substitution di Hot Path

**File:** `crates/autograd/src/gpu.rs` (baris 1796, 2403)

```rust
MATMUL_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string())
REDUCE_WGSL_TEMPLATE.replace("{{OP}}", &(op as u32).to_string())
```

**Deskripsi:** WGSL shader templates menggunakan `String::replace` pada setiap
invocation dengan tile size atau op yang berbeda. Ini berarti kompilasi shader baru
atau minimal string manipulation di setiap dispatch.

**Impact ke production:** CPU overhead di setiap GPU dispatch. Untuk model dengan
banyak matmul berbeda, ini bisa menjadi bottleneck.
**Saran:** Cache compiled pipeline per (TILE_SIZE, OP) combination. Jangan replace
string di hot path.

---

## M4. Hardcoded Thresholds di Setiap Adaptive Logic

**File:** `crates/reasoning/src/saca/execute/strategies/adaptive.rs` (baris 39-48)
`crates/shared/src/agent_coordinator.rs` (baris 126)
`crates/star-x/src/asc.rs` (baris 343-344)

**Contoh:**
- `adaptive.rs`: >5 → parallel, ≤2 → sequential, else hybrid
- `agent_coordinator.rs`: `task_description.contains(&rule.pattern)` — substring matching
- `asc.rs`: magic threshold untuk short vs long sequence

**Deskripsi:** Banyak keputusan algoritmik menggunakan hardcoded threshold tanpa
konfigurasi atau dynamic adjustment.

**Impact ke production:** Tidak bisa di-tune per use case. Perilaku berubah
signifikan dengan input size yang berbeda.
**Saran:** Export ke config struct dengan default yang masuk akal.

---

## M5. CPU Affinity Hardcoded ke 64 Core Max

**File:** `crates/inference/src/runtime.rs` (baris 804)
`crates/autograd/src/gpu_core_layout.rs` (baris 125)

```rust
if cpu < 64 {  // max 64 core
    unsafe { libc::CPU_SET(cpu, &mut cpu_set); }
}
```

**Deskripsi:** CPU affinity dibatasi 64 core (stolen dari libc CPU_SETSIZE limit).
Server modern punya 128-256 core. Affinity tidak akan apply ke core > 63.

**Impact ke production:** NUMA locality tidak optimal di large server.
**Saran:** Gunakan `CPU_SET_S` yang mendukung dynamic size. Atau gunakan `affinity` crate.

---

# LOW PRIORITY

## L1. Excess `.to_string()` di Error Messages Hot Path

**File:** `crates/transformer/src/model.rs` (baris 666-675, 857-861, 959-964, 1468-1473, 1567-1571)
`crates/transformer/src/gqa.rs` (baris 860-933, 1014-1425)

Setiap error di GPU forward pass melakukan `e.to_string()` — alokasi String.
Jika error jarang, ini tidak masalah. Tapi jika ada pattern error sementara
(misal GPU temp OOM), log flooding bisa menyebabkan alloc pressure.

---

## L2. SIMD Transmute (3x)

**File:** `crates/infrastructure/utils/src/simd_ops.rs` (baris 86, 233, 609)

Unsafe transmute dari SIMD register ke `[f32; 4]`. Well-documented dan safe untuk
x86_64, tapi mengunci platform.

---

## L3. No Lockfile (Cargo.lock di gitignore)

**File:** Root `.gitignore`

Build tidak reproducible. Setiap `cargo build` resolve dependency dari awal.
Bisa menyebabkan bug yang tidak reproducible antar environment.

---

## L4. String Comparison untuk Routing

**File:** `crates/agent/src/routing_agent.rs` (baris 220-227)

Regex failure silent fallback ke substring matching:
```rust
if let Ok(re) = Regex::new(&config.pattern) { ... }
else { // fallback ke case-insensitive contains }
```

Bisa menghasilkan false positive routing jika regex salah konfigurasi.

---

# DAFTAR PLACEHOLDER TERSEMBUNYI

| Temuan | File | Baris | Deskripsi |
|--------|------|-------|-----------|
| WARNING: NOT integrated | `autograd/src/gpu_mixed.rs` | 1 | Mixed precision GPU tidak terintegrasi |
| QUANTIZATION_IS_STORAGE_ONLY: true | `quantization/src/lib.rs` | 16 | Kuantisasi hanya storage, bukan compute |
| #\[deprecated\] SequentialBatchingEngine | `inference/src/continuous_batching.rs` | 35 | Continuous batching deprecated |
| // for demonstration | `echo-net/src/isc.rs` | 413 | IFFT demo code O(N⁴) |
| cast_model_to_compute_dtype() = {} | `autograd/src/mixed_precision.rs` | 328 | Stub no-op |
| dummy_cos_sin() | `transformer/src/block.rs` | 228 | Fungsi dummy di production |
| Currently only CPU is implemented | `gnac/src/execution/mod.rs` | 42 | 4/5 backend tidak jalan |
| // Fallback: SequentialCoordinator x7 | `shared/src/agent_coordinator.rs` | 288-294 | 7 strategi palsu |
| // Simulated average lookup time | `models/swift/agents/fast_cache.rs` | 614 | Cache benchmark pakai nilai simulasi |
| // Return simulated recomputed activation | `gnac/src/scheduler/memory.rs` | 119 | Memory checkpoint simulasi |

---

# FAKE COMPLETION ANALYSIS

Fitur yang **tampak selesai tapi sebenarnya palsu:**

| Fitur | Alasan |
|-------|--------|
| **Quantized Inference** | 4 implementasi, 0 dipakai. Semua jalan di fp32. |
| **Mixed Precision Training** | GPU shaders siap, pipeline fp32. Akan compile tapi tidak ada effect. |
| **CUDA Backend** | Hanya enum variant dan wgpu wrapper. No actual CUDA runtime. |
| **Multi-Backend Execution** | 5 backend dideklarasikan, 1 berfungsi (CPU). |
| **Continuous Batching** | Sequential dengan nama "batch". Deprecated. |
| **Data Parallel Training** | 2 implementasi, 0 call site. CPU-only. |
| **Adaptive Coordination** | Adaptive → Sequential silent fallback. |
| **Consensus Strategy** | Consensus → Sequential silent fallback. |
| **Priority-based Routing** | Priority → Sequential silent fallback. |
| **GPU-Native Computation** | Setiap operasi diakhiri to_cpu(). Fully utilized? Tidak. |
| **Multi-Head Latent Attention** | KV cache compression belum selesai. |
| **GPU Batch Dispatch** | Tersedia tapi tidak digunakan oleh transformer forward path. |
| **KV Cache Paging** | Ada implementasi tapi GPU paged cache tidak fully integrated. |

---

# ARSITEKTUR: YANG TERLIHAT PARALLEL TAPI SEBENARNYA SERIAL

1. **ModelForward::forward_batched** — default implementation adalah loop sequential.
   GPU batched path ada tapi conditional: jika GPU gagal, silent fallback ke sequential.

2. **GNAC CompiledExecutor** — multi-backend execution engine. Seluruh graph
   diproses oleh `CpuBackend`. GPU hanya untuk operasi tertentu via wgpu.

3. **Agent Coordination** — 7 strategi palsu yang rebound ke Sequential.
   "Parallel" dan "Hierarchical" benar parallel. Sisanya sequential.

4. **Data Parallel Training** — dua implementasi ada tapi tidak pernah digunakan.
   Satu-satunya training yang benar-benar jalan adalah single-device.

5. **Inference Scheduler** — `RequestScheduler` mendelegasikan ke runtime, tapi
   continuous batching deprecated. Scheduler menata request, tapi eksekusi tetap
   satu-per-satu.

---

# PERFORMANCE BOTTLENECK UTAMA

```
1. GPU→CPU transfer setiap operasi (to_cpu())    ⚠️ MAIN BOTTLENECK
2. Clone prompt_ids di setiap request             ⚠️ MEMORY PRESSURE
3. Sequential batching (tidak ada true batching)  ⚠️ THROUGHPUT LIMIT
4. Spin loop blocking async runtime thread        ⚠️ LATENCY SPIKES
5. Blocking std::sync::Mutex di async context     ⚠️ DEADLOCK RISK
6. Unwrap chain crash pada error                  ⚠️ NO FAULT TOLERANCE
```

---

# TECHNICAL DEBT PALING BERBAHAYA

1. **`to_cpu()` setelah setiap GPU op** — ini adalah masalah paling fundamental.
   Seluruh GPU stack kehilangan benefit karena PCIe bottleneck. Solusi:
   implementasi proper compute graph dengan lazy evaluation dan minimal readback.

2. **Sequential batching dengan nama "continuous"** — ini adalah UX/deception issue.
   Pengguna configure batch size expecting throughput gain, dapatnya sequential.
   Butuh true continuous batching dengan dynamic padding.

3. **Silent fallback ke CPU tanpa alerting** — GPU degradation bisa terjadi
   tanpa diketahui operator. Butuh health endpoint + metrics exposure.

4. **Unwrap chain di checkpoint loading** — Satu file corrupt → server crash.
   Butuh proper error handling dengan recovery path.

---

# KESIMPULAN

**Readiness Production: ~35%**

Codebase ini memiliki **arsitektur yang sangat ambisius dan struktur yang baik**,
tapi sebagian besar modul berada dalam state "structurally complete, functionally incomplete."

Kekuatan:
- Autograd engine dengan 25+ op + GPU support via WGSL ✅
- Tokenizer BPE production-ready ✅
- Model definitions (7 series) lengkap ✅
- Async runtime infrastructure ✅
- Safety features (hallucination, isolation) ✅
- Test coverage cukup baik ✅

Kelemahan:
- Semua "GPU-native" claim perlu verifikasi — banyak yang CPU-heavy
- Quantization dan mixed precision adalah smoke and mirrors
- Multi-backend execution palsu (CPU-only)
- Batching sequential, bukan continuous
- Error handling buruk (unwrap chain)
- Blocking code di async runtime

Untuk production, **prioritas #1 adalah menghilangkan silent fallback** — lebih baik
error jelas daripada output salah/lambat tanpa tahu penyebabnya. Prioritas #2 adalah
memastikan GPU benar-benar dipakai (minimalisasi to_cpu, true batching, mixed precision).
