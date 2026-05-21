# PLAN: Optimasi GPU Utilization — 3 Tahap

## Ringkasan Masalah

GPU utilization ~10%, CPU 100% karena **5 sync point blocking per training step** dan **data pipeline serial**:

| Sync Point | File:Baris | Biaya |
|---|---|---|
| Input tokens → `Vec<f32>` → upload GPU | `training/src/lib.rs:566-579` | alloc + H2D tiap batch |
| `loss.data()[0]` (NaN check) | `training/src/lib.rs:610` | BLOCKING — `device.poll(MapMode::Read)` |
| `p.grad()` — readback gradient | `training/src/lib.rs:615-622` | blocking read seluruh parameter |
| Gradient → upload ulang ke GPU | `training/src/lib.rs:618` | D2H + H2D untuk optimizer |
| Hasil backward per-op ke CPU | `engine.rs:71` | tiap node di tape |

**Ada 3 kode yang sudah ditulis tapi tidak dipakai:** `gpu_async.rs` (double buffering), `gpu_batch.rs` (command batching), `gpu_kv_cache.rs` (paged KV cache GPU).

---

## Tahap 1: Quick Wins — Integrasi Infrastruktur yang Sudah Ada

**Target:** 10 → 30% GPU utilization. **Zero kode baru, hanya wiring.**

### 1.1 Pasang `GpuCommandBatch` di training loop

**Akar masalah:** `auto_flush_ops` di `gpu.rs:450` flush tiap 64 op. Batch kecil (6-12 layer × 4 op) hanya menghasilkan 24-48 dispatch — terpecah jadi 1-2 submit. Setiap submit punya fixed overhead 5-10µs.

**File target:**
- `crates/autograd/src/engine.rs` — ganti `with_encoder` individu dengan `GpuCommandBatch`

**Perubahan konkret:**
```rust
// SEKRANG (engine.rs:58-97) — tiap op dispatch sendiri
let gpu_result = ctx.add(ga, gb);  // dispatch + flush internal

// MENJADI — bungkus seluruh forward + backward dalam 1 batch
let mut batch = GpuCommandBatch::new(&ctx);
let gpu_result = batch.add(ga, gb);  // no flush — kumpulkan di encoder
batch.submit();  // 1 queue.submit() untuk SEMUA op
```

**Efek:** Dispatch overhead turun dari N×10µs jadi 10µs per step.

### 1.2 Integrasi `AsyncDataPipeline` untuk double-buffering

**Akar masalah:** Data preparation (tokenize → `Vec<f32>` → `ArrayD` → upload) terjadi **serial sebelum GPU compute**. GPU idle selama CPU prep.

**File target:**
- `crates/training/src/lib.rs` — modifikasi `train_batch_gpu()` sekitar baris 538-650

**Perubahan konkret:**
```rust
// SEKRANG — serial: prep → upload → compute → prep → upload → compute
let input_arr = ArrayD::from_shape_vec(vec![seq], input_buf).unwrap();
let input_t = Tensor::from_gpu(GpuTensor::from_cpu(&input_arr).unwrap(), ...);
let logits = trainable.forward(&input_t);  // GPU idle nunggu input_arr selesai

// MENJADI — overlap: prep batch N+1 while compute batch N
let pipeline = AsyncDataPipeline::new(&ctx, 2, seq * 4);  // 2 buffer
pipeline.upload_next(&ctx, &tokens, 0);  // upload batch N
let logits = trainable.forward(&gpu_input);
pipeline.upload_next(&ctx, &tokens_next, 1);  // overlap: upload batch N+1
```

**Efek:** CPU→GPU transfer overlap dengan GPU compute. Utilisasi naik dari 10% → ~25%.

### 1.3 GPU-native grad norm tanpa readback

**Akar masalah:** `compute_grad_norm(&trainable.parameters())` baris ~637 baca semua parameter `p.data()` ke CPU cuma buat L2 norm.

**File target:**
- `crates/training/src/lib.rs` — ganti `compute_grad_norm` dengan GPU path
- WGSL reduce shader **sudah ada** di `gpu.rs` (shader "reduce_sum", "reduce_max")

**Perubahan konkret:**
```rust
// SEKRANG — blocking readback semua parameter
fn compute_grad_norm(params: &[Tensor]) -> f32 {
    let total: f32 = params.iter()
        .map(|p| p.grad().unwrap().iter().map(|x| x * x).sum::<f32>())
        .sum();
    total.sqrt()
}

// MENJADI — GPU-native L2 norm
fn compute_grad_norm_gpu(params: &[&GpuTensor], ctx: &GpuContext) -> f32 {
    let mut sum_sq = 0.0f32;
    for p in params {
        let sq = ctx.reduce_sum(ctx.mul(p, p))?;  // semua di GPU
        sum_sq += sq.to_cpu_scalar();  // 1 readback 4 byte, bukan 100MB
    }
    sum_sq.sqrt()
}
```

**Efek:** Eliminasi readback 100MB per step. Utilisasi naik ke ~30%.

### Deliverable Tahap 1

| Item | File | Baris Kode Berubah |
|---|---|---|
| GpuCommandBatch di engine | `crates/autograd/src/engine.rs` | ~40 |
| AsyncDataPipeline di trainer | `crates/training/src/lib.rs` | ~80 |
| GPU grad norm | `crates/training/src/lib.rs` | ~30 |
| **Total** | | **~150 baris** |

---

## Tahap 2: Arsitektur Ulang Sync Point

**Target:** 30 → 60% GPU utilization. **Pattern architecture changes.**

### 2.1 Eliminasi `loss.data()[0]`

**Akar masalah:** Trainer butuh loss value untuk logging + NaN detection. Sekarang blocking read dari GPU.

**Strategi:** Gunakan flag buffer + GPU-side NaN detection.

**File target:**
- `crates/training/src/lib.rs` — method `train_batch_gpu()`
- `crates/autograd/src/gpu.rs` — tambah shader `nan_detect` (20 baris WGSL)

**Implementasi:**
```rust
// SEKRANG — blocking
let loss_val = loss.data()[0];
if loss_val.is_nan() { break; }

// MENJADI — deferred read
let gpu_loss = loss.storage().as_gpu().unwrap();
let nan_flag = ctx.detect_nan(gpu_loss)?;  // GPU shader, return u32 buffer
ctx.flush();
// ... compute lainnya lanjut ...
let is_nan = nan_flag.to_cpu_scalar::<u32>() != 0;  // poll nanti
```

**Shader `nan_detect` (WGSL):**
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&input) { return; }
    if isNan(input[id.x]) {
        output[0] = 1u;  // atomicOr?
    }
}
```

### 2.2 Persistent gradient storage di GPU

**Akar masalah:** `p.grad()` di `training.rs:615` selalu readback ke CPU `ArrayD<f32>`, lalu diupload ulang buat optimizer step.

**Strategi:** Simpan gradient reference di GPU, jangan pernah readback kecuali untuk checkpoint.

**File target:**
- `crates/autograd/src/tensor.rs` — method `grad()` → return `Option<&Storage>` instead of `ArrayD<f32>`
- `crates/training/src/lib.rs` — `train_batch_gpu()` langsung pakai GPU gradient ref

**Perubahan konkret:**
```rust
// SEKRANG — Tensor.grad() selalu return CPU ArrayD
pub fn grad(&self) -> Option<ArrayD<f32>> {
    self.grad.as_ref().map(|g| g.storage.to_cpu())
}

// MENJADI — return storage reference, tetap di GPU
pub fn grad_storage(&self) -> Option<&Storage> {
    self.grad.as_ref().map(|g| &g.storage)
}

// Trainer GPU path langsung:
let grad_refs: Vec<&GpuTensor> = trainable.parameters()
    .iter()
    .map(|p| p.grad_storage().unwrap().as_gpu().unwrap())
    .collect();
gpu_opt.step(&ctx, &params, &grad_refs);  // tanpa readback + re-upload
```

### 2.3 Pre-allocated weight + gradient buffer GPU

**Akar masalah:** `train_batch_gpu()` alloc `Vec<f32>` per batch — heap allocator contention + CPU cache miss.

**Strategi:** Alokasi `GpuTensor` sekali di `prepare()`, reuse buffer tiap step.

**File target:**
- `crates/training/src/lib.rs` — `Trainer::prepare()` alokasi GPU buffer pool

### 2.4 ModelForward trait return GPU-aware type

**Akar masalah:** `inference_trait.rs:14` return `Array1<f32>` — paksa GPU→CPU tiap forward.

**Strategi:** Ubah return type jadi `Storage` atau `GpuTensor` dengan opsi tetap di GPU.

**File target:**
- `crates/inference/src/inference_trait.rs`
- Semua implementor trait ini

```rust
pub trait ModelForward: Send + Sync {
    async fn forward(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> Array1<f32>;
    // ↑ return CPU ARRAY — MEMAKSA SYNC

    // MENJADI:
    type Output: Into<ArrayD<f32>> + Send;
    async fn forward_gpu(&self, input_ids: &[u32], kv_cache: &mut GpuKVCache) -> Self::Output;
    // ↑ GPU-native, tidak pindah ke CPU kecuali diminta
}
```

### Deliverable Tahap 2

| Item | File | Baris |
|---|---|---|
| NaN detection GPU | `gpu.rs` + `training/lib.rs` | ~50 |
| Gradient storage GPU | `tensor.rs` + `training/lib.rs` | ~100 |
| Pre-alloc buffer pool | `training/src/lib.rs` | ~80 |
| ModelForward GPU-aware | `inference_trait.rs` | ~60 |
| **Total** | | **~290 baris** |

---

## Tahap 3: Advanced — Compute Pipeline GPU-First

**Target:** 60 → 85%+ GPU utilization. **Arsitektur ulang pipeline.**

### 3.1 True batched inference (continuous batching)

**Akar masalah:** `continuous_batching.rs:147-205` proses 1 sequence at a time — batch size efektif = 1.

**Strategi:** Concatenate multiple sequences jadi 1 forward pass.

**Perubahan:**
```rust
// SEKRANG — per-sequence forward
for seq_id in ready_ids {
    let logits = self.model.forward(&[input_token], cache);
}

// MENJADI — batched forward
let batch_input = concat_sequences(&ready_sequences, &PAD_TOKEN);
let all_logits = self.model.forward_batched(&batch_input, &batch_cache);
for (i, seq_id) in ready_ids.iter().enumerate() {
    let logits_i = all_logits.slice(s![i, -1, ..]);
}
```

GPU modern optimal dengan batch size 16-64. Continuous batching HARUS concatenate.

### 3.2 GPU sampling — tetap di GPU

**Akar masalah:** `sampler.rs:74-96` — 3 sync point per token: logits → CPU → GPU sample → CPU baca.

**Strategi:** Sampling state machine di GPU, baca cuma token ID (4 byte).

```rust
// SEKRANG — 3 sync per token
let cpu = ArrayD::from_shape_vec(shape, logits.to_vec())?;
let gpu = GpuTensor::from_cpu(&cpu)?;
let out = ctx.gpu_sample(&gpu, temp, top_k, seed)?;
let raw = out.to_cpu_raw_bytes();       // blocking read
let token = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);

// MENJADI — sampling state di GPU, baca 4 byte
let seed_state = GpuSamplerState::new(ctx, seed);
let token = ctx.sample_next(&logits_gpu, &mut seed_state)?;  // all GPU
// token langsung Vec<u32> kecil dari staging buffer
```

`gpu_sampler.rs` sudah punya `gpu_sample` — implementasi ulang jadi stateful sampler yang tetap di GPU antar-token.

### 3.3 Fused kernel untuk backward pass

**Akar masalah:** `gpu_fused.rs` hanya untuk forward. Backward tiap op decomposed.

**Strategi:** Tulis fused backward shader untuk pattern umum:
- `matmul_backward + bias_grad` → 1 kernel
- `layer_norm_backward + residual_grad` → 1 kernel
- `cross_entropy_backward + softmax_backward` → 1 kernel (dulu sudah fused di forward)

**File target:**
- `crates/autograd/src/gpu_fused.rs`

### 3.4 Paged KV cache GPU (sudah ada, integrasi)

**Akar masalah:** `gpu_kv_cache.rs` sudah punya paged cache dengan gather/scatter shader, tapi inference masih pakai `Vec<KVCacheEntry>` di CPU.

**Integrasi:**
- Ganti `Vec<KVCacheEntry>` di continuous batching dengan `GpuPagedKVCache`
- Eliminasi transfer KV cache CPU↔GPU per step

### 3.5 Async compute stream overlap

**Akar masalah:** wgpu `queue.submit()` ke 1 queue. Tidak overlap compute + transfer.

**Strategi:** wgpu support multiple queue (tidak semua backend). Alternatif: manual overlap dengan `AsyncDataPipeline` + timing:

```
Stream 1 (compute): forward → backward → optimizer
Stream 2 (transfer): upload weights → upload next batch input
```

Implementasi dengan `AsyncDataPipeline` (sudah ada) + `GpuCommandBatch`.

### Deliverable Tahap 3

| Item | File | Baris |
|---|---|---|
| Batched continuous batching | `continuous_batching.rs` | ~200 |
| GPU stateful sampling | `gpu_sampler.rs` + `sampler.rs` | ~150 |
| Fused backward kernel | `gpu_fused.rs` | ~300 |
| Paged KV cache integrasi | `inference/` + `transformer/` | ~250 |
| Async compute overlap | `gpu_async.rs` + `training/src/lib.rs` | ~100 |
| **Total** | | **~1000 baris** |

---

## Ringkasan Target Utilization

| Tahap | GPU Util | CPU Util | Sync Point per Step | Baris Kode |
|---|---|---|---|---|
| **Sekarang** | ~10% | 100% | 5 | - |
| **Tahap 1** | ~30% | 80% | 3 | ~150 |
| **Tahap 2** | ~60% | 60% | 1 | ~290 |
| **Tahap 3** | ~85%+ | 40% | 0 (deferred) | ~1000 |

## Prioritas Eksekusi

1. **Tahap 1** — zero risk, semua kode sudah ada, tinggal wiring. Mulai sekarang.
2. **Tahap 2** — break change di `Tensor::grad()` API. Butuh update semua caller.
3. **Tahap 3** — perubahan arsitektur besar. Butuh testing + benchmark.

Mulai dari Tahap 1 dulu — deliverable dalam 1-2 hari.
