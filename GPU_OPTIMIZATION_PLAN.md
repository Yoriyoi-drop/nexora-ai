# GPU Optimization Plan: Full GPU Residency + Hybrid CPU/GPU

> Revisi berdasarkan review ChatGPT. Detail: [REVIEW_CHATGPT.md](./REVIEW_CHATGPT.md)

---

## Masalah Saat Ini

- **GPU ~5%** — hanya matmul di-GPU-kan, itupun langsung `to_cpu()` karena `requires_grad=true`
- **CPU 100% (1 core)** — semua tensor `Storage::Cpu`, autograd CPU, Adam CPU, inference CPU
- **Tidak ada overlap CPU↔GPU** — sinkronisasi blocking tiap langkah
- **Cuma 1 op GPU (naive matmul)** — sisanya CPU-only
- **Tidak ada memory pool** — alloc/free buffer tiap langkah → thrashing
- **Tidak ada mixed precision** — semua f32, VRAM boros
- **Tidak ada NaN detection / profiler** — debugging GPU sulit

---

## Arsitektur Target

```
┌──────────────────────────────────────────────────────┐
│                   GPU 95-100%                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Forward  │→│ Backward │→│ Optimizer│  ...loop   │
│  │ (all ops)│  │ (all GPU)│  │ (AdamW)  │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│  ┌──────────────────────────────────────────┐        │
│  │         GPU Memory Pool / Arena          │        │
│  │  (reusable buffers, workspace cache)     │        │
│  └──────────────────────────────────────────┘        │
│  ┌──────────────────────────────────────────┐        │
│  │      Mixed Precision (FP16/BF16+FP32)    │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
         ▲ sync only loss scalar ▲
         │                        │
┌────────┴────────────────────────┴────────────────────┐
│                  CPU 25-50%                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │Data Load │→│Tokenize  │→│ Shuffle  │  ...loop   │
│  │(rayon)   │  │(rayon)   │  │          │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│  ┌──────────────────────────────────────────┐        │
│  │       Checkpoint I/O (async)              │        │
│  └──────────────────────────────────────────┘        │
│  ┌──────────────────────────────────────────┐        │
│  │       Logging / Profiling / NaN Detect   │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

### Prinsip Utama

1. **Full GPU Residency** — parameter, grad, optimizer state (m, v), logits, KV cache tinggal di GPU. Zero `to_cpu()` di tengah pipeline.
2. **Memory Pool** — reusable buffer arena, hindari alloc/free thrashing.
3. **Mixed Precision** — FP16/BF16 compute + FP32 master weight + loss scaling.
4. **Async Pipeline** — CPU prep data batch N+1 sementara GPU compute batch N.
5. **No CPU Fallback** — semua op punya GPU path, fallback hanya untuk ops yg benar-benar tidak terpakai.
6. **Capability Detection** — deteksi fitur GPU (subgroup, shared memory limit, dll) + vendor-aware path.

---

## Phase 0: Foundation (minggu 1-2)

### 0.1 GPU Capability Detection

**File:** `crates/autograd/src/gpu_caps.rs` (baru)

```rust
pub struct GpuCapabilities {
    pub vendor: GpuVendor,          // NVIDIA / AMD / Intel / Apple
    pub subgroup_size: Option<u32>, // 4, 8, 16, 32, or None
    pub max_workgroup_size: u32,
    pub max_shared_memory: u32,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub max_storage_buffers: u32,
    pub dedicated_video_memory: u64,
}

pub fn detect_capabilities(device: &wgpu::Device, adapter: &wgpu::Adapter) -> GpuCapabilities;
```

- Query `wgpu::Adapter::get_info()` untuk vendor
- Query `wgpu::Limits` untuk max workgroup, shared memory
- Coba buat pipeline FP16/BF16 untuk deteksi support
- Pilih kernel optimal berdasarkan vendor + caps

### 0.2 NaN / Inf Detection System

**File:** `crates/autograd/src/gpu_debug.rs` (baru)

```
#[cfg(debug_assertions)]
mod gpu_debug {
    fn check_nan(tensor: &GpuTensor) -> bool;
    fn check_inf(tensor: &GpuTensor) -> bool;
    fn tensor_checksum(tensor: &GpuTensor) -> u64;
    fn gradient_anomaly_detect(grads: &[GpuTensor]) -> Vec<usize>;
}
```

- WGSL kernel: reduce → cek NaN/Inf per blok
- Nonaktif di release build
- Integrasi dengan logging

### 0.3 GPU Profiling Infrastructure

**File:** `crates/autograd/src/gpu_profiler.rs` (baru)

```
pub struct GpuProfiler {
    kernel_times: Vec<(&'static str, Duration)>,
    transfer_times: Vec<(&'static str, Duration)>,
    allocator_stats: AllocatorStats,
    queue_depth: usize,
}
```

- Wrap `wgpu::Queue::submit()` untuk timing
- `wgpu::QuerySet` untuk GPU timestamp (kalo supported)
- Export ke Prometheus / console logging

### 0.4 Memory Pool / Tensor Arena

**File:** `crates/autograd/src/gpu_memory.rs` (baru)

```rust
pub struct GpuMemoryPool {
    device: wgpu::Device,
    free_buffers: HashMap<BufferKey, Vec<wgpu::Buffer>>,
    active_buffers: HashMap<u64, AllocInfo>,
    workspace: wgpu::Buffer, // ring buffer untuk temporary
    stats: AllocatorStats,
}

impl GpuMemoryPool {
    pub fn alloc(&self, size: u64, usage: wgpu::BufferUsages) -> PooledBuffer;
    pub fn dealloc(&self, buf: PooledBuffer);
    pub fn reset_workspace(&self); // reset ring buffer tiap step
}
```

Strategi:
- **Pool per size class** (bucket 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, ...)
- **Ring buffer workspace** untuk tensor temporary (intermediate)
- **LRU eviction** untuk buffer jarang dipakai
- **defragmentation** background (kalo perlu)

### Deliverable Phase 0

- GPU capability detection → pilih kernel optimal
- NaN/Inf safety net
- GPU profiler (timing tiap kernel)
- Memory pool (zero alloc/free thrashing)

---

## Phase 1: GPU Kernels — Tahap Awal (minggu 3-4)

Prioritas: **tiled matmul → reduce → element-wise → activation**

### 1.1 Tiled Matmul (upgrade dari naive)

**File:** `crates/autograd/src/gpu.rs`

```
┌───────┬───────┐
│ TileA │ TileB │  workgroup_size(16,16) = 256 threads
│ (SMEM)│ (SMEM)│  BLOCK_SIZE = 16 (tunable per GPU caps)
├───────┼───────┤
│ TileC │       │  Loop BLOCK_SIZE/K iterasi per tile
└───────┴───────┘
```

```wgsl
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

// Cooperative load: tiap thread load 1 elemen A + 1 elemen B ke SMEM
workgroupBarrier();

// Compute partial dot product
for (var k = 0u; k < 16; k++) {
    sum += tile_a[local_x][k] * tile_b[k][local_y];
}
workgroupBarrier();
```

Ukuran tile dipilih berdasarkan `GpuCapabilities.max_shared_memory`.

### 1.2 Reduce Kernel

```wgsl
// Tree-reduce dalam shared memory
var<workgroup> shared: array<f32, 256>;

// Phase 1: tiap thread load 1 elemen
shared[lid] = input[gid * 256 + lid];
workgroupBarrier();

// Phase 2: tree-reduce
for (var stride = 128u; stride > 0; stride >>= 1) {
    if (lid < stride) {
        shared[lid] += shared[lid + stride];
    }
    workgroupBarrier();
}
```

- `sum`, `mean`, `max`, `min` — 4 varian
- Multi-blok kalo input > 256 elemen

### 1.3 Element-wise + Activation

Satu kernel generic:

```wgsl
struct ElementwiseConfig {
    op: u32,     // 0=add, 1=sub, 2=mul, 3=div, 4=neg
                 // 5=exp, 6=ln, 7=powf, 8=sqrt
                 // 9=relu, 10=gelu, 11=sigmoid, 12=tanh, 13=silu
    stride_a: u32,
    stride_b: u32,
    stride_out: u32,
    numel: u32,
};
```

Satu pipeline dengan specialization constants untuk op.

### Deliverable Phase 1

- Matmul: 5-10x dari naive shader
- Reduce: sum, mean, max, min GPU
- Semua element-wise + activation GPU

---

## Phase 2: GPU Kernels — Tahap Tengah (minggu 5-6)

Prioritas: **RMSNorm → Softmax → Embedding → LayerNorm**

### 2.1 RMSNorm

```wgsl
// 1. Hitung sum(x²) per row (reduce)
var sum_sq = 0.0;
for (var i = 0u; i < hidden; i++) {
    sum_sq += x[row * hidden + i] * x[row * hidden + i];
}
// 2. Normalize
let rms = sqrt(sum_sq / f32(hidden) + epsilon);
for (var i = 0u; i < hidden; i++) {
    out[row * hidden + i] = x[row * hidden + i] / rms * weight[i];
}
```

- 1 thread per row (cooperative kalo hidden > 256)
- Fused: reduce + normalize dalam 1 kernel

### 2.2 Softmax (stable)

```wgsl
// 1. Cari max per row (tree-reduce)
var max_val = -1e20;
for (var i = 0u; i < vocab; i++) {
    max_val = max(max_val, logits[row * vocab + i]);
}
// 2. Exp sum
var sum_exp = 0.0;
for (var i = 0u; i < vocab; i++) {
    sum_exp += exp(logits[row * vocab + i] - max_val);
}
// 3. Divide
for (var i = 0u; i < vocab; i++) {
    out[row * vocab + i] = exp(logits[row * vocab + i] - max_val) / sum_exp;
}
```

### 2.3 Embedding

```wgsl
out[batch_idx * seq_len * hidden + pos * hidden + d]
    = weight[token_id * hidden + d];
```

- 1 thread per (batch, pos, hidden dim)
- Gather dari buffer weight

### 2.4 Cross-Entropy

```wgsl
// Softmax + log + index select
let log_softmax = log(softmax_output[row * vocab + target[row]]);
let loss = -log_softmax;
// Reduce mean (sum loss / batch*seq)
```

Atau fused langsung tanpa intermediate softmax buffer.

### Deliverable Phase 2

- RMSNorm, LayerNorm, Softmax, Embedding GPU
- Cross-entropy loss GPU
- Semua NN op dasar GPU

---

## Phase 3: GPU Kernels — Tahap Lanjut (minggu 7-9)

Prioritas: **Modular Attention → Fused Attention → Sampler**

### 3.1 Modular Attention (sebelum fused)

```
1. Q @ K^T → score (tiled matmul)
2. Causal mask → score (element-wise)
3. Softmax per row (softmax kernel)
4. Score @ V → output (tiled matmul)
```

Setiap langkah kernel terpisah. Lebih mudah debug & validasi.

### 3.2 Fused Attention (setelah modular stabil)

```wgsl
// 1. Load Q tile ke SMEM
// 2. Loop over K/V tiles:
//    a. Load K tile ke SMEM
//    b. Compute Q@K^T partial
//    c. Softmax partial (online softmax)
//    d. Load V tile → compute partial output
//    e. Store partial
```

Pake algoritma **FlashAttention-style**:
- Tiling over sequence dimension
- Online softmax (2 pass atau running)
- Zero large intermediate (seq×seq)

### 3.3 Fused SwiGLU FFN

```wgsl
// 1. gate = x @ W_gate
// 2. up = x @ W_up
// 3. silu(gate) * up → hidden
// 4. hidden @ W_down → output
```

Fused dalam 1-2 kernel (matmul + silu + mul + matmul).

### 3.4 GPU Sampler

```wgsl
// 1. Temperature scaling
// 2. Softmax
// 3. Top-K: partial sort → filter
// 4. Top-P: cumulative sum → filter
// 5. Multinomial: uniform random → inverse CDF
```

Mungkin perlu split jadi 2 kernel (softmax + sampling) karena random state.

### Deliverable Phase 3

- Attention modular + fused (FlashAttention-style)
- Fused SwiGLU FFN
- GPU sampler (softmax + top-k/top-p + sampling)

---

## Phase 4: GPU Autograd + Optimizer (minggu 9-11)

### 4.1 GPU Backward Engine

**File:** `crates/autograd/src/engine.rs`

- Tape tetap sama, tapi GradFn closures operasi di `GpuTensor`
- Backward BFS: dispatch kernel dalam antrian GPU queue
- Pipeline backward: `matmul_grad_a` → `matmul_grad_b` → `elementwise_grad` → ...

```rust
fn backward(tensor: &Tensor) {
    let tape = tensor.tape.lock();
    let mut queue = VecDeque::new();
    queue.push_back(tape.last_node());
    
    while let Some(node) = queue.pop_front() {
        let grad = grads.entry(node.id).or_insert(GpuTensor::ones());
        let input_grads = node.grad_fn.backward(grad); // panggil kernel GPU
        for (input, g) in node.inputs.iter().zip(input_grads) {
            accumulate_grad!(grads, input.id, g); // GPU kernel add
        }
    }
}
```

### 4.2 Gradient Accumulator GPU

**File:** `crates/autograd/src/engine.rs`

- `zero_grad()` → kernel `fill(buffer, 0.0)` di GPU
- Accumulate → kernel `add` di GPU
- Semua tetap di GPU

### 4.3 Fused Gradient Clipping

**File:** `crates/autograd/src/lib.rs`

```
Kernel 1: Hitung total_norm = sqrt(sum(||grad_i||²)) for all params
          (tree-reduce over concatenated grad buffer)
          
Kernel 2 (kalo total_norm > max_grad_norm):
          scale_coeff = max_grad_norm / total_norm
          for each param: grad_i *= scale_coeff (element-wise mul)
```

Fused: 2 kernel instead of N+1 (N params + 1 reduce).

### 4.4 Adam Optimizer GPU

**File:** `crates/autograd/src/gpu_adam.rs` (baru)

```rust
pub struct GpuAdam {
    m: Vec<GpuTensor>,   // momentum (di GPU)
    v: Vec<GpuTensor>,   // velocity (di GPU)
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

impl GpuAdam {
    pub fn step(&mut self, params: &[GpuTensor], grads: &[GpuTensor]) {
        // Satu dispatch per parameter tensor
        // Kernel: update m, v, param dalam 1 pass
    }
}
```

WGSL kernel:

```wgsl
fn adam_step(
    param: ptr<function, array<f32>>,
    grad: ptr<function, array<f32>>,
    m: ptr<function, array<f32>>,
    v: ptr<function, array<f32>>,
    lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: u32,
) {
    let g = grad[id];
    let m_new = b1 * m[id] + (1.0 - b1) * g;
    let v_new = b2 * v[id] + (1.0 - b2) * g * g;
    let m_hat = m_new / (1.0 - pow(b1, step));
    let v_hat = v_new / (1.0 - pow(b2, step));
    param[id] -= lr * m_hat / (sqrt(v_hat) + eps) + wd * param[id];
    m[id] = m_new;
    v[id] = v_new;
}
```

### Deliverable Phase 4

- Backward engine GPU-native
- Gradient accumulator (zero_grad, accumulate, clipping) semua di GPU
- Adam optimizer full GPU

---

## Phase 5: Mixed Precision (minggu 11-12)

### 5.1 FP16/BF16 Storage + Compute

**File:** `crates/autograd/src/gpu_mixed.rs` (baru)

```rust
pub enum GpuDType {
    F32,
    F16,
    BF16,
}

pub struct MixedPrecisionConfig {
    pub compute_dtype: GpuDType,     // FP16 atau BF16
    pub master_weights: bool,        // simpan FP32 copy
    pub loss_scaling: LossScaling,
}

pub struct LossScaler {
    pub scale: f32,
    pub growth_interval: u32,    // naikin scale tiap N step tanpa overflow
    pub growth_factor: f32,      // 2.0
    pub backoff_factor: f32,     // 0.5
}
```

- FP16/BF16 kernel variants (specialization constants)
- Loss scaling: scale up sebelum forward, scale down grad sebelum optimizer
- Overflow detection → rollback step + scale down

### 5.2 FP16/BF16 WGSL Kernels

```wgsl
// FP16 variant menggunakan array<f16> (WGSL f16 extension)
enable f16;

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f16>;
```

- Deteksi support via `device.has_feature(wgpu::Features::FLOAT16)`
- Fallback ke F32 kalo tidak support

### Deliverable Phase 5

- FP16/BF16 ops untuk semua kernel utama
- Master weight FP32
- Loss scaling dengan overflow detection
- VRAM turun ~40-50%

---

## Phase 6: Async Data Pipeline (minggu 12-13)

### 6.1 Double Buffering

```
                    ┌────────────┐
CPU→GPU staging ───>│  Buffer A  │───> GPU compute
                    ├────────────┤
CPU→GPU staging ───>│  Buffer B  │───> GPU compute
                    └────────────┘
                    (swap tiap step)
```

- 2 set buffer GPU (weights, grads, optimizer state tetap sama, hanya input batch yg double buffer)
- CPU isi buffer A → GPU compute buffer A sementara CPU isi buffer B

### 6.2 Async Tokenizer

```rust
let (tx, mut rx) = mpsc::channel(2);

// CPU thread: tokenize + kirim ke channel
tokio::spawn_blocking(move || {
    for batch in data.chunks(batch_size) {
        let tokens = tokenizer.encode_batch_par(batch); // rayon par_iter
        tx.blocking_send(tokens).unwrap();
    }
});

// GPU thread: ambil dari channel, kirim ke GPU, compute
while let Some(tokens) = rx.recv().await {
    let gpu_batch = GpuTensor::from_cpu_async(&tokens, &pool);
    trainer.step(gpu_batch).await;
}
```

### 6.3 Async Checkpoint

- Save/Load pake `tokio::task::spawn_blocking`
- GPU→CPU transfer untuk checkpoint via staging buffer async
- Tidak blocking training loop

### Deliverable Phase 6

- Zero idle GPU (data selalu ready sebelum GPU selesai step sebelumnya)
- CPU 25-50% (tokenize parallel)
- Checkpoint I/O tidak blocking

---

## Phase 7: GPU Inference (minggu 13-15)

### 7.1 CausalLM GPU Forward

**File:** `crates/transformer/src/model.rs`

- Deteksi apakah parameter di GPU → panggil GPU forward path
- Semua layer (embedding, RoPE, attention, FFN, RMSNorm, lm_head) GPU
- Output logits tetap di GPU → langsung ke GPU sampler

### 7.2 GPU KV Cache

**File:** `crates/inference/src/kv_cache.rs`

- Paged cache blocks sebagai `GpuTensor`
- Append: `queue.write_buffer()` ke slot kosong (GPU async copy)
- Flatten: kernel copy GPU (bukan CPU loop)
- Prefix cache: matching pake hash + GPU compare

### 7.3 GPU Sampler (inference)

**File:** `crates/inference/src/sampler.rs`

Sama dengan kernel Phase 3.4 — temperature + top-k + top-p + sampling.

### 7.4 Continuous Batching GPU

**File:** `crates/inference/src/continuous_batching.rs`

- Prefill: concatenate prompts → 1 forward GPU → split
- Generate: pad sequences → 1 forward GPU → causal mask
- PagedAttention + GPU scheduler

### Deliverable Phase 7

- Inference end-to-end GPU
- 5-20x tokens/s dibanding CPU
- Continuous batching untuk throughput maksimal

---

## Phase 8: CPU Parallelism & Stabilization (minggu 15-16)

### 8.1 Thread Pinning + Affinity

```rust
// core_layout.rs
pub struct CoreLayout {
    pub data_cores: Vec<usize>,    // 0-3
    pub compute_cores: Vec<usize>, // 4+
}

// Pin rayon thread pool ke data_cores
// Biarkan wgpu backend thread bebas
```

### 8.2 OpenBLAS Tuning (fallback)

```sh
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
```

### 8.3 Debug Mode GPU

```rust
// cargo run --features gpu -- --debug-gpu
pub struct GpuDebugConfig {
    pub sync_execution: bool,    // device.poll() tiap kernel
    pub verbose_tensor_check: bool, // NaN/Inf tiap op
    pub deterministic_seed: u64,
    pub kernel_validation: bool, // compare output vs CPU reference
}
```

### 8.4 Integration Test Suite

```
tests/gpu/
├── test_matmul.rs     // compare GPU vs CPU
├── test_softmax.rs
├── test_layernorm.rs
├── test_attention.rs
├── test_backward.rs   // grad check (numerical)
├── test_adam.rs
├── test_training_step.rs  // 1 full step GPU vs CPU
├── test_mixed_precision.rs
├── test_memory_pool.rs
└── test_kv_cache.rs
```

### Deliverable Phase 8

- Thread pinning stabil
- Debug mode untuk development
- Test suite GPU (regression)

---

## Total Timeline

| Phase | Minggu | Deliverable | Speedup (estimasi) |
|---|---|---|---|
| 0: Foundation | 1-2 | Caps detection, NaN detect, profiler, memory pool | — (infra) |
| 1: Kernel Awal | 3-4 | Tiled matmul, reduce, element-wise, activation | 2-5x |
| 2: Kernel Tengah | 5-6 | RMSNorm, softmax, embedding, cross-entropy | 5-10x |
| 3: Kernel Lanjut | 7-9 | Modular attention, fused attention, sampler | 10-20x |
| 4: Autograd + Adam | 9-11 | Full GPU backward + optimizer | 20-40x |
| 5: Mixed Precision | 11-12 | FP16/BF16, loss scaling | 30-60x (VRAM↓) |
| 6: Async Pipeline | 12-13 | Double buffer, async tokenizer, async ckpt | 40-80x |
| 7: GPU Inference | 13-15 | GPU forward, KV cache, sampler, continuous batch | 5-20x (inf) |
| 8: Stabilization | 15-16 | Thread pinning, debug mode, test suite | — (quality) |

**Total: ~16 minggu (4 bulan) — solo developer.**

---

## Quick Win (minggu 0, bisa dikerjakan sekarang)

| # | Item | Estimasi | Speedup |
|---|---|---|---|
| 1 | **Full GPU Residency untuk training** — bikin tensor pindah ke GPU (`to_device(Device::Gpu)`) sebelum forward. Fix matmul.rs GPU path biar kepanggil meski `requires_grad=true`. | 3-5 hari | 2-3x |
| 2 | **Memory pool** — reusable buffer per size class + ring buffer workspace. | 3-5 hari | stabilisasi latency |
| 3 | **GPU Adam optimizer** — WGSL kernel Adam step. State m, v tetap di GPU. | 2-3 hari | 3-5x |
| 4 | **OpenBLAS + thread tuning** — `libopenblas-dev`, Cargo.toml, env vars. | 0 hari | 2-5x (gratis) |
| 5 | **NaN detection system** — kernel cek NaN/Inf + logging. | 1 hari | safety |
| 6 | **GPU profiler** — timing + logging tiap kernel dispatch. | 1-2 hari | insight |

---

## Catatan Penting (dari review)

1. **Jangan obsesi CPU 70%.** Target: GPU setinggi mungkin (95%+), CPU cukup efisien. Kalau CPU 25% GPU 98%, itu justru bagus.
2. **Fused attention belakangan.** Debugging fused attention sangat menyiksa. Modular dulu → stabil → baru fused.
3. **Fokus memory bandwidth.** AI modern sering bottleneck bandwidth, bukan FLOPS. Fused kernels penting.
4. **Shared memory secukupnya.** Tiap GPU punya limit berbeda. Ukuran tile harus adaptif.
5. **Timeline realistis.** Solo development GPU debugging sangat brutal. Jangan buru-buru.
6. **Stabilitas > fitur.** Jangan lompat ke model besar sebelum runtime stabil.
7. **Logging & profiling wajib.** Tanpa profiler, optimisasi cuma ritual spiritual developer.
8. **NaN detection wajib.** Mixed precision nanti bikin NaN muncul seperti hantu kontrakan.
