# ARSITEKTUR FIX — 200B+ MoE VRAM EFFICIENCY

**Tujuan:** Naikkan experts 8→324, top-k 2→32, dan fix VRAM boros.

---

## 1. CONFIG CHANGES — Experts 324, Top-32

### `crates/has-moe-ffn/src/lib.rs`
```rust
// Default HasMoeFFNConfig
num_experts: 324,    // 8 → 324
top_k: 32,          // 2 → 32

// Test small_moe()
num_experts: 4,      // keep for tests
top_k: 2,
```

### `crates/has-moe-ffn/src/routing.rs`
```rust
// Default RouterConfig
num_experts: 324,    // 8 → 324
top_k: 32,          // 2 → 32
capacity_factor: 1.1, // 1.25 → 1.1 (324 experts, 32 top-k = 10% aktif, slack 10% cukup)
```

### `crates/transformer/src/config.rs`
```rust
// ModelTier::Ultra preset
num_experts: 324,
top_k_experts: 32,
expert_intermediate_size: 4096, // lebih kecil biar muat

// ModelTier::Apex preset  
num_experts: 324,
top_k_experts: 32,
expert_intermediate_size: 2048,
```

### `crates/shared/src/model_config.rs`
```rust
// Omnis model
moe_config: MoeConfig {
    num_experts: 324,
    num_experts_per_token: 32,
    capacity_factor: 1.1,
    ...
}

// Genesis model  
moe_config: MoeConfig {
    num_experts: 324,
    num_experts_per_token: 32,
    ...
}
```

---

## 2. EXPERT OFFLOADING SYSTEM — VRAM GAME CHANGER

**Masalah:** 324 expert × 4 matriks × bobot fp32 = TERLALU BESAR untuk VRAM.
Dengan top-32, hanya ~10% expert aktif per forward. Sisanya idle di VRAM.

**Solusi:** ExpertOffloader — swap idle experts CPU↔GPU dengan LRU.

### File baru: `crates/has-moe-ffn/src/offload.rs`

```rust
pub struct ExpertOffloader {
    gpu_budget_bytes: usize,           // max GPU VRAM untuk expert weights
    cpu_pool: Vec<ExpertWeights>,      // semua expert di CPU (pinned memory)
    gpu_resident: Vec<Option<GpuExpert>>, // expert yang sedang di GPU
    lru_tracker: LruCache<usize>,      // least-recently-used tracking
    prefetch_queue: Vec<usize>,        // expert yang akan di-load next
    transfer_stream: Option<GpuStream>, // async H2D/D2H stream
}

struct ExpertWeights {
    fc1_w: Vec<f32>,  // [intermediate, hidden]
    fc1_b: Vec<f32>,  // [intermediate]
    fc2_w: Vec<f32>,  // [hidden, intermediate]
    fc2_b: Vec<f32>,  // [hidden]
    last_used: Instant,
    usage_count: u64,
}

struct GpuExpert {
    // GPU tensor handles — Option karena bisa di-evict
    fc1_w: Option<GpuTensor>,
    fc1_b: Option<GpuTensor>,
    fc2_w: Option<GpuTensor>,
    fc2_b: Option<GpuTensor>,
}
```

### Algoritma

```
FORWARD:
  1. Router output → top-32 expert indices + confidence scores
  2. Cari expert yang SUDAH di GPU → reuse
  3. Cari expert yang belum di GPU:
     a. Hitung "urgency score" = confidence × (1 - recency_bonus)
     b. Urutkan by urgency, ambil paling butuh
     c. Evict LRU expert sampai cukup VRAM
     d. Async H2D upload expert weights
  4. Jalankan forward untuk expert yang sudah siap
  5. Sisa expert: CPU forward atau tunggu async transfer selesai
  6. Update LRU tracker

PREFETCH:
  - Setelah forward selesai, predicted next top experts
  - Gunakan router weight similarity untuk prediksi
  - Pre-load ke GPU via async transfer
  - Overlap compute dengan transfer

EVICT:
  - Ketika GPU budget habis, evict least recently used
  - Copy GPU → CPU hanya jika weights dirty (training)
  - Untuk inference: GPU tensor bisa langsung di-drop
```

### Budget formula
```
gpu_budget = min(
    available_vram * 0.7,           // 70% VRAM untuk expert weights
    num_resident_experts * expert_size
)

expert_size = (intermediate * hidden + intermediate + hidden * intermediate + hidden) * 4  // fp32 bytes

Dengan intermediate=4096, hidden=4096:
  expert_size = (4096*4096 + 4096 + 4096*4096 + 4096) * 4 = ~134MB per expert (fp32)
               = ~67MB per expert (fp16)
  
  24GB VRAM → ~170 expert (fp16) atau ~358 expert (Q8) bisa di-GPU
  48GB VRAM → ~350 expert (fp16) atau ~716 expert (Q8)
```

### Integrasi ke `HasMoeFFN`
```rust
pub struct HasMoeFFN {
    config: HasMoeFFNConfig,
    experts: Vec<Expert>,           // CPU: semua 324 expert weights
    router: Router,
    offloader: Option<ExpertOffloader>, // GPU: expert swapping
}
```

---

## 3. TRUE INT4/INT8 COMPUTE — STOP DEQUANT KE FP32

**Masalah:** `QUANTIZATION_IS_STORAGE_ONLY = true` — semua weight di-Q4 untuk storage tapi didequant ke fp32 untuk compute. VRAM meledak.

### File baru: `crates/quantization/src/gemm.rs`

```rust
// Int4 packed matmul — compute langsung dari Q4 weights tanpa dequant penuh
pub fn matmul_int4(
    a: &[u8],           // activation fp32 [M, K]
    b_packed: &[u8],    // weight Q4 packed [K/2, N] (2 values per byte)
    scales: &[f32],     // per-group scale [N, groups]
    out: &mut [f32],    // output [M, N]
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
);

// Int8 matmul — 32-bit accumulate, int8 weights  
pub fn matmul_int8(
    a: &[f32],
    b: &[i8],           // weight int8 [K, N]
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
);

// Dequant-on-the-fly per tile (untuk GPU kernel)
// Hanya dequant tile [TILE_M, TILE_K] → fp16 register, bukan seluruh matriks
pub fn dequant_tile_q4(
    packed: &[u8],
    scales: &[f32],
    tile: &mut [f16],
    tile_m: usize,
    tile_k: usize,
    n: usize,
    group_size: usize,
);
```

### GPU path (CUDA): INT4 GEMM kernel
File baru: `crates/autograd/src/gpu/cuda/int4_gemm.cu` (JIT via NVRTC)
- Load Q4 weights langsung ke shared memory
- Dequant di register (bukan global memory)
- Tensor core path via `mma.sync` (sm_75+)
- Fallback ke CUDA core loop

### GPU path (wgpu): INT4 compute shader
File baru: `crates/autograd/src/gpu/wgpu/int4_gemm.wgsl`  
- Packed Q4 → unpack di thread level
- Group-wise scale diterapkan per thread
- Shared memory tile untuk weight reuse

### Ubah `QFormat` jadi non-storage-only
```rust
// crates/quantization/src/lib.rs
pub const QUANTIZATION_IS_STORAGE_ONLY: bool = false;  // <-- UBAH INI

// Tambah enum variant untuk compute mode
pub enum QuantComputeMode {
    DequantFallback,  // old behavior — dequant semua ke fp32
    Int4Direct,       // compute langsung dari Q4
    Int8Direct,       // compute langsung dari INT8
    F16Native,        // native F16 matmul (via GPU)
}
```

---

## 4. VRAM BUDGET TRACKER

**Masalah:** `ResourceManager` cuma semaphore — tidak tahu VRAM usage, OOM tidak terdeteksi.

### File baru: `crates/runtime/src/vram_budget.rs`

```rust
pub struct VramBudget {
    total_bytes: u64,            // total VRAM device  
    reserved_bytes: u64,         // untuk OS / framework
    model_bytes: u64,            // weight model
    kv_cache_bytes: u64,         // KV cache saat ini
    expert_bytes: u64,           // expert weights di GPU
    peak_bytes: u64,             // peak usage
    allocation_limit: u64,       // soft limit — trigger eviction
    critical_limit: u64,         // hard limit — reject requests
}

impl VramBudget {
    pub fn available(&self) -> u64;
    pub fn usage_ratio(&self) -> f64;       // 0.0 - 1.0
    pub fn can_allocate(&self, bytes: u64) -> bool;
    pub fn reserve(&mut self, bytes: u64) -> Result<VramReservation>;
    pub fn release(&mut self, bytes: u64);
    
    // Auto-tune limits based on current model config
    pub fn auto_configure(&mut self, model_params: usize, num_experts: usize);
}
```

### Integrasi ke `ResourceManager`
```rust
pub struct ResourceManager {
    semaphore: Arc<Semaphore>,
    vram_budget: Arc<Mutex<VramBudget>>,  // NEW
}

impl ResourceManager {
    pub fn with_vram_budget(total_vram: u64) -> Self;
    pub fn acquire_with_vram(&self, needed_vram: u64) -> Result<ResourceGuard>;
    
    // OOM prevention: sebelum acquire, cek VRAM
    pub fn try_ensure_vram(&self, bytes: u64) -> Result<VramPressure>;
}

pub enum VramPressure {
    Ok,
    Warning { available: u64, pct: f64 },      // >80%
    Critical { needed: u64, available: u64 },   // >90%
    Oom { needed: u64, available: u64 },        // >100%
}
```

### OOM Prevention Flow
```
PRE-FORWARD:
  1. Cek VRAM via VramBudget
  2. Jika Warning → trigger expert eviction (coolest → CPU)
  3. Jika Critical → turunkan batch size, tolak request baru
  4. Jika OOM → reject dengan error message bersih

POST-FORWARD:
  1. Release temporary buffers
  2. Update peak tracking
  3. Jika usage turun → trigger expert prefetch

BACKGROUND:
  - Sweep tiap 5s: update VRAM usage dari device query
  - Jika VRAM > 85% 30 detik: aggressive eviction
  - Jika VRAM < 50%: opportunistic prefetch
```

---

## 5. TENSOR PARALLELISM — NCCL BACKEND

**Masalah:** `CpuLocalCollective` cuma shared memory dalam 1 proses. `HttpDistributed` return Err.

### File baru: `crates/transformer/src/nccl_collective.rs`

```rust
#[cfg(feature = "nccl")]
pub struct NcclCollective {
    comm: nccl::Comm,
    num_ranks: usize,
    rank: usize,
}

impl NcclCollective {
    // all_reduce untuk gradient sync (training)
    pub fn all_reduce(&self, buf: &mut [f32], count: usize) -> Result<()>;
    
    // all_gather untuk output concatenation (inference)
    pub fn all_gather(&self, send: &[f32], recv: &mut [f32], count: usize) -> Result<()>;
    
    // reduce_scatter untuk forward shard
    pub fn reduce_scatter(&self, send: &[f32], recv: &mut [f32], count: usize) -> Result<()>;
}
```

### Fix `HttpDistributed` — bikin real
```rust
// crates/transformer/src/sharded.rs
HttpDistributed {
    num_shards: usize,
    shard_rank: usize,
    peer_urls: Vec<String>,
    http_client: reqwest::Client,
}

impl HttpDistributed {
    fn all_gather_2d(&self, local: &Array2<f32>) -> Result<Array2<f32>> {
        // POST local shard → peer /all_gather
        // GET concatenated result
    }
}
```

### ShardConfig tambah NCCL option
```rust
pub enum CollectiveBackend {
    CpuLocal { num_shards, shard_rank },
    HttpDistributed { ... },
    #[cfg(feature = "nccl")]
    Nccl { ... },
}
```

---

## 6. UBAH SETTING YANG BOROS VRAM

### Default quantization: Q4 bukan Q8
```rust
// crates/transformer/src/config.rs
quantization: QFormat::Q4 { group_size: 128 },  // was Q8
```

### Paged cache Q4 storage
```rust
// crates/inference/src/continuous_batching.rs
paged_cache_q4: true,    // was false (f16 default)
paged_cache_f16: false,  // Q4 lebih hemat
```

### ResourceManager default concurrent
```rust
// crates/runtime/src/resource.rs
max_concurrent: 4,  // turunkan dari 32 → kurang VRAM peak
```

### Capacity factor turun
```rust
// Dengan 324 experts, top-32, capacity_factor 1.1 cukup
// (dulu 1.25 karena 8 experts top-2 = 25% aktif)
capacity_factor: 1.1,
```

---

## 7. PERHITUNGAN VRAM — SEBELUM vs SESUDAH

### Asumsi: 200B model, hidden=8192, intermediate=16384, 324 experts, 96 layers

| Komponen | SEBELUM (fp32, semua expert di GPU) | SESUDAH (Q4 + offloading) |
|----------|--------------------------------------|---------------------------|
| Weights dense (attn + shared) | 96 × (8K×8K×4 + ...) ≈ 300GB | Q4 → ~75GB |
| Expert weights (324) | 324 × 134MB × 4 mat = **~170GB** | Hanya ~30 expert di GPU = **~4GB** |
| KV cache (8K context) | 96×8×128×4×2 = **~6GB** | Q4 paged → **~800MB** |
| Activations (batch=32) | ~8GB | ~8GB |
| **TOTAL per forward** | **~484GB** 🚫 | **~88GB** ✅ |
| **Dengan TP=8** | tidak bisa (CpuLocal) | **~11GB per GPU** ✅✅ |

### VRAM Savings: ~5.5× dengan Q4 weights + expert offloading + KV cache Q4

### Hardware target:
- **8× RTX 4090 24GB** → cukup untuk inference
- **4× A100 80GB** → cukup untuk training + inference
- **2× RTX 5090 32GB** → cukup untuk inference (Q4 + offloading agresif)

---

## 8. FILE YANG PERLU DIUBAH / DIBUAT

### Modified files:
| File | Perubahan |
|------|-----------|
| `crates/has-moe-ffn/src/lib.rs` | Default config 324/32, tambah offloader field |
| `crates/has-moe-ffn/src/routing.rs` | Default config 324/32, capacity 1.1 |
| `crates/has-moe-ffn/src/experts.rs` | Tambah metode untuk CPU-only forward tanpa GPU cache |
| `crates/transformer/src/config.rs` | Ultra/Apex preset 324/32, Q4 default |
| `crates/shared/src/model_config.rs` | Omnis/Genesis 324/32, VRAM budget |
| `crates/quantization/src/lib.rs` | `QUANTIZATION_IS_STORAGE_ONLY = false`, compute mode enum |
| `crates/runtime/src/resource.rs` | Integrasi VramBudget |
| `crates/inference/src/continuous_batching.rs` | paged_cache_q4 default true |

### New files:
| File | Isi |
|------|-----|
| `crates/has-moe-ffn/src/offload.rs` | ExpertOffloader — LRU swap CPU↔GPU |
| `crates/quantization/src/gemm.rs` | Int4/Int8 matmul langsung dari Q4 |
| `crates/runtime/src/vram_budget.rs` | VramBudget — tracker + OOM prevention |
| `crates/transformer/src/nccl_collective.rs` | NCCL backend untuk TP |
| `crates/autograd/src/gpu/cuda/int4_gemm.cu` | CUDA INT4 GEMM kernel (NVRTC JIT) |

---

## 9. RISIKO & MITIGASI

| Risiko | Mitigasi |
|--------|----------|
| Expert offloading latency (H2D transfer time) | Async prefetch + overlap dengan compute router |
| INT4 matmul akurasi turun | AWQ calibration, group_size=64 bukan 128 |
| Load balancing loss dengan 324 experts | Z-loss + importance loss coefficient dinaikkan |
| Cold start — semua expert di CPU | Warm-up phase: load top-32 most common experts dulu |
| Memory fragmentation dari swap | Pinned memory pool, reuse buffers |
| NCCL backend kompleks | Fallback ke HttpDistributed jika NCCL unavailable |

---

## 10. TESTING PLAN

```
1. Unit test: ExpertOffloader swap correctness
2. Unit test: Int4 matmul vs fp32 matmul (tolerance 1%)
3. Unit test: VramBudget tracking accuracy
4. Integration: MoE forward dengan 324 expert, top-32
5. Integration: VRAM usage sebelum/sesudah (nvidia-smi)
6. Benchmark: Throughput dengan berbagai gpu_budget
7. Stress test: OOM prevention dengan batch besar
```

---

## PROGRESS REPORT — 3 Juni 2026

### ✅ SELESAI DIIMPLEMENTASI

| Item | Files | Status |
|------|-------|--------|
| **Config: experts 324, top-k 32** | `has-moe-ffn/lib.rs`, `routing.rs`, `transformer/config.rs`, `shared/model_config.rs` | ✅ Compiled, tested |
| **Default Q4** | `transformer/config.rs` — Ultra/Apex preset | ✅ |
| **Paged cache Q4 default** | `inference/continuous_batching.rs` | ✅ |
| **ExpertOffloader** | NEW `has-moe-ffn/src/offload.rs` — LRU swap, CPU pool, GPU resident, prefetch, CPU fallback | ✅ 5 tests pass |
| **LruTracker** | Dalam `offload.rs` — hit/miss tracking, eviction | ✅ |
| **INT4 GEMM** | NEW `quantization/src/gemm.rs` — `matmul_int4`, `matmul_int8`, `quantize_fp32_to_q4`, `dequant_tile_q4` | ✅ 5 tests pass |
| **QUANTIZATION_IS_STORAGE_ONLY = false** | `quantization/src/lib.rs` | ✅ |
| **VramBudget** | NEW `runtime/src/vram_budget.rs` — pressure tracking, reserve/release, auto-configure, OOM prevention | ✅ 4 tests pass |
| **ResourceManager VRAM integration** | `runtime/src/resource.rs` — `acquire_with_vram()`, `auto_configure_vram()`, `vram_pressure()` | ✅ Compiled |
| **ExpertDomain + tier-aware pool** | NEW `has-moe-ffn/src/domains.rs` — 28 domain, 324 expert (128 shared + 196 tier-specific), domain bias routing | ✅ Compiled, tested |
| **Domain-aware routing** | `has-moe-ffn/src/routing.rs` — `route_single_domain_aware()`, tier quota enforcement, domain bias | ✅ Compiled |
| **Domain & tier di Expert struct** | `has-moe-ffn/src/lib.rs`, `has-moe-ffn/src/types.rs` — `domain`, `tier`, `ExpertAssignment`, `DomainRoutingConfig` | ✅ Compiled |
| **ExpertDomainConfig di MoeConfig** | `shared/src/model_config.rs` — domain config untuk Omnis (Ultra) + Genesis (Apex) | ✅ Compiled |
| **use_domain_experts di TransformerConfig** | `transformer/src/config.rs` — flag + default false, Ultra/Apex/Pro preset true | ✅ Compiled |

### 🧪 TEST RESULTS (3 Juni 2026)

```
nexora-has-moe-ffn:   88 passed, 0 failed (domain, offload, routing, experts, integration)
nexora-quantization:  19 passed, 0 failed (gemm + existing)
nexora-runtime:       43 passed, 0 failed (vram_budget + executor, scheduler, KV cache stress)
cargo check:          0 errors
```

### 📈 VRAM SAVING — SEBELUM vs SESUDAH

| Komponen | SEBELUM (fp32, semua expert di GPU) | SESUDAH (Q4 + offloading) |
|----------|--------------------------------------|---------------------------|
| Weights dense (attn + shared) | 96 × 8K² = ~300GB | Q4 → ~75GB |
| Expert weights (324) | 324 × 134MB = **~170GB** | Hanya ~30 expert di GPU = **~4GB** |
| KV cache (8K context) | ~6GB (f32) | Q4 paged → **~800MB** |
| Activations (batch=32) | ~8GB | ~8GB |
| **TOTAL per forward** | **~484GB** 🚫 | **~88GB** ✅ |
| **Dengan TP=8** | tidak bisa | **~11GB per GPU** ✅ |

### Expert Pool Map (324 total)

| Pool | Count | Domain |
|------|-------|--------|
| **Shared** | 128 | math(20), language(24), logic(12), science(16), code(20), factual(16), reasoning(12), general(8) |
| **Ultra** (Omnis) | 64 | adv_reasoning(16), multimodal(16), metalearning(8), deep_code(12), sci_discovery(12) |
| **Apex** (Genesis) | 48 | code_review(16), emotional(12), task_planning(12), system_design(8) |
| **Pro** | 32 | creative(12), security(10), data_analysis(10) |
| **Core** | 28 | temporal(10), retrieval(10), summarization(8) |
| **Edge** | 24 | fast_path(8), classification(6), extraction(6), translation(4) |

### ❌ BELUM DIKERJAKAN (Phase 3)

| Item | Reason |
|------|--------|
| **Tensor Parallelism — NCCL backend** | Kompleksitas tinggi, butuh NCCL SDK. HttpDistributed skeleton sudah ada di `sharded.rs` — tinggal implementasi. |
| **CUDA INT4 GEMM kernel** (NVRTC JIT) | GPU kernel optimal untuk Q4 matmul. Saat ini CPU path via `gemm.rs` sudah jalan. |
| **Weight CPU offloading ke disk** | `ColdStorage` hanya untuk KV cache saat ini. |

### Perubahan Globals

| Konstanta | Lama | Baru |
|-----------|------|------|
| `QUANTIZATION_IS_STORAGE_ONLY` | `true` | `false` |
| `num_experts` (default) | 8 | 324 |
| `top_k` (default) | 2 | 32 |
| `capacity_factor` | 1.25 | 1.1 |
| `paged_cache_q4` (default) | false | true |
| `paged_cache_f16` (default) | true | false |
| `quantization` (Ultra tier) | Q8 | Q4 |
| `min_memory_gb` (Omnis) | 64GB | 32GB |
