# CUDA Backend & Model Scaling Plan

## 1. Arsitektur CUDA Backend

### Strategy: `candle` sebagai CUDA Runtime

Proyek saat ini pakai **wgpu** (Vulkan/Metal) untuk GPU. CUDA backend akan ditambahkan via **`candle`**:

- `candle-core` — tensor ops, device management
- `candle-nn` — linear, embedding, rmsnorm, rope
- `candle-flash-attn` — FlashAttention 2/3 untuk context panjang
- `candle-cuda` — CUDA graph, fused kernels, FP8

### Dual Backend Architecture

```
                    +-----------+
                    |  Abstraction Layer |
                    +-----+-----+
                          |
              +-----------+-----------+
              |                       |
        +-----v-----+          +------v------+
        |  wgpu      |          |  CUDA       |
        | (Vulkan)   |          | (candle)    |
        +------------+          +-------------+
        - dev/debug              - production
        - Apple Silicon          - NVIDIA H100/A100
        - fallback               - training skala besar
```

**Device auto-select**: Runtime detection — CUDA kalau NVIDIA GPU tersedia, fallback ke wgpu.

### Komponen CUDA yang Diperlukan

| Komponen | Library | Status |
|----------|---------|--------|
| Tensor ops (matmul, add, etc) | `candle-core` | ✅ Siap |
| Linear layer | `candle-nn` | ✅ Siap |
| Embedding | `candle-nn` | ✅ Siap |
| RMSNorm | WGSL → `candle-nn` | 🔄 Porting |
| RoPE (NTK-aware) | WGSL → `candle` kernel | 🔄 Porting |
| SwiGLU FFN | Custom → `candle` | 🔄 Porting |
| GQA FlashAttention | `candle-flash-attn` 2/3 | 🔄 Integrasi |
| MoE Gating + Experts | Custom CUDA kernel | 🆕 Bangun |
| KV Cache paged | `candle` buffers | 🆕 Bangun |
| Sampling (top-k/p) | `candle` ops | 🔄 Porting |
| Ring Attention (multi-GPU) | NCCL via `candle` | 🆕 Bangun |
| FP8/FP16 AMP training | `candle-cuda` | 🆕 Bangun |

### Cargo Feature Gate

```toml
[features]
default = ["wgpu"]
wgpu = ["dep:wgpu"]
cuda = ["dep:candle-core", "dep:candle-nn", "dep:candle-flash-attn"]
cuda-fp8 = ["cuda", "dep:candle-cuda"]
```

---

## 2. Architecture Gap Analysis

### Apakah Arsitektur Sekarang Siap untuk 146,6B?

**Tidak — 7 bottleneck fundamental:**

### 2.1 Weight Storage: `ndarray::Array2<f32>`

| Komponen | Kapasitas Sekarang | Butuh untuk 29B (1 model) |
|----------|-------------------|---------------------------|
| Tipe data | `Array2<f32>` (4B/param) | f16 (2B) / BF16 / FP8 |
| Inference | 146M × 4B = **0,6 GB** | 29B × 2B = **58,5 GB** |
| Adam states | 0,6×3 = 1,8 GB | 58,5×3 = **175,5 GB** |

`ndarray::Array2` tidak support f16/BF16 native dan **tidak bisa di-shard** antar GPU. Setiap weight matriks harus utuh di 1 device.

### 2.2 Distributed Infrastructure: Nihil

```
Sekarang:                    Butuh untuk 29B:
┌──────── Model ────────┐   ┌─GPU0──┐ ┌─GPU1──┐ ┌─GPU2──┐ ┌─GPU3──┐
│ semua weights di 1 GPU │   │layers │ │layers │ │layers │ │layers │
│ 1 proses, 1 device     │   │ 0-10  │ │ 11-20 │ │ 21-30 │ │ 31-40 │
└────────────────────────┘   │shard 1│ │shard 2│ │shard 3│ │shard 4│
                             └───────┘ └───────┘ └───────┘ └───────┘
                             FSDP + Tensor Parallelism + Pipeline Parallelism
```

| Komponen | Status | Kebutuhan 29B |
|----------|--------|---------------|
| NCCL / MPI | ❌ Nihil | FSDP sharding + all-reduce |
| Ring Attention | ❌ Nihil | Distributed KV cache 429 GB |
| FSDP | ❌ Nihil | Shard weights/grads/opt states |
| Pipeline parallelism | ❌ Nihil | Layer-level pipelining |

### 2.3 MoE: `Vec<Vec<Vec<f32>>>`

`HasMoeFFN` saat ini:
- **Storage**: nested `Vec` — allocasi terfragmentasi, cache miss
- **Compute**: loop over all 8 experts — **tidak sparse** (top-2 tetap compute semua)
- **GPU**: fallback ke CPU untuk GELU (lihat `experts.rs:103-181`)
- **Parallelism**: tidak ada expert placement — semua expert di device yang sama
- **Router**: `select_nth_unstable_by` — sorting CPU, bukan CUDA kernel

Target: fused CUDA kernel `[E, 3, H, I]` tensor dengan sparse dispatch.

### 2.4 Continuous Batching: Single Device

Engine (`ContinuousBatchingEngine`) sekarang:
- Single-threaded, single-device
- KV cache per-sequence di HashMap lokal
- Prefix trie di memory lokal
- Tidak ada mekanisme distributed scheduling

Target: distributed inference engine dengan request routing + cross-GPU cache.

### 2.5 RoPE: Terbatas ke 2K Token

`CausalLM::precomputed_cos/sin` = `[max_seq_len × head_dim/2]`.

Untuk 5M context (head_dim=128):
- `precomputed_cos`: 5.242.880 × 64 = 335M elemen @ f32 = **1,34 GB**
- `precomputed_sin`: **1,34 GB**
- Total: **2,68 GB** — masih manageable, tapi alokasi statis.

Masalah: tabel fix untuk 1 max_seq_len. Ganti ke NTK-aware dynamic (sudah ada `ExtendedRope::update_scaling()`).

### 2.6 Training Pipeline: Single GPU

Trainer sekarang semua data + grads + optimizer states di 1 proses:

```
1× GPU: Weights(4B) + Gradients(4B) + Adam(m)(4B) + Adam(v)(4B) = 16B/param
29B × 16B = 468 GB ─❌─→ gak muat GPU mana pun
```

Dengan FSDP across 8× H100:
```
29B × (2B(f16) + 2B(grad) + 4B(m) + 4B(v)) / 8 GPU = 29B × 12B / 8 = 43,5 GB ✅
```

Tapi FSDP harus diimplementasi dari nol — belum ada.

### 2.7 Mixed Precision: Bolak-balik f16→f32→f16

Sekarang: **f16 cuma untuk storage, compute tetap f32** — konversi bolak-balik setiap forward.

```
Upload:   CPU f32 → GPU f32 (wq_t)     + GPU packed f16 (wq_f16)     → duplikasi 1,5× VRAM
Forward:  packed f16 → WGSL unpack → f32 → matmul f32 → output f32  → 1× convert overhead
KV Cache: f32 → pack f16 → store → unpack f16 → f32 → attn         → 2× convert overhead
```

Tidak ada f16 matmul kernel, tidak ada BF16 compute, tidak ada FP8.

#### Opsi untuk 146,6B

| Approach | VRAM (29B) | Bandwidth | Compute | Kebutuhan |
|----------|------------|-----------|---------|-----------|
| **BF16 native** (via `candle`) | 58,5 GB ✅ | 1× ✅ | Stabil (7-bit mantissa) ✅ | Backend `candle` + BF16 kernel |
| **FP8** (H100 only) | 29,3 GB ✅✅ | 1× ✅ | ⚠️ Stabilitas loss scaling | `candle-cuda` FP8 + FlashAttention-3 |
| **f16→f32→f16 bolak-balik** (skrg) | 58,5 GB ✅ | ❌ 2-3× overhead | ✅ f32 compute | ~~Gak scale ke 29B~~ |
| **f32 murni** | ❌ 117 GB | - | ✅ | Gak muat GPU mana pun |

**Rekomendasi**: Stop pola bolak-balik. Waktu migrasi ke `candle`, langsung implementasi **BF16 native compute** — modern LLM training/inference standard (Llama, Mistral, GPT-4). Matmul, attention, semuanya di BF16. cuma optimizer state yg tetap f32 (master weights).

```
Target flow:
  Storage:  BF16 tensor (2 bytes/param)
  Compute:  BF16 matmul → BF16 attention → BF16 output
  Optimizer: f32 master weights (FSDP sharded)
  Gradients: BF16 all-reduce → f32 update
```

### 2.8 Lain-lain

| Item | Sekarang | Target |
|------|----------|--------|
| **Mixed precision** | Manual f16 packing | `candle-cuda` AMP: FP8/BF16 otomatis |
| **Activation memory** | Full retain for backward | Gradient checkpointing (recompute) |
| **Kernel fusion** | Per-op WGSL dispatch | Fused FlashAttention + MoE CUDA kernels |
| **Model serialization** | .safetensors via `ndarray` | .safetensors via `candle` tensor view |

### Gap Summary

| Area | Kapasitas Sekarang | Butuh untuk 29B + 5M | Rewrite |
|------|-------------------|----------------------|---------|
| Tensor backend | ~1B (f32, ndarray, single GPU) | 29B (f16/FP8, distributed) | 🔴 Total |
| Distributed | ❌ none | NCCL + FSDP + Ring Attention | 🆕 Bangun |
| MoE | toy (nested Vec, CPU fallback) | production sparse CUDA kernel | 🔴 Total |
| Inference engine | 1 GPU, 2K ctx | 8 GPU, 5M ctx, continuous batch | 🆕 Bangun |
| Training | 146M model | 29B × 8 GPU | 🆕 Bangun |
| Mixed precision | manual f16 pack | AMP BF16/FP8 | 🔴 Total |

Inilah kenapa roadmap estimasi 42 minggu — mayoritas adalah **rewrite komponen dari bawah**, bukan sekadar scale-up config.

---

## 3. Model Scaling Specifications

Semua multiplier adalah **kali (×)** dari parameter sekarang, bukan persen.

### Ultra Tier — Omnis, Axiom, Genesis

**Target**: 200× params (~29,3B), MoE 8 experts (top-2), context 5M

| Dimensi | Sekarang | Target |
|---------|----------|--------|
| hidden_size | 768 | **4096** |
| num_heads | 12 | **32** |
| num_kv_heads | 4 | **4** |
| num_layers | 8 | **40** |
| intermediate_size | 3072 | **6912** |
| MoE experts | - | **8 (top-2)** |
| max_seq_len | 2048 | **5.242.880 (5M)** |
| **Total params** | **146,4M** | **~29,27B** |
| Activated params/token | 146,4M | **~8,9B** |

```
Params breakdown:
  embedding (50257×4096)                     =   205,9M
  40 layers × (attn + MoEFFN 8 expert + norms)
    attn: Q 16,78M + K 2,10M + V 2,10M + O 16,78M  =  37,75M
    MoE FFN: 8 × 3 × 4096 × 6912                   = 679,48M
    norms: 2 × 4096                                     8K
    per layer                                       = 717,24M
  40 × 717,24M                                    =  28,69B
  final_norm                                       =   4,1K
  lm_head (50257×4096)                            = 205,9M
  ───────────────────────────────────────────────────────
  Total                                            ≈ 29,27B

  Activated per token: 205,9M + 40×(37,75M + 2×3×4096×6912 + 8K) + 205,9M
                      = 205,9M + 40×207,6M + 205,9M
                      ≈ 8,9B
```

### Apex Tier — Vortex, Aether, Nexum

**Target**: 175× params (~13,1B), MoE 6 experts (top-2), context 2,5M

| Dimensi | Sekarang | Target |
|---------|----------|--------|
| hidden_size | 512 | **3072** |
| num_heads | 8 | **24** |
| num_kv_heads | 4 | **8** |
| num_layers | 6 | **32** |
| intermediate_size | 2048 | **6912** |
| MoE experts | - | **6 (top-2)** |
| max_seq_len | 2048 | **2.621.440 (2,5M)** |
| **Total params** | **75,1M** | **~13,3B** |
| Activated params/token | 75,1M | **~3,9B** |

```
Params breakdown:
  embedding (50257×3072)                     =   154,4M
  32 layers × (attn + MoEFFN 6 expert + norms)
    attn: 9,44M + 3,15M + 3,15M + 9,44M            =  25,17M
    MoE FFN: 6 × 3 × 3072 × 6912                   = 382,21M
    norms: 2 × 3072                                  =   6,1K
    per layer                                       = 407,38M
  32 × 407,38M                                    =  13,04B
  final_norm                                       =   3,1K
  lm_head (50257×3072)                            = 154,4M
  ───────────────────────────────────────────────────────
  Total                                            ≈ 13,34B

  Activated per token: 154,4M + 32×(25,17M + 2×3×3072×6912 + 6K) + 154,4M
                      ≈ 3,9B
```

### Pro Tier — Spectra, Cipher

**Target**: 150× params (~7,1B), MoE 4 experts (top-2), context 1M

| Dimensi | Sekarang | Target |
|---------|----------|--------|
| hidden_size | 384 | **2048** |
| num_heads | 6 | **16** |
| num_kv_heads | 3 | **8** |
| num_layers | 4 | **32** |
| intermediate_size | 1536 | **8192** |
| MoE experts | - | **4 (top-2)** |
| max_seq_len | 2048 | **1.048.576 (1M)** |
| **Total params** | **47,4M** | **~7,05B** |
| Activated params/token | 47,4M | **~2,1B** |

```
Params breakdown:
  embedding (50257×2048)                     =   102,9M
  32 layers × (attn + MoEFFN 4 expert + norms)
    attn: 4,19M + 2,10M + 2,10M + 4,19M            =  12,58M
    MoE FFN: 4 × 3 × 2048 × 8192                   = 201,33M
    norms: 2 × 2048                                  =   4,1K
    per layer                                       = 213,91M
  32 × 213,91M                                    =   6,85B
  final_norm                                       =   2,0K
  lm_head (50257×2048)                            = 102,9M
  ───────────────────────────────────────────────────────
  Total                                            ≈ 7,05B

  Activated per token: 102,9M + 32×(12,58M + 2×3×2048×8192 + 4K) + 102,9M
                      ≈ 2,1B
```

### Core Tier — Kronos

**Target**: 100× params (~2,9B), no MoE, context 500K

| Dimensi | Sekarang | Target |
|---------|----------|--------|
| hidden_size | 256 | **3072** |
| num_heads | 4 | **24** |
| num_kv_heads | 2 | **8** |
| num_layers | 3 | **24** |
| intermediate_size | 1024 | **8704** |
| MoE | - | none |
| max_seq_len | 2048 | **524.288 (500K)** |
| **Total params** | **28,7M** | **~2,84B** |

```
Params breakdown:
  embedding (50257×3072)                     =   154,4M
  24 layers × (attn + FFN + norms)
    attn: 9,44M + 3,15M + 3,15M + 9,44M            =  25,17M
    FFN: 3 × 3072 × 8704                             =  80,22M
    norms: 2 × 3072                                  =   6,1K
    per layer                                       = 105,39M
  24 × 105,39M                                    =   2,53B
  final_norm                                       =   3,1K
  lm_head (50257×3072)                            = 154,4M
  ───────────────────────────────────────────────────────
  Total                                            ≈ 2,84B
```

### Edge Tier — Swift

**Target**: 100× params (~1,3B), no MoE, context 500K

| Dimensi | Sekarang | Target |
|---------|----------|--------|
| hidden_size | 128 | **2048** |
| num_heads | 4 | **16** |
| num_kv_heads | 2 | **8** |
| num_layers | 2 | **20** |
| intermediate_size | 512 | **7168** |
| MoE | - | none |
| max_seq_len | 2048 | **524.288 (500K)** |
| **Total params** | **13,4M** | **~1,34B** |

```
Params breakdown:
  embedding (50257×2048)                     =   102,9M
  20 layers × (attn + FFN + norms)
    attn: 4,19M + 2,10M + 2,10M + 4,19M            =  12,58M
    FFN: 3 × 2048 × 7168                             =  44,04M
    norms: 2 × 2048                                  =   4,1K
    per layer                                       =  56,63M
  20 × 56,63M                                     =   1,13B
  final_norm                                       =   2,0K
  lm_head (50257×2048)                            = 102,9M
  ───────────────────────────────────────────────────────
  Total                                            ≈ 1,34B
```

---

## 4. Model Summary

| Model | Tier | Sekarang | Target | × | MoE | Experts | Aktif | Context |
|-------|------|----------|--------|---|-----|---------|-------|---------|
| Omnis | Ultra | 146,4M | **29,3B** | 200× | ✅ | 8 (top-2) | 8,9B | 5M |
| Axiom | Ultra | 146,4M | **29,3B** | 200× | ✅ | 8 (top-2) | 8,9B | 5M |
| Genesis | Ultra | 146,4M | **29,3B** | 200× | ✅ | 8 (top-2) | 8,9B | 5M |
| Vortex | Apex | 75,1M | **13,3B** | 175× | ✅ | 6 (top-2) | 3,9B | 2,5M |
| Aether | Apex | 75,1M | **13,3B** | 175× | ✅ | 6 (top-2) | 3,9B | 2,5M |
| Nexum | Apex | 75,1M | **13,3B** | 175× | ✅ | 6 (top-2) | 3,9B | 2,5M |
| Spectra | Pro | 47,4M | **7,1B** | 150× | ✅ | 4 (top-2) | 2,1B | 1M |
| Cipher | Pro | 47,4M | **7,1B** | 150× | ✅ | 4 (top-2) | 2,1B | 1M |
| Kronos | Core | 28,7M | **2,8B** | 100× | - | - | 2,8B | 500K |
| Swift | Edge | 13,4M | **1,3B** | 100× | - | - | 1,3B | 500K |
| **Total** | | | **~146,6B** | | | | | |

---

## 5. Context Length Strategy

Context 5M di Ultra (40 layers, kv_heads=4) membutuhkan:

### 5.1 RoPE Scaling

| Teknik | Formula |
|--------|---------|
| NTK-aware | `theta' = theta × 100^(log(seq_len/2048) / log(head_dim/2-1))` |
| YaRN | Scale factor ratio (α) + length scaling (β) |
| Dynamic NTK | Theta berubah per posisi — sudah ada di `ExtendedRope` |

### 5.2 FlashAttention 2/3

Via `candle-flash-attn`:
- IO-aware: minimalkan HBM reads/writes
- Fused kernel: attention + softmax + dropout
- Varlen support untuk continuous batching
- FP8 support di FlashAttention-3 (H100)

### 5.3 Ring Attention (Multi-GPU Distributed)

KV cache 5M context tidak muat di 1 GPU mana pun. Wajib distribute.

#### Ultra 5M — KV Cache Distribution

```
KV cache size (f16):
  = 2 × layers × kv_heads × head_dim × seq_len × 2 bytes
  = 2 × 40 × 4 × 128 × 5.242.880 × 2
  = 429,5 GB

Dengan 8× H100 80GB (Ring Attention):
  KV cache per GPU: 429,5 / 8 = 53,7 GB
  Weights (FSDP): 29,3B × 2 / 8 = 7,3 GB
  Activations: ~2 GB
  ─────────────────────────────
  Total per GPU: ~63 GB ✅ (headroom 17 GB)
```

```
GPU 0: ctx[0..655K]    GPU 1: ctx[655K..1,3M]   GPU 2: ctx[1,3M..2M]   GPU 3: ctx[2M..2,6M]
GPU 4: ctx[2,6M..3,3M]  GPU 5: ctx[3,3M..3,9M]   GPU 6: ctx[3,9M..4,6M]  GPU 7: ctx[4,6M..5,2M]

Tiap GPU punya full Q (local), pegang 1/8 K/V blocks
Ring pass: kirim K/V block ke GPU tetangga via NCCL
Overlap compute + P2P transfer
```

#### Apex 2,5M — KV Cache Distribution

```
KV cache (f16): 2 × 32 × 8 × 128 × 2.621.440 × 2 = 343,6 GB

Dengan 4× H100:
  343,6 / 4 = 85,9 GB ❌ (overshoot H100 80GB)

Dengan 8× H100:
  343,6 / 8 = 43,0 GB
  Weights (FSDP): 13,3B × 2 / 8 = 3,3 GB
  Total: ~48,3 GB ✅
```

#### Pro 1M — KV Cache Distribution

```
KV cache (f16): 2 × 32 × 8 × 128 × 1.048.576 × 2 = 137,4 GB

Dengan 4× H100:
  137,4 / 4 = 34,4 GB
  Weights (FSDP): 7,1B × 2 / 4 = 3,5 GB
  Total: ~39,9 GB ✅
```

#### Kronos/Swift 500K — Single GPU

```
KV cache (f16): 2 × 24 × 8 × 128 × 524.288 × 2 = 51,5 GB
Weights (f16): 2,8B × 2 = 5,7 GB
Total: ~57,2 GB → muat di 1× H100 / 2× A6000 ✅
```

### 5.4 Context Scaling Summary

| Model | Context | KV Cache | GPU Min | Config |
|-------|---------|----------|---------|--------|
| Ultra | 5M | 429,5 GB | **8× H100** | Ring Attn + FlashAttn 3 + FSDP |
| Apex | 2,5M | 343,6 GB | **8× H100** | Ring Attn + FlashAttn 3 + FSDP |
| Pro | 1M | 137,4 GB | **4× H100** | Ring Attn + FlashAttn 2 + FSDP |
| Core | 500K | 51,5 GB | **1× H100** | FlashAttn 2 |
| Edge | 500K | 25,7 GB | **1× A6000** | FlashAttn 2 |

---

## 6. MoE Implementation Detail

### Arsitektur MoE per Layer

```
Input (hidden_size)
    │
    ├── Router: MLP gate → softmax → top-2
    │            [E × hidden → logits → softmax → topk]
    │
    ├── Expert 0 ─┐  fc1(SiLU)  fc3  multiply  fc2
    ├── Expert 1 ─┤
    ├── Expert 2 ─┤
    ├── Expert 3 ─┤  (top-2 active per token)
    ├── Expert 4 ─┤
    ├── Expert 5 ─┤
    ├── Expert 6 ─┤
    └── Expert 7 ─┘
         │
    Weighted sum (gating_weights × expert_output)
         │
    + residual → output
```

### Load Balancing + Z-Loss

```
L_balancing = α × E × Σ(f_i × P_i)   // α = 0,01
  f_i = fraction tokens → expert i
  P_i = avg router probability untuk expert i

L_zloss = β × log(Σ(exp(logits)))²   // β = 0,001
```

### Expert Structure (SwiGLU)

```
Expert: fc1(H→I) → SiLU → fc3(H→I) → multiply → fc2(I→H)
         ↑ gate_proj     ↑ up_proj              ↑ down_proj
```

### CUDA Kernel Skeleton

```cuda
__global__ void moe_forward(
  const float* input,     // [B, H]
  const float* experts,   // [E, 3, H, I] — fc1, fc3, fc2 stacked
  const float* router_w,  // [E, H]
  float* output,          // [B, H]
  int B, int H, int I, int E, int top_k
) {
  // 1. Router: matmul(input, router_w^T) → softmax → top-2
  // 2. Dispatch: token → expert via top-2 indices
  // 3. Expert compute: gate=fc1(x), up=fc3(x), h=silu(gate)*up, out=fc2(h)
  // 4. Scatter-add: output[t] += weight[t][e] * expert_out
}
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (10 minggu)

| Step | Deliverable | Durasi |
|------|-------------|--------|
| 1.1 | Integrasi `candle-core` + `candle-nn` sbg backend opsional | 3 mg |
| 1.2 | Port tensor ops + RMSNorm + RoPE dari WGSL ke candle | 3 mg |
| 1.3 | Port GQA + SwiGLU + sampling | 2 mg |
| 1.4 | Device abstraction layer + auto-select CUDA > wgpu > CPU | 1 mg |
| 1.5 | FP16 inference path | 1 mg |

### Phase 2: Model Scaling (8 minggu)

| Step | Deliverable | Durasi |
|------|-------------|--------|
| 2.1 | Implementasi config baru di `TransformerConfig` + init | 2 mg |
| 2.2 | Extend RoPE: NTK-aware + YaRN + Dynamic NTK | 2 mg |
| 2.3 | FlashAttention 2 integrasi via `candle-flash-attn` | 2 mg |
| 2.4 | Scale-down test: train Swift-scale (1,3B) + validasi loss | 2 mg |

### Phase 3: MoE Integration (8 minggu)

| Step | Deliverable | Durasi |
|------|-------------|--------|
| 3.1 | Refactor `TransformerBlock.ffn` → trait: DenseFFN | MoeFFN | 2 mg |
| 3.2 | Router forward + top-2 + load balancing loss (CUDA) | 2 mg |
| 3.3 | Expert forward CUDA kernel + token dispatch | 2 mg |
| 3.4 | MoE checkpointing (.safetensors) + test convergence | 2 mg |

### Phase 4: Distributed Context (10 minggu)

| Step | Deliverable | Durasi |
|------|-------------|--------|
| 4.1 | NCCL integration via `candle` + multi-GPU tensor ops | 3 mg |
| 4.2 | Ring Attention: block-sparse KV + P2P ring pass | 3 mg |
| 4.3 | FSDP (Fully Sharded Data Parallel) untuk weights | 2 mg |
| 4.4 | 5M context stress test + OOM recovery + graceful degradation | 2 mg |

### Phase 5: Production Optimization (6 minggu)

| Step | Deliverable | Durasi |
|------|-------------|--------|
| 5.1 | FP8 training via `candle-cuda` AMP | 2 mg |
| 5.2 | CUDA graph optimization + fused MoE kernels | 2 mg |
| 5.3 | Continuous batching engine CUDA port | 1 mg |
| 5.4 | End-to-end benchmark: latency, throughput, memory | 1 mg |

**Total estimasi: 42 minggu (~10 bulan)** dengan 1-2 engineer Rust + CUDA.

---

## 8. VRAM Requirements

### Inference (f16 weights)

| Model | Params | Weights (f16) | KV Cache (max ctx) | Total | GPU Config |
|-------|--------|---------------|---------------------|-------|------------|
| Ultra | 29,3B | 58,5 GB | 429,5 GB | 488,0 GB | **8× H100** |
| Apex | 13,3B | 26,7 GB | 343,6 GB | 370,3 GB | **8× H100** |
| Pro | 7,1B | 14,1 GB | 137,4 GB | 151,5 GB | **4× H100** |
| Kronos | 2,8B | 5,7 GB | 51,5 GB | 57,2 GB | **1× H100** |
| Swift | 1,3B | 2,7 GB | 25,7 GB | 28,4 GB | **1× A6000** |

### Training (FSDP + mixed precision + gradient checkpointing)

| Model | Batch | Weights+grads+opt | Activations | Total | GPU |
|-------|-------|-------------------|-------------|-------|-----|
| Ultra | 1/GPU | 58,5×2×4/8=58,5 GB | ~8 GB | 66,5 GB | 8× H100 |
| Apex | 2/GPU | 26,7×2×4/8=26,7 GB | ~4 GB | 30,7 GB | 8× H100 |
| Pro | 4/GPU | 14,1×2×4/4=28,2 GB | ~3 GB | 31,2 GB | 4× H100 |
| Kronos | 8/GPU | 5,7×2×4/1=45,6 GB | ~4 GB | 49,6 GB | 1× H100 |
| Swift | 16/GPU | 2,7×2×4/1=21,6 GB | ~3 GB | 24,6 GB | 1× A6000 |

> *Weight×2×4 = f16 weights + f32 gradients + f32 Adam states = 2 + 4 + 4 = 10× f16 setara*
> *Dibagi jumlah GPU via FSDP*

### GPU Minimum Recommendations

| Use Case | GPU | VRAM | Qty | Notes |
|----------|-----|------|-----|-------|
| Dev / Swift | A6000 | 48 GB | 1 | Bisa training Swift + inference |
| Kronos training | H100 | 80 GB | 1 | Muat training dengan batch kecil |
| Pro training | H100 | 80 GB | 4 | FSDP + Ring Attention |
| Apex training | H100 | 80 GB | 8 | Full pipeline |
| **Ultra training + 5M ctx** | **H100** | **80 GB** | **8** | **Minimal config untuk target akhir** |
| Production Ultra | H100 | 80 GB | 16+ | Inference + serving + redundancy |

---

## 9. New Dependencies

```toml
# crates/transformer/Cargo.toml
[target.'cfg(any(target_os = "linux", target_os = "windows"))'.dependencies]
candle-core = { version = "0.8", optional = true }
candle-nn = { version = "0.8", optional = true }
candle-flash-attn = { version = "0.8", optional = true }

[features]
cuda = ["dep:candle-core", "dep:candle-nn", "dep:candle-flash-attn"]
cuda-fp8 = ["cuda", "dep:candle-cuda"]

# crates/has-moe-ffn/Cargo.toml
cuda = ["dep:candle-core"]

# crates/inference/Cargo.toml (Ring Attention)
cuda-distributed = ["cuda", "dep:nccl"]
```

---

## 10. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Context 5M KV cache 429GB | OOM | Ring Attention 8 GPU + FP8 KV cache (214 GB) + offloading |
| Training 29B stabil | Convergence failure | Warmup + grad clipping + Z-loss + BF16 mixed precision |
| MoE load imbalance | Expert collapse | Load balancing loss (α=0,01) + expert dropout + capacity capping |
| Rust + CUDA FFI complexity | Development slow | `candle` handle CUDA API — tidak perlu CUDA C manual |
| wgpu ↔ CUDA dual maintenance | Code bloat | Trait abstraction layer di `crates/transformer/src/backend/` |
