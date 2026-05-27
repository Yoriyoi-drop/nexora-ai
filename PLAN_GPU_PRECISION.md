# PLAN: GPU Migration & Precision Strategy — 10 Model ~146.6B

## 1. Arsitektur Model (Bukan Monolitik)

10 model independen, bukan satu model 146.6B. Setiap model punya konfigurasi sendiri:

| Tier | Model | Total | MoE | Experts | Active | Context | F16 VRAM | F32 VRAM |
|------|-------|-------|-----|---------|--------|---------|----------|----------|
| Ultra | Omnis | 29.3B | ✅ | 8 (top-2) | 8.9B | 5M | ~58.6GB | ~117.2GB |
| Ultra | Axiom | 29.3B | ✅ | 8 (top-2) | 8.9B | 5M | ~58.6GB | ~117.2GB |
| Ultra | Genesis | 29.3B | ✅ | 8 (top-2) | 8.9B | 5M | ~58.6GB | ~117.2GB |
| Apex | Vortex | 13.3B | ✅ | 6 (top-2) | 3.9B | 2.5M | ~26.6GB | ~53.2GB |
| Apex | Aether | 13.3B | ✅ | 6 (top-2) | 3.9B | 2.5M | ~26.6GB | ~53.2GB |
| Apex | Nexum | 13.3B | ✅ | 6 (top-2) | 3.9B | 2.5M | ~26.6GB | ~53.2GB |
| Pro | Spectra | 7.1B | ✅ | 4 (top-2) | 2.1B | 1M | ~14.2GB | ~28.4GB |
| Pro | Cipher | 7.1B | ✅ | 4 (top-2) | 2.1B | 1M | ~14.2GB | ~28.4GB |
| Core | Kronos | 2.8B | - | - | 2.8B | 500K | ~5.6GB | ~11.2GB |
| Edge | Swift | 1.3B | - | - | 1.3B | 500K | ~2.6GB | ~5.2GB |

**Kunci:** Karena MoE top-2, hanya ~30% parameter aktif per forward. Tapi **semua expert tetap harus di-load** di VRAM.

---

## 2. Precision Strategy

### 2.1 Final Target (Production Inference)

| Tier | Precision | Alasan |
|------|-----------|--------|
| Ultra (29.3B) | **f16** | 58.6GB — muat 1× H100 (80GB) dengan sisa untuk KV cache + activations |
| Apex (13.3B) | **f16** | 26.6GB — nyaman di H100, bahkan bisa 2 model paralel |
| Pro (7.1B) | **f16 / int8** | 14.2GB — bisa di GPU consumer (RTX 4090 24GB via int8) |
| Core (2.8B) | **f16 / f32** | 5.6GB — GPU mana pun |
| Edge (1.3B) | **f16 / f32** | 2.6GB — GPU mana pun, bahkan iGPU |

**Mengapa bukan f32?**
- Ultra 29.3B f32 = 117.2GB > H100 80GB — tidak muat
- Apex 13.3B f32 = 53.2GB — muat tapi boros, tidak ada manfaat untuk inference
- f16 memberikan 2× lebih padat, throughput hampir sama (Tensor Core H100 mendukung f16 native)

**Mengapa bukan int8 untuk semuanya?**
- int8 (1 byte/param) — kualitas degradasi masih bisa ditoleransi untuk MoE
- Target pertama: f16 stabil. int8 sebagai opsional deployment.

### 2.2 Training / Fine-tuning Strategy

| Tier | Forward/Backward | Master Weights | Optimizer (Adam) | Total VRAM | GPU |
|------|-----------------|----------------|-----------------|------------|-----|
| Ultra 29.3B | **bf16** | fp32 | fp32 | ~293GB | 4-8× H100 (ZeRO-3) |
| Apex 13.3B | **bf16** | fp32 | fp32 | ~133GB | 2-4× H100 (ZeRO-2/3) |
| Pro 7.1B | **bf16** | fp32 | fp32 | ~71GB | 1-2× H100 |
| Core 2.8B | **f16/bf16** | fp32 | fp32 | ~28GB | 1× GPU |
| Edge 1.3B | **f16** | fp32 | fp32 | ~13GB | 1× GPU |

**Mengapa bf16 untuk training (bukan f16)?**
- bf16 punya range lebih besar (8-bit exponent vs 5-bit) — lebih stabil untuk gradient
- f16 rawan overflow/underflow pada loss scale
- H100 Tensor Core mendukung bf16 native dengan throughput sama dengan f16

---

## 3. GPU Architecture — Multi-GPU Readiness

### 3.1 Current State Problem

| Problem | Detail |
|---------|--------|
| **Single GPU** | `OnceLock<GpuWeights>` — semua weights di 1 GPU context |
| **wgpu backend** | Tidak bisa Tensor Core, tidak bisa NVLink/NCCL |
| **CPU fallback sirkuit breaker** | 50 gagal → semua fallback ke CPU permanen |
| **Tidak ada tensor parallelism** | Matriks besar tidak bisa di-shard |
| **Tidak ada pipeline parallelism** | Layer tidak bisa dibagi antar GPU |
| **Tidak ada NCCL** | Cross-GPU sync pakai naive flat buffer copy |
| **Memory pool max 4GB** | Bucket alokasi maksimal 4GB |
| **Softmax backward CPU readback** | Training GPU jadi hambatan |

### 3.2 Target Architecture

```
                         ┌──────────────────┐
                         │   Request Router  │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
       ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
       │   GPU 0      │    │   GPU 1      │    │   GPU 2      │
       │ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────┐ │
       │ │ Layers   │ │    │ │ Layers   │ │    │ │ Layers   │ │
       │ │ 0-31     │ │    │ │ 32-63    │ │    │ │ 64-95    │ │
       │ │ (pipeline)│ │    │ │ (pipeline)│ │    │ │ (pipeline)│ │
       │ └──────────┘ │    │ └──────────┘ │    │ └──────────┘ │
       │ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────┐ │
       │ │ TP rank 0│ │    │ │ TP rank 1│ │    │ │ TP rank 2│ │
       │ └──────────┘ │    │ └──────────┘ │    │ └──────────┘ │
       └──────────────┘    └──────────────┘    └──────────────┘
```

**3 strategi paralelisme yang dibutuhkan:**

| Strategi | Fungsi | Untuk Model |
|----------|--------|-------------|
| **Tensor Parallelism (TP)** | Shard weight matrix per layer (column/row split) | Ultra, Apex |
| **Pipeline Parallelism (PP)** | Bagi layer per GPU | Ultra, Apex |
| **Data Parallelism (DP)** | Replikasi model, beda data batch | Semua (training) |

### 3.3 ZeRO Stages untuk Training

| Stage | Shard | VRAM Hemat | Untuk |
|-------|-------|------------|-------|
| ZeRO-1 | Optimizer states only | ~4× | Apex, Pro |
| ZeRO-2 | + Gradients | ~8× | Ultra |
| ZeRO-3 | + Parameters | ~16× | Ultra (multi-node) |

---

## 4. Implementation Roadmap

### Phase A: Foundation GPU (AVRILAN — 2-3 minggu) ✅ Progress: 27 Mei 2026

Goal: Semua model bisa inference di GPU tanpa CPU fallback.

```
[P1] Hapus CPU fallback circuit breaker. ✅ DONE
     - Ganti GPU_CIRCUIT_BROKEN (permanent AtomicBool) dengan
       GpuRecoveryManager (stateful Open/HalfOpen/Closed, auto-recover 30s)
     - Di-wire ke matmul.rs + math.rs (9 fungsi math + 1 matmul)
     - record_recovery() dipanggil setiap sukses, record_failure() setiap gagal
     - Tidak ada lagi fallback permanen — GPU auto-recover setelah cooldown 30s
     
[P1] Fixed memory pool: max bucket dari 4GB → 80GB. ✅ DONE
     - SIZE_BUCKETS ditambah: 8GB, 16GB, 32GB, 64GB, 80GB
     - Cukup untuk H100 (80GB) dan GH200 (144GB)
     
[P2] Fix ops yang masih CPU-only:
     - Reshape → GpuTensor::reshape() ✅ DONE (zero-copy metadata, GPU forward+backward)
     - sum_backward / mean_backward → fill_constant WGSL ✅ DONE
       (ganti from_cpu + CPU readback dengan GPU fill_constant kernel)
     - Concatenation → WGSL kernel ⏳ (low priority — hanya dipakai GNAC CPU backend)
     - Softmax backward → full GPU (already has GPU backward, non-last-dim fallback OK)

[P2] Model scaling config + MoE support. ✅ DONE
     - TransformerConfig: num_experts, top_k_experts, expert_intermediate_size
     - ModelTier enum: Ultra/Apex/Pro/Core/Edge
     - TransformerConfig::preset(tier) untuk 5 tier
     - parameter_count() + active_parameters() untuk MoE
     - Default vocab_size: 100000
     
[P3] Ganti wgpu → CUDA backend (atau tambah CUDA). ⏳
     wgpu bagus untuk portability tapi tidak bisa Tensor Core, NVLink.
      
[P3] Embedding lookup GPU-native ✅ SUDAH ADA sejak awal.
     - ctx.embedding(gpu_ids, gpu_w) WGSL kernel
     - GPU backward via embedding_backward_gpu
```

### Phase B: Single-GPU Inference (JUNI — 2 minggu)

Goal: Setiap model bisa jalan di GPU sendiri.

```
[P1] Model dimensi scaling:
     - Buat preset config untuk setiap tier
     - TransformerConfig.scale(tier: Tier) -> konfigurasi otomatis
     - parameter_count() validasi sesuai target

[P1] Weight storage dari OnceLock → GpuWeightShard:
     - Bisa partial load (tidak semua weights harus di VRAM)
     - MoE expert pinning (experts paling sering dipanggil tetap di VRAM)

[P2] KV cache untuk 5M context (Ultra):
     - PagedAttention yang sudah ada (PagedKVCache)
     - Prefix caching (PrefixDAG sudah ada)
     - Context window management

[P2] MoE routing GPU-native:
     - Router dari `has-moe-ffn` → WGSL kernel
     - Top-2 selection GPU (bukan CPU)

[P3] Sampling full GPU:
     - Top-K, Top-P, temperature sudah ada WGSL
     - Integrasi ke forward pipeline tanpa CPU readback
```

### Phase C: Multi-GPU Inference (JULI — 4 minggu)

Goal: Model Ultra (29.3B) jalan di 2-4 GPU dengan tensor/pipeline parallelism.

```
[P1] Tensor Parallelism Framework:
     - TPLinear: column shard + row shard + all-reduce
     - TPAttention: split heads across GPUs
     - TPMoE: split experts across GPUs
     - AllReduce GPU kernel via NVLink (bukan CPU)

[P1] Pipeline Parallelism Framework:
     - Layer partitioning: N layers per GPU
     - Micro-batch scheduling (1F1B)
     - Bubble optimal: 4-8 micro-batches

[P2] Multi-GPU weight loader:
     - Load safetensors → shard → distribute ke GPU masing-masing
     - Sharded checkpoint format: `model-rank-{i}-of-{n}.safetensors`

[P2] NCCL/NCCl bindings:
     - Atau pake NVSHMEM via rust
     - Atau CUDA IPC untuk inter-GPU dalam 1 node
     - Fallback: CPU mpi (lambat, untuk development)
```

### Phase D: Multi-GPU Training (AGUSTUS — 4-6 minggu)

Goal: Fine-tune model Ultra 29.3B dengan ZeRO-3.

```
[P1] ZeRO-1: Shard optimizer states
     - Adam states (momentum + variance) dibagi antar GPU
     - AllGather sebelum update, ReduceScatter setelah

[P1] ZeRO-2: + shard gradients
     - Gradient di-reduce-scatter ke GPU pemilik
     - overlap communication + computation

[P2] ZeRO-3: + shard parameters
     - Parameter di-all-gather sebelum forward/backward
     - Prefetch parameter untuk hide latency

[P2] Activation checkpointing:
     - Recomputed activations di backward (bukan disimpan semua)
     - Trade: ~33% compute tambahan, 50-80% VRAM hemat

[P3] Mixed precision trainer:
     - fp32 master weights (cpu atau GPU terpisah)
     - bf16 forward/backward
     - Loss scaling otomatis (sudah ada LossScaler)
```

### Phase E: Serving Architecture (SEPTEMBER — 2-3 minggu)

Goal: Production-grade multi-model serving.

```
[P1] Model scheduler:
     - Dynamic GPU assignment: model kecil (Edge/Core) bisa sharing 1 GPU
     - Model warm-up: load weights sebelum request pertama
     - GPU memory pressure monitoring

[P2] Request router:
     - Route ke model tier sesuai kebutuhan
     - Queue priority: Ultra > Apex > Pro > Core > Edge
     - Rate limiting per model

[P3] Continuous batching (existing):
     - PagedAttention + PrefixDAG sudah ada
     - Flight recorder + observability
```

---

## 5. Hardware Requirements

### Development / Training

| Setup | GPU | Total VRAM | Mampu |
|-------|-----|-----------|-------|
| Dev laptop | 1× RTX 4090 | 24GB | Edge, Core, Pro (int8) |
| Dev server | 1× H100 | 80GB | Pro, Apex, Ultra (f16 inference) |
| Training small | 4× H100 | 320GB | Apex training, Ultra inference |
| Training full | 8× H100 NVLink | 640GB | Ultra training (ZeRO-3) |

### Production Inference

| Model | GPU | Strategi |
|-------|-----|----------|
| Edge 1.3B | 1× T4 (16GB) | f16, sequential |
| Core 2.8B | 1× T4 (16GB) | f16, sequential |
| Pro 7.1B | 1× L4 (24GB) | int8 |
| Apex 13.3B | 1× H100 (80GB) | f16 |
| Ultra 29.3B | 1× H100 (80GB) | f16 → sisa 21GB untuk KV cache 5M |
| Ultra training | 4-8× H100 (320-640GB) | bf16 + ZeRO-3 |

---

## 6. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| **wgpu tidak support NVLink** | Multi-GPU lambat | Tambah CUDA backend sebagai opsional (`--features cuda`) |
| **MoE load imbalance** | Expert collapse | Load balancing loss sudah ada di `has-moe-ffn` |
| **5M context = 3.8GB KV cache** | VRAM habis | PagedAttention + prefix sharing + context eviction |
| **f16 overflow (training)** | NaN loss | bf16 > f16, loss scaler, gradient clipping |
| **Tensor Core tidak terpakai** | Performance 50% | CUDA backend wajib, wgpu via Vulkan tidak akses Tensor Core |
| **Weight loading 58GB** | Loading lambat | Memory-mapped loading, async pipeline, sharded format |
| **CPU→GPU migration error** | Silent correctness bug | Per-layer output diff test (CPU vs GPU) |

---

## 7. Decision Points (Perlu Diskusi)

1. **wgpu vs CUDA**: wgpu bagus untuk portability (Mac, Windows, Linux) tapi tidak bisa NVLink/Tensor Core. Opsinya: tambah CUDA sebagai feature flag (`cuda`), wgpu tetap untuk development.

2. **f16 vs bf16 untuk inference**: f16 lebih umum, bf16 lebih stabil. Untuk inference (tanpa gradient), f16 cukup.

3. **int8 deployment**: Untuk Pro tier di GPU consumer (RTX 4090, L4). Apakah perlu atau f16 sudah cukup?

4. **Prioritas model**: Apakah 10 model harus siap bersamaan, atau bertahap? (Edge → Core → Pro → Apex → Ultra)

5. **Sharding strategy**: Untuk Ultra training, pakai FSDP (PyTorch style) atau custom ZeRO? FSDP-like lebih mudah diimplement dengan nccl.

---

**Dokumen ini akan diupdate berdasarkan diskusi dan approval.**
