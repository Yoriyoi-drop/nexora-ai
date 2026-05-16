# Roadmap Eksekusi Autograd → GPU → LLM

> **Combined plan:** Visi user + hasil audit codebase + prioritas realistis
> Dimulai: 16 Mei 2026

---

## Fase 0 — Foundation (Combined Assessment)

Berdasarkan audit, gap terbesar adalah **Autograd Engine**. Tanpa ini, proyek tidak bisa scaling.

| Komponen | Status Sekarang | Target Fase-1 |
|----------|----------------|----------------|
| Tensor ops | Ada (ATQS + ndarray) | Autograd-aware |
| Backward trait | Declared only (0 impl) | Fully working |
| Computation graph | GNAC DAG visual only | Autograd DAG |
| Gradient storage | Manual HashMap | `grad: Option<Tensor>` per tensor |
| API | Fragmented | `x.matmul(&w).gelu().mean().backward()` |

---

## PHASE 1 — AUTOGRAD FOUNDATION ✅ (MULAI SEKARANG)
**PRIORITAS ABSOLUT — Semua tergantung ini**

### Minggu 1: Core Engine

#### 1.1 Computation Graph (`autograd/graph.rs`)
- `TensorId`, `GradFnIndex`, `GradFn` struct
- `AutogradTape` (thread-local, manages all grad functions)
- `BACKWARD` closure storage per node

#### 1.2 Tensor Core (`autograd/tensor.rs`)
- `Tensor` struct: `data`, `grad`, `requires_grad`, `grad_fn`
- `randn()`, `zeros()`, `ones()`, `from_slice()`
- `backward()` method entry point

#### 1.3 Operation Trait (`autograd/op.rs`)
- `Op` trait: `forward()`, `backward()`, `name()`

#### 1.4 Operations Registry
| Op | File | Prioritas |
|----|------|-----------|
| Add, Sub, Mul, Div | `ops/math.rs` | Tinggi |
| MatMul | `ops/matmul.rs` | KRITIS |
| Sum, Mean | `ops/reduce.rs` | Tinggi |
| Reshape, Transpose | `ops/shape.rs` | Tinggi |
| ReLU, GELU, Sigmoid, SwiGLU backward | `ops/activation.rs` | Tinggi |

#### 1.5 Backward Engine (`autograd/engine.rs`)
- Topological sort dari computation graph
- Reverse traversal
- Gradient accumulation ke input tensors

#### 1.6 Autograd Public API
```rust
let x = Tensor::randn(&[32, 512], true);
let w = Tensor::randn(&[512, 256], true);
let y = x.matmul(&w).gelu().mean();
y.backward();
// x.grad, w.grad tersedia
```

### Deliverable Fase 1
- [x] Tensor dengan autograd tracking
- [x] 10+ operation dengan backward
- [x] `backward()` traversal berfungsi
- [x] Gradient tersimpan di `tensor.grad`
- [x] Test: linear regression end-to-end

---

## PHASE 2 — MODERN TENSOR ENGINE

### 2.1 Broadcasting Universal
- [ ] NumPy-style broadcasting di semua op
- [ ] Shape inference + validation

### 2.2 Tensor Views
- [ ] `slice()`, `narrow()`, `view()`, `contiguous()`
- [ ] Memory-efficient (no-copy where possible)

### 2.3 Lazy Execution (Post-MVP)
- [ ] Graph optimization (op fusion, dead code elimination)
- [ ] Compiled graph execution

### 2.4 Additional Ops
- [ ] Conv1D/2D, Pooling, Softmax
- [ ] BatchNorm forward + backward
- [ ] LayerNorm forward + backward
- [ ] Dropout forward + backward
- [ ] Embedding forward + backward

### Audit Note:
Tensor system sudah punya `ShapePropagator` di GNAC → bisa di-reuse.
Broadcasting sudah ada di `oracle/backbone.rs` via ndarray.
Yang kurang: autograd-aware broadcasting.

---

## PHASE 3 — GPU RUNTIME

### 3.1 CUDA Backend

**Status audti:** `nvml-wrapper` ada untuk monitoring. `cudarc` di ignore list. **Tidak ada CUDA kernel.**

Roadmap:
- [ ] Matmul CUDA kernel
- [ ] Element-wise ops (add, mul, activation) CUDA
- [ ] Softmax CUDA
- [ ] Tensor transfer CPU ↔ GPU
- [ ] Autograd-aware GPU tensors

### 3.2 WGPU/Vulkan Backend
**Status audit:** Hanya enum target. **Tidak ada implementasi.**

- [ ] WGPU device/context initialization
- [ ] Compute shader pipeline
- [ ] Cross-platform fallback

### 3.3 Kernel Fusion
**Status audit:** Sudah ada `FusedLinearActivation` + `FusedAttentionSoftmax` di STAR-X (AVX2). Tinggal diperluas.

- [ ] Linear + Bias + GELU fused
- [ ] QKV projection fused
- [ ] Attention softmax fused

---

## PHASE 4 — TRAINING MODERNIZATION

### 4.1 AMP — Automatic Mixed Precision
**Status audit:** Ada config `mixed_precision: bool`, `MixedPrecisionScheduler` dengan loss scaling. Tapi **belum aktif runtime.**

- [ ] `autocast` context manager
- [ ] Dynamic loss scaling
- [ ] FP16/BF16 tensor ops

### 4.2 Distributed Training
**Status audit:** Hanya `tensor_parallel_size` di config. **Tidak ada NCCL/MPI.**

- [ ] NCCL all-reduce
- [ ] Data parallel wrapper
- [ ] Tensor parallel
- [ ] Pipeline parallel

### 4.3 Advanced Optimizers
**Status audit:** Sudah punya Adam, SGD, AdaGrad, RMSProp, LAMB.
- [ ] Integrasi dengan autograd
- [ ] Optimizer `zero_grad()`, `step()` API
- [ ] Parameter groups

---

## PHASE 5 — LLM OPTIMIZATION

### 5.1 Speculative Decoding
**Status audit:** **Tidak ada.** Baru.
- [ ] Draft model integration
- [ ] Assisted generation

### 5.2 Flash Attention
**Status audit:** Ada `NodeType::FlashAttention` di GNAC. **Tidak ada implementasi.**
- [ ] Tiling algorithm
- [ ] CUDA implementation
- [ ] Backward pass

### 5.3 Paged KV Cache
**Status audit:** KV cache sudah ada (3 implementasi). Tapi belum paged.
- [ ] Page table management
- [ ] Block-level memory allocation
- [ ] Long context support

### 5.4 Ring Attention / Long Context
- [ ] Ring attention untuk >1M tokens
- [ ] Context compression (ATQS bisa dipakai)

---

## PHASE 6 — DEBUGGING & TOOLING

### 6.1 Gradient Visualizer
- [ ] Histogram gradient distribution
- [ ] Layer-wise gradient flow
- [ ] NaN/inf detection overlay

### 6.2 Attention Viewer
- [ ] Attention weight heatmap
- [ ] Dead head detection
- [ ] Collapse detection
- [ ] Routing pattern analysis (untuk MoE)

### 6.3 Training Profiler
- [ ] Kernel-level timing
- [ ] Memory bottleneck detection
- [ ] FLOPs utilization
- [ ] Timeline visualization

**Status audit:** ATQS punya `LayerAnalyzer`, `SensitivityMapper`, `EntanglementProfiler` → bisa di-extend jadi visualizer.

---

## Prioritas Teknis

```
NOW ─────────────────────────────────────────── LATER
  │                                                │
  │  Fase 1: Autograd Foundation                   │
  │    ├─ Core graph + tensor        ████████░░░░  │
  │    ├─ Ops (math/matmul/reduce)   ████████░░░░  │
  │    ├─ Activations backward       ████████░░░░  │
  │    └─ Backward engine            ████████░░░░  │
  │                                                │
  │  Fase 2: Tensor Engine                        │
  │    ├─ Broadcasting               ██░░░░░░░░░░  │
  │    ├─ Views                      ██░░░░░░░░░░  │
  │    └─ Lazy execution             ░░░░░░░░░░░░  │
  │                                                │
  │  Fase 3: GPU Runtime                          │
  │    ├─ CUDA kernels               ░░░░░░░░░░░░  │
  │    ├─ WGPU backend               ░░░░░░░░░░░░  │
  │    └─ Kernel fusion              ██████░░░░░░  │  ← reuse STAR-X
  │                                                │
  │  Fase 4-6: Training/LLM/Debug    ░░░░░░░░░░░░  │  ← post-MVP
  └──────────────────────────────────────────────────┘

LEGEND: ██ = existing, ░░ = needs work
```

---

## Catatan Teknis dari Audit

### Yang Bisa Langsung Di-reuse
| Komponen | Lokasi | Untuk |
|----------|--------|-------|
| ndarray `broadcast()` | foundation/oracle/backbone.rs | Broadcasting |
| `ShapePropagator` | gnac/smart_tensor/propagation.rs | Shape inference |
| `FusedLinearActivation` | star_x/fused_ops.rs | Kernel fusion baseline |
| `GradientFailureLens` | gnac/lensing/gradient_lens.rs | Gradient debugging |
| `AnomalyDetector` | gnac/intervention/detector.rs | NaN/exploding detection |
| LayerNorm, RMSNorm | hldva_t + transformer | Layer norm ops |

### Yang Harus Dibuat dari Nol
| Komponen | Alasan |
|----------|--------|
| **Autograd tape** | Tidak ada sama sekali |
| **Backward impl untuk setiap op** | 0 implementasi `impl Backward for ...` |
| CUDA kernels | Hanya monitoring GPU |
| Speculative decoding | Tidak ada |
| Visualizer tools | Tidak ada |

---

## Milestone

| Milestone | Target | Isi |
|-----------|--------|-----|
| **M1** | 1-2 hari | Autograd core + 5 ops + backward engine |
| **M2** | 3-5 hari | 10+ ops + activation backward + broadcasting |
| **M3** | Minggu 2 | GPU matmul + element-wise CUDA |
| **M4** | Minggu 3-4 | AMP + distributed + advanced optimizers |
| **M5** | Bulan 2 | Flash attention + speculative decoding |
| **M6** | Bulan 3 | Tooling + visualizer + profiler |

---

**Mulai: FASE 1 — AUTOGRAD FOUNDATION**
