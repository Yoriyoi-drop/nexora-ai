# Audit GPU — Semua Crate di Nexora Workspace

**Tanggal:** 7 Juni 2026
**Scope:** 51 crate (workspace member)
**Engine GPU utama:** `nexora-autograd` (wgpu WebGPU + CUDA via cudarc)

---

## Ringkasan

| Status | Jumlah |
|--------|--------|
| ✅ Punya GPU (feature + code) | 25 crate |
| ⚠️ Feature-only (ada feature, 0 GPU code) | 2 crate |
| 🔴 CPU-only murni | 19 crate |
| 🔶 Transitive-only GPU (via dep) | 11 crate |

---

## ✅ Crate dengan GPU — 25 crate

Semua punya `gpu` feature aktif + GPU code di source. GPU engine melalui `nexora-autograd` (wgpu/CUDA).

| # | Crate | Path | GPU Deps Langsung | GPU Refs |
|---|-------|------|-------------------|----------|
| 1 | `nexora-autograd` | `crates/autograd` | wgpu, cudarc, bytemuck | ~1974 |
| 2 | `nexora-autograd-gpu` | `crates/autograd-gpu` | wgpu, cudarc | ~0 (struktural) |
| 3 | `nexora-deeplearning` | `crates/deeplearning` | via autograd | ~0 (re-export) |
| 4 | `nexora-star-x` | `crates/star-x` | via autograd | ~116 |
| 5 | `nexora-gnac` | `crates/gnac` | via autograd | ~48 |
| 6 | `nexora-echo-net` | `crates/echo-net` | wgpu (direct) | ~100 |
| 7 | `nexora-transformer` | `crates/transformer` | wgpu (direct), half | ~981 |
| 8 | `nexora-has-moe-ffn` | `crates/has-moe-ffn` | via autograd (gpu+cuda) | ~176 |
| 9 | `nexora-quantization` | `crates/quantization` | via autograd | ~9 |
| 10 | `nexora-foundation` | `crates/foundation` | via deeplearning, transformer | ~46 |
| 11 | `nexora-intelligence` | `crates/intelligence` | via autograd | ~5 |
| 12 | `nexora-inference` | `crates/inference` | via autograd, transformer, nvml-wrapper | ~464 |
| 13 | `nexora-training` | `crates/training` | via autograd, transformer | ~132 |
| 14 | `nexora-shared` | `crates/shared` | via erp, deeplearning, gnac | ~28 |
| 15 | `nexora-erp` | `crates/erp` | via autograd (gpu+cuda) | ~161 |
| 16 | `nexora-atqs` | `crates/atqs` | via autograd | ~34 |
| 17 | `nexora-hldva-t` | `crates/hldva-t` | via autograd | ~222 |
| 18 | `nexora-vogp` | `crates/vogp` | via autograd | ~17 |
| 19 | `nexora-multimodal` | `crates/multimodal` | via autograd | ~105 |
| 20 | `nexora-oracle` | `crates/oracle` | via autograd | ~89 |
| 21 | `nexora-alignment` | `crates/alignment` | via autograd | ~3 |
| 22 | `nexora-datastream` | `crates/datastream` | via autograd | ~32 |
| 23 | `nexora-isolation` | `crates/isolation` | via autograd | ~11 |
| 24 | `nexora-ai` | `apps/nexora-ai` | via deeplearning, transformer, inference | ~159 |
| 25 | `nexora-monitoring` | `crates/monitoring` | none (GPU monitoring via nvml) | ~68 |

---

## ⚠️ Feature-only GPU — 2 crate

Punya `gpu` feature di `Cargo.toml` tapi **0 GPU references** di source code. Feature cuma pass-through ke dependency.

| # | Crate | Path | Masalah |
|---|-------|------|---------|
| 1 | `nexora-core` | `crates/core` | `gpu` feature enable `nexora-autograd` tapi tidak ada GPU code sendiri |
| 2 | `nexora-agent` | `crates/agent` | Sama — feature hanya pass-through |

> **Rekomendasi:** Tidak perlu ditambah GPU code. Feature pass-through sudah benar untuk crate leaf yang murni orchestrator.

---

## 🔴 CPU-only murni — 19 crate

Tidak punya `gpu`/`cuda` feature, tidak punya GPU dependencies, **0 GPU references** di seluruh source code.

### 1. Layer Infrastruktur & Umum

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 1 | `nexora-infrastructure` | `crates/infrastructure` | Re-export hub |
| 2 | `nexora-common` | `crates/infrastructure/common` | Shared types, config |
| 3 | `nexora-utils` | `crates/infrastructure/utils` | Utilities |
| 4 | `nexora-foundation-types` | `crates/foundation-types` | Type definitions |

### 2. Data & Storage

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 5 | `nexora-database` | `crates/database` | PostgreSQL, SQLite, MySQL |
| 6 | `nexora-tokenizer` | `crates/tokenizer` | Tokenization |

### 3. API & Integrasi

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 7 | `nexora-api` | `crates/api` | REST API server |
| 8 | `nexora-blaa` | `crates/blaa` | External AI API bridge |

### 4. Training & Autograd

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 9 | `nexora-autograd-core` | `crates/autograd-core` | Intentionally CPU-only |
| 10 | `nexora-autograd-training` | `crates/autograd-training` | Training loop helper |

### 5. Scheduler & Runtime

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 11 | `nexora-runtime` | `crates/runtime` | **Scheduler, batching, cluster, gossip, distributed** |

### 6. Reasoning & Memory

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 12 | `nexora-reasoning` | `crates/reasoning` | **SACA 6-phase reasoning pipeline** |
| 13 | `nexora-memory` | `crates/memory` | Memory management |

### 7. Validasi & Evaluasi

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 14 | `nexora-validation` | `crates/validation` | Validation |
| 15 | `nexora-benchmark` | `crates/benchmark` | Benchmarking |
| 16 | `nexora-evaluation` | `crates/evaluation` | Evaluation |

### 8. Lain-lain

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 17 | `nexora-cognition` | `crates/cognition` | Cognition engine |
| 18 | `nexora-hallucination` | `crates/hallucination` | Hallucination detection |

### 9. Dashboard

| # | Crate | Path | Notes |
|---|-------|------|-------|
| 19 | `nexora-dashboard` | `apps/dashboard` | TUI Dashboard (ratatui) |

---

## 🔶 Transitive-only GPU — 11 crate

Tidak punya `gpu` feature sendiri, tapi punya GPU references di source code karena dependency membawa GPU. **Tidak perlu ditambahkan GPU feature** — GPU sudah diakses via dep graph.

| # | Crate | Path | GPU Refs | GPU via Dep |
|---|-------|------|----------|-------------|
| 1 | `nexora-model-core` | `crates/model-core` | ~3 | transformer, shared |
| 2 | `nexora-model-omnis` | `crates/model-omnis` | ~27 | transformer, has-moe-ffn, erp, vogp |
| 3 | `nexora-model-vortex` | `crates/model-vortex` | ~25 | transformer, oracle, has-moe-ffn |
| 4 | `nexora-model-aether` | `crates/model-aether` | ~23 | transformer, multimodal, erp |
| 5 | `nexora-model-axiom` | `crates/model-axiom` | ~25 | transformer, reasoning, oracle |
| 6 | `nexora-model-cipher` | `crates/model-cipher` | ~25 | transformer, oracle, erp |
| 7 | `nexora-model-genesis` | `crates/model-genesis` | ~27 | transformer, reasoning |
| 8 | `nexora-model-kronos` | `crates/model-kronos` | ~25 | transformer, reasoning |
| 9 | `nexora-model-nexum` | `crates/model-nexum` | ~23 | transformer, oracle, reasoning, alignment |
| 10 | `nexora-model-spectra` | `crates/model-spectra` | ~22 | transformer, multimodal |
| 11 | `nexora-model-swift` | `crates/model-swift` | ~31 | transformer, has-moe-ffn, erp |

---

## Prioritas GPU-ification

### 🏆 Prioritas Tinggi (kritis untuk performa, sering dipanggil)

| Crate | Alasan |
|-------|--------|
| **`nexora-runtime`** | Scheduler + batching + cluster/gossip — otak routing request. Setiap inference request melewati runtime. GPU-ify batching engine & cluster load balancing bisa kurangi latensi. |
| **`nexora-reasoning`** | SACA 6-phase reasoning pipeline — dipanggil tiap kali Axiom, Genesis, Kronos, Nexum delegasi. Tiap phase bisa paralel via GPU. |
| **`nexora-api`** | API server — bisa GPU-accelerate request serialization/deserialization? Risiko: bottleneck I/O, bukan compute. Mungkin prioritas lebih rendah. |

### 📊 Prioritas Sedang

| Crate | Alasan |
|-------|--------|
| **`nexora-memory`** | Paged cache / prefix DAG — data movement intensif. GPU bisa akselerasi eviction policy atau similarity search. |
| **`nexora-validation`** | Validation pipeline — bisa paralel di GPU. |

### 🟢 Prioritas Rendah (CPU-only sudah sesuai)

| Crate | Alasan |
|-------|--------|
| `autograd-core` | Intentionally CPU-only — core trait definitions |
| `autograd-training` | Training loop helper — overhead GPU-ify > benefit |
| `tokenizer` | Tokenization — CPU-bound, latensi rendah |
| `database` | Database — I/O bound |
| `infrastructure` / `common` / `utils` | Pure utilities |
| `blaa` | HTTP bridge — I/O bound |
| `cognition` / `hallucination` | Lightweight logic |
| `benchmark` / `evaluation` | Tooling — tidak di hot path |
| `foundation-types` | Tipe data — zero logic |
| `dashboard` | TUI — CPU-only sudah tepat |

---

## Arsitektur GPU Saat Ini

```
nexora-autograd (GPU engine pusat)
├── wgpu (WebGPU compute shaders)
│   ├── Matmul, Add, Sub, Mul, Div
│   ├── Neg, Exp, Sqrt, ReLU, GELU, SiLU, Sigmoid, Ln, Tanh
│   ├── Powf, Softmax, Transpose
│   ├── Fused Attention (FlashAttention-style)
│   └── Broadcast add, gelu_inplace
├── CUDA (via cudarc + NVRTC JIT)
│   ├── cuBLAS matmul
│   ├── FlashAttention kernel (tiled online-softmax)
│   ├── Softmax, GELU in-place
│   ├── Transpose 2D
│   └── Broadcast add

Feature propagation:
  crates dengan `gpu` feature → enable nexora-autograd/gpu
  crates dengan `cuda` feature → enable nexora-autograd/cuda

Fallback chain (MoE Router/Expert):
  1. CUDA (forward_cuda)
  2. wgpu (forward_gpu)
  3. CPU (naive loop)
```

---

## Dependency Graph GPU

```
                  ┌──────────────────┐
                  │ nexora-autograd  │
                  │ (wgpu + CUDA)    │
                  └────────┬─────────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                     │
   ┌──────▼──────┐  ┌─────▼──────┐    ┌────────▼────────┐
   │ deeplearning │  │transformer │    │ has-moe-ffn     │
   │ (re-export)  │  │ (wgpu+gpu) │    │ (gpu + cuda)    │
   └──────┬──────┘  └─────┬──────┘    └────────┬────────┘
          │               │                     │
   ┌──────▼──────┐  ┌─────▼──────┐             │
   │ foundation  │  │ inference  │             │
   │ star-x      │  │ training   │             │
   │ gnac        │  │            │             │
   │ echo-net    │  └────────────┘             │
   │ quantization │                            │
   │ atqs        │                             │
   │ hldva-t     │                             │
   │ vogp        │                             │
   │ erp         │                             │
   │ multimodal  │                             │
   │ oracle      │                             │
   │ alignment   │                             │
   │ isolation   │                             │
   └─────────────┘                             │
          │                                    │
   ┌──────┴──────────────┐      ┌──────────────┘
   │ shared              │      │
   │ model-* (11 crates) │◄─────┘
   └─────────────────────┘
```

---

## Command Audit

```sh
# Cari semua Cargo.toml dengan feature GPU
rg -l 'gpu' --include '*.toml' | grep -v node_modules | sort

# Cari crate tanpa GPU feature
for f in $(find crates apps -name Cargo.toml); do
  if ! grep -q 'gpu' "$f" 2>/dev/null; then
    name=$(grep '^name' "$f" | head -1 | sed 's/.*"\(.*\)"/\1/')
    path=$(dirname "$f")
    echo "[NO GPU] $name ($path)"
  fi
done

# Cari GPU references per crate
for f in $(find crates apps -name Cargo.toml); do
  name=$(grep '^name' "$f" | head -1 | sed 's/.*"\(.*\)"/\1/')
  path=$(dirname "$f")
  count=$(rg -c 'gpu|wgpu|cuda|cudarc|nvml' "$path/src" 2>/dev/null | wc -l)
  echo "$count refs - $name"
done | sort -rn
```
