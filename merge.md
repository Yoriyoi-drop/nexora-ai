# 📦 Nexora AI — Merge Plan: 56 → 15 Crates

> **Tujuan:** Mereduksi workspace members dari **56 crate** menjadi **15 crate** dengan mengelompokkan crate berdasarkan kohesi fungsional, dependency layering, dan domain arsitektur.

---

## 📊 Status Saat Ini

Workspace memiliki **56 anggota** (`Cargo.toml` root):

| Kategori | Jumlah |
|----------|--------|
| Library crates (crates/) | 52 |
| Binary crates (apps/nexora-ai) | 1 |
| Binary crates (apps/dashboard) | 1 |
| Excluded (Telemetry) | 1 |
| **Total** | **54 aktif + 1 excluded** |

---

## 🎯 Target Akhir: 15 Workspace Members

```
┌─────────────────────────────────────────────────────────────┐
│  Target 15 Crate Architecture                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  15. nexora-ai         (Main App)                    │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │ depends on: 4,5,6,7,8,9,11,12,13    │
│    ┌────────────────┼──────────────────┬────────────────┐   │
│    ▼                ▼                  ▼                ▼   │
│ ┌──────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
│ │  4   │  │  5  Models   │  │  6 Inference │  │7 Runtime│  │
│ │Fndtn │  │  (12-in-1)   │  │  (1-in-1)    │  │(2-in-1)│  │
│ └──┬───┘  └──────┬───────┘  └──────┬───────┘  └───┬────┘  │
│    │             │                 │              │        │
│    └──────┬──────┘                 │              │        │
│           ▼                        ▼              ▼        │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ 3 Transformer  │  │ 12 Agent       │  │ 11 Alignment │ │
│  │ (3-in-1)       │  │ (2-in-1)       │  │ (3-in-1)     │ │
│  └───────┬────────┘  └───────┬────────┘  └──────┬───────┘ │
│          │                   │                   │         │
│          ▼                   ▼                   ▼         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ 1 Core         │  │ 2 DeepLearning │  │ 8 Memory     │ │
│  │ (9-in-1)       │  │ (6-in-1)       │  │ (1-in-1)     │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│                                                             │
│  ┌────────────┐ ┌──────────────┐ ┌──────────┐ ┌─────────┐  │
│  │ 9 Datastrm │ │10 Cognition  │ │13 BLAA   │ │14 Monit │  │
│  │ (2-in-1)   │ │(3-in-1)     │ │(2-in-1)  │ │(1-in-1) │  │
│  └────────────┘ └──────────────┘ └──────────┘ └─────────┘  │
│                                                             │
│  📊 Dashboard (binary, luar 15 count — app pelengkap)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Detail Merge per Grup

### Grup 1: `nexora-core` — Core Infrastructure
**9 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/core` | `nexora-core` | Types, error, async executor, coordination, controller |
| 2 | `crates/shared` | `nexora-shared` | Shared model types, agent types, foundation components |
| 3 | `crates/eventbus` | `nexora-eventbus` | Event-driven pub/sub backbone |
| 4 | `crates/cost-optimizer` | `nexora-cost-optimizer` | Cascade routing cost optimizer |
| 5 | `crates/benchmark` | `nexora-benchmark` | Benchmark types (MetricSample, BenchmarkReport) |
| 6 | `crates/validation` | `nexora-validation` | Environment, security, migration validation |
| 7 | `crates/infrastructure` | `nexora-infrastructure` | Re-export hub untuk common + utils |
| 8 | `crates/infrastructure/common` | `nexora-common` | Common types, config, retry, logging, error |
| 9 | `crates/infrastructure/utils` | `nexora-utils` | Crypto, hash, text processing, file utils, SIMD |

**Dependensi internal:** `nexora-common` + `nexora-utils` → di-re-export oleh `nexora-infrastructure`
**Merge strategy:** Buat modul baru di `crates/core/src/`:
- `crates/core/src/types/` (dari shared)
- `crates/core/src/eventbus/` (dari eventbus)
- `crates/core/src/cost_optimizer/` (dari cost-optimizer)
- `crates/core/src/validation/` (dari validation)
- `crates/core/src/common/` (dari infrastructure/common)
- `crates/core/src/utils/` (dari infrastructure/utils)
- `crates/core/src/benchmark/` (dari benchmark)

**Risiko:** `nexora-shared` memiliki dependensi ke Group 2 (deeplearning) dan Group 4 (foundation). Perlu direfactor agar shared tidak bergantung ke luar group.

---

### Grup 2: `nexora-deeplearning` — Tensor & GPU Compute
**6 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/deeplearning` | `nexora-deeplearning` | Re-export hub (autograd, star-x, gnac, echo-net) |
| 2 | `crates/autograd` | `nexora-autograd` | CPU tensor ops, GPU (wgpu/CUDA), training pipeline |
| 3 | `crates/star-x` | `nexora-star-x` | STar-X tensor framework |
| 4 | `crates/gnac` | `nexora-gnac` | Gradient Navigation & Analysis Canvas |
| 5 | `crates/echo-net` | `nexora-echo-net` | Echo State Networks |
| 6 | `crates/quantization` | `nexora-quantization` | INT8/INT4 weight quantization helpers |

**Dependensi internal:** Semua tergantung `nexora-autograd`. `nexora-deeplearning` adalah re-export hub.
**Merge strategy:** `nexora-deeplearning` menjadi parent crate; `autograd.rs`, `star_x.rs`, `gnac.rs`, `echo_net.rs`, `quantization.rs` sebagai modul di `crates/deeplearning/src/`.

---

### Grup 3: `nexora-transformer` — Model Architecture
**3 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/transformer` | `nexora-transformer` | CausalLM, GQA, RoPE, SwiGLU, MTP, backbone registry |
| 2 | `crates/has-moe-ffn` | `nexora-has-moe-ffn` | MoE FFN (8 experts, top-2 gating, load balancing) |
| 3 | `crates/oracle` | `nexora-oracle` | Oracle backbone (12-layer MoE, code verifiers, trainer) |

**Dependensi internal:** `has-moe-ffn` + `oracle` → `transformer`. Oracle juga bergantung `has-moe-ffn`.
**Merge strategy:** Modul baru di `crates/transformer/src/`:
- `moe/` (dari has-moe-ffn)
- `oracle/` (dari oracle)
- Semua dependency `nexora-autograd` + `nexora-quantization` → arahkan ke Group 2.

**Keuntungan:** Oracle dan MoE sangat tightly coupled dengan transformer backbone.
**Risiko:** Oracle punya dependensi `syn`, `quote`, `proc-macro2` — AST parsing untuk code verification. Perlu dipastikan tidak bentrok.

---

### Grup 4: `nexora-foundation` — Model Framework & Training
**7 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/foundation` | `nexora-foundation` | Hub terbesar — model impl, ATQS, SACA, CAFFEINE, ORACLE, SPARO, VOGP, MoE |
| 2 | `crates/atqs` | `nexora-atqs` | Advanced Tensor Quantization System + calibration |
| 3 | `crates/erp` | `nexora-erp` | Entropic Resonance Pruning |
| 4 | `crates/hldva-t` | `nexora-hldva-t` | Hierarchical Latent Diffusion (VAED, DiT, CLIP) |
| 5 | `crates/vogp` | `nexora-vogp` | Variational Outlier-Guided Pruning |
| 6 | `crates/training` | `nexora-training` | Training pipeline (gradient accumulation, LR schedule) |
| 7 | `crates/evaluation` | `nexora-evaluation` | Evaluation metrics (perplexity, accuracy) |

**Dependensi internal:** foundation sudah menjadi hub. ATQS/VOGP/HLDVA-T/ERP sudah digunakan oleh foundation.
**Merge strategy:** Semua langsung masuk sebagai modul di `crates/foundation/src/`:
- `atqs/` (sudah ada di foundation, tapi re-export dari crate terpisah)
- `erp/` (sudah ada)
- `hldva_t/` (sudah ada)
- `vogp/` (sudah ada)
- `training/` (sudah ada)
- `evaluation/` → evaluasi metrics

**Catatan:** Sebenarnya foundation SUDAH me-re-export semua crate ini via `crates/foundation/src/{atqs, erp, etc}/mod.rs`. Jadi merge ini tinggal "buka bungkus" — pindahkan source code langsung.

---

### Grup 5: `nexora-models` — All NXR Model Definitions
**12 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/models` | `nexora-models` | Re-export hub untuk 10 model NXR |
| 2 | `crates/model-core` | `nexora-model-core` | Foundation model impl, classifier utilities, delegation base |
| 3 | `crates/model-omnis` | `nexora-model-omnis` | NXR Omnis (Ultra — expert routing, harmony weaver) |
| 4 | `crates/model-aether` | `nexora-model-aether` | NXR Aether (Apex — emotional framing, empathy) |
| 5 | `crates/model-axiom` | `nexora-model-axiom` | NXR Axiom (Ultra — structured logic, reasoning) |
| 6 | `crates/model-cipher` | `nexora-model-cipher` | NXR Cipher (Pro — security, encryption) |
| 7 | `crates/model-genesis` | `nexora-model-genesis` | NXR Genesis (Ultra — iterative refinement, creation) |
| 8 | `crates/model-kronos` | `nexora-model-kronos` | NXR Kronos (Core — temporal context, time analysis) |
| 9 | `crates/model-nexum` | `nexora-model-nexum` | NXR Nexum (Apex — task decomposition, orchestration) |
| 10 | `crates/model-spectra` | `nexora-model-spectra` | NXR Spectra (Pro — creative, spectral analysis) |
| 11 | `crates/model-swift` | `nexora-model-swift` | NXR Swift (Edge — latency-aware, fast inference) |
| 12 | `crates/model-vortex` | `nexora-model-vortex` | NXR Vortex (Apex — code review, analysis) |

**Dependensi internal:** Semua model crates memiliki pola identik:
- `agents/` → agent logic
- `classifier.rs` → MLP classifier
- `delegation.rs` → delegation ke foundation backbone
- `capabilities.rs` + `identity.rs` + `config.rs`

Semua tergantung `nexora-model-core` + `nexora-transformer` + `nexora-shared`.

**Merge strategy:** Satu crate besar di `crates/models/src/`:
```
crates/models/src/
├── lib.rs              # Re-export semua model
├── model_core/         # Dari crates/model-core
├── omnis/              # Dari crates/model-omnis
├── aether/
├── axiom/
├── cipher/
├── genesis/
├── kronos/
├── nexum/
├── spectra/
├── swift/
├── vortex/
└── specialist.rs       # Dari crates/models/src/specialist.rs
```

**Keuntungan:** Menghilangkan 12 crate dengan pola duplikasi tinggi. Semua model share backbone yang sama — tidak perlu separation.
**Risiko:** Compile time besar. Tapi foundation sudah besar, tambahan 12 model crate tidak signifikan.

---

### Grup 6: `nexora-inference` — Inference Engine
**1 crate — tetap standalone**

| # | Crate Asal | Fungsi |
|---|-----------|--------|
| 1 | `crates/inference` | KV cache, sampling, beam search, speculative decoding, distributed routing, paged cache, continuous batching, prefix sharing |

**Alasan tetap standalone:**
- Crate terbesar kedua setelah foundation
- Dependency berat (sysinfo, procfs, nvml-wrapper, zstd)
- Independent hot path — butuh compile isolation
- Digunakan langsung oleh nexora-ai binary

---

### Grup 7: `nexora-runtime` — Runtime & Scheduling
**2 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/runtime` | `nexora-runtime` | Scheduler, batching, cluster/gossip, distributed scheduler, GPU runtime |
| 2 | `crates/scheduler-v2` | `nexora-scheduler-v2` | DAG-based task scheduler, work-stealing, GPU/NUMA-aware, priority queue |

**Merge strategy:** `scheduler-v2` masuk sebagai `dag_scheduler/` module di `crates/runtime/src/`.

---

### Grup 8: `nexora-memory` — Memory System
**1 crate — tetap standalone**

| # | Crate Asal | Fungsi |
|---|-----------|--------|
| 1 | `crates/memory` | 4-layer memory (short/session/long/knowledge), hybrid cache (7-layer), memory pools, zero-copy |

**Alasan tetap standalone:**
- Ukuran besar (7+ file hybrid cache, 7 file pool, 7 file zero-copy)
- Dependency khusus (memmap2, bytes, dashmap)
- Digunakan oleh agent, inference, foundation

---

### Grup 9: `nexora-datastream` — Data Pipeline & Database
**2 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/datastream` | `nexora-datastream` | DAG-based streaming data pipeline, 15+ filter types, dataset loading, HuggingFace integration |
| 2 | `crates/database` | `nexora-database` | DB abstraction (Postgres, SQLite, MySQL, SQLx), connection pool |

**Merge strategy:** `database` masuk sebagai `database/` module di `crates/datastream/src/`.

---

### Grup 10: `nexora-cognition` — Reasoning & Multimodal
**3 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/cognition` | `nexora-cognition` | Planning, reflection, context, reasoning — ORCHESTRATOR |
| 2 | `crates/reasoning` | `nexora-reasoning` | SACA — 6-phase reasoning (CoT, decompose, sampling, execute-fail-fix, rerank) |
| 3 | `crates/multimodal` | `nexora-multimodal` | Caffeine — 5 encoders (image/audio/video/text/regional), Q-Former, action head |

**Merge strategy:** Satu crate `crates/cognition/`:
- `saca/` (dari reasoning)
- `caffeine/` (dari multimodal)

**Keuntungan:** Reasoning sudah tergantung multimodal. Keduanya tightly coupled.

---

### Grup 11: `nexora-alignment` — Safety & Alignment
**3 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/alignment` | `nexora-alignment` | SPARO (RLAIF, DPO, IPO, KTO, SPIN, RLVF) |
| 2 | `crates/isolation` | `nexora-isolation` | L0-L6 isolation, firewall, kill switch, multi-cluster, quarantine |
| 3 | `crates/hallucination` | `nexora-hallucination` | Pre/in/post-generation hallucination detection, risk scoring |

**Merge strategy:** Satu crate `crates/alignment/`:
- `isolation/` (dari isolation)
- `hallucination/` (dari hallucination)

---

### Grup 12: `nexora-agent` — Agent System
**2 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/agent` | `nexora-agent` | Agent manager, worker agent, planner agent, routing, memory agent, autoscaler |
| 2 | `crates/intelligence` | `nexora_intelligence` | Model registry, serving, unified API, specialist routing |

**Merge strategy:** `intelligence` masuk sebagai `intelligence/` module di `crates/agent/src/`.

**Keuntungan:** Keduanya sudah tightly coupled. `agent` tergantung `intelligence`. Satu crate untuk seluruh agent ecosystem.

---

### Grup 13: `nexora-blaa` — External API Bridge
**2 crates → 1 crate**

| # | Crate Asal | Package | Fungsi |
|---|-----------|---------|--------|
| 1 | `crates/blaa` | `nexora-blaa` | External LLM API bridge (Black Language Model API) |
| 2 | `crates/api` | `nexora-api` | API layer, middleware, handlers, metrics, TLS, routing |

**Merge strategy:** `api` masuk sebagai `api/` module di `crates/blaa/src/`.

**Alasan:** API crate berat di TLS, axum, tower. BLAA ringan. API crate sudah tergantung infra dan validation.

---

### Grup 14: `nexora-monitoring` — Observability
**1 crate — tetap standalone**

| # | Crate Asal | Fungsi |
|---|-----------|--------|
| 1 | `crates/monitoring` | Observability (33 metrics), health check, profiling, tracing, Prometheus output, Grafana dashboards |

**Alasan tetap standalone:**
- Berat di sysinfo, Prometheus, Grafana
- Dependency khusus (grafana dashboards, prometheus.yml)
- Tidak punya internal nexora dependency → leaf crate

---

### Grup 15: `nexora-ai` — Main Application Binary
**1 crate — tetap standalone**

| # | Crate Asal | Fungsi |
|---|-----------|--------|
| 1 | `apps/nexora-ai` | CLI (train, start, health, info), API server (axum, TLS), handlers, config |

**Alasan tetap standalone:** Binary crate — tidak bisa di-merge dengan library crate.

---

### 📊 Dashboard (Binary Pelengkap)

| Crate | Status |
|-------|--------|
| `apps/dashboard` | Tetap standalone — binary TUI dengan ratatui, crossterm, sysinfo. Tidak perlu di-merge. |

---

## 📈 Ringkasan Merge

### Sebelum (56 members)

| Kategori | Count |
|----------|-------|
| Leaf / standalone crates | 12 |
| Small utility crates (< 500 LOC) | 18 |
| Medium crates (500-2000 LOC) | 14 |
| Large crates (2000-5000 LOC) | 8 |
| Very large crates (>5000 LOC) | 4 |
| **Total** | **56** |

### Sesudah (15 members)

| # | Nama Crate | Tipe | Asal (crates) | Estimasi LOC |
|---|-----------|------|---------------|-------------|
| 1 | **nexora-core** | Library | 9 crates | ~8K-12K |
| 2 | **nexora-deeplearning** | Library | 6 crates | ~15K-20K |
| 3 | **nexora-transformer** | Library | 3 crates | ~10K-15K |
| 4 | **nexora-foundation** | Library | 7 crates | ~25K-35K |
| 5 | **nexora-models** | Library | 12 crates | ~8K-12K |
| 6 | **nexora-inference** | Library | 1 crate | ~8K-10K |
| 7 | **nexora-runtime** | Library | 2 crates | ~6K-8K |
| 8 | **nexora-memory** | Library | 1 crate | ~5K-7K |
| 9 | **nexora-datastream** | Library | 2 crates | ~6K-8K |
| 10 | **nexora-cognition** | Library | 3 crates | ~10K-15K |
| 11 | **nexora-alignment** | Library | 3 crates | ~4K-6K |
| 12 | **nexora-agent** | Library | 2 crates | ~6K-8K |
| 13 | **nexora-blaa** | Library | 2 crates | ~2K-3K |
| 14 | **nexora-monitoring** | Library | 1 crate | ~2K-3K |
| 15 | **nexora-ai** | Binary | 1 crate | ~5K-7K |
| — | **nexora-dashboard** | Binary | 1 crate (pelengkap) | ~2K |
| | **Total** | **15 + 1** | **56 crates** | **~120K-170K** |

### Keuntungan

| Metrik | Before | After | Dampak |
|--------|--------|-------|--------|
| Workspace members | 56 | 15 | **-73%** |
| Cargo.toml files | 56 | 16 | **-71%** |
| Inter-crate dependencies | ~300+ | ~100 | **-67%** |
| Compile time (incremental) | Lambat (56 crate graph) | Cepat (15 crate graph) | **~40% lebih cepat** |
| Build parallelism | Tersebar | Optimal | **~30% lebih baik** |
| Crate boundary overhead | Tinggi | Rendah | API changes lebih mudah |

---

## ⚠️ Risiko & Strategi Mitigasi

### Risiko Tinggi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| **Circular dependency** antara Group 4 (foundation) ↔ Group 5 (models) | Compile error | Foundation → Models (harus satu arah). Models TIDAK tergantung foundation — hanya transformer + core. ✅ Already satisfied. |
| **`nexora-shared`** tergantung deeplearning (Group 2) dan erp/atqs/vogp (Group 4) | Circular jika shared masuk Group 1 | **Solusi:** Pisahkan foundation-specific types dari shared ke Group 4. Shared di Group 1 hanya berisi generic types. |
| **Compile time** Group 5 (models) | 12 crate jadi 1, compile lebih lambat | Feature flags (`model-omnis`, `model-swift`, dll) untuk compile parsial |
| **`nexora-intelligence`** (Group 12) tergantung foundation (Group 4) | Bisa circular | ✅ Aman: Group 4 → Group 12, tapi Group 12 tidak depend ke Group 12. Cuma Group 12 tergantung Group 4. OK. |
| **`nexora-cognition`** (Group 10) tergantung atqs (Group 4) dan has-moe-ffn (Group 3) | Cross-group dependency | ✅ Aman: leaf → branch. Tidak ada circular. |

### Risiko Sedang

| Risiko | Mitigasi |
|--------|----------|
| **Name collision** — 56 crate punya namespace Rust berbeda | Prefix module: `nexora_core::eventbus`, `nexora_core::common`, dll |
| **Feature flags** — 56 crate punya feature propagation chain | Simpan semua feature di root crate, propagate manual |
| **Test isolation** — Test per modul perlu restructuring | `#[cfg(test)] mod tests` di tiap modul |
| **Lint configuration** — Clippy config per crate hilang | Single `clippy.toml` di root — sudah ada |

---

## 🗺️ Roadmap Implementasi

### Phase 1: Low-Hanging Fruit (minggu 1)
1. **Group 1 → 13 (BLAA + API)** — 2 crate, dependency paling ringan
2. **Group 7 (Runtime + Scheduler-v2)** — 2 crate, runtime punya scheduler
3. **Group 10 (Cognition + Reasoning + Multimodal)** — 3 crate, tightly coupled
4. **Group 11 (Alignment + Isolation + Hallucination)** — 3 crate, safety domain

### Phase 2: Medium Complexity (minggu 2)
5. **Group 2 (Deep Learning + Autograd + STar-X + GNAC + EchoNet + Quant)** — 6 crate
6. **Group 4 (Foundation + ATQS + ERP + HLDA-T + VOGP + Training + Eval)** — 7 crate
7. **Group 14 → tetap (Monitoring)** — sudah standalone

### Phase 3: High Complexity (minggu 3-4)
8. **Group 3 (Transformer + MoE FFN + Oracle)** — 3 crate, tightly coupled
9. **Group 5 (Models + Model-Core + 10 model crates)** — 12 crate terbesar
10. **Group 12 (Agent + Intelligence)** — 2 crate
11. **Group 6 → tetap (Inference)** — sudah standalone
12. **Group 8 → tetap (Memory)** — sudah standalone
13. **Group 9 (Datastream + Database)** — 2 crate
14. **Group 1 (Core + Shared + EventBus + CostOpt + Benchmark + Validation + Infra)** — 9 crate paling tricky karena shared punya cross-dependency

### Phase 4: Final (minggu 4-5)
15. Update `apps/nexora-ai/Cargo.toml` — swap dependencies
16. Update `Cargo.toml` root — workspace members = 16 (15 + dashboard)
17. `cargo check --all-targets` — fix compile errors
18. `cargo test` — fix test imports
19. Update `AGENTS.md`, `ARCHITECTURE.md`

---

## 📝 Contoh Transformasi `Cargo.toml`

### Sebelum (root Cargo.toml)
```toml
[workspace]
members = [
    "crates/core",
    "crates/shared",
    "crates/eventbus",
    "crates/cost-optimizer",
    "crates/benchmark",
    "crates/validation",
    "crates/infrastructure",
    "crates/infrastructure/common",
    "crates/infrastructure/utils",
    "crates/autograd",
    "crates/deeplearning",
    "crates/star-x",
    "crates/gnac",
    "crates/echo-net",
    "crates/quantization",
    "crates/transformer",
    "crates/has-moe-ffn",
    "crates/oracle",
    "crates/foundation",
    "crates/atqs",
    "crates/erp",
    "crates/hldva-t",
    "crates/vogp",
    "crates/training",
    "crates/evaluation",
    "crates/models",
    "crates/model-core",
    "crates/model-omnis",
    "crates/model-vortex",
    "crates/model-aether",
    "crates/model-axiom",
    "crates/model-cipher",
    "crates/model-genesis",
    "crates/model-kronos",
    "crates/model-nexum",
    "crates/model-spectra",
    "crates/model-swift",
    "crates/inference",
    "crates/runtime",
    "crates/scheduler-v2",
    "crates/memory",
    "crates/datastream",
    "crates/database",
    "crates/cognition",
    "crates/reasoning",
    "crates/multimodal",
    "crates/alignment",
    "crates/isolation",
    "crates/hallucination",
    "crates/agent",
    "crates/intelligence",
    "crates/blaa",
    "crates/api",
    "crates/monitoring",
    "apps/nexora-ai",
    "apps/dashboard",
]
```

### Sesudah (root Cargo.toml)
```toml
[workspace]
members = [
    "crates/core",           # core + shared + eventbus + cost-optimizer + benchmark + validation + infrastructure + common + utils
    "crates/deeplearning",   # deeplearning + autograd + star-x + gnac + echo-net + quantization
    "crates/transformer",    # transformer + has-moe-ffn + oracle
    "crates/foundation",     # foundation + atqs + erp + hldva-t + vogp + training + evaluation
    "crates/models",         # models + model-core + all 10 model-*
    "crates/inference",      # tetap
    "crates/runtime",        # runtime + scheduler-v2
    "crates/memory",         # tetap
    "crates/datastream",     # datastream + database
    "crates/cognition",      # cognition + reasoning + multimodal
    "crates/alignment",      # alignment + isolation + hallucination
    "crates/agent",          # agent + intelligence
    "crates/blaa",           # blaa + api
    "crates/monitoring",     # tetap
    "apps/nexora-ai",        # tetap
    "apps/dashboard",        # tetap (binary pelengkap)
]
```

---

## 📋 Tabel Mapping: Crate Asal → Grup Tujuan

| Crate Asal | Grup Tujuan | Nama Baru |
|------------|-------------|-----------|
| crates/core | 1 | `nexora_core` |
| crates/shared | 1 | `nexora_core::shared` |
| crates/eventbus | 1 | `nexora_core::eventbus` |
| crates/cost-optimizer | 1 | `nexora_core::cost_optimizer` |
| crates/benchmark | 1 | `nexora_core::benchmark` |
| crates/validation | 1 | `nexora_core::validation` |
| crates/infrastructure | 1 | `nexora_core::infrastructure` |
| crates/infrastructure/common | 1 | `nexora_core::common` |
| crates/infrastructure/utils | 1 | `nexora_core::utils` |
| crates/deeplearning | 2 | `nexora_deeplearning` |
| crates/autograd | 2 | `nexora_deeplearning::autograd` |
| crates/star-x | 2 | `nexora_deeplearning::star_x` |
| crates/gnac | 2 | `nexora_deeplearning::gnac` |
| crates/echo-net | 2 | `nexora_deeplearning::echo_net` |
| crates/quantization | 2 | `nexora_deeplearning::quantization` |
| crates/transformer | 3 | `nexora_transformer` |
| crates/has-moe-ffn | 3 | `nexora_transformer::moe` |
| crates/oracle | 3 | `nexora_transformer::oracle` |
| crates/foundation | 4 | `nexora_foundation` |
| crates/atqs | 4 | `nexora_foundation::atqs` |
| crates/erp | 4 | `nexora_foundation::erp` |
| crates/hldva-t | 4 | `nexora_foundation::hldva_t` |
| crates/vogp | 4 | `nexora_foundation::vogp` |
| crates/training | 4 | `nexora_foundation::training` |
| crates/evaluation | 4 | `nexora_foundation::evaluation` |
| crates/models | 5 | `nexora_models` |
| crates/model-core | 5 | `nexora_models::model_core` |
| crates/model-omnis | 5 | `nexora_models::omnis` |
| crates/model-aether | 5 | `nexora_models::aether` |
| crates/model-axiom | 5 | `nexora_models::axiom` |
| crates/model-cipher | 5 | `nexora_models::cipher` |
| crates/model-genesis | 5 | `nexora_models::genesis` |
| crates/model-kronos | 5 | `nexora_models::kronos` |
| crates/model-nexum | 5 | `nexora_models::nexum` |
| crates/model-spectra | 5 | `nexora_models::spectra` |
| crates/model-swift | 5 | `nexora_models::swift` |
| crates/model-vortex | 5 | `nexora_models::vortex` |
| crates/inference | 6 | `nexora_inference` |
| crates/runtime | 7 | `nexora_runtime` |
| crates/scheduler-v2 | 7 | `nexora_runtime::scheduler_v2` |
| crates/memory | 8 | `nexora_memory` |
| crates/datastream | 9 | `nexora_datastream` |
| crates/database | 9 | `nexora_datastream::database` |
| crates/cognition | 10 | `nexora_cognition` |
| crates/reasoning | 10 | `nexora_cognition::saca` |
| crates/multimodal | 10 | `nexora_cognition::caffeine` |
| crates/alignment | 11 | `nexora_alignment` |
| crates/isolation | 11 | `nexora_alignment::isolation` |
| crates/hallucination | 11 | `nexora_alignment::hallucination` |
| crates/agent | 12 | `nexora_agent` |
| crates/intelligence | 12 | `nexora_agent::intelligence` |
| crates/blaa | 13 | `nexora_blaa` |
| crates/api | 13 | `nexora_blaa::api` |
| crates/monitoring | 14 | `nexora_monitoring` |
| apps/nexora-ai | 15 | `nexora_ai` |
| apps/dashboard | — | `nexora_dashboard` |

---

## 🔧 Estimasi Pekerjaan

| Aktivitas | Jam |
|-----------|-----|
| Restruktur direktori (cp -r, mv) | 2 |
| Merge Cargo.toml dependencies | 4 |
| Update `use` statements (56 → 15 crate paths) | 8 |
| Fix `pub use` re-exports | 4 |
| Fix feature flag propagation | 3 |
| Update workspace `Cargo.toml` | 1 |
| `cargo check --all-targets` cycles | 4 |
| Update docs (AGENTS.md, ARCHITECTURE.md) | 2 |
| **Total** | **~28 jam** |

---

> **Rekomendasi:** Mulai dari Phase 1 (4 grup termudah, 10 crate → 4 crate) untuk memvalidasi approach, lalu lanjut ke grup yang lebih besar.
