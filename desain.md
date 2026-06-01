# Nexora AI — Desain Arsitektur Sistem

> **Versi:** 0.2.0 | **Lisensi:** MIT OR Apache-2.0 | **Platform:** Rust (stable, edition 2021)

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Workspace & Struktur Proyek](#2-workspace--struktur-proyek)
3. [Dependency Graph & Layer Mapping](#3-dependency-graph--layer-mapping)
4. [Core Engine: Autograd & Transformer](#4-core-engine-autograd--transformer)
5. [Sistem Model: 10 NXR Models](#5-sistem-model-10-nxr-models)
6. [Inference Engine](#6-inference-engine)
7. [GPU Backend: CUDA + wgpu](#7-gpu-backend-cuda--wgpu)
8. [Runtime & Distributed Scheduler](#8-runtime--distributed-scheduler)
9. [Agent Ecosystem](#9-agent-ecosystem)
10. [Memory Hierarchy](#10-memory-hierarchy)
11. [Multimodal: Caffeine System](#11-multimodal-caffeine-system)
12. [Reasoning: SACA Framework](#12-reasoning-saca-framework)
13. [MoE: Mixture of Experts](#13-moe-mixture-of-experts)
14. [Oracle Backbone & Code Verifiers](#14-oracle-backbone--code-verifiers)
15. [Alignment: SPARO Framework](#15-alignment-sparo-framework)
16. [Quantization: ATQS](#16-quantization-atqs)
17. [Application Layer: CLI & Server](#17-application-layer-cli--server)
18. [Data Pipeline](#18-data-pipeline)
19. [Infrastructure & Deployment](#19-infrastructure--deployment)
20. [Security & Isolation](#20-security--isolation)
21. [Production Readiness](#21-production-readiness)
22. [Alur Data End-to-End](#22-alur-data-end-to-end)

---

## 1. Ringkasan Eksekutif

Nexora AI adalah **sistem AI multimodal generatif** berbasis Rust yang terdiri dari **42 crate** dalam satu Cargo workspace. Sistem ini mengimplementasikan **10 model NXR** (Nexora Reasoning) dengan spesialisasi berbeda, dari edge (128-dim) hingga ultra (768-dim), didukung oleh:

- **Autograd engine** tape-based reverse-mode dengan dual GPU backend (wgpu/Vulkan + CUDA)
- **CausalLM transformer** dengan GQA, RoPE, SwiGLU, MTP (Multi-Token Prediction)
- **Paged KV cache** dengan block-level prefix sharing (copy-on-write)
- **Distributed scheduler** dengan gossip-based node discovery
- **Multi-agent orchestration** (planner-worker hierarchy)
- **Multimodal pipeline** (Caffeine: image/audio/video/text encoders + Q-Former)
- **Reasoning pipeline** (SACA: 6-phase closed-loop reasoning)
- **Mixture of Experts** (8 experts, top-2 gating, CUDA-accelerated)
- **Oracle backbone** (12-layer MoE + MultiHeadLatentAttention transformer)
- **Alignment** (SPARO: DPO, KTO, IPO, RLAIF, RLVF, SPIN)
- **Quantization** (ATQS: AWQ, Tucker decomposition, adaptive rank)

**Status:** Late prototype / early production (~95%+ production readiness per audit).

---

## 2. Workspace & Struktur Proyek

### 2.1 Root Structure

```
nexora-ai/
├── Cargo.toml                  # Workspace root (42 members)
├── Cargo.lock                  # Ada lokal, di-gitignore
├── AGENTS.md                   # 569-line agent guide
├── ARCHITECTURE.md             # Ringkasan arsitektur
├── AUDIT_PRODUCTION_READINESS.md  # ~2600+ line audit
├── Makefile                    # 50+ targets
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # 9 services
├── nextest.toml                # Test runner config
├── .cargo/config.toml          # lld linker, LTO, panic=abort
├── clippy.toml                 # cognitive-complexity=25, dll
├── deny.toml                   # cargo-deny config
├── rustfmt.toml                # max_width=100, tab_spaces=4
├── rust-toolchain.toml         # stable, x86_64-unknown-linux-gnu
├── apps/
│   ├── nexora-ai/              # Main CLI binary "nexora"
│   ├── dashboard/              # TUI dashboard (ratatui)
│   └── landing-page/           # Vue 3 + Vite
├── crates/                     # 38 library crates
│   ├── core/ tokenizer/ shared/ infrastructure/
│   ├── autograd/ deeplearning/ star-x/ gnac/ echo-net/
│   ├── transformer/ training/ quantization/
│   ├── foundation/ intelligence/ models/
│   ├── inference/ runtime/ memory/ agent/ cognition/
│   ├── has-moe-ffn/ multimodal/ reasoning/ oracle/ alignment/
│   ├── atqs/ hldva-t/ vogp/ erp/
│   ├── api/ database/ datastream/ monitoring/
│   ├── isolation/ validation/ hallucination/ blaa/
│   └── Telemetry/              # Excluded from workspace
├── configs/                    # Runtime config (.toml)
├── training_data/              # .arrow shards
└── data/                       # state_store.db, stream_store/
```

### 2.2 42 Workspace Members

#### Application Binaries (2)
| Path | Package | Binary | Deskripsi |
|------|---------|--------|-----------|
| `apps/nexora-ai` | `nexora-ai` | `nexora`, `generate-training-data` | CLI utama + server |
| `apps/dashboard` | `nexora-dashboard` | `dashboard` | TUI monitoring |

#### Infrastructure Layer (8)
| Path | Package | Deskripsi |
|------|---------|-----------|
| `crates/core` | `nexora-core` | Controller, types, execution, error recovery |
| `crates/infrastructure/common` | `nexora-common` | Shared domain types |
| `crates/infrastructure/utils` | `nexora-utils` | Pure utilities (hashing, crypto) |
| `crates/infrastructure` | `nexora-infrastructure` | Hub re-export common + utils |
| `crates/tokenizer` | `nexora-tokenizer` | BPE + unicode normalization |
| `crates/memory` | `nexora-memory` | Episodic/semantic/working memory |
| `crates/monitoring` | `nexora-monitoring` | Prometheus metrics |
| `crates/api` | `nexora-api` | Axum server + middleware |

#### Deep Learning Layer (5)
| Path | Package | Deskripsi |
|------|---------|-----------|
| `crates/autograd` | `nexora-autograd` | Tape-based reverse-mode, GPU backends |
| `crates/deeplearning` | `nexora-deeplearning` | Hub re-export |
| `crates/star-x` | `nexora-star-x` | STAR-X tensor engine |
| `crates/gnac` | `nexora-gnac` | Graph Neural Attention Controller |
| `crates/echo-net` | `nexora-echo-net` | Echo recurrent network |

#### Model Layer (12)
| Path | Package | Deskripsi |
|------|---------|-----------|
| `crates/foundation` | `nexora-foundation` | **Crate terbesar**. Model NXR series |
| `crates/intelligence` | `nexora_model` | Model registry, serving |
| `crates/models` | `nexora-models` | 10 model delegation impl |
| `crates/transformer` | `nexora-transformer` | CausalLM, GQA, RoPE, SwiGLU, MTP |
| `crates/training` | `nexora-training` | Training pipeline, LoRA, AMP |
| `crates/quantization` | `nexora-quantization` | INT8/INT4 quantization |
| `crates/has-moe-ffn` | `nexora-has-moe-ffn` | MoE FFN (gating + experts + CUDA) |
| `crates/multimodal` | `nexora-multimodal` | Caffeine (5 encoders, Q-Former) |
| `crates/reasoning` | `nexora-reasoning` | SACA (6-phase reasoning) |
| `crates/oracle` | `nexora-oracle` | Oracle backbone + code verifiers |
| `crates/alignment` | `nexora-alignment` | SPARO (6 alignment methods) |
| `crates/atqs` | `nexora-atqs` | AWQ, Tucker, adaptive rank |

#### Runtime Layer (5)
| Path | Package | Deskripsi |
|------|---------|-----------|
| `crates/inference` | `nexora-inference` | KV cache, sampling, speculative decoding |
| `crates/runtime` | `nexora-runtime` | Scheduler, cluster, gossip |
| `crates/agent` | `nexora-agent` | Multi-agent orchestration |
| `crates/cognition` | `nexora-cognition` | Planning, reflection, reasoning |
| `crates/isolation` | `nexora-isolation` | L0-L6 isolation architecture |

#### Specialized (10)
| Path | Package | Deskripsi |
|------|---------|-----------|
| `crates/shared` | `nexora-shared` | Shared model components |
| `crates/hldva-t` | `nexora-hldva-t` | Hierarchical Latent Diffusion |
| `crates/vogp` | `nexora-vogp` | Outlier-Guided Pruning |
| `crates/erp` | `nexora-erp` | Entropic Resonance Pruning |
| `crates/database` | `nexora-database` | PostgreSQL/SQLite/MySQL |
| `crates/datastream` | `nexora-datastream` | DAG data pipeline |
| `crates/validation` | `nexora-validation` | Validation engines |
| `crates/hallucination` | `nexora-hallucination` | Anti-hallucination |
| `crates/blaa` | `nexora-blaa` | Black Language Model API bridge |
| `crates/Telemetry` | (excluded) | Tauri desktop app |

---

## 3. Dependency Graph & Layer Mapping

### 3.1 Layered Architecture

```
  LAYER 6 — Application
  ┌──────────────────────────────────────────────────────────┐
  │  nexora-ai, nexora-dashboard                              │
  │  Depends on: semua layer di bawah                        │
  └──────────────────────────────────────────────────────────┘
            │
  LAYER 5 — Orchestration
  ┌──────────────────────────────────────────────────────────┐
  │  nexora-agent, nexora-cognition, nexora-memory            │
  │  nexora-isolation, nexora-api, nexora-monitoring          │
  │  Depends on: Layer 4 + Layer 3                            │
  └──────────────────────────────────────────────────────────┘
            │
  LAYER 4 — Intelligence & Models
  ┌──────────────────────────────────────────────────────────┐
  │  nexora-foundation (HUB), nexora-intelligence              │
  │  nexora-models (10 delegation crates)                      │
  │  Depends on: Layer 3 + Layer 2                             │
  └──────────────────────────────────────────────────────────┘
            │
  LAYER 3 — Inference & Runtime
  ┌──────────────────────────────────────────────────────────┐
  │  nexora-inference, nexora-runtime                          │
  │  nexora-blaa                                              │
  │  Depends on: Layer 2 + Layer 1                             │
  └──────────────────────────────────────────────────────────┘
            │
  LAYER 2 — Subsystem Spesialisasi
  ┌──────────────────────────────────────────────────────────┐
  │  nexora-has-moe-ffn, nexora-multimodal, nexora-reasoning   │
  │  nexora-oracle, nexora-alignment, nexora-atqs              │
  │  nexora-hldva-t, nexora-vogp, nexora-erp                   │
  │  Depends on: Layer 1 + shared                              │
  └──────────────────────────────────────────────────────────┘
            │
  LAYER 1 — Transformer & Training
  ┌──────────────────────────────────────────────────────────┐
  │  nexora-transformer, nexora-training, nexora-quantization │
  │  nexora-deeplearning (hub untuk star-x, gnac, echo-net)   │
  │  Depends on: Layer 0                                       │
  └──────────────────────────────────────────────────────────┘
            │
  LAYER 0 — Core & Infrastructure
  ┌──────────────────────────────────────────────────────────┐
  │  nexora-core, nexora-tokenizer, nexora-shared              │
  │  nexora-common, nexora-utils, nexora-infrastructure        │
  │  nexora-autograd (GPU engine), nexus star-x/gnac/echo-net │
  │  nexora-database, nexora-datastream, nexora-validation     │
  └──────────────────────────────────────────────────────────┘
            │
  LAYER 0a — Foundation Libraries
  ┌──────────────────────────────────────────────────────────┐
  │  Rust std, tokio, ndarray, serde, axum, reqwest, wgpu    │
  │  cudarc, half, arrow, parquet, sqlx, ratatui, prometheus │
  └──────────────────────────────────────────────────────────┘
```

### 3.2 Dependency Flow Arah

```
nexora-ai (app)
  ├── nexora-agent → nexora-core, nexora-intelligence, nexora-memory
  ├── nexora-inference → nexora-core, nexora-intelligence, nexora-foundation
  │     ├── nexora-runtime → nexora-core
  │     └── nexora-blaa → nexora-core
  ├── nexora-foundation (HUB dengan 18+ deps)
  │     ├── nexora-models (10 delegation crate)
  │     │     ├── nexora-shared → erp, vogp, atqs, moe, deeplearning
  │     │     ├── nexora-has-moe-ffn → nexora-autograd (opt)
  │     │     ├── nexora-multimodal → shared, atqs, moe
  │     │     ├── nexora-reasoning → shared, atqs, multimodal, moe, oracle
  │     │     ├── nexora-oracle → nexora-autograd (opt)
  │     │     ├── nexora-alignment → shared, autograd, oracle (opt)
  │     │     └── nexora-atqs → ndarray, fastrand
  │     ├── nexora-transformer → nexora-autograd
  │     ├── nexora-training → transformer, autograd
  │     └── nexora-quantization → ndarray
  ├── nexora-intelligence → nexora-core, tokenizer, foundation
  ├── nexora-memory → nexora-core, database
  ├── nexora-tokenizer → nexora-core
  ├── nexora-api → nexora-common
  ├── nexora-infrastructure → common, utils
  ├── nexora-datastream → arrow, reqwest
  ├── nexora-deeplearning → star-x, gnac, echo-net, autograd
  └── nexora-dashboard (standalone, no nexora deps)
```

### 3.3 Feature Flags per Crate

| Crate | Features | Gating |
|-------|----------|--------|
| `nexora-autograd` | `default=["std","serde"]`, `gpu`, `cuda` | GPU via wgpu; CUDA via cudarc |
| `nexora-core` | `default=["gpu"]`, `gpu` | GPU ops enable |
| `nexora-foundation` | `default=["gpu","legacy-fastpath"]` | Fast inference path |
| `nexora-has-moe-ffn` | `default=["gpu"]`, `gpu`, `cuda` | CUDA terpisah dari GPU |
| `nexora-models` | `simulated-models`, `hallucination` | Mock models, hallucination guard |
| `nexora-api` | `tls`, `rustls` | TLS via axum-server |
| `nexora-database` | `postgres`, `sqlite`, `sqlx`, `mysql`, `all` | Database backends |
| `nexora-datastream` | `arrow`, `parquet`, `toxicity`, `prompt-injection` | Pipeline features |
| `nexora-alignment` | `oracle` | Enable Oracle dependency |
| `nexora-ai` | `production` (mimalloc), `blas`, `server-*`, `gpu` | App-level features |

---

## 4. Core Engine: Autograd & Transformer

### 4.1 Autograd Engine (`nexora-autograd`)

**Lokasi:** `crates/autograd/src/`

#### Arsitektur Tape-Based Reverse-Mode

```
Tensor (storage enum)
  ├── Storage::Cpu(ArrayD<f32>)     — CPU ndarray
  ├── Storage::Gpu(GpuTensor)       — wgpu GPU buffer
  └── Storage::Cuda(CudaTensor)     — CUDA device buffer

AutogradTape (thread-local)
  ├── BackwardNode { grad_fn, inputs }
  └── backward() → BFS topological sort → chain rule
```

**25+ Ops:**
| Kategori | Ops |
|----------|-----|
| Aritmetika | add, sub, mul, div, neg |
| Unary | exp, log, sqrt, abs, powf |
| Aktivasi | relu, gelu, silu, sigmoid, tanh, leaky_relu |
| Normalisasi | softmax, log_softmax, layer_norm, rms_norm |
| Reduksi | sum, mean, max, min |
| Matriks | matmul, transpose, reshape, gather, scatter |
| Loss | cross_entropy, binary_cross_entropy, mse |
| Training | adam_step, gradient_clip, gradient_allreduce |
| Sampling | top_k_mask, top_p_mask, multinomial_sample |

#### GPU Ops (40+ WGSL Shaders)

| Kategori | Pipeline |
|----------|----------|
| Matmul | matmul_tiled, matmul_f16_tiled, matmul_int8_tiled |
| Normalisasi | softmax, causal_softmax, rms_norm, layer_norm, l2_norm |
| Attention | fused_attention (FlashAttention-style), fused_attention_backward |
| Sampling | temperature_scale, top_k_mask, top_p_mask, multinomial_sample |
| MoE | moe_scatter_add (weighted scatter-add) |
| Training | adam_step, gradient_allreduce, gradient_clip, cross_entropy |
| Mixed Precision | f32_to_f16_packed, f16_packed_to_f32 |
| Transformer | rotary_embedding, repeat_heads (GQA) |

### 4.2 Transformer Architecture (`nexora-transformer`)

**Lokasi:** `crates/transformer/src/`

#### CausalLM — Struktur Utama

```
CausalLM
  ├── config: TransformerConfig
  │     ├── hidden_size: 128-768 (per model tier)
  │     ├── num_heads: 4-12
  │     ├── num_kv_heads: 2-4 (GQA)
  │     ├── num_layers: 2-8
  │     ├── intermediate_size: 512-3072 (SwiGLU)
  │     └── max_seq_len: 512-2048
  ├── token_embedding: [vocab_size × hidden]
  ├── lm_head: [hidden × vocab_size] (weight-tying optional)
  ├── norm: RMSNorm
  ├── blocks: Vec<TransformerBlock>
  │     ├── attention_norm: RMSNorm
  │     ├── ffn_norm: RMSNorm
  │     ├── attention: GQA (Grouped Query Attention)
  │     │     ├── Wq: [hidden × num_heads * head_dim]
  │     │     ├── Wk: [hidden × num_kv_heads * head_dim]
  │     │     ├── Wv: [hidden × num_kv_heads * head_dim]
  │     │     └── Wo: [num_heads * head_dim × hidden]
  │     └── ffn: SwiGLU
  │           ├── W1: [hidden × intermediate] (gate)
  │           ├── W2: [intermediate × hidden] (down)
  │           └── W3: [hidden × intermediate] (up)
  └── weights: QuantizedWeights (opsional)
```

#### Multi-Token Prediction (MTP)

```
MTPHeads
  ├── config: MTPConfig { num_predictions: 2, head_hidden_dim: 256 }
  ├── hidden_norms: Vec<Array1<f32>>   — per-head projection
  ├── head_weights: Vec<Array2<f32>>    — per-head output [vocab × head_hidden]
  └── head_biases: Vec<Array1<f32>>     — per-head bias

Alur MTPInference:
  1. Prefill: forward seluruh prompt
  2. Generation loop:
     a. Forward main token → logits
     b. Sample main token (greedy/top-k)
     c. Untuk tiap depth (1..num_predictions):
        - Draft prediction via MTPHeads
        - Jika draft_id == main_id → accept
        - Jika temperature > 0.5 → verifikasi dengan forward
     d. Stop jika EOS atau max_tokens
```

#### TrainableCausalLM

Wrapper autograd untuk training:
```
TrainableCausalLM
  ├── from_inference(&CausalLM) → semua weight jadi Tensor requires_grad=true
  ├── forward(input_ids: &Tensor) → logits (full autograd graph)
  ├── sync_to_inference(model) → copy gradient-updated weights kembali
  ├── parameters() → Vec<Tensor> (semua weight)
  └── save_checkpoint / load_checkpoint (safetensors)
```

#### Safetensors I/O Format

```
File format:
  [8 bytes: header_length (little-endian u64)]
  [N bytes: header (JSON)]
  [M bytes: data (F32/F16 packed)]

Header:
  { "tensor_name": { "dtype": "F32"|"F16", "shape": [d1, d2, ...],
                     "data_offsets": [start, end] } }
```

---

## 5. Sistem Model: 10 NXR Models

### 5.1 Model Tier & Spesifikasi

| Model | ID | Tier | hidden | heads | kv_heads | layers | intermediate | Specialisasi |
|-------|----|------|--------|-------|----------|--------|-------------|-------------|
| **Omnis** | NXR-01 | Ultra | 768 | 12 | 4 | 8 | 3072 | General-purpose + MoE gating |
| **Aether** | NXR-03 | Apex | 512 | 8 | 4 | 6 | 2048 | Emotional/cognitive |
| **Axiom** | NXR-06 | Ultra | 768 | 12 | 4 | 8 | 3072 | Strategic reasoning |
| **Spectra** | NXR-04 | Pro | 384 | 6 | 3 | 4 | 1536 | Multimodal creative |
| **Cipher** | NXR-07 | Pro | 384 | 6 | 3 | 4 | 1536 | Security/network |
| **Vortex** | NXR-02 | Apex | 512 | 8 | 4 | 6 | 2048 | Code analysis |
| **Kronos** | NXR-09 | Core | 256 | 4 | 2 | 3 | 1024 | Temporal reasoning |
| **Swift** | NXR-08 | Edge | 128 | 4 | 2 | 2 | 512 | Edge inference |
| **Genesis** | NXR-10 | Ultra | 768 | 12 | 4 | 8 | 3072 | Evolutionary learning |
| **Nexum** | NXR-05 | Apex | 512 | 8 | 4 | 6 | 2048 | Agent orchestration |

### 5.2 Model Foundation (`crates/foundation`)

**Crate terbesar** — hub sentral dengan implementasi native + re-export.

#### CausalLmModel — Struktur Utama

```
CausalLmModel
  ├── meta: ModelMeta { id, tier, version }
  ├── capabilities: CapabilityVector
  ├── model: Arc<RwLock<Option<Arc<CausalLM>>>>
  ├── tokenizer: Arc<RwLock<Option<MiniTokenizer>>>
  ├── transformer_config: Arc<RwLock<TransformerConfig>>
  ├── echo_net_config: Option<EchoNetInjectionConfig>
  ├── sedc_enabled: bool
  ├── use_gpu: AtomicBool
  └── use_half_precision: AtomicBool
```

#### Alur Inference (Legacy Fastpath)

```
CausalLmModel::infer(&NxrInput)
  1. MiniTokenizer::encode(text) → [BOS, token_ids..., EOS]
  2. CausalLM::generate_with_gpu(input_ids, max_tokens, temperature, top_k, use_gpu)
     a. Prefill: forward all prompt tokens
     b. Generation loop: sample → forward → repeat
     c. Stop on EOS or max_tokens
  3. MiniTokenizer::decode(output_ids) → String
  4. Return NxrOutput { data: OutputData::Text(decoded), metadata, performance }
```

#### Alur Training

```
CausalLmModel::train_on_data(data, cfg, val_data)
  1. Tokenisasi validation data upfront
  2. Loop step < max_steps:
     a. Tokenisasi 1 epoch (streaming, O(epoch) memory)
     b. Shuffle data per epoch
     c. Batch chunks → train_batch_multi()
     d. Report loss per optimizer step
  3. Checkpoint tiap 1000 step
  4. Evaluasi validasi tiap max_steps/5 step
  5. Early stopping (3× tanpa improvement setelah mid-point)
  6. Final checkpoint + sync weights
  7. Return TrainingReport { steps, final_loss, total_tokens, duration }
```

#### EchoNet APSS Injector

```
EchoNetInjector (LayerInjector)
  ├── AdaptivePhaseSeparationStabilizer
  ├── sliding window of hidden states + phase vectors
  └── after_layer():
      1. Simpan hidden state + phase vector ke buffer
      2. Bangun HolographicWave (amplitude, phase, frequency)
      3. APSS forward() → stabilisasi phase
      4. Modulasi: h' = h * (1 + alpha * tanh(avg_phase_delta))
```

### 5.3 Model Delegation Pattern (`crates/models`)

Setiap model mengikuti pola identik:

```
OnceLock Foundation  ──→  inject_model(Arc<CausalLM>)
       │
       ▼
init_classifier()  ──→  baca token_embedding → Xavier init weights
       │
       ▼
classify(prompt)  ──→  embed_average → MLP → softmax → [(class, prob)]
       │
       ▼
delegate(prompt)  ──→  format system prompt → call_model(foundation, framed)
       │
       ▼
call_model()  ──→  NxrInput → model.infer() → OutputData::Text
```

#### Classifier Arsitektur per Model

| Model | Hidden Layer | Output | Kelas |
|-------|-------------|--------|-------|
| **Omnis** | Langsung ke MoE Router | 7 domains | math, science, code, creative, reasoning, factual, general |
| **Aether** | `embed_dim → 64` GELU | 8 emotions | joy, sadness, anger, fear, surprise, disgust, trust, neutral |
| **Axiom** | `embed_dim → 32` GELU | 6 reasoning | deductive, inductive, abductive, analogical, causal, analytical |
| **Spectra** | `embed_dim → 32` GELU | 6 styles | narrative, poetic, persuasive, technical, dialogue, descriptive |
| **Cipher** | `embed_dim → 32` GELU | 6 threats | injection, xss, auth, crypto, config, network |
| **Vortex** | `embed_dim → 64` GELU | 6 categories + language detection | bugs, security, performance, style, architecture, general |
| **Kronos** | `embed_dim → 32` GELU | 5 modes | urgent, scheduled, historical, realtime, evergreen |
| **Swift** | `embed_dim → 32` GELU | 5 tasks | qa, summarize, translate, generate, analyze |
| **Genesis** | `embed_dim → 32` GELU | 6 quality | clarity, depth, accuracy, structure, conciseness, engagement |
| **Nexum** | `embed_dim → 32` GELU | 4 complexity | simple, moderate, complex, multi_domain |

#### Phase 4 Wiring — Native Subsystems

| Model | Subsystem | Fungsi |
|-------|-----------|--------|
| **Omnis** | `has-moe-ffn::Router` | `Router::forward()` → top-2 domain routing |
| **Aether** | `multimodal::CaffeineProcessor` | `process_multimodal()` → emotion + multimodal fusion |
| **Axiom** | `reasoning::SacaEngine` | `engine.reason(prompt, focus)` → 6-phase reasoning |
| **Spectra** | `multimodal::CaffeineProcessor` | `process_multimodal()` → style + multimodal fusion |
| **Cipher** | `oracle::CodeLinterManager` | `verify_detailed()` → security scan findings |
| **Vortex** | `oracle::CodeLinterManager` | `verify_detailed()` → 4 rule-based verifiers |
| **Kronos** | `reasoning::SacaEngine` | `engine.reason(prompt, context)` → temporal reasoning |
| **Swift** | `has-moe-ffn::Router` | `Router::forward()` → top expert (5→task types) |
| **Genesis** | `reasoning::SacaEngine` | Multi-iteration refinement loop (max 3, threshold 0.6) |
| **Nexum** | `oracle::CodeLinterManager` + `reasoning::SacaEngine` | Task decomposition + quality scoring |

Semua wiring memiliki **graceful fallback** ke prompt-based delegation jika subsystem tidak tersedia.

---

## 6. Inference Engine

### 6.1 Continuous Batching Engine

**Lokasi:** `crates/inference/src/continuous_batching.rs`

```
ContinuousBatchingEngine
  ├── config: ContinuousBatchingConfig
  │     ├── max_batch_size: 32
  │     ├── max_seq_len: 4096
  │     ├── enable_kv_cache: true
  │     ├── use_paged_cache: true
  │     ├── paged_block_size: 16 (auto-suggest)
  │     ├── paged_cache_f16: true
  │     ├── enable_prefix_sharing: true
  │     └── min_shared_prefix_len: 4
  ├── scheduler: RequestScheduler (priority + aging + adaptive batching)
  ├── kv_caches: HashMap<u64, Box<dyn KVCacheProvider>>
  ├── prefix_trie: PrefixTrie (tree dari token → seq_ids)
  ├── shared_paged: Arc<Mutex<PagedKVCache>> (optional)
  └── gpu_page_table: Option<GpuPageTable> (cfg-gated)
```

#### Step() — Satu Iterasi Batch

```
step():
  Phase 0: Prefix Sharing
    ├── try_gpu_prefix_sharing()     — GPU GpuKVCache-based
    └── try_paged_prefix_sharing()   — Block-level paged cache

  Phase 1: Prefill
    ├── Prefill prompt tokens untuk sequence baru
    └── Update KV cache

  Phase 2: Decode
    ├── Forward batch → logits
    ├── Sample next token per sequence
    ├── Append ke KV cache
    └── Check EOS / max_tokens

  Phase 3: Post-process
    ├── Hapus sequence selesai
    ├── Update metrics
    └── Return generated tokens
```

### 6.2 Paged KV Cache

**Lokasi:** `crates/inference/src/paged_cache.rs`

#### Design: Fixed-Size Blocks

```
PagedKVCache
  ├── config: PagedCacheConfig
  │     ├── block_size: 16 (tokens per block)
  │     ├── max_blocks: 65536
  │     ├── num_layers, num_kv_heads, head_dim
  │     └── f16_storage: true (2× memory reduction)
  ├── blocks: Vec<Vec<PhysicalBlock>>    — per-layer physical blocks
  ├── free_lists: Vec<Vec<usize>>        — per-layer free list
  ├── sequences: HashMap<u64, BlockTable> — per-seq logical → physical
  └── gpu_page_table: Option<GpuPageTable>
```

#### Block Sharing & Copy-on-Write

```
Sequence A (prompt="abcdef"): blocks [0:a, 1:b, 2:c, 3:d, 4:e, 5:f]  ref_count=1
Sequence B (prompt="abcxyz"):
  1. PrefixTrie → match at depth 3 ("abc")
  2. share_prefix_in_blocks(A→B, 3) → copy block table entries
     B's block table: [0→phys0, 1→phys1, 2→phys2]
     phys0, phys1, phys2 → ref_count=2
  3. B appends token at pos 3:
     get_or_alloc_block() sees ref_count>1 → deep_copy()
     B gets new independent block, A retains original
```

#### Memory Calculation

```
Memory = max_blocks × block_size × num_kv_heads × head_dim × 2 (K+V) × elem_size × num_layers

Contoh (default):
  = 65536 × 16 × 8 × 128 × 2 × 2 (f16) × 32
  = ~8.5 GB untuk 32 layer
```

### 6.3 PrefixTrie

```
PrefixTrieNode
  ├── children: HashMap<u32, PrefixTrieNode>  — next token → child
  └── seq_ids: Vec<u64>                       — sequences sharing this prefix

Methods:
  insert(seq_id, prompt)            — full walk semua token
  find_shared_prefix(prompt, exclude)  — longest prefix match
  remove(seq_id)                    — cleanup saat sequence selesai
```

### 6.4 Sampling Strategies

| Strategy | Parameter | Implementasi |
|----------|-----------|-------------|
| Greedy | temperature=0 | argmax logits |
| Top-K | top_k=50 | Filter ke K highest prob → renormalize |
| Top-P (nucleus) | top_p=0.9 | Cumulative prob threshold |
| Temperature | temp=0.7 | logits /= temperature before softmax |
| MTP Speculative | — | Draft → verify via MTPHeads |

### 6.5 KVCacheProvider Trait

```rust
pub trait KVCacheProvider: Send + Sync {
    fn append(&mut self, layer: usize, token_pos: usize, k: &[f32], v: &[f32]);
    fn as_cpu_entries(&self) -> &[KVCacheEntry];
    fn as_gpu_entries(&mut self) -> &mut [GpuKVCacheEntry];
    fn sync_gpu_to_paged(&mut self);
    fn clear(&mut self);
}
```

Dual implementation: `GpuKVCacheEntry` (flat GPU buffer) dan `PagedKVCacheProvider` (block-based paged).

---

## 7. GPU Backend: CUDA + wgpu

### 7.1 Auto-Detection & Dual Backend

```
GpuContext::from_adapter()
  ├── #[cfg(feature = "cuda")] CudaRuntime::try_init()
  │     ├── Success → GpuBackend::Cuda, cuda: Some(&'static CudaRuntime)
  │     └── Failure → GpuBackend::Wgpu, cuda: None
  └── #[cfg(not(feature = "cuda"))] → GpuBackend::Wgpu unconditionally
```

### 7.2 GpuContext — Singleton Runtime

```
GpuContext
  ├── device: wgpu::Device
  ├── queue: wgpu::Queue
  ├── caps: GpuCapabilities
  ├── pipelines: HashMap<String, CompiledPipeline>  — 40+ compiled
  ├── shader_cache: HashMap<u64, ShaderModule>
  ├── bind_group_cache_mutex: Mutex<HashMap<...>>
  ├── memory_pool: Mutex<GpuMemoryPool>
  ├── current_encoder: Mutex<Option<CommandEncoder>>
  ├── auto_flush_ops: 256 (dGPU) / 64 (integrated)
  ├── backend: GpuBackend
  ├── cuda: Option<&'static CudaRuntime>
  ├── readback_limiter: ReadbackLimiter (max 16 concurrent)
  ├── profiling_query_set: Option<QuerySet>
  └── disk_cache: Option<PipelineDiskCache>
```

### 7.3 FlashAttention CUDA Kernel

**Algoritma:** FlashAttention-style tiled online softmax

```
Parallelism: 1 block per (batch, head, q_pos)
Tile size: 32 KV positions per iteration
Shared memory: Q(256) + score(32) + exp(32) + scratch(256) = 1344 bytes

6-step per tile:
  1. SCORE: q·k dot product via __shfl_xor_sync warp reduction
  2. TILE MAX: tree reduction across threads
  3. EXP & SUM: exp(score - tile_max), tree reduction sum
  4. ONLINE UPDATE: m_new, old_scale, tile_scale, d = old*d + tile*sum
  5. V ACCUM: o *= old_scale; o += tile_scale * exp_tile[i] * v[i]
  6. NORMALIZE: output = o / d
```

### 7.4 3-Tier Fallback Chain

```
Router::forward(input)           Expert::forward_batched(input)
  ├── forward_cuda()               ├── forward_batched_cuda()
  │     cuBLAS matmul              │     cuBLAS fc1 + GELU inplace + cuBLAS fc2
  │     + JIT softmax              │     (no CPU round-trip)
  ├── forward_gpu() (wgpu)         ├── forward_batched_gpu() (wgpu)
  └── CPU (ndarray)                └── CPU (ndarray)
```

### 7.5 GPU Error Recovery

```
GpuRecoveryManager (Circuit Breaker)
  ├── Closed    → normal operation
  ├── Open      → all ops blocked (10 consecutive failures)
  └── HalfOpen  → trial ops after 30s cooldown

Per-error retry:
  DeviceLost → 2 retries + context rebuild
  OOM       → 1 retry + cache purge
  Timeout   → 3 retries + fresh encoder
  Fatal     → 0 retries
```

### 7.6 GPU Observability

| Metrik | Source |
|--------|--------|
| `GPU_BUSY_NS` | AtomicU64 → total GPU busy time |
| `PCIE_READ_BYTES` / `PCIE_WRITE_BYTES` | PCIe traffic tracking |
| Pool allocs/hits/misses | Memory pool stats |
| `gpu_health_status()` | Healthy/Degraded/Unavailable |
| `vram_fragmentation_ratio()` | VRAM usage heuristic |

### 7.7 GPU Watchdog

```
GpuWatchdog
  ├── timeout: 30s
  ├── check_interval: 5s
  ├── max_hangs_before_reset: 3
  ├── ping()      — reset timer setelah GPU op sukses
  └── trigger_reset() — rebuild GPU context jika hang terdeteksi
```

---

## 8. Runtime & Distributed Scheduler

### 8.1 NodeRegistry

**Lokasi:** `crates/runtime/src/cluster.rs`

```
NodeInfo
  ├── node_id: Uuid, address: String
  ├── state: NodeState (Alive | Suspect | Dead | Leaving)
  ├── load: NodeLoad
  │     ├── active_requests, queue_depth
  │     ├── gpu_util_pct, memory_used_mb
  │     ├── tokens_per_sec, kv_fragmentation_pct
  ├── capabilities: NodeCapabilities
  │     ├── models: Vec<String>, max_batch_size, max_seq_len
  │     └── supports_streaming, supports_cuda, supports_paged_cache
  ├── last_heartbeat, started_at
  └── version: String
```

**Load Score Formula:**
```
score = active × 0.3 + queue × 0.3 - (100-gpu)/100 × 0.2 + mem/1024 × 0.1 - min(tok/s/1000,10) × 0.1
```

### 8.2 GossipProtocol

**Lokasi:** `crates/runtime/src/gossip.rs`

```
GossipProtocol::start()
  loop (1s interval):
    1. refresh_dynamic_load()
    2. mark_stale_nodes()      — 3 missed heartbeats → Suspect
    3. evict_dead_nodes()      — 30s timeout → evict
    4. pick random alive peer
    5. alternate push/pull rounds:
       Push → POST {peer}/cluster/gossip/push
       Pull → POST {peer}/cluster/gossip/pull → merge
```

### 8.3 DistributedScheduler

**5 Routing Strategies:**
| Strategy | Logic |
|----------|-------|
| LeastLoaded | Pilih node dengan load_score terendah |
| ModelAffinity | Filter by model capability → sort by load |
| Random | Shuffle candidates |
| RoundRobin | Rotate index per request |
| CacheLocality | Sort by KV fragmentation (terendah dulu) |

**Routing Decision:**
```
submit_request(request):
  local_score = active + queue × 0.5
  remote_candidates = select_remote_nodes(model_id, strategy)
  if remote_candidates[0].load_score() < local_score:
    route_remote(peer, request)
  else:
    local.submit_request(request)
```

### 8.4 DistributedRouter

**Lokasi:** `crates/inference/src/distributed.rs`

```
DistributedRouter::route_remote(node, request)
  → POST {node}/generate (InferenceRequest JSON)
  → Response: InferenceResponse JSON (60s timeout)

DistributedRouter::route_remote_streaming(node, request)
  → POST {node}/generate/stream
  → SSE events: data: {token, position} ... data: [DONE]
  → mpsc::channel(64) → Receiver<GeneratedToken>
```

### 8.5 HTTP API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/generate` | POST | Forward inference → `InferenceResponse` |
| `/generate/stream` | POST | SSE streaming |
| `/cluster/gossip/push` | POST | Gossip push round |
| `/cluster/gossip/pull` | POST | Gossip pull round |
| `/inference/submit` | POST | Scheduler submit |
| `/inference/health` | GET | Node health → `NodeLoad` |

---

## 9. Agent Ecosystem

### 9.1 AgentManager

**Lokasi:** `crates/agent/src/agent_manager.rs`

```
AgentManager
  ├── agents: HashMap<Uuid, Box<dyn BaseAgent>>
  ├── planner: Option<Box<dyn PlannerAgent>>
  ├── workers: Vec<Box<dyn WorkerAgent>>
  ├── inference_engine: StdArc<RwLock<Option<...>>>
  └── memory_store: Arc<MemoryStore>
```

**Agent Types (7 agents):**
| Agent | Role |
|-------|------|
| context_agent | Context management |
| routing_agent | Request routing |
| inference_agent | Inference execution |
| memory_agent | Memory operations |
| planner_agent | Plan creation & management |
| response_agent | Response formatting |
| validation_agent | Output validation |

### 9.2 Planner-Worker Hierarchy

```
POST /api/generate/agent { prompt, context }
  1. ListAgentIds → find planner agent
  2. SendMessage("create_plan") → Plan { plan_id, steps }
  3. DispatchPlan(plan_id):
      dispatch_plan_internal loop:
        for each pending step:
          pick worker (round-robin)
          send_message_internal(worker, exec_msg)
          worker.process_step_inner()
            └─ inference_engine.generate_tokens()
          report complete_step/fail_step to planner
        repeat until all steps done or any failed
  4. PlanStatus(plan_id) → return final status
```

### 9.3 API Endpoints

| Route | Method | Handler |
|-------|--------|---------|
| `/api/agents` | GET | List agents grouped by type |
| `/api/plans` | POST | Create plan via planner |
| `/api/plans` | GET | List all plans |
| `/api/plans/:plan_id` | GET | Plan status |
| `/api/plans/dispatch` | POST | Dispatch steps |
| `/api/generate/agent` | POST | Plan + dispatch (blocking) |

---

## 10. Memory Hierarchy

### 10.1 Memory Layers

**Lokasi:** `crates/memory/src/`

```
MemoryManager
  ├── Episodic (max 10000 entries, 30d retention)
  │     └── Experience records with importance threshold
  ├── Semantic (max 100000 vectors, 1536-dim)
  │     └── Embedding-based similarity search
  └── Working (max 10000 context, 4096 window)
        └── Compression ratio 0.5
```

### 10.2 Memory Operations

| Method | Deskripsi |
|--------|-----------|
| `store(entry)` | Save to appropriate layer |
| `retrieve(query)` | Similarity-based retrieval |
| `consolidate()` | Episodic → Semantic transfer |
| `forget()` | LRU/TTL eviction |
| `stats()` | Memory usage statistics |

---

## 11. Multimodal: Caffeine System

### 11.1 Caffeine Architecture

**Lokasi:** `crates/multimodal/src/caffeine/`

```
CaffeineProcessor
  ├── MultiModalEncoders (5 encoders)
  │     ├── image_encoder: CLIP ViT-L/14 (768d, 14px patch)
  │     ├── audio_encoder: Whisper-style (16kHz, 400 window)
  │     ├── video_encoder: ViT3D (8 frames, 1fps)
  │     ├── text_encoder: BERT (30522 vocab, 512 maxlen)
  │     └── regional_alignment: CLIP-inspired (7x7 grid, 49 patches)
  ├── TriQueryFormer (BLIP-2 style)
  │     ├── 32 semantic queries (high-level understanding)
  │     ├── 16 spatial queries (position/region awareness)
  │     └── 16 temporal queries (time-based context)
  ├── UnifiedTokenizer (VQ-VAE)
  │     ├── vocab_size: 8192, codebook: 1024, num_codebooks: 8
  │     └── commitment_weight: 0.25
  ├── AgenticActionHead
  │     ├── semantic_output (text/response)
  │     ├── spatial_grounding (bounding boxes)
  │     ├── action_planning (multi-step plans)
  │     └── execution (Click, Type, Scroll, dll)
  ├── ATQS Compression (opsional, Tucker decomposition)
  └── HAS-MoE-FFN Router (opsional, modality-aware routing)
```

### 11.2 MultiModalInputs → MultiModalOutputs

```rust
pub struct MultiModalInputs {
    text: Option<TextInput>,
    image: Option<ImageInput>,
    audio: Option<AudioInput>,
    video: Option<VideoInput>,
    context: Option<ContextInfo>,
}

pub struct MultiModalOutputs {
    text: Option<TextOutput>,
    image: Option<ImageOutput>,
    audio: Option<AudioOutput>,
    video: Option<VideoOutput>,
    action: Option<ActionOutput>,
}
```

### 11.3 Pipeline Flow

```
MultiModalInputs
  → Encoder: masing-masing modality → EncodedFeatures
  → TriQueryFormer: 3 query types cross-attend ke encoded features
  → UnifiedTokenizer: discrete tokenization
  → ATQS: optional compression
  → MoE: optional modality-aware routing
  → ActionHead: generate output
  → MultiModalOutputs
```

---

## 12. Reasoning: SACA Framework

### 12.1 6-Phase Pipeline

**Lokasi:** `crates/reasoning/src/saca/`

```
SACA::solve(task)
  ├── Phase 1: Chain-of-Thought
  │     └── CoTEngine → task_analysis, reasoning_steps, edge_cases
  │         (max 10 reasoning steps, configurable depth)
  │
  ├── Phase 2: Modular Decomposition
  │     └── DecomposeEngine → Vec<Module>
  │         (max 20 modules, dependency analysis)
  │
  ├── Phase 3: Repository Context
  │     └── ContextEngine → RepositoryContext
  │         (max 1000 files, 1h cache TTL)
  │
  ├── Phase 4: Large-Scale Sampling
  │     └── SamplingEngine → Vec<Candidate>
  │         (5 candidates, diversity threshold 0.3)
  │
  ├── Phase 5: Execute-Fail-Fix
  │     └── ExecuteEngine → execution result + fixes
  │         (max 3 fix attempts, 30s timeout)
  │
  ├── Phase 6: Mathematical Reranking
  │     └── RerankEngine → scored candidates
  │         (correctness 0.4, performance 0.25, readability 0.15, ...)
  │
  └── Feedback Loop (quality < 0.8 → regenerate)
```

### 12.2 Simplified API

```rust
SacaEngine::reason(problem: &str, context: &str) -> Result<ReasoningResult>
  → Membuat CodingTask → inisialisasi SACA pipeline → solve() → ambil final_code
```

### 12.3 SACAConfig

| Sub-config | Key Parameters |
|------------|---------------|
| CoT | max_steps=10, depth=Medium |
| Decompose | max_modules=20, min_size=5 |
| Context | max_files=1000, cache_ttl=3600s |
| Sampling | candidates=5, strategy=Hybrid |
| Execute | strategy=Adaptive, timeout=30s, fixes=3 |
| Rerank | correctness_weight=0.4 |
| Feedback | max_loops=5, quality_threshold=0.8 |

---

## 13. MoE: Mixture of Experts

### 13.1 Architecture

**Lokasi:** `crates/has-moe-ffn/src/`

```
HasMoeFFN
  ├── config: HasMoeFFNConfig
  │     ├── num_experts: 8, top_k: 2, hidden_size, intermediate_size
  │     ├── capacity_factor: 1.25
  │     ├── z_loss_coefficient: 1e-4
  │     └── use_capped_routing: true
  ├── router: Router (learned gating)
  │     ├── weight: [num_experts × hidden]
  │     ├── forward(input) → softmax → top-2
  │     └── route(input) → full routing with auxiliary losses
  └── experts: Vec<Expert> (8)
        ├── fc1: hidden → intermediate
        ├── GELU activation
        └── fc2: intermediate → hidden
```

### 13.2 Gating Mechanism

**Batched Forward (inference path):**
```
1. matmul(input, router_weight.T) → gate_logits [batch, num_experts]
2. softmax(gate_logits) → gate_probs
3. top-2 selection + renormalize
```

**Full Route (training path with auxiliary losses):**
```
Phase 1: Per-token top-2 → O(E) partial sort
Phase 2: Capped routing — limit token/expert via capacity
Phase 3: Load balancing loss — Switch Transformer: sum(f_i * P_i)
Phase 4: Z-loss — log(sum(exp(gate)))^2 regularization
```

### 13.3 CUDA Fused Forward

```
forward_cuda_fused():
  1. CPU routing (cheap — small gating network)
  2. Allocate output buffer CUDA (zeros)
  3. Loop per expert:
     a. Upload grouped tokens to CUDA
     b. cuBLAS fc1 matmul + add bias
     c. GELU in-place (NVRTC JIT)
     d. cuBLAS fc2 matmul + add bias
     e. scatter_add_weighted → output buffer
  4. Single readback CPU (menghindari N-1 roundtrips)
```

---

## 14. Oracle Backbone & Code Verifiers

### 14.1 Oracle Architecture

**Lokasi:** `crates/oracle/src/`

```
OracleBackbone (12-layer transformer, configurable)
  ├── Embedding: [vocab_size × d_model] (Xavier init)
  ├── 12× SparseMoELayer
  │     ├── Gate: Linear(d_model → n_experts=8)
  │     ├── Router: top-2
  │     └── MLPExpert per expert
  ├── 12× MultiHeadLatentAttention (MLA)
  │     ├── LatentProjection: d_model → latent_dim (512)
  │     ├── LatentCompression: top-25% → compressed KV
  │     ├── LatentAttentionHead (n_heads=32)
  │     └── OutputProjection: latent_dim → d_model
  ├── 13× LayerNorm (pre-norm)
  └── Output: [d_model × vocab_size]

Config: d_model=4096, n_heads=32, n_experts=8, top_k=2,
         latent_dim=512, context_size=32768, mlp_hidden=16384
```

### 14.2 Multi-Head Latent Attention (MLA)

**Key innovation:** Mengurangi KV cache via latent compression.

```
Input [B, S, d_model]
  1. LatentProjection: d_model → 512 (latent)
  2. LatentCompression: top-25% → 128 (compressed KV)
  3. Per-head: Q/K/V projections in latent space
  4. Scaled dot-product attention (causal mask)
  5. Concatenate heads → OutputProjection
```

### 14.3 RoPE with Dynamic Frequency

```
Extended Rotary Position Embedding:
  - Base frequency: 10000.0
  - Scaling factor: 1.0 (dynamic: 1.0 + 0.5 * position_ratio)
  - Cross-file awareness via CrossFilePositionTracker
  - PositionAwareMask: cross-file attention penalty -0.1
```

### 14.4 Code Verifiers (Linters)

**Lokasi:** `crates/oracle/src/linters/`

| Linter | File | Detection | Lines |
|--------|------|-----------|-------|
| SecurityLinter | `security.rs` | SQL injection, XSS, cmd injection, path traversal, hardcoded secrets, unsafe deserialization, SSRF, crypto weakness | 557 |
| PerformanceLinter | `performance.rs` | N+1 queries, O(n²) loops, memory leaks, unbounded recursion, sync-in-async | ~200 |
| CorrectnessLinter | `correctness.rs` | Null pointer, use-after-free, type confusion, race conditions, integer overflow | ~200 |
| StyleLinter | `style.rs` | Naming conventions, formatting, complexity, magic numbers | ~200 |

**Language-specific AST analyzers:**
- `ast_analyzer.rs` — Rust (syn-based): unsafe, cmd injection, transmute
- `python_analyzer.rs` — Python patterns
- `java_analyzer.rs` — Java patterns
- `javascript_analyzer.rs` — JS patterns
- `go_analyzer.rs` — Go patterns
- `dep_scanner.rs` — Dependency vulnerability scanning

**CodeLinterManager:**
```rust
CodeLinterManager::verify_detailed(code, language) -> Vec<LintResult>
  → orchestrates all linters → aggregated issues with severity
```

---

## 15. Alignment: SPARO Framework

### 15.1 6 Alignment Methods

**Lokasi:** `crates/alignment/src/sparo/`

```
SPARO = Self-Play Aligned Reasoning via Prospect-Theoretic Stepwise Optimization

SparoTrainer
  ├── DPO: β=0.1, label_smoothing=0.0
  │     Loss: -log(σ(β * (π_lograt - ref_lograt)))
  │
  ├── KTO: λ=2.25 (loss aversion), α=0.88 (probability weighting)
  │     Prospect-theoretic value function (Kahneman & Tversky)
  │
  ├── IPO: τ=0.1, KL_weight=1.0, identity_strength=0.5
  │     Identity preservation regularization
  │
  ├── RLAIF: consensus_threshold=0.8, max_judges=3
  │     AI judges: GPT-4, constitutional AI, multi-judge consensus
  │
  ├── RLVF: step_weight=0.8, final_weight=0.2
  │     Step-level verification (math, logic, type, security, perf)
  │
  └── SPIN: rounds=3, temperature=0.7, improvement_threshold=0.05
        Self-play tournament → score → select best → retrain
```

### 15.2 Real-Time Alignment Assessment

```
SparoSystem::align_behavior(behavior, context) -> AlignmentResult
  ├── Dimension 1: Keyword pre-filter (10 toxic keywords)
  ├── Dimension 2: Statistical embedding analysis
  │     └── n-gram frequency → embedding_harm_score + structural_anomaly
  ├── Dimension 3: Oracle verification (cfg-gated, SecurityLinter)
  ├── Dimension 4: Evasion risk detection
  └── Dimension 5: Context relevance (word overlap)

Composite harm: keyword(30%) + embedding(30%) + oracle(15%) + evasion(15%) + 1-richness(10%)
Gate: composite < 0.3 → BLOCKED
```

---

## 16. Quantization: ATQS

### 16.1 Pipeline

**Lokasi:** `crates/atqs/src/`

```
ATQSConfig
  ├── Calibration
  │     ├── LoRA Calibration (rank=64, alpha=16, lr=1e-4, 10 epochs)
  │     ├── Accuracy Recovery (threshold=0.95, max_iter=1000)
  │     └── Calibration Optimizer (genetic, pop=100)
  ├── Compression
  │     ├── Adaptive Rank (target=50%, min=8, max=512)
  │     ├── Quantum Sparse (sparsity=0.1, entanglement=0.05)
  │     └── Sparse Augmentation (factor=1.2, magnitude pruning)
  └── Profiling
        ├── Entanglement (sample=1000, entropy=0.1, correlation=0.8)
        ├── Layer Analysis (depth=10, sensitivity=0.01, SVD rank)
        └── Sensitivity Mapping (resolution=100, gradient metric)
```

### 16.2 AWQ (Activation-aware Weight Quantization)

```
AWQEngine::quantize(weight, activations) -> AWQQuantizedTensor
  1. Compute activation_scale: per-column abs max
  2. Compute saliency: col_norm × activation_scale
  3. Find optimal scales: saliency-aware clipping (clip_alpha=1.0)
  4. Quantize: round(W/s + zp) → clamp → pack bits
  5. Presets: w4a16 (4-bit, group=128), w3a16 (3-bit)
```

---

## 17. Application Layer: CLI & Server

### 17.1 NexoraAI Initialization

```
NexoraAI::new(config)
  ├── 0. Init GPU context (wgpu/CUDA auto-detect)
  ├── 1. Init 10 NXR foundation models (CausalLM instances)
  ├── 1b. Init model-internal agents (10 delegation systems)
  ├── 2. Init system monitoring (sysinfo)
  ├── 3. Create InferenceEngine with model from registry
  │     └── create_inference_engine_inner():
  │           ├── Get model from registry → CausalLmModel
  │           ├── Create BpeTokenizer
  │           ├── Optionally wire distributed cluster
  │           └── engine.initialize()
  ├── 4. Init multi-agent system (AgentManager)
  │     ├── Wire InferenceEngine via NexoraInferenceEngine adapter
  │     ├── agent_manager.start()
  │     └── Spawn 7 agents
  └── Return NexoraAI { registry, inference_engine, config, agent_manager, ... }
```

### 17.2 CLI Commands (14 commands)

| Command | Alias | Deskripsi | Key Flags |
|---------|-------|-----------|-----------|
| `start` | dev, serve | Start HTTP server | `--host`, `--port`, `--tls` |
| `process` | — | Single request | `--input`, `--format` |
| `generate` | — | Text generation | `--prompt`, `--max-tokens` |
| `chat` | — | Interactive chat | `--message`, `--conversation-id` |
| `analyze` | — | Code analysis | `--file`, `--language` |
| `codegen` | — | Code generation | `--description`, `--language` |
| `train-foundation` | tf | Foundation training | `--data`, `--hf-dataset`, `--model-id`, `--steps` |
| `train` | t | Full training | `--data`, `--epochs`, `--batch-size`, `--lr` |
| `collect-data` | cd | Web → .arrow | `--sources`, `--max-samples` |
| `load-checkpoint` | lc | Restore model | `--model`, `--path` |
| `evaluate` | e, eval | Evaluation | `--model`, `--test-data` |
| `info` | — | System info | `--performance`, `--memory` |
| `health` | — | Health check | `--detailed` |
| `benchmark` | bench, perf | 9 metrics | `--warmup`, `--samples` |

### 17.3 HTTP Server — 21 Endpoints

```
NexoraServer (axum 0.7)
  ├── Middleware: CORS + Auth + RequestLogging + TraceLayer + ConcurrencyLimit + Timeout
  ├── BackgroundMetricsCollector (15s polling)
  └── Graceful shutdown (Ctrl+C + SIGTERM)

Endpoints:
  GET  /                    — Index page
  GET  /health              — Health status
  GET  /health/detailed     — Full health report
  GET  /metrics             — Prometheus format
  GET  /info                — System info
  POST /generate            — Text generation
  POST /generate/stream     — SSE streaming
  POST /chat                — Chat
  POST /process             — Request processing
  POST /code/analyze        — Code analysis
  POST /code/generate       — Code generation
  GET  /api/agents          — List agents
  POST /api/plans           — Create plan
  GET  /api/plans           — List plans
  GET  /api/plans/:id       — Plan status
  POST /api/plans/dispatch  — Dispatch steps
  POST /api/generate/agent  — Agent generation
```

### 17.4 Benchmark — 9 Metrics

| Metric | Description |
|--------|-------------|
| Tokens/s CPU | CPU-only throughput |
| Tokens/s GPU | GPU-accelerated throughput |
| Prefill latency (ms) | Prompt processing time |
| Decode latency (ms/token) | Per-token generation time |
| VRAM usage (MB) | GPU memory consumption |
| RAM usage (MB) | System memory consumption |
| Prefix cache hit rate (%) | KV cache sharing efficiency |
| Continuous batching throughput | Batch processing efficiency |
| Multi-user concurrency | Max concurrent users |

### 17.5 Error System — 14 Variants

| Error | HTTP Code | Description |
|-------|-----------|-------------|
| Config | 500 | Configuration errors |
| Initialization | 500 | System init failures |
| Processing | 422 | Request processing errors |
| Agent | 422 | Agent system errors |
| Memory | 503 | Memory subsystem errors |
| Model | 422 | Model errors |
| Io | 500 | File I/O errors |
| Serialization | 400 | Parse errors |
| System | 500 | General system errors |
| Validation | 400 | Input validation |
| Timeout | 408 | Operation timeout |
| NotFound | 404 | Resource not found |
| PermissionDenied | 403 | Authorization |
| RateLimit | 429 | Rate limit exceeded |

---

## 18. Data Pipeline

### 18.1 Datastream DAG Pipeline

**Lokasi:** `crates/datastream/src/`

```
DataStream Pipeline:
  Source → Filter DAG → Sink

Sources:
  - HuggingFaceDatasetProvider (live via datasets-server API)
  - FileSource (.arrow, .txt, .csv, .json)
  - WebScraper (hackernews, wikipedia, reddit)

Filters:
  - LengthFilter: min/max sequence length
  - QualityFilter: heuristic quality scoring
  - DedupFilter: MinHash 7-gram deduplication
  - ToxicityFilter: (feature-gated)
  - PromptInjectionFilter: (feature-gated)

Sinks:
  - ArrowSink (.arrow file with sharding)
  - StreamSink (live pipeline)
```

### 18.2 Training Data Flow

```
CLI → HuggingFace / Local File
  → DataStream (DAG filter pipeline)
  → Tokenizer (BPE encode)
  → Training loop (epochs, shuffle, batch, forward/backward)
  → Checkpoint (.safetensors)
  → Evaluation (validation loss, perplexity)
```

---

## 19. Infrastructure & Deployment

### 19.1 Docker Compose — 9 Services

| Service | Image | Port(s) | Volume | Purpose |
|---------|-------|---------|--------|---------|
| Qdrant | qdrant/qdrant | 6333, 6334 | qdrant_storage | Vector DB |
| Redis | redis:7-alpine | 6379 | redis_data | Cache (2GB, allkeys-lru) |
| MongoDB | mongo:7 | 27017 | mongo_data | Document store |
| PostgreSQL | postgres:16-alpine | 5432 | postgres_data | Relational DB |
| PgBouncer | — | 6432 | — | Connection pool |
| Prometheus | prom/prometheus | 9090 | prometheus_data | Metrics (30d) |
| Grafana | grafana/grafana | 3000 | grafana_data | Dashboards |
| InfluxDB | influxdb:2 | 8086 | influxdb_data | Long-term metrics (365d) |
| MinIO | minio/minio | 9000, 9001 | minio_data | Object storage (5 buckets) |

### 19.2 Dockerfile — Multi-Stage Build

```
Stage 1: chef (rust:1.85-slim-bookworm + cargo-chef)
Stage 2: planner (dependency caching)
Stage 3: builder (cargo build --release --bin nexora)
Stage 4: runtime (debian:bookworm-slim + ca-certificates + libssl3)
  ├── COPY binary
  ├── COPY configs/
  ├── EXPOSE 8080
  └── ENTRYPOINT ["nexora", "start", "--config", ...]
```

### 19.3 Monitoring Stack

```
Prometheus (scrape tiap 15s)
  ├── nexora metrics endpoint (/metrics)
  ├── node_exporter (system metrics)
  ├── postgres_exporter
  ├── redis_exporter
  └── qdrant_exporter

Grafana
  ├── Provisioned datasource (Prometheus)
  ├── Pre-built dashboards
  └── Alerting rules
```

---

## 20. Security & Isolation

### 20.1 Isolation Architecture — 7 Levels

**Lokasi:** `crates/isolation/src/`

```
IsolationOrchestrator
  ├── Layer 0: Global-level isolation
  ├── Layer 1: Mode-level isolation
  ├── Layer 2: Agent-level isolation
  ├── Layer 3: Tool-level isolation
  ├── Layer 4: Runtime isolation
  ├── Layer 5: Cognitive isolation
  └── Layer 6: Permission isolation

Plus:
  ├── Firewall: Request filtering + rate limiting
  ├── Kill Switch: Emergency shutdown
  ├── Quarantine: Isolate compromised components
  └── Multi-Cluster: Cross-cluster isolation
```

### 20.2 Security Features

| Feature | Implementation |
|---------|---------------|
| API Auth | Bearer token + x-api-key header |
| CORS | Configurable origins |
| Input Validation | SecurityValidator (malicious prompt detection) |
| Rate Limiting | Token bucket algorithm |
| Request Timeout | 30s default, configurable |
| TLS | Optional via axum-server + rustls |
| Gossip Auth | x-nexora-auth shared secret |
| Cipher Delegation | Threat classifier (6 categories) + Oracle security scan |

---

## 21. Production Readiness

**Per Audit** (Batch Fix 29, 31 Mei 2026): **~95%+**

### 21.1 8 Dimensions — ALL 100%

| Dimension | Status | Key Achievements |
|-----------|--------|------------------|
| GPU acceleration | 100% | CUDA + wgpu + CPU fallback, FlashAttention, F16 |
| Async correctness | 100% | try_lock, OnceLock, Arc<str> optimization |
| Model delegation | 100% | 0 unwrap/expect di 10 model crates |
| Error handling | 100% | All weight accessors return Result |
| Dead code | 100% | 981 lines deprecated unwired |
| Security | 100% | Multi-language AST analyzer + dep scanner |
| Multimodal | 100% | 5 encoders real, image format auto-detect |
| Safety/stability | 100% | try_lock, CUDA poison recovery, quant warnings |

### 21.2 Batch Fix History (29 batches)

| Batch | Focus | Date |
|-------|-------|------|
| BF1-14 | GPU backward, error handling, memory safety | — |
| BF15 | 128KB/token readback elimination | — |
| BF16-20 | 10 model delegation wiring | — |
| BF21 | println! → tracing, serialization safety | — |
| BF22 | Security/Style linters, Arc<Vec<u32>> | — |
| BF23 | SessionEntry Arc<str>, profiling clone | — |
| BF24 | Security 100% — AST analyzer + dep scanner | — |
| BF25 | Multimodal 100% — image format auto-detect | — |
| BF26 | Model delegation 100% — 0 unwrap/expect | 31 Mei 2026 |
| BF27 | MoE FFN 100% — unused import cleanup | 31 Mei 2026 |
| BF28 | ATQS 100% — dead fields + caffeine guard | 31 Mei 2026 |
| BF29 | Ringkasan gap 100% — all 8 dimensions | 31 Mei 2026 |

### 21.3 Remaining Gaps

| Area | Status | Notes |
|------|--------|-------|
| Observability | Not scored | KV fragmentation, token/sec tracing |
| Tests | 233 passed, 1 pre-existing failure | Spectral frequency analyzer |
| Distributed | Functional | Need production hardening |
| CI/CD | Exists but unverified | `.github/workflows/` ada tapi AGENTS.md bilang "No CI" |
| Cargo.lock | Gitignored | Every build resolves from scratch |
| Docs | ARCHITECTURE.md + AGENTS.md | Need sync with current code |

---

## 22. Alur Data End-to-End

### 22.1 Server Request Flow (POST /generate)

```
Client → POST /generate { prompt, max_tokens, temperature }
  │
  ├── auth_middleware (API key check)
  ├── request_logging (method + URI)
  ├── concurrency_limit (semaphore)
  ├── timeout_layer (30s)
  │
  └── generate_text handler
        │
        ├── SecurityValidator::validate_prompt(prompt)
        │     └── Reject if malicious
        │
        ├── nexora.generate_text(prompt, max_tokens, temperature)
        │     │
        │     └── delegate_for_model(active_model_id, prompt)
        │           │
        │           ├── Example: Omnis delegation
        │           │     ├── OmnisMoERouter::predict(prompt)
        │           │     │     └── embed_average → Router::forward() → 7 domains
        │           │     ├── Format system prompt with domain context
        │           │     ├── call_model(foundation, framed_prompt)
        │           │     │     └── CausalLmModel::infer(&NxrInput)
        │           │     │           ├── MiniTokenizer::encode(prompt) → tokens
        │           │     │           ├── CausalLM::generate_with_gpu()
        │           │     │           │     ├── Prefill: forward semua prompt
        │           │     │           │     ├── Loop: forward → sample → append
        │           │     │           │     │     ├── GPU: CUDA matmul / wgpu dispatch
        │           │     │           │     │     ├── PagedKVCache::append
        │           │     │           │     │     └── Sampling: top-k/top-p/temperature
        │           │     │           │     └── Stop: EOS atau max_tokens
        │           │     │           └── MiniTokenizer::decode(output_ids) → text
        │           │     └── Return generated text
        │           │
        │           ├── Example: Aether delegation
        │           │     ├── EmotionClassifier::predict(prompt) → 8 emotions
        │           │     ├── CaffeineProcessor::process_multimodal() → summary
        │           │     └── Format empathetic prompt → call_model()
        │           │
        │           └── Return NxrOutput { text, metadata, performance }
        │
        └── JSON response { success, generated_text, parameters, timestamp }
```

### 22.2 Distributed Request Flow

```
Client → POST /generate (Node A)
  │
  ├── InferenceEngine.submit_request(request)
  │     │
  │     ├── Distributed route?
  │     │     ├── best_remote_node() → Node B (lower load_score)
  │     │     └── route_remote(Node B, request)
  │     │           └── POST {Node B}/generate
  │     │                 └── Node B processes locally → response
  │     │
  │     └── Local route?
  │           └── RequestScheduler → request_tx channel
  │                 └── ContinuousBatchingEngine.process_batch()
  │                       ├── Phase 0: Prefix sharing
  │                       │     ├── PrefixTrie.find_shared_prefix()
  │                       │     └── share_prefix_in_blocks()
  │                       ├── Phase 1: Prefill
  │                       ├── Phase 2: Decode loop
  │                       └── Phase 3: Post-process → response
  │
  └── Response back to client
```

### 22.3 Agentic Request Flow

```
Client → POST /api/generate/agent { prompt, context }
  │
  ├── 1. ListAgentIds → find planner agent
  ├── 2. SendMessage("create_plan") → planner
  │     └── Plan { plan_id, steps: [Step1, Step2, ...] }
  │
  ├── 3. DispatchPlan(plan_id)
  │     └── dispatch_plan_internal loop:
  │           for each pending step:
  │             pick worker (round-robin)
  │             send_message_internal(worker, exec_msg)
  │             worker.process_step_inner()
  │               └─ inference_engine.generate_tokens()
  │             report complete_step / fail_step to planner
  │
  └── 4. PlanStatus(plan_id) → final status
```

### 22.4 Training Flow

```
CLI: nexora train --hf-dataset wikitext --epochs 3 --output ./model
  │
  ├── 1. Init GPU context
  ├── 2. Load data
  │     └── HuggingFaceDatasetProvider → rows → tokenize → cache
  │
  ├── 3. DAG Filter Pipeline
  │     ├── LengthFilter: 50-2048 tokens
  │     ├── QualityFilter: heuristic score > 0.3
  │     └── DedupFilter: MinHash 7-gram
  │
  ├── 4. Create Trainer
  │     ├── Optimizer: AdamW (lr=0.001, weight_decay=0.01)
  │     ├── Schedule: linear warmup (10%) → cosine decay
  │     ├── Gradient clipping: max_norm=1.0
  │     └── Batch size with gradient accumulation
  │
  ├── 5. Training Loop
  │     ├── Per epoch: shuffle → batch → forward → backward → step
  │     ├── Checkpoint: tiap 500 step (small), 1000 step (big)
  │     ├── Validation: tiap max_steps/5
  │     └── Early stopping: 3× tanpa improvement
  │
  └── 6. Final Checkpoint → model.safetensors + training_report.json
```

---

## Lampiran A: Key Traits

```rust
// Model trait utama
#[async_trait]
pub trait NxrModel: Send + Sync {
    type Config; type Metrics; type State;
    fn identity(&self) -> &ModelMeta;
    async fn initialize(&mut self, config: Self::Config) -> Result<()>;
    async fn infer(&self, input: &NxrInput) -> Result<NxrOutput>;
    async fn infer_stream(&self, input: &NxrInput, callback: ...) -> Result<()>;
    async fn validate(&self) -> Result<ValidationResult>;
    async fn is_ready(&self) -> bool;
}

// Neural network layer trait
pub trait Layer: Send + Sync {
    type Input; type Output;
    fn forward(&self, input: &Self::Input) -> Result<Self::Output>;
    fn num_params(&self) -> usize;
}

// KV Cache provider
pub trait KVCacheProvider: Send + Sync {
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]);
    fn as_cpu_entries(&self) -> &[KVCacheEntry];
    fn as_gpu_entries(&mut self) -> &mut [GpuKVCacheEntry];
    fn clear(&mut self);
}

// Inference backend
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn generate(&self, model_id: &str, prompt: &str, max_tokens: usize, temperature: f32)
        -> Result<(String, usize)>;
    fn backend_name(&self) -> &str;
}

// Agent base
pub trait BaseAgent: Send + Sync {
    fn id(&self) -> &Uuid;
    fn agent_type(&self) -> AgentType;
    async fn process_message(&mut self, message: AgentMessage) -> Result<AgentResponse>;
}

// Inference engine trait (untuk agent system)
pub trait InferenceEngine: Send + Sync {
    async fn start_session(&self, config: SessionConfig) -> Result<SessionId>;
    async fn generate_tokens(&self, session: &SessionId, prompt: &str)
        -> Result<Vec<GeneratedToken>>;
    async fn stream_tokens(&self, session: &SessionId, prompt: &str)
        -> Result<Receiver<GeneratedToken>>;
}
```

## Lampiran B: Key Enums

```rust
pub enum GpuBackend { Wgpu, Cuda }
pub enum GpuErrorKind { DeviceLost, OutOfMemory, ShaderCompile, Timeout, Transient, Fatal }
pub enum GpuHealth { Healthy, Degraded, Unavailable }
pub enum NodeState { Alive, Suspect, Dead, Leaving }
pub enum RoutingStrategy { LeastLoaded, ModelAffinity, Random, RoundRobin, CacheLocality }
pub enum NxrModelId { Omnis, Vortex, Aether, Spectra, Nexum, Axiom, Cipher, Swift, Kronos, Genesis }
pub enum ModelTier { Ultra, Master, Apex, Pro, Core, Edge }
pub enum InputData { Text(String), Tokens(Vec<u32>), Image(...), Audio(...), Multimodal(...), Structured(Value) }
pub enum OutputData { Text(String), Tokens(Vec<u32>), Image(...), Audio(...), Multimodal(...), Structured(Value) }
pub enum SpecialistType { Text, Image, Audio, Video, Multimodal, TextGenerator, Analyzer, CodeGenerator, CreativeWriter }
pub enum AgentType { Context, Routing, Inference, Memory, Planner, Response, Validation }
pub enum Storage { Cpu(ArrayD<f32>), Gpu(GpuTensor), Cuda(CudaTensor) }
pub enum BlockData { F32(Array2<f32>), F16(Vec<u16>, usize, usize) }
pub enum ElemOp { Add, Sub, Mul, Div, Neg, Exp, Sqrt, Relu, Gelu, Silu, Sigmoid, Ln, Tanh, Powf, Step, ... }
pub enum ClusterGranularity { Neuron, Layer, Embedding, Document, Memory, Token }
pub enum TemporalMode { Urgent, Scheduled, Historical, Realtime, Evergreen }
```

---

> **Dokumen ini adalah hasil sintesis dari 10 sub-agent eksplorasi yang menganalisis 42 crate, 100+ file source, dan ribuan baris kode secara independen.**
>
> **Dibuat:** 31 Mei 2026
