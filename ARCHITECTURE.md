# Nexora AI Architecture

## Overview

Nexora AI is a Rust-based multi-model cognitive architecture with 21 workspace members (19 library crates + 2 binary crates). It implements 10 transformer-based models (NXR series), a custom autograd engine, reasoning frameworks, multimodal processing, safety alignment, and a full API server.

## Dependency Layers

```
Leaf crates (no internal deps):
  core, common, utils, deeplearning, database, data,
  monitoring, runtime

Layer 2:
  tokenizer     → core
  memory        → core
  datastream    → tokenizer

Layer 3:
  foundation    → core, tokenizer, deeplearning, common
  isolation     → core
  cognition     → foundation

Layer 4:
  intelligence  → core, tokenizer, foundation, common

Layer 5:
  inference     → core, intelligence, foundation, common, blaa, tokenizer
  agent         → core, intelligence, memory, common

Infrastructure:
  infrastructure → re-exports common + utils
  api           → infrastructure

Application:
  nexora-ai     → core, foundation, tokenizer, intelligence, memory,
                  inference, blaa, infrastructure, datastream,
                  deeplearning, monitoring
  dashboard     → standalone (ratatui TUI)
```

## NXR Model Series

10 models sharing a `CausalLM` transformer backbone, differing in size:

| Model  | Tier  | Hidden | Heads | Layers |
|--------|-------|--------|-------|--------|
| Omnis  | Ultra | 768    | 12    | 8      |
| Axiom  | Ultra | 768    | 12    | 8      |
| Genesis| Ultra | 768    | 12    | 8      |
| Vortex | Apex  | 512    | 8     | 6      |
| Aether | Apex  | 512    | 8     | 6      |
| Nexum  | Apex  | 512    | 8     | 6      |
| Spectra| Pro   | 384    | 6     | 4      |
| Cipher | Pro   | 384    | 6     | 4      |
| Kronos | Core  | 256    | 4     | 3      |
| Swift  | Edge  | 128    | 4     | 2      |

All models implement `NxrModel` via the `define_foundation_model!` macro in `crates/foundation/src/models/foundation.rs`.

## Key Framework Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Autograd  | `deeplearning/` | Tensor ops, STar-X, Echo Net, GNAC |
| ATQS      | `foundation/atqs/` | Adaptive tensor quantization & calibration |
| SACA      | `foundation/reasoning/saca/` | Self-attention cognitive architecture |
| CAFFEINE  | `foundation/multimodal/caffeine/` | Multimodal fusion & encoding |
| SPARO     | `foundation/alignment/sparo/` | Safety, privacy, alignment |
| ORACLE    | `foundation/oracle/` | Code analysis, verification, training |
| HLDA-VT   | `foundation/hldva_t/` | Hierarchical latent diffusion |
| VOGP+     | `foundation/vogp/` | Variance-optimized Gaussian processes |
| ERP       | `foundation/erp/` | Echo resonance processing |
| MoE FFN   | `foundation/has_moe_ffn/` | Mixture-of-experts |
| Inference | `inference/` | KV cache, sampling, beam search, speculative decoding |
| Memory    | `memory/` | 4-layer memory (short, session, long, knowledge) |
| Isolation | `isolation/` | L0-L6 isolation, firewall, kill switch |
| Safety    | `nexora-ai/src/security/` | Safety gate system |

## Pipeline Architecture

```
Stream Sources → Intake Engine → Filter DAG → Intelligence Core → Delivery Layer
                                     ↓
                              Toxicity/Injection/Regex filters
```

## Binaries

- **nexora** (`apps/nexora-ai`): CLI + API server (`health`, `info`, `start` on `:8080`)
- **dashboard** (`apps/dashboard`): TUI monitoring dashboard

## Infrastructure

9 Docker services: Qdrant, Redis, MongoDB, PostgreSQL + PgBouncer, Prometheus, Grafana, InfluxDB, MinIO S3.

Config files in `configs/`: `inference.toml`, `logging.toml`, `memory.toml`, `runtime.toml`.
