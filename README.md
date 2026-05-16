# Nexora AI

Modular AI framework in pure Rust — NXR model series, advanced alignment, streaming inference, and production infrastructure.

## Architecture

21-member Cargo workspace. Dependency layering:

```
nexora-core              # leaf — no internal deps
nexora-tokenizer         # BPE + unicode normalization
nexora-deeplearning      # STar-X tensor, EchoNet, GNAC, Autograd engine
nexora-foundation        # NXR model series + ERP/VOGP/ATQS/ORACLE/SACA/CAFFEINE/SPARO/HLDA-VT/MoE
nexora-intelligence      # model registry, unified serving API
nexora-inference         # KV cache, sampling, beam search, speculative decoding
nexora-memory            # episodic/semantic/working memory
nexora-agent             # multi-agent orchestration
nexora-isolation         # L0-L6 isolation, firewall, kill-switch
nexora-datastream        # DAG-based streaming pipeline
nexora-database          # PostgreSQL, SQLite, MySQL drivers
nexora-api               # axum server + middleware (CORS, auth, metrics, TLS)
nexora-infrastructure    # common utilities, logging, error recovery
nexora-cognition         # reasoning, planning, reflection
nexora-blaa              # external LLM bridge
nexora-monitoring        # system monitoring
nexora-runtime           # scheduler, thread pool
nexora-data              # data processing
nexora-ai (app)          # CLI binary
nexora-dashboard (app)   # TUI dashboard
```

## NXR Model Series

10 specialized models in `crates/foundation`:

| Model | Domain | Integration |
|---|---|---|
| **Omnis** | General-purpose | ERP pruning, MoE FFN |
| **Vortex** | Code analysis | ORACLE code verifier |
| **Swift** | Edge inference | ATQS compression |
| **Sentry** | Safety/guard | VOGP optimizer |
| **Aether** | Emotional/cognitive | SACA reasoning |
| **Echo** | Conversational | — |
| **Vertex** | Vision/graph | VOGP optimizer |
| **Spectra** | Multimodal | CAFFEINE encoder |
| **Solara** | Lightweight | — |
| **Nexum** | Agent orchestration | SPARO alignment |

All models share `FoundationComponents` (ERP engine, VOGP engine, ATQS compressor, MoE FFN, Tokenizer) and deep learning integration (STar-X tensor ops, EchoNet, GNAC, Autograd).

## Key Components

| Component | Crate | Description |
|---|---|---|
| **ERP** | `nexora-foundation` | Evolutionary Reinforcement Pruning |
| **VOGP** | `nexora-foundation` | Variational Optimization with Gradient Propagation |
| **ATQS** | `nexora-foundation` | Attention Tensor Quantum System (compression) |
| **ORACLE** | `nexora-foundation` | Optimized Retrieval-Augmented Code Learning Engine |
| **SACA** | `nexora-foundation` | Self-Aware Cognitive Architecture |
| **CAFFEINE** | `nexora-foundation` | encoder-decoder framework |
| **SPARO** | `nexora-foundation` | Supervised Preference Alignment and Reward Optimization |
| **HLDA-VT** | `nexora-foundation` | Hierarchical Latent Dirichlet Allocation with Vision Transformers |
| **MoE** | `nexora-foundation` | Mixture of Experts FFN |
| **STar-X** | `nexora-deeplearning` | Tensor computation engine |
| **EchoNet** | `nexora-deeplearning` | Recurrent processing |
| **GNAC** | `nexora-deeplearning` | Graph Neural Attention Controller |

## Binaries

```bash
# CLI — health, info, start, generate, chat, analyze, codegen, train, evaluate
cargo run --bin nexora

# TUI dashboard
cargo run --bin dashboard
```

### CLI Commands

| Command | Description |
|---|---|
| `health` | Component health check |
| `info` | System info (CPU/memory/models) |
| `start` | Start HTTP server |
| `generate` | Generate text |
| `chat` | Interactive chat |
| `analyze` | Code analysis |
| `codegen` | Code generation |
| `train` | Full training pipeline |
| `evaluate` | Model evaluation (perplexity) |
| `config show\|validate\|generate` | Config management |

## Quick Start

```bash
# Health check
cargo run --bin nexora -- health

# Start API server
cargo run --bin nexora -- start

# Generate text
cargo run --bin nexora -- generate --prompt "Hello, world!"

# Train a model
cargo run --bin nexora -- train --data ./data
```

## Config

Runtime config in `configs/`:

| File | Purpose |
|---|---|
| `inference.toml` | Engine, decoding, streaming, latency |
| `logging.toml` | Level, format, sinks, tracing |
| `runtime.toml` | Scheduler, thread pool, resource limits |
| `memory.toml` | Episodic/semantic/working memory |

```bash
cargo run --bin nexora -- start --config configs/inference.toml
```

## Development

```bash
cargo check              # typecheck (fastest)
cargo fmt                # format
cargo clippy             # lint
cargo nextest run        # tests (preferred)
cargo test -p <crate>    # single crate
```

Infrastructure services (pick what you need):

```bash
docker compose up postgres redis -d
```

## HTTP API

Server starts at `http://localhost:8080`:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/generate` | POST | Text generation |
| `/chat` | POST | Chat completion |
| `/code/analyze` | POST | Code analysis |
| `/code/generate` | POST | Code generation |
| `/train` | POST | Start training |
| `/config` | GET/POST | Runtime config |

Middleware: CORS, rate limiting, auth (disabled by default), compression, security headers, metrics (Prometheus format).

## Infrastructure

9 services via docker-compose: Qdrant (vector DB), Redis (cache), MongoDB (doc store), PostgreSQL + PgBouncer (relational), Prometheus + Grafana (metrics), InfluxDB (long-term), MinIO S3 (object storage).

## Project Status

**Late prototype / early production.** CLI and training pipeline are production-ready. Middleware stack is comprehensive. ~100+ placeholder methods remain in foundation/deeplearning crates. No CI/CD pipeline configured. See `AGENTS.md` for detailed crate map.

## License

MIT OR Apache-2.0
