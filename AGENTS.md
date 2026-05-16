# Nexora AI — Agent Guide

## Workspace

21-member Cargo workspace (resolver = "2"). Root `Cargo.lock` is gitignored — lockfile is NOT committed.

## Binaries

| Binary | Crate | Command |
|---|---|---|
| `nexora` | `apps/nexora-ai` | `cargo run --bin nexora -- [health\|info\|start]` |
| `dashboard` | `apps/dashboard` | `cargo run --bin dashboard` |

## Key crates

| Crate | Package name | Notes |
|---|---|---|
| `crates/foundation` | `nexora-foundation` | Largest crate. NXR model series, ATQS, SACA, CAFFEINE, ORACLE, SPARO, HLDA-VT, VOGP, MoE, multimodal, alignment |
| `crates/intelligence` | `nexora_model` (`[lib] name = "nexora_model"`) | Model registry, serving, unified API |
| `crates/deeplearning` | `nexora-deeplearning` | Autograd engine, STar-X tensor, GNAC, Echo Net |
| `crates/core` | `nexora-core` | Controller, types, execution, async executor |
| `crates/inference` | `nexora-inference` | KV cache, sampling, beam search, speculative decoding |
| `crates/blaa` | `nexora-blaa` | External "Black Language Model API" bridge |
| `crates/infrastructure` | `nexora-infrastructure` | Re-exports sub-crates `common` (`nexora-common`) and `utils` (`nexora-utils`) |
| `crates/datastream` | `nexora-datastream` | DAG-based streaming data pipeline; features: `toxicity`, `prompt-injection`, `candle`, `arrow` |
| `crates/database` | `nexora-database` | Features: `postgres`, `sqlite`, `sqlx`, `mysql`, `all` |
| `crates/api` | `nexora-api` | Features: `cors`, `metrics`, `tls` |

## Commands

```sh
cargo check                          # typecheck (fastest feedback)
cargo build                          # debug build
cargo build --release                # opt build
cargo fmt                            # format (no config — uses defaults)
cargo clippy                         # lint
cargo nextest run                    # preferred test runner
cargo test -p <package>              # single package
cargo nextest run -p <package>       # single package via nextest
cargo test --test <test_name>        # single integration test
cargo run --bin nexora               # run CLI
cargo run --bin nexora -- start      # start API server (http://localhost:8080)
cargo run --bin nexora -- health     # health check
cargo run --bin dashboard            # TUI dashboard
```

## Nextest specifics

- Config in `nextest.toml`. Default: 4 threads, 2 retries (fixed 1s backoff).
- Integration tests: 1 thread, 600s timeout.
- `package:nexora-api`: 2 threads, 30s timeout.
- `package:nexora-models`: 1 thread, 120s timeout.

## Infrastructure

`docker-compose.yml` starts 9 services for full-stack dev: Qdrant (vector DB :6333), Redis (:6379), MongoDB (:27017), PostgreSQL (:5432) + PgBouncer (:6432), Prometheus (:9090), Grafana (:3000), InfluxDB (:8086), MinIO S3 (:9000/:9001).

Pick and choose — not all are needed for local dev. Start selectively:
```sh
docker compose up postgres redis -d
```

## Config

Runtime config files in `configs/`:
- `inference.toml` — engine, decoding, streaming, latency params
- `logging.toml` — level, format, file/console sinks, tracing
- `memory.toml`, `runtime.toml`

CLI accepts `--config <path>` flag.

## Crate dependency layers

```
nexora-core          # leaf — no internal deps
nexora-tokenizer     # depends on nexora-core
nexora-deeplearning  # standalone (ndarray-based autograd/STar-X)
nexora-foundation    # largest hub: depends on core, tokenizer, deeplearning, common
nexora-intelligence  # depends on core, tokenizer, foundation, common
nexora-inference     # depends on core, intelligence, foundation, common, blaa
nexora-agent         # depends on core, intelligence, memory, common
nexora-memory        # depends on core
nexora-ai (app)      # depends on core, foundation, tokenizer, intelligence, memory, inference, blaa, infrastructure
nexora-dashboard     # standalone TUI (ratatui)
```

## Notable quirks

- `Cargo.lock` in `.gitignore` — every `cargo build` resolves from scratch unless a lockfile exists locally.
- No `rustfmt.toml`, `clippy.toml`, `deny.toml`, or Makefile.
- No CI workflows (no `.github/workflows/`).
- No `build.rs` files anywhere.
- `crates/data/raw/*.arrow` in `.gitignore`.
- Several crates have feature-gated modules: always check `[features]` before assuming something is included.
- App library (`apps/nexora-ai/src/lib.rs`) re-exports from `config::NexoraConfig`, `server::NexoraServer`, `api::NexoraApi` — models in `app/config/`, `app/server/`, `app/api/` dirs.
