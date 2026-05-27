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
| `crates/isolation` | `nexora-isolation` | Multi-layered isolation architecture (L0-L6, Firewall, Kill Switch, Multi-Cluster) |

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
cargo run --bin nexora -- load-checkpoint --model <id> --path <file>  # restore checkpoint
cargo run --bin nexora -- train-foundation --data <file> --output <dir>  # in-place training
cargo run --bin nexora -- collect-data --sources hackernews,wikipedia,reddit --max-samples 1000 --output ./data/  # collect dataset from web → .arrow
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
nexora-isolation     # depends on core; L0-L6 isolation, firewall, kill-switch, multi-cluster
nexora-ai (app)      # depends on core, foundation, tokenizer, intelligence, memory, inference, blaa, infrastructure
nexora-dashboard     # standalone TUI (ratatui)
```

## Training

Real training via `cargo run --bin nexora -- train --data <file> --output <model>`. Key components:

| Component | Location | Status |
|---|---|---|
| Autograd engine | `crates/deeplearning/src/autograd/` | Tape-based reverse-mode, 25+ ops, full test suite |
| AdamW optimizer | `crates/deeplearning/src/autograd/mod.rs:150` | Bias correction, gradient clipping, decoupled weight decay |
| Trainer | `crates/foundation/src/training/mod.rs` | Gradient accumulation (batch_size), LR warmup + cosine decay, checkpoint at save_every |
| CLI | `apps/nexora-ai/src/cli/training.rs` | `train` + `evaluate` commands, DataStream filter pipeline |
| CLI (foundation) | `apps/nexora-ai/src/cli/handlers.rs` | `train-foundation` — in-place training via `--output` |
| Data pipeline | `crates/datastream/src/` | DAG-based streaming, length/quality/dedup filters |

### Training features
- **Gradient accumulation**: `batch_size` controls how many forward/backward passes accumulate before optimizer step.
- **Learning rate schedule**: Linear warmup (`warmup_steps`) followed by cosine decay to 0.
- **Weight decay (AdamW)**: Decoupled weight decay via `weight_decay` param (default 0.01 in CLI).
- **Gradient clipping**: Global L2 norm clipping via `max_grad_norm` (default 1.0 in CLI).
- **Data shuffling**: Each epoch shuffles samples.
- **Loss reporting**: Average loss per-token (not per-step), reported each optimizer step.
- **Graceful shutdown**: Ctrl+C completes current batch before saving.
- **BLAS acceleration**: Add `"blas"` to ndarray features in `crates/deeplearning/Cargo.toml` and install `libopenblas-dev` for 5-10x CPU matmul speedup.

### Checkpoint system (`train-foundation`)

When `--output <base>` is set, `train_on_data` saves automatically:

| Type | Every N steps | File pattern |
|---|---|---|
| Small checkpoint | 100 | `{base}.step-{N}.safetensors` |
| Big checkpoint | 1000 | `{base}.big-{N}.safetensors` |
| Final | end of training | `{base}.final.safetensors` |

Restore a checkpoint into a model:
```sh
cargo run --bin nexora -- load-checkpoint --model omnis --path ./ckpt.final.safetensors
```

### Foundation training (`train-foundation`)
```sh
cargo run --bin nexora -- train-foundation \
  --data training_data.arrow \
  --model-id swift \
  --steps 500 \
  --output ./checkpoints/swift

# All models in parallel:
cargo run --bin nexora -- train-foundation \
  --data training_data.arrow \
  --model-id all \
  --steps 1000 \
  --output ./checkpoints/all \
  --parallel
```

### Full pipeline example
```sh
cargo run --bin nexora -- train \
  --data training_data.txt \
  --output ./checkpoints/model \
  --tokenizer ./checkpoints/tokenizer.json \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 0.001
```

## Fake Architecture Cleanup

Phase 1-3 complete. Tiap model crate punya `delegation.rs` dengan pola unik: Tiap model crate punya `delegation.rs` dengan pola unik:

| Crate | Pola | Delegasi |
|-------|------|----------|
| Omnis | **Expert routing (7 domain)** | `omnis::delegation::delegate(prompt)` |
| Aether | Emotional framing | `aether::delegation::delegate(prompt)` |
| Axiom | Structured logic | `axiom::delegation::delegate(prompt)` |
| Spectra | Parallel creative (3 temps) | `spectra::delegation::delegate(prompt)` |
| Cipher | Security checklist | `cipher::delegation::delegate(prompt)` |
| Vortex | **Code review** | `vortex::delegation::delegate(prompt)` |
| Kronos | Temporal context | `kronos::delegation::delegate(prompt)` |
| Swift | Minimal pass-through | `swift::delegation::delegate(prompt)` |
| Genesis | Iterative refinement | `genesis::delegation::delegate(prompt)` |
| Nexum | Task decomposition | `nexum::delegation::delegate(prompt)` |

Semua delegation memanggil `crate::foundation::Nxr*Model::infer()` via `OnceLock` singleton. Native impl (classifier, encoder, expert routing) selesai — semua 10 crate punya real MLP classifier.

**Aether Emotion Classifier** (`crates/models/src/aether/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 64 GELU → 8 emotions`) — bukan keyword matching
- Input: average pooled CausalLM `token_embedding` per token ID
- Output: 8 emotions (joy, sadness, anger, fear, surprise, disgust, trust, neutral) dengan softmax probability
- Bobot Xavier init via `rand::thread_rng()`, bisa di-load dari checkpoint
- Integrasi: Aether `delegate()` deteksi emosi sebelum call foundation, inject `detected emotion: {dominant}` ke prompt

**Omnis Expert Router** (`crates/models/src/omnis/router.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 7 domains`) — domain-aware expert routing
- Input: average pooled CausalLM `token_embedding` per token ID
- Output: 7 domains (math, science, code, creative, reasoning, factual, general) dengan softmax probability
- Routing: top-1 inject expert system prompt; top-2 confidence > 0.3 → dual-expert + synthesis call
- Bobot Xavier init via `rand::thread_rng()`

**Vortex Code Review Analyzer** (`crates/models/src/vortex/analyzer.rs`):
- Real 2-layer MLP (`embed_dim → 64 GELU → 6 categories`) — code review focus classifier
- Input: average pooled CausalLM `token_embedding` per token ID
- Output: 6 categories (bugs, security, performance, style, architecture, general) dengan softmax probability
- Language detection heuristic: 8 bahasa (Rust, Python, JS, Java, C++, Go, SQL, unknown)
- Integrasi: Vortex `delegate()` deteksi kategori + bahasa, inject focus prompt ke foundation call

**Axiom Reasoning Classifier** (`crates/models/src/axiom/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 6 reasoning types`)
- Types: deductive, inductive, abductive, analogical, causal, analytical

**Cipher Threat Classifier** (`crates/models/src/cipher/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 6 threat categories`)
- Threats: injection, xss, auth, crypto, config, network

**Kronos Temporal Classifier** (`crates/models/src/kronos/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 5 temporal modes`)
- Modes: urgent, scheduled, historical, realtime, evergreen
- Inject `chrono::Utc::now()` + temporal framing

**Swift Task Classifier** (`crates/models/src/swift/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 5 task types`)
- Types: qa, summarize, translate, generate, analyze
- Adjusts `max_tokens` + `temperature` per task type

**Genesis Quality Classifier** (`crates/models/src/genesis/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 6 quality dimensions`)
- Dimensions: clarity, depth, accuracy, structure, conciseness, engagement
- Focuses iterative refinement on weakest dimension

**Nexum Complexity Classifier** (`crates/models/src/nexum/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 4 complexity levels`)
- Levels: simple (direct), moderate (2-3 subtasks), complex (3-5), multi_domain (8 + synthesis)

**Spectra Style Classifier** (`crates/models/src/spectra/classifier.rs`):
- Real 2-layer MLP (`embed_dim → 32 GELU → 6 creative styles`)
- Styles: narrative, poetic, persuasive, technical, dialogue, descriptive
- Adjusts temperature range per style (not fixed 0.7/0.9/1.2)

## Phase 4 — Native Specialized Systems

**Vision**: Wiring real subsystem infrastructure into model crate delegation. Bukan prompt wrapper — genuine capability.

### Infrastructure yang SUDAH ADA (belum di-wire)

| Subsystem | Crate | Files | Status |
|-----------|-------|-------|--------|
| **Multimodal** (Caffeine) | `crates/multimodal/` | 5 encoders (image/audio/video/text/regional), Q-Former, unified tokenizer, action head, ATQS+MoE integration — 20+ source files | ✅ Real, bobot random |
| **SACA Reasoning** | `crates/reasoning/` | 6-phase closed-loop: CoT → Decompose → Context → Sampling → Execute-Fail-Fix → Rerank. Feedback loop dengan quality threshold | ✅ Real pipeline |
| **MoE Gating** | `crates/has-moe-ffn/` | Token-level learned gating (8 experts, top-2), load balancing loss, capped routing, GPU acceleration | ✅ Real, 22 unit tests |
| **Oracle Backbone** | `crates/oracle/` | 12-layer MoE + MultiHeadLatentAttention transformer, RoPE, FIM pretraining, DPO alignment trainer | ✅ Real arsitektur, bobot random |
| **Code Verifiers** | `crates/oracle/src/verifiers/` | 4 rule-based verifiers (security, performance, correctness, style) — 775 LOC real analysis | ✅ Real |

### Target Wiring

| Crate | Current (Phase 3) | Phase 4 Target | Subsystem | Status |
|-------|-------------------|----------------|-----------|--------|
| **Omnis** | Expert router MLP + prompt | Real MoE gating + Oracle reasoning backbone | `has-moe-ffn` + `oracle` | ✅ MoE gating wired |
| **Aether** | Emotion classifier MLP (8 emotions) | Sentiment + intent + emotional state pipeline | `multimodal` encoders + classifier | ⏳ Deferred (no lightweight sentiment API yet) |
| **Axiom** | Reasoning classifier MLP | SACA 6-phase reasoning loop (CoT, decompose, execute-fail-fix, rerank) | `reasoning` | ⏳ Deferred (SACA is code-specific) |
| **Spectra** | Style classifier + 3 temps | Full multimodal Caffeine pipeline (vision, audio, fusion) | `multimodal` | ✅ CaffeineProcessor wired |
| **Vortex** | Code review analyzer MLP | Oracle code verifiers + analysis pipeline | `oracle/src/verifiers/` | ✅ CodeVerifierManager wired |
| **Cipher** | Threat classifier MLP | Security scanning engine + verifier integration | `oracle/src/verifiers/security.rs` | ✅ Security verifier wired |
| **Kronos** | Temporal classifier MLP | Temporal reasoning with context window | `reasoning` + `multimodal` | ⏳ Deferred |
| **Swift** | Task classifier MLP | Latency-aware dispatch + task routing | `has-moe-ffn` routing | ✅ MoE Router wired |
| **Genesis** | Quality classifier MLP | Self-improvement loop with SACA feedback | `reasoning` feedback system | ⏳ Deferred (SACA is code-specific) |
| **Nexum** | Complexity classifier MLP | Automatic task decomposition + Oracle sub-tasks | `oracle` + `reasoning` | ⏳ Deferred |

### Phase 4 Wiring Detail

| Crate | What changed | File |
|-------|-------------|------|
| **Omnis** | Custom MLP router replaced with `nexora_has_moe_ffn::Router` (real learned gating, top-2, Xavier init). Prompt embedding averaged → `Router::forward()` → per-domain probabilities. | `crates/models/src/omnis/router.rs` |
| **Vortex** | `CodeVerifierManager::new()` → `verify_detailed(code, lang)` → inject `[Verifier findings]` block. Runs 4 real rule-based verifiers. | `crates/models/src/vortex/delegation.rs` |
| **Cipher** | `CodeVerifierManager::verify_detailed()` for general text security scan. Findings injected as `[Oracle security scan: ...]` in prompt. | `crates/models/src/cipher/delegation.rs` |
| **Spectra** | `CaffeineProcessor::process_multimodal()` via `nexora-multimodal` crate. Multimodal summary injected alongside style-framed creative generation. | `crates/models/src/spectra/delegation.rs` |
| **Swift** | `nexora_has_moe_ffn::Router` creates expert routing (5 experts → task types). MoE weights determine best expert path. Latency-aware dispatch via router weights. | `crates/models/src/swift/delegation.rs` |

### Phase 4 Decisions

1. **MoE gating repurposed for sequence-level routing**: Per-token routing (`O(num_tokens * experts * hidden)`) too expensive for delegation hot path. Use averaged prompt embedding → `Router::forward()` → per-expert probabilities.
2. **SACA deferred for Axiom/Genesis/Nexum**: SACA pipeline is code-specific (`CodingTask`, execute-fail-fix, rerank). Not suitable for general reasoning/quality/complexity tasks until a non-code API exists.
3. **Aether multimodal deferred**: `Caffeine` encoders require concrete input tensors (image pixel data, audio samples). No lightweight sentiment-only API available without full pipeline overhead.
4. **Kronos temporal reasoning deferred**: Requires both `reasoning` (code-specific) and `multimodal` (heavyweight). Cleanest approach waits for a dedicated temporal reasoning module.

## Phase 5a — Memory Architecture (Paged Cache + Prefix DAG)

### Completed (27 Mei 2026)

| Component | What changed | File |
|-----------|-------------|------|
| **PagedKVCacheProvider** | New `KVCacheProvider` impl wrapping `PagedKVCache`. Bridges block-based paged cache with engine's trait-based cache interface. Lazy flat mirror for `as_cpu_entries()` backward compat. | `crates/inference/src/paged_provider.rs` |
| **Shared paged cache pool** | `ContinuousBatchingEngine.shared_paged: Arc<Mutex<PagedKVCache>>`. All sequences share the same block pool via `PagedKVCacheProvider::new_shared()`. Initialized in `set_model_dims()`. | `crates/inference/src/continuous_batching.rs` |
| **`share_prefix_in_blocks`** | Block-level prefix sharing: copies block table entries from source to destination sequence, increments `ref_count` on shared physical blocks. Subsequent divergent appends trigger copy-on-write deep copy in `get_or_alloc_block`. | `crates/inference/src/paged_cache.rs` |
| **`try_paged_prefix_sharing`** | Wired into CB engine's Phase 0 (pre-share) alongside existing GPU prefix sharing. Finds shared prefix via `PrefixTrie`, calls `share_prefix_in_blocks`, advances `prompt_pos`. | `crates/inference/src/continuous_batching.rs` |
| **Engine dimensions unconditional** | `num_layers`, `num_kv_heads`, `head_dim`, `max_seq_len` moved out of `#[cfg(feature = "gpu")]` — needed by paged cache regardless of GPU. | `crates/inference/src/continuous_batching.rs` |
| **Config toggle** | `ContinuousBatchingConfig.use_paged_cache`, `paged_block_size`, `paged_max_blocks`. | `crates/inference/src/continuous_batching.rs` |

### How PrefixDAG works

```
Sequence A (prompt=abcdef):  blocks [0:a, 1:b, 2:c, 3:d, 4:e, 5:f]  ref_count=1 each
Sequence B (prompt=abcxyz):  prefix_trie finds match at depth 3 ("abc")
                             share_prefix_in_blocks copies block 0,1,2 from A→B
                             B's block table: [0→phys0, 1→phys1, 2→phys2]
                             phys0, phys1, phys2 ref_count=2 now
                             When B appends token 3, get_or_alloc_block sees ref_count>1
                             → deep copy block 3 for B (copy-on-write)
                             A retains original block 3, unaffected
```

### Next Steps (5b+)
- **Distributed scheduler**: Multi-node request dispatch with load-aware routing.
- **Observability**: KV fragmentation ratio, token/sec tracing.
- **Agent ecosystem**: Planner-worker hierarchy with persistent state.

## Notable quirks

- `Cargo.lock` in `.gitignore` — every `cargo build` resolves from scratch unless a lockfile exists locally.
- No `rustfmt.toml`, `clippy.toml`, `deny.toml`, or Makefile.
- No CI workflows (no `.github/workflows/`).
- No `build.rs` files anywhere.
- `crates/data/raw/*.arrow` in `.gitignore`.
- Several crates have feature-gated modules: always check `[features]` before assuming something is included.
- App library (`apps/nexora-ai/src/lib.rs`) re-exports from `config::NexoraConfig`, `server::NexoraServer`, `api::NexoraApi` — models in `app/config/`, `app/server/`, `app/api/` dirs.
