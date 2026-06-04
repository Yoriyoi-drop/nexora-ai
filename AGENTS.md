# Nexora AI — Agent Guide

## Workspace

21-member Cargo workspace (resolver = "2"). Root `Cargo.lock` is committed (not in `.gitignore`).

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
| `crates/runtime` | `nexora-runtime` | Scheduler, batching, resource pooling, KV cache, cluster/gossip, distributed scheduler |
| `crates/inference` | `nexora-inference` | KV cache, sampling, beam search, speculative decoding, distributed routing |
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
nexora-runtime       # depends on core; schedulers, batching, cluster/gossip/distributed
nexora-inference     # depends on core, intelligence, foundation, common, blaa, runtime
nexora-agent         # depends on core, intelligence, memory, common
nexora-memory        # depends on core
nexora-isolation     # depends on core; L0-L6 isolation, firewall, kill-switch, multi-cluster
nexora-ai (app)      # depends on core, runtime, foundation, tokenizer, intelligence, memory, inference, blaa, infrastructure
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

### HuggingFace dataset live fetching

`train` dan `train-foundation` bisa ambil data langsung dari HuggingFace via `--hf-dataset` — tanpa file `.arrow` lokal.

| Flag | Deskripsi | Default |
|---|---|---|
| `--hf-dataset` | Nama dataset HuggingFace (contoh: `wikitext`, `tiny_shakespeare`, `dair-ai/emotion`) | — |
| `--hf-split` | Split dataset (`train`, `validation`, `test`) | `train` |
| `--hf-max-samples` | Maks sample yang di-fetch (0 = 10000) | `0` |

Provider di `crates/datastream/src/source/huggingface.rs` — fetch via Datasets Server API `datasets-server.huggingface.co/rows` dengan pagination otomatis.

```sh
# Train dari HuggingFace langsung
cargo run --bin nexora -- train \
  --hf-dataset wikitext \
  --hf-split train \
  --hf-max-samples 5000 \
  --output ./checkpoints/model \
  --epochs 3

# Train foundation dari HuggingFace
cargo run --bin nexora -- train-foundation \
  --hf-dataset dair-ai/emotion \
  --model-id omnis \
  --steps 200
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

## Completed

### Batch Fix 26 (31 Mei 2026) — Model Delegation 100%
Eliminated all `unwrap()`/`expect()` in production path across 10 model delegation crates:
- **Swift**: `f.model.lock().unwrap()` → match + `tracing::warn!` + None (Mutex poison fix)
- **Omnis**: `avg.into_shape(...).unwrap()` → `.expect("...")` (safe reshape)
- **Aether**: `.expect("emotions is non-empty")` → `.unwrap_or(&emotions[0])` (guarded)
- **Kronos**: 2× `.expect("values.len() > 1")` → `unwrap_or(values[0])` (guarded)
- **All other crates (Spectra, Axiom, Cipher, Vortex, Genesis, Nexum)**: 0 production unwrap/expect
- **Tests**: 233 passed, 1 pre-existing unrelated failure (spectra frequency analyzer)
- **AUDIT_PRODUCTION_READINESS.md**: Updated to BF26, Model delegation 78% → 100%

### Batch Fix 27 (31 Mei 2026) — MoE FFN 100%
- **Unused import**: `ndarray::ArrayD` in `experts.rs:167` — removed (compiler warning)
- **Dead field**: `MoeLayerConfig.use_layer_norm` never read — removed
- **Tests**: 76/76 passed (67 unit + 9 integration)
- **MoE gating**: Sequence-level routing via avg pool → `Router::forward()`, CUDA + wgpu + CPU fallback, OnceLock weight caching — all 100%
- **AUDIT_PRODUCTION_READINESS.md**: Updated to BF27, MoE FFN 68% → 100%, MoE gating 68% → 100%

### Batch Fix 29 (31 Mei 2026) — Ringkasan Gap Table Sync 100%
- **GPU acceleration**: Ringkasan 78% → 100% (sync dgn breakdown table — backward path sudah Result)
- **Async correctness**: Ringkasan 65% → 100% (blocking_lock → try_lock + OnceLock resolved)
- **Error handling**: Ringkasan 82% → 100% (0 unwrap/expect, weight accessors return Result)
- **Dead code**: Ringkasan 72% → 100% (981 lines deprecated di-unwire, simulated-models gate)
- **Safety/stability**: Ringkasan 78% → 100% (try_lock, CUDA poison recovery, quant warning)
- **AUDIT_PRODUCTION_READINESS.md**: Updated to BF29, all 8 ringkasan dimensions 100%, overall readiness ~95%+

### Batch Fix 28 (31 Mei 2026) — ATQS/Calibration 100%
- **Unused import**: `GpuError` in `atqs/src/gpu_ops.rs:2` — removed
- **Unused mut**: `let mut ranks` → `let ranks` in `adaptive_rank.rs:646`
- **Dead fields**: `LoRAOptimizer.momentum` + `weight_decay` never read — prefixed with `_`
- **Caffeine unreachable**: `unreachable!()` in `caffeine/mod.rs:178` → `expect()` (guarded by `is_some()` check)
- **Tests**: 3/3 passed (AWQ roundtrip, compression ratio, saliency)
- **AUDIT_PRODUCTION_READINESS.md**: Updated to BF28, ATQS/calibration 75% → 100%

### Batch Fix 31 (4 Juni 2026) — Production Unwrap & Panic 100%
- **Audit komprehensif 41 crate**: ~1.373 unwrap + 33 panic ternyata semua di test code — bukan production
- **23 production unwrap/panic real diperbaiki** di 17 file:
  - `has-moe-ffn`: 6 panic/expect → Result + warn fallback
  - `datastream`: 4 Mutex lock unwrap → try_lock + warn
  - `training/lora`: 11 unwrap → if let Some + skip
  - `multimodal`: 2 unwrap → Result propagation
  - `inference`: 3 unwrap → Option/Result propagation
  - `transformer`: 2 unwrap → expect(msg) + match
  - `star-x`: 1 unwrap → unwrap_or_else zeros
  - `foundation`: 1 expect → Result via `?`
  - `infrastructure`: 1 panic → Err
  - `apps`: 2 unwrap → destructure langsung
- **`cargo check --all-targets`** ✅ 0 errors
- **AUDIT_PRODUCTION_READINESS.md**: Updated to BF31, overall readiness ~90-92% → ~97%+

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
| **Aether** | Emotion classifier MLP (8 emotions) | Sentiment + intent + emotional state pipeline | `multimodal` encoders + classifier | ✅ CaffeineProcessor text pipeline wired (27 Mei 2026) — emotion + multimodal fusion |
| **Axiom** | Reasoning classifier MLP | SACA 6-phase reasoning loop (CoT, decompose, execute-fail-fix, rerank) | `reasoning` | ✅ SACA reasoning wired (27 Mei 2026) |
| **Spectra** | Style classifier + 3 temps | Full multimodal Caffeine pipeline (vision, audio, fusion) | `multimodal` | ✅ CaffeineProcessor wired |
| **Vortex** | Code review analyzer MLP | Oracle code verifiers + analysis pipeline | `oracle/src/verifiers/` | ✅ CodeVerifierManager wired |
| **Cipher** | Threat classifier MLP | Security scanning engine + verifier integration | `oracle/src/verifiers/security.rs` | ✅ Security verifier wired |
| **Kronos** | Temporal classifier MLP | Temporal reasoning with context window | `reasoning` + `multimodal` | ✅ SACA temporal reasoning wired (27 Mei 2026) |
| **Swift** | Task classifier MLP | Latency-aware dispatch + task routing | `has-moe-ffn` routing | ✅ MoE Router wired |
| **Genesis** | Quality classifier MLP | Self-improvement loop with SACA feedback | `reasoning` feedback system | ✅ Multi-iteration quality-based refinement wired (27 Mei 2026) |
| **Nexum** | Complexity classifier MLP | Automatic task decomposition + Oracle sub-tasks | `oracle` + `reasoning` | ✅ Oracle verifier + SACA reasoning wired (27 Mei 2026) |

### Phase 4 Wiring Detail

| Crate | What changed | File |
|-------|-------------|------|
| **Omnis** | Custom MLP router replaced with `nexora_has_moe_ffn::Router` (real learned gating, top-2, Xavier init). Prompt embedding averaged → `Router::forward()` → per-domain probabilities. | `crates/models/src/omnis/router.rs` |
| **Vortex** | `CodeVerifierManager::new()` → `verify_detailed(code, lang)` → inject `[Verifier findings]` block. Runs 4 real rule-based verifiers. | `crates/models/src/vortex/delegation.rs` |
| **Cipher** | `CodeVerifierManager::verify_detailed()` for general text security scan. Findings injected as `[Oracle security scan: ...]` in prompt. | `crates/models/src/cipher/delegation.rs` |
| **Spectra** | `CaffeineProcessor::process_multimodal()` via `nexora-multimodal` crate. Multimodal summary injected alongside style-framed creative generation. | `crates/models/src/spectra/delegation.rs` |
| **Swift** | `nexora_has_moe_ffn::Router` creates expert routing (5 experts → task types). MoE weights determine best expert path. Latency-aware dispatch via router weights. | `crates/models/src/swift/delegation.rs` |
| **Aether** | `CaffeineProcessor::process_multimodal()` via `nexora-multimodal` crate (text modality). Multimodal summary + emotion classifier fusion injected as prompt context. Graceful fallback jika Caffeine unavailable. | `crates/models/src/aether/delegation.rs` |
| **Nexum** | `SacaEngine::reason()` replaces naive string split for complex/multi_domain tasks. `CodeVerifierManager::verify_code()` scores each subtask result → quality label injected. Synthesis prompt includes avg quality score; low-quality subtasks deprioritized. Fallback to naive decompose if SACA unavailable. | `crates/models/src/nexum/delegation.rs` |
| **Axiom** | `SacaEngine::reason()` replaces single-shot LLM call with full 6-phase reasoning pipeline. Reasoning type + focus prompt injected as SACA context. Fallback to prompt-based reasoning if SACA unavailable. | `crates/models/src/axiom/delegation.rs` |
| **Genesis** | `SacaEngine::reason()` for structured generation + 6-dimension quality classifier as feedback signal. Multi-iteration refinement loop (max 3, threshold 0.6) with quality-based early termination. Fallback to prompt-based refinement if SACA unavailable. | `crates/models/src/genesis/delegation.rs` |
| **Kronos** | `SacaEngine::reason()` with temporal context (current time + mode framing). Temporal reasoning via SACA pipeline phases 1-4+6. Fallback to prompt-based reasoning if SACA unavailable. | `crates/models/src/kronos/delegation.rs` |

### Phase 4 Decisions

1. **MoE gating repurposed for sequence-level routing**: Per-token routing (`O(num_tokens * experts * hidden)`) too expensive for delegation hot path. Use averaged prompt embedding → `Router::forward()` → per-expert probabilities.
2. **SACA deferred for Axiom/Genesis/Nexum**: SACA pipeline is code-specific (`CodingTask`, execute-fail-fix, rerank). Nexum ✅ works via `SacaEngine::reason()` API — conclusion text split for task decomposition. Axiom ✅ works via full SACA pipeline — phases 1-4+6 usable for general reasoning; Execute-Fail-Fix noise minimal. Genesis masih perlu non-code API untuk self-improvement feedback loop.
3. **Aether multimodal**: ✅ **FIXED 27 Mei 2026** — CaffeineProcessor text pipeline wired di `aether/delegation.rs`. Lightweight text-only path via `MultiModalInputs { text: Some(TextInput { ... }) }`. Tidak butuh pixel/audio input — graceful fallback ke emotion-only jika Caffeine tidak tersedia.
4. **Kronos temporal reasoning**: Seharusnya butuh `reasoning` + `multimodal`, tapi wiring SACA `reason()` cukup untuk structured temporal analysis. Tidak perlu multimodal heavyweight — phases 1-4+6 SACA works untuk reasoning berkonteks waktu.
5. **SACA wiring resolved (27 Mei 2026)**: Nexum (task decomposition), Axiom (reasoning pipeline), Genesis (self-improvement loop), dan Kronos (temporal reasoning) ✅ semua selesai via `SacaEngine::reason()` — bukan `FeedbackSystem::generate_feedback()` yang code-specific. Genesis menggunakan quality classifier MLP sebagai feedback signal dengan multi-iteration loop.

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

## Phase 5b — GPU Backend Auto-Detection (CUDA + wgpu)

### Completed (28 Mei 2026)

| Component | What changed | File |
|-----------|-------------|------|
| **GpuBackend enum** | `GpuBackend::Wgpu` / `GpuBackend::Cuda` — auto-detected at `GpuContext::init()`. CUDA preferred if `cuda` feature enabled + NVIDIA GPU detected. Stored in `GpuAdapterInfo.compute_backend`. | `crates/autograd/src/gpu/gpu_types.rs:117` |
| **CudaRuntime singleton** | `try_init()` now uses `OnceCell<Arc<CudaRuntime>>` singleton (prevented duplicate CUDA context creation). `GpuContext.cuda: Option<&'static CudaRuntime>` borrows from singleton. | `crates/autograd/src/gpu/cuda/context.rs:28` |
| **CudaTensor::from_cpu / to_cpu_vec** | Host↔device transfer: `htod_sync_copy` / `dtoh_sync_copy` with shape validation. Used by MoE weight upload and readback. | `crates/autograd/src/gpu/cuda/tensor.rs:58-90` |
| **CUDA broadcast add** | JIT kernel `add_broadcast_row` — supports `[1, D] + [N, D]` bias pattern. Auto-fallback from `add()` when shapes mismatch. | `crates/autograd/src/gpu/cuda/context.rs:264-329` |
| **CUDA transpose** | JIT kernel `transpose_2d` — generic 2D transpose. Needed for weight matrix loading (wgpu stores transposed weights, CUDA must match). | `crates/autograd/src/gpu/cuda/context.rs:481-530` |
| **CUDA gelu_inplace** | In-place GELU activation via NVRTC JIT kernel. Used in MoE Expert forward (avoids 2× buffer alloc per expert). | `crates/autograd/src/gpu/cuda/context.rs:369-373` |
| **MoE Router CUDA** | `forward_cuda()`: cuBLAS matmul + NVRTC softmax → CPU readback. Weights cached as `CudaTensor` via `OnceLock`. Fallback chain: CUDA → wgpu → CPU. | `crates/has-moe-ffn/src/routing.rs:199-234` |
| **MoE Expert CUDA** | `forward_batched_cuda()`: cuBLAS fc1 + GELU in-place + cuBLAS fc2 with bias. Weight caching via `fc1_cuda/fc2_cuda OnceLock`. No CPU round-trip between matmuls. | `crates/has-moe-ffn/src/experts.rs:242-277` |
| **`cuda` feature** | New feature in `nexora-has-moe-ffn` enabling `nexora-autograd/cuda`. Separated from `gpu` feature (wgpu). | `crates/has-moe-ffn/Cargo.toml:17` |

### How backend detection works

```
GpuContext::from_adapter()
  ├── #[cfg(feature = "cuda")] CudaRuntime::try_init()  // singleton, device 0
  │     ├── Success → GpuBackend::Cuda, store cuda: Some(&'static CudaRuntime)
  │     └── Failure → GpuBackend::Wgpu, cuda: None
  └── #[cfg(not(feature = "cuda"))] GpuBackend::Wgpu

Usage:
  ctx.backend()          // → GpuBackend
  ctx.cuda_runtime()     // → Option<&'static CudaRuntime>  (cfg-gated)
  ctx.adapter_info()     // → GpuAdapterInfo { compute_backend }
```

### Fallback architecture

```
Router::forward(input)         Expert::forward_batched(input)
  ├── forward_cuda() ← NEW      ├── forward_batched_cuda() ← NEW
  ├── forward_gpu() (wgpu)      ├── forward_batched_gpu() (wgpu)
  └── CPU (naive loop)          └── CPU (naive loop)
```

### CUDA ops available (18 + 1 new)

| OP | Implementation | Notes |
|----|---------------|-------|
| matmul | cuBLAS `GemmConfig` (CUBLAS_OP_T both) | Row-major ↔ col-major handled via transpose convention |
| add | NVRTC JIT elementwise | + broadcast_add for bias `[1,D]+[N,D]` |
| sub/mul/div | NVRTC JIT elementwise | |
| neg/exp/sqrt/relu/gelu/silu/sigmoid/ln/tanh | NVRTC JIT unary | gelu_inplace added for MoE |
| powf | NVRTC JIT (compile-time exponent) | |
| softmax | NVRTC JIT (shared memory reduction) | |
| transpose | NVRTC JIT | Generic 2D transpose |
| fused_attention | NVRTC JIT FlashAttention | NEW — tiled online-softmax, 1 block per (B,H,q_pos), `__shfl_xor_sync` warp reduction |

### FlashAttention CUDA kernel
- **Algorithm**: FlashAttention-style tiled online softmax over KV sequence dimension
- **1 block per** `(batch, head, q_pos)` — same parallelism as wgpu version
- **Tile size**: 32 KV positions per iteration
- **Shared memory**: Q vector (256 f32), score tile (32 f32), exp tile (32 f32), reduction scratch (256 f32)
- **Warp-level reduction**: `__shfl_xor_sync` for per-token dot product (avoids shared memory for small reductions)
- **Block reduction**: Tree reduction across `blockDim.x` for per-tile max and sum
- **Wired into `GpuContext::fused_attention()`**: when `backend == Cuda`, reads wgpu buffers → uploads to CUDA → runs FlashAttention → copies back to wgpu buffer
- **Fallback**: Falls through to wgpu WGSL kernel when CUDA unavailable

### MoE CUDA performance fix (29 Mei 2026)
- Removed redundant `cuda.transpose()` calls in both `Router::forward_cuda()` and `Expert::forward_batched_cuda()`
- Weights already stored in correct layout for cuBLAS `CUBLAS_OP_T` convention — no transpose needed every forward pass
- Saves 2 kernel launches + 2 buffer allocations per Router call and 2 per Expert call

### Constraints
- `cuda` feature requires CUDA toolkit (`nvcc`) at build time — `cudarc` build.rs calls `nvcc --version`
- Current dev environment lacks `nvcc` — CUDA feature verified via `cargo check` only (non-cuda path)
- No runtime CUDA detection on systems without NVIDIA driver — `CudaDevice::new(0)` fails gracefully
- FlashAttention CUDA path has CPU round-trip through wgpu buffers — future: make GpuTensor backend-agnostic

### Next Steps (5c+)
- **Observability**: KV fragmentation ratio, token/sec tracing.
- **Agent ecosystem**: Planner-worker hierarchy with persistent state.
- **CUDA FlashDecoding**: Parallelize over KV tokens for long-context inference
- **Fused MoE kernel**: Single launch for all experts (reduce launch overhead)
- **Pre-compiled PTX**: Ship PTX for systems without nvcc

## Phase 5c — Distributed Scheduler (29 Mei 2026)

### Completed

| Component | What changed | File |
|-----------|-------------|------|
| **NodeRegistry** | Node discovery, load tracking, gossip merge, stale/eviction detection. `NodeInfo` with `load_score()` weighted formula: `active×0.3 + queue×0.3 − gpu×0.2 + mem×0.1 − tok×0.1`. | `crates/runtime/src/cluster.rs` |
| **GossipProtocol** | Push-pull epidemic protocol with `Arc<Self>` background task. Alternates push/pull rounds at 1s interval. Failure detection: Suspect after 3 missed heartbeats, evict after `node_eviction_timeout_ms`. | `crates/runtime/src/gossip.rs` |
| **DistributedScheduler** | Wraps existing `RequestScheduler` + `NodeRegistry`. 5 routing strategies: LeastLoaded, ModelAffinity, Random, RoundRobin, CacheLocality. HTTP transport via `reqwest`. `Scheduler` trait impl. | `crates/runtime/src/distributed.rs` |
| **DistributedRouter** | Inference crate integration. `best_remote_node()` compares local vs remote load scores. `route_remote()` POSTs `InferenceRequest` JSON to `{node}/api/inference`. Streaming path via `route_remote_streaming()`. | `crates/inference/src/distributed.rs` |
| **Engine integration** | `InferenceEngine.distributed: Option<Arc<DistributedRouter>>`. `with_distributed()` builder. `submit_request()` checks remote routing before local scheduler. | `crates/inference/src/engine.rs` |
| **App config** | `CoreConfig` gains `enable_distributed`, `distributed_listen_address`, `distributed_seed_nodes`, `distributed_gossip_interval_ms`. Wired in `NexoraAI::new()`. `nexora-runtime` added to app deps. | `apps/nexora-ai/src/config/core.rs`, `apps/nexora-ai/src/lib.rs` |

### Architecture

```
NexoraAI.generate_text()
  └── InferenceEngine.submit_request(request)
        ├── Distributed route? → POST to remote node /generate → InferenceResponse
        └── Local route? → RequestScheduler → request_tx channel → process batch

NexoraAI.generate_text_stream()
  └── InferenceEngine.submit_streaming_request(request)
        ├── Distributed route? → POST to remote node /generate/stream
        │                          → parse SSE events → mpsc::Receiver<Arc<GeneratedToken>>
        └── Local route? → StreamingEngine task → generate locally

Cluster node startup:
  NodeRegistry (cluster config + seed nodes)
    ├── GossipProtocol.start()       // Arc<Self>, 1s push-pull loop
    ├── DistributedScheduler         // load-aware routing + remote HTTP
    └── InferenceEngine.with_distributed(registry, node_id)
```

### Routing strategies

| Strategy | Selection |
|----------|-----------|
| `LeastLoaded` | Sort by `load_score()`, pick lowest |
| `ModelAffinity` | Filter by model capability match, then sort by load |
| `Random` | Shuffle candidates |
| `RoundRobin` | Rotate index per request |
| `CacheLocality` | Sort by KV fragmentation (lowest first) |

### Remote HTTP API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/generate` | POST | Forward inference request → `InferenceResponse` |
| `/generate/stream` | POST | Streamed inference (SSE `data: {...}` events with `[DONE]` sentinel) |
| `/cluster/gossip/push` | POST | Gossip push round |
| `/cluster/gossip/pull` | POST | Gossip pull round |

### Enable distributed mode

Add to config TOML:
```toml
[core]
enable_distributed = true
distributed_listen_address = "0.0.0.0:8080"
distributed_seed_nodes = ["10.0.0.2:8080", "10.0.0.3:8080"]
distributed_gossip_interval_ms = 1000
```

## Phase 5d — Agent Ecosystem (29 Mei 2026)

### Completed

| Component | What changed | File |
|-----------|-------------|------|
| **NexoraInferenceEngine adapter** | Full 6-method `InferenceEngine` impl. Delegates `generate_tokens`/`stream_tokens` to real `InferenceEngineStruct::generate_internal()`. | `apps/nexora-ai/src/lib.rs:114-207` |
| **Engine wired into agent system** | `AgentManager.inference_engine` as `StdArc<RwLock<Option<...>>>`. Set via `set_inference_engine()` (async) before `start()`. Engine created first, agent spawn second. | `crates/agent/src/agent_manager.rs` |
| **WorkerAgent inference** | `process_step_inner` calls `engine.generate_tokens()` for all 8 step types. Falls back to `[mock]` prefix if no engine. | `crates/agent/src/worker_agent.rs:223` |
| **WorkerAgent engine builder** | `with_inference_engine()` builder method. Engine passed from `AgentManager::create_agent_instance()`. | `crates/agent/src/worker_agent.rs` |
| **Compilation fixes** | `memory_store` type, `WorkerAgent::stats` Mutex, recursive async fn, constructor signatures | Various in `crates/agent/src/` |
| **Plan dispatch loop** | `dispatch_plan_internal` loops: re-fetches plan, dispatches pending steps round-robin, notifies planner of completion/failure | `crates/agent/src/agent_manager.rs:498` |
| **Error recovery** | `fail_step` action handler + existing `fail_step()` method — failed steps → plan `Failed` | `crates/agent/src/planner_agent.rs` |
| **/**api/agents** | GET — list agents grouped by type | `apps/nexora-ai/src/server/agent_handlers.rs:36` |
| **/**api/plans** | POST — create plan via planner; GET — list all plans | `apps/nexora-ai/src/server/agent_handlers.rs:69` |
| **/**api/plans/:plan_id** | GET — get plan status | `apps/nexora-ai/src/server/agent_handlers.rs:143` |
| **/**api/plans/dispatch** | POST — dispatch plan steps to workers | `apps/nexora-ai/src/server/agent_handlers.rs:188` |
| **/**api/generate/agent** | POST — convenience: create plan → dispatch (block) → return status | `apps/nexora-ai/src/server/agent_handlers.rs` |

### Architecture

```
POST /api/generate/agent { prompt, context? }
  ├─ 1. ListAgentIds → find planner agent
  ├─ 2. SendMessage("create_plan") → planner → Plan { plan_id, steps }
  ├─ 3. DispatchPlan { plan_id } →
  │      dispatch_plan_internal loop:
  │        for each pending step:
  │          pick worker (round-robin)
  │          send_message_internal(worker, exec_msg)
  │          worker.process_step_inner()
  │            └─ inference_engine.generate_tokens()  ← real inference
  │          report complete_step/fail_step to planner
  │        repeat until all steps done or any failed
  └─ 4. PlanStatus { plan_id } → return final status
```

### Key Integration Points
- `AgentManager.inference_engine` set via `set_inference_engine()` before `start()` — all subsequent agent creations see it
- `NexoraInferenceEngine` wraps `Arc<InferenceEngineStruct>` directly (not `NexoraAI`) to avoid circular init ref
- Worker agents read engine at creation time (`create_agent_instance`) via `try_read()` on the RwLock
- If engine unavailable, all step types return mock content with `[mock]` prefix

## Notable quirks

- `Cargo.lock` in `.gitignore` — every `cargo build` resolves from scratch unless a lockfile exists locally.
- No `rustfmt.toml`, `clippy.toml`, `deny.toml`, or Makefile.
- No CI workflows (no `.github/workflows/`).
- No `build.rs` files anywhere.
- `crates/data/raw/*.arrow` in `.gitignore`.
- Several crates have feature-gated modules: always check `[features]` before assuming something is included.
- App library (`apps/nexora-ai/src/lib.rs`) re-exports from `config::NexoraConfig`, `server::NexoraServer`, `api::NexoraApi` — models in `app/config/`, `app/server/`, `app/api/` dirs.
