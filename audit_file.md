# Audit SRP — File dengan >1 Tanggung Jawab

## Ringkasan

| Sektor | Total File | SRV Violation | Persentase |
|--------|-----------|---------------|------------|
| `crates/` (non-model) | ~890 | ~127 | 14% |
| `crates/models/` | 122 | 81 | 66% |
| `apps/` | 70 | 17 | 24% |
| Root + config | ~25 | 16 | 64% |
| **Total** | ~1107 | **241** | **22%** |

---

## Peta Lengkap

### KRITIS — God Object / Monolithic (>1000 lines, 5+ tanggung jawab)

| # | File | Lines | Domain | Tanggung Jawab | Saran Split |
|---|------|-------|--------|----------------|-------------|
| 1 | `crates/autograd/src/gpu/utils.rs` | 4216 | autograd | GpuContext → 100+ method: matmul, add, softmax, MHA, attention, layernorm, embedding, cross_entropy, RoPE, GELU, SiLU, bias_add, dropout, fused ops, CUDA bridge | `gpu_ops/matmul.rs`, `activation.rs`, `normalization.rs`, `attention.rs` |
| 2 | `crates/autograd/src/gpu/cuda/context.rs` | 3295 | autograd | CudaRuntime: JIT compilation, PTX disk cache, kernel execution, memory management, tensor transfer, cuBLAS, NVRTC | `cuda/runtime.rs`, `jit.rs`, `ptx_cache.rs`, `kernels/` |
| 3 | `crates/autograd/src/gpu/wgsl.rs` | 2569 | autograd | WGSL shader strings tanpa organized structure | `wgsl/matmul.rs`, `attention.rs`, `normalization.rs` |
| 4 | `crates/autograd/src/ops/nn.rs` | 2192 | autograd | Semua neural network ops: softmax, dropout, layernorm, RMS norm, BCE, cross_entropy, MSE, embedding, causal_attention | `ops/activation.rs`, `loss.rs`, `normalization.rs`, `attention.rs` |
| 5 | `crates/inference/src/engine.rs` | 2199 | inference | `InferenceConfig` + `InferenceEngine` + `EngineState` + `RequestStatus` + `EngineStats` | `engine/config.rs`, `logic.rs`, `handle.rs`, `stats.rs` |
| 6 | `crates/transformer/src/model/registry.rs` | 2257 | transformer | Config loading, safetensors I/O, KV cache init, GQA setup, model dimension compute | `model/loader.rs`, `registry.rs`, `dims.rs` |
| 7 | `crates/autograd/src/gpu_sedc.rs` | 1982 | autograd | `SedcError` + `SedcConfig` + rank selection + EGSS + GPU SVD + `CompressedWeight` + `SedcCompressor` | `gpu_sedc/config.rs`, `algorithms.rs`, `compressor.rs`, `types.rs` |
| 8 | `apps/nexora-ai/src/cli/training.rs` | 2693 | app | Training loop, checkpointing, GPU init, HTTP metrics, parallel training, ANSI colors | `training/checkpoint.rs`, `gpu.rs`, `runner.rs`, `parallel.rs`, `metrics.rs` |
| 9 | `apps/nexora-ai/src/cli/handlers.rs` | 1531 | app | CLI dispatcher 18+ command branch: config, tokenizer, health, server, train, collect, generate, chat, analyze, codegen, evaluate | `handlers/train.rs`, `server.rs`, `data.rs`, `chat.rs`, `evaluate.rs` |
| 10 | `apps/nexora-ai/src/core/debate.rs` | 1435 | app | Cost controller, capability profile, context bus, confidence calibration, verifier, orchestrator | `debate/controller.rs`, `capability.rs`, `context.rs`, `confidence.rs`, `verifier.rs`, `orchestrator.rs` |
| 11 | `apps/nexora-ai/src/cli/benchmark.rs` | 1412 | app | Metric sampling, benchmark runner, report formatter, baseline system | `benchmark/runner.rs`, `baseline.rs`, `metrics.rs`, `report.rs` |
| 12 | `apps/nexora-ai/src/server/handlers.rs` | 1202 | app | 17 handler: health, metrics, process, generate, stream, chat, code analysis, code gen, config CRUD, jailbreak detection, static files | `handlers/security.rs`, `inference.rs`, `code.rs`, `config.rs`, `admin.rs` |
| 13 | `apps/nexora-ai/src/lib.rs` | 1040 | app | Model init, inference engine, `NexoraInferenceEngine`, `NexoraAI` (25 field), text gen, chat, code, kill switch | `app.rs`, `agent_adapter.rs`, `engine_factory.rs` |
| 14 | `apps/dashboard/src/main.rs` | 991 | dashboard | 6 halaman dashboard + event loop + API fetch + system monitoring | per `page_*.rs` (already partial) |
| 15 | `apps/nexora-ai/src/core/processing.rs` | 969 | app | Input validation, type detection, process routing, code analysis, code generation + test inline | `input_detector.rs`, `request_router.rs`, `code_analyzer.rs`, `code_generator.rs` |

### Model Agents — Monolith

| # | File | Lines | Tanggung Jawab | Saran Split |
|---|------|-------|----------------|-------------|
| 16 | `crates/model-vortex/src/agents/arch_weaver.rs` | 3374 | Agent + 50+ tipe + analysis + patterns + BaseAgent impl | `types.rs`, `config.rs`, `analysis.rs`, `patterns.rs` |
| 17 | `crates/model-vortex/src/agents/code_sentinel.rs` | 3217 | Agent + 50+ tipe + review logic + quality enforcement | `types.rs`, `config.rs`, `review.rs`, `quality.rs` |
| 18 | `crates/model-vortex/src/agents/debug_phantom.rs` | 3127 | Agent + 50+ tipe + debugging logic + bug detection | `types.rs`, `config.rs`, `tracing.rs`, `bug_detection.rs` |
| 19 | `crates/model-vortex/src/agents/test_forge.rs` | 2958 | Test generation agent + 40+ tipe + generation logic | pisah types vs generation logic |
| 20 | `crates/model-nexum/src/agents/resource_optimizer.rs` | 1876 | Resource optimizer + scheduling types | pisah types vs logic |
| 21 | `crates/model-axiom/src/config.rs` | 2086 | 100+ config types + validation + `Default` | pisah per domain + `validator.rs` |
| 22 | `crates/model-axiom/src/identity.rs` | 1565 | 40+ tipe identity + reasoning + logical + proof | `logical.rs`, `proof.rs`, `reasoning.rs`, `summary.rs` |
| 23 | `crates/model-nexum/src/config.rs` | 1250 | 50+ config types + orchestration + resource + alignment | pisah per subsystem |
| 24 | `crates/model-nexum/src/capabilities.rs` | 1192 | 100+ capability spec builder | pecah per domain |
| 25 | `crates/model-nexum/src/agents/consensus_builder.rs` | 1187 | Consensus agent + types | pisah types vs builder |
| 26 | `crates/model-nexum/src/agents/mod.rs` | 1017 | `mod.rs` berisi 6 full agent struct + 30+ method | pisah tiap agent ke file sendiri |
| 27 | `crates/model-aether/src/config.rs` | 1011 | Config + logic + macro | `macros.rs`, `config/logic.rs` |

### TINGGI — Multiple Concerns >1000 lines

| # | File | Lines | Saran Split |
|---|------|-------|-------------|
| 28 | `crates/database/src/lib.rs` | 1125 | per backend: `mysql.rs`, `postgres.rs`, `sqlite.rs` |
| 29 | `crates/agent/src/agent_manager.rs` | 1155 | `manager/config.rs`, `lifecycle.rs`, `commands.rs`, `scaling.rs` |
| 30 | `crates/agent/src/planner_agent.rs` | 1400 | `planner/types.rs`, `strategies.rs`, `agent.rs` |
| 31 | `crates/agent/src/context_agent.rs` | 1284 | `context/types.rs`, `strategies.rs`, `agent.rs` |
| 32 | `crates/agent/src/memory_agent.rs` | 1102 | `memory_agent/types.rs`, `agent.rs` |
| 33 | `crates/agent/src/state.rs` | 1028 | `state/global.rs`, `session.rs`, `agent.rs`, `events.rs` |
| 34 | `crates/agent/src/validation_agent.rs` | 1075 | `validation/types.rs`, `rules.rs`, `agent.rs` |
| 35 | `crates/agent/src/response_agent.rs` | 1020 | `response/formatters.rs`, `agent.rs` |
| 36 | `crates/memory/src/memory_model.rs` | 1310 | `memory/types.rs`, `manager.rs`, `cognitive.rs` |
| 37 | `crates/oracle/src/backbone.rs` | 1249 | `backbone/config.rs`, `attention.rs`, `layers.rs`, `compression.rs` |
| 38 | `crates/oracle/src/code_utils.rs` | 1176 | `code_utils/tokenizer.rs`, `parser.rs`, `ast.rs`, `formatter.rs`, `metrics.rs` |
| 39 | `crates/oracle/src/alignment.rs` | 973 | `alignment/dpo.rs`, `analyzers.rs`, `stats.rs` |
| 40 | `crates/oracle/src/trainer.rs` | 913 | `trainer/config.rs`, `state.rs`, `train.rs` |
| 41 | `crates/oracle/src/pretraining.rs` | 841 | `pretraining/config.rs`, `fim.rs`, `contrastive.rs` |
| 42 | `crates/reasoning/src/saca/cot.rs` | 1564 | `cot/generator.rs`, `verifier.rs`, `engine.rs` |
| 43 | `crates/reasoning/src/saca/rerank.rs` | 969 | `rerank/scorer.rs`, `ranker.rs` |
| 44 | `crates/inference/src/sampler.rs` | 1116 | `sampler/methods.rs`, `sampler.rs`, `advanced.rs`, `stats.rs` |
| 45 | `crates/inference/src/paged_cache/cache.rs` | 1103 | `paged_cache/cache.rs`, `eviction.rs`, `storage.rs` |
| 46 | `crates/inference/src/runtime.rs` | 861 | `runtime/config.rs`, `state.rs`, `metrics.rs`, `events.rs` |
| 47 | `crates/inference/src/prefix_cache.rs` | 897 | `prefix_cache/radix.rs`, `cache.rs`, `config.rs` |
| 48 | `crates/has-moe-ffn/src/experts.rs` | 1155 | `experts/forward.rs`, `weights.rs`, `activation.rs` |
| 49 | `crates/has-moe-ffn/src/routing.rs` | 993 | `routing/gate.rs`, `weights.rs`, `cuda.rs` |
| 50 | `crates/star-x/src/blas_backend.rs` | 1498 | `blas/backend.rs`, `operations.rs`, `init.rs` |
| 51 | `crates/star-x/src/fused_ops.rs` | 859 | `fused/linear.rs`, `attention.rs`, `elementwise.rs` |
| 52 | `crates/star-x/src/quantization.rs` | 822 | `quant/types.rs`, `engine.rs`, `mixed.rs`, `stats.rs` |
| 53 | `crates/training/src/lib.rs` | 1065 | `training/config.rs`, `loop.rs`, `utils.rs` |
| 54 | `crates/foundation/src/causal_lm_model/mod.rs` | 1027 | `causal_lm/model.rs`, `forward.rs`, `nxr_impl.rs` |
| 55 | `crates/transformer/src/trainable.rs` | 1011 | `trainable/config.rs`, `layers.rs`, `model.rs` |
| 56 | `crates/atqs/src/calibration/calibration_optimizer.rs` | 1147 | 5 optimizer: `adam.rs`, `sgd.rs`, `adagrad.rs`, `rmsprop.rs`, `lamb.rs` |
| 57 | `crates/erp/src/training.rs` | 1278 | `training/loop.rs`, `metrics.rs`, `checkpoint.rs` |
| 58 | `crates/erp/src/cache.rs` | 942 | `cache/inference.rs`, `hybrid.rs`, `pattern.rs`, `hash.rs` |
| 59 | `crates/echo-net/src/tkrr.rs` | 917 | `tkrr/candidate.rs`, `routing.rs`, `stats.rs` |
| 60 | `crates/echo-net/src/training.rs` | 951 | pisah training logic vs metrics |
| 61 | `crates/multimodal/src/caffeine/qformer/mod.rs` | 1013 | `qformer/queries.rs`, `attention.rs`, `cross_modal.rs`, `stats.rs` |
| 62 | `crates/multimodal/src/caffeine/cache.rs` | 1099 | `cache/types.rs`, `storage.rs`, `video.rs`, `compress.rs` |
| 63 | `crates/multimodal/src/caffeine/action_head/execution.rs` | 906 | `action_head/engine.rs`, `handlers.rs`, `records.rs` |
| 64 | `crates/infrastructure/common/src/error.rs` | 997 | `error/types.rs`, `recovery.rs`, `circuit_breaker.rs`, `retry.rs` |
| 65 | `crates/vogp/src/utils.rs` | 727 | `utils/augmentation.rs`, `gradient.rs`, `metrics.rs`, `memory.rs` |
| 66 | `crates/transformer/src/gqa/gpu.rs` | 1278 | `gqa/gpu/attention.rs`, `memory.rs` |
| 67 | `crates/shared/src/model_config.rs` | 907 | `model_config/arch.rs`, `training.rs`, `inference.rs` |
| 68 | `crates/model-core/src/foundation.rs` | 811 | `config.rs`, `model.rs`, `inference.rs`, `tokenizer.rs`, `error.rs` |

### SEDANG — Cross-domain mix 500-1000 lines

| # | File | Lines | Masalah |
|---|------|-------|---------|
| 69 | `crates/agent/src/worker_agent.rs` | 978 | Worker + inference + step processing |
| 70 | `crates/agent/src/communication.rs` | 693 | Message types + routing + bus impl |
| 71 | `crates/memory/src/episodic.rs` | 877 | Episode type + store + stats |
| 72 | `crates/memory/src/layers.rs` | 810 | Layer types + operations |
| 73 | `crates/memory/src/lib.rs` | 817 | Re-export + struct + DB bridge |
| 74 | `crates/inference/src/decoding.rs` | 709 | Config + trait + 4 strategy impls |
| 75 | `crates/inference/src/metrics.rs` | 776 | 14 types: collector + config + alerts |
| 76 | `crates/inference/src/session.rs` | 513 | Config + state + session + stats |
| 77 | `crates/inference/src/latency.rs` | 624 | 12 struct: tracker + config + stats |
| 78 | `crates/inference/src/stop_conditions.rs` | 579 | StopCondition enum + trait + stats |
| 79 | `crates/runtime/src/scheduler.rs` | 679 | Types + scheduler + stats |
| 80 | `crates/runtime/src/streaming.rs` | 827 | Config + token stream + engine + stats |
| 81 | `crates/runtime/src/kv_cache.rs` | 687 | Config + entry + cache + stats |
| 82 | `crates/star-x/src/tensor_pool.rs` | 548 | Pool + pooled types + stats |
| 83 | `crates/star-x/src/sca.rs` | 806 | Attention + BLAS + fused ops |
| 84 | `crates/star-x/src/sliding_window.rs` | 662 | Sliding window + hierarchical variant |
| 85 | `crates/has-moe-ffn/src/lib.rs` | 748 | Config + MoE logic |
| 86 | `crates/quantization/src/lib.rs` | 838 | QFormat + 15+ converter functions |
| 87 | `crates/atqs/src/types.rs` | 672 | 30+ types dump |
| 88 | `crates/atqs/src/compression/adaptive_rank.rs` | 688 | Rank selection + compression |
| 89 | `crates/erp/src/resonance.rs` | 909 | Resonance patterns + compute |
| 90 | `crates/erp/src/utils.rs` | 852 | Diverse utilities |
| 91 | `crates/echo-net/src/isc.rs` | 830 | Config + collapse + events + stats |
| 92 | `crates/echo-net/src/derr.rs` | 771 | Differential error reasoning |
| 93 | `crates/isolation/src/multicluster.rs` | 787 | 20+ types + orchestration + health |
| 94 | `crates/foundation/src/clustering_orchestrator.rs` | 740 | Types + orchestration |
| 95 | `crates/foundation/src/distillation/mod.rs` | 634 | Teacher/student + loss + training |
| 96 | `crates/transformer/src/gqa/gqa_cpu.rs` | 770 | GQA CPU attention + cache |
| 97 | `crates/reasoning/src/saca/execute/engine.rs` | 566 | Engine + trait + env + sandbox + monitor |
| 98 | `crates/reasoning/src/saca/execute/testing/generator.rs` | 932 | Signature parsing + test gen + types |
| 99 | `crates/reasoning/src/saca/feedback.rs` | 601 | Feedback + pattern analysis |
| 100 | `crates/alignment/src/sparo/rlaif.rs` | 852 | Config + trait + judge + manager + stats |
| 101 | `crates/alignment/src/sparo/trainer.rs` | 728 | Trainer + results + stats + checkpoint |
| 102 | `crates/alignment/src/sparo/data.rs` | 712 | Config + dataset + batch + stats + processor |
| 103 | `crates/validation/src/validator.rs` | 682 | Result types + config validator + security |
| 104 | `crates/validation/src/security.rs` | 685 | Trait + 5 validator impls |
| 105 | `crates/multimodal/src/caffeine/mod.rs` | 804 | Caffeine + result + processor + pipeline |
| 106 | `crates/infrastructure/utils/src/performance.rs` | 795 | Monitor + metrics + benchmark |
| 107 | `crates/infrastructure/utils/src/text_processing.rs` | 750 | TextProcessor 20+ method |
| 108 | `crates/infrastructure/utils/src/file_utils.rs` | 732 | FileUtils 40+ method |
| 109 | `crates/infrastructure/utils/src/simd_ops.rs` | 931 | Vector + text + matrix SIMD + benchmark |
| 110 | `crates/infrastructure/utils/src/validation.rs` | 611 | Rules + types |
| 111 | `crates/tokenizer/src/bpe_tokenizer.rs` | 797 | Config + tokenizer + stats |
| 112 | `crates/tokenizer/src/pretokenizer.rs` | 751 | Types + config + tokenizer |
| 113 | `crates/shared/src/deeplearning_integration.rs` | 702 | Config + state + metrics + engine + traits |
| 114 | `crates/datastream/src/graph.rs` | 633 | Types + execution + results |
| 115 | `crates/datastream/src/dataset/loader.rs` | 740 | Config + loader + error |
| 116 | `crates/datastream/src/format_loader.rs` | 804 | Iterator + sync/async loaders |
| 117 | `crates/monitoring/src/observability/collector.rs` | ~500 | Counters + collector + broadcast |

### Model Agents — Medium (>500 lines)

| # | File | Lines | Masalah |
|---|------|-------|---------|
| 118 | `crates/model-aether/src/agents/emotion_weaver.rs` | 1038 | Agent + 30 types + processing + task I/O |
| 119 | `crates/model-aether/src/agents/culture_adapter.rs` | 1037 | Agent + 35+ types + adaptation |
| 120 | `crates/model-aether/src/agents/psyche_analyzer.rs` | 1312 | Agent + psychological models + analysis |
| 121 | `crates/model-aether/src/agents/mod.rs` | 680 | `mod.rs` berisi 3 agent + 15 types |
| 122 | `crates/model-aether/src/capabilities.rs` | 936 | 70+ capability specs |
| 123 | `crates/model-aether/src/agents/empathy_prime.rs` | 735 | Agent + empathy models |
| 124 | `crates/model-nexum/src/agents/alignment_arbiter.rs` | 1594 | Agent + consensus types |
| 125 | `crates/model-nexum/src/agents/orchestrator_prime.rs` | 736 | Orchestrator agent |
| 126 | `crates/model-nexum/src/identity.rs` | 229 | Identity + orchestration profile |
| 127 | `crates/model-spectra/src/config.rs` | 922 | 30+ config types |
| 128 | `crates/model-spectra/src/capabilities.rs` | 945 | 80+ capability specs |
| 129 | `crates/model-spectra/src/agents/innovation_engine.rs` | 927 | Agent + creative types |
| 130 | `crates/model-spectra/src/agents/style_adapter.rs` | 804 | Agent + style types |
| 131 | `crates/model-spectra/src/agents/artistic_weaver.rs` | 688 | Agent + art types |
| 132 | `crates/model-spectra/src/agents/spectral_analyzer.rs` | 546 | Potensi duplikasi dgn spectrum_analyzer |
| 133 | `crates/model-swift/src/agents/edge_opt.rs` | 1032 | Agent + 25+ types |
| 134 | `crates/model-swift/src/agents/fast_cache.rs` | 753 | Agent + cache types |
| 135 | `crates/model-swift/src/config.rs` | 608 | Config + optimization constraints |
| 136 | `crates/model-omnis/src/config.rs` | 851 | Config + validation + agent query |
| 137 | `crates/model-omnis/src/lib.rs` | 688 | FoundationModel + 5 state + 3 enum + metrics + NxrModel impl |
| 138 | `crates/model-omnis/src/capabilities.rs` | 502 | Capability + query + validation |
| 139 | `crates/model-vortex/src/lib.rs` | 751 | FoundationModel + state + identity + agents + config |
| 140 | `crates/model-vortex/src/capabilities.rs` | 895 | 60+ code capability specs |
| 141 | `crates/model-vortex/src/config.rs` | 513 | Config + `get_agent_config()` |
| 142 | `crates/model-cipher/src/config.rs` | 710 | Config + encryption + security + threat |
| 143 | `crates/model-kronos/src/config.rs` | 520 | Config + macro accessors |
| 144 | `crates/model-genesis/src/config.rs` | 466 | Config + evolution config |
| 145 | `crates/model-genesis/src/capabilities.rs` | 469 | Capability specs |
| 146 | `crates/model-axiom/src/capabilities.rs` | 470 | Capability specs |
| 147 | `crates/model-omnis/src/agents/empathy_catalyst/mod.rs` | 357 | `mod.rs` berisi full agent impl |

### Apps — Significant (300-800 lines)

| # | File | Lines | Masalah |
|---|------|-------|---------|
| 148 | `apps/nexora-ai/src/server/router.rs` | 464 | Routes + auth + rate limiter + CORS + security headers + logging |
| 149 | `apps/nexora-ai/src/security/mod.rs` | 460 | Config + validator + utils |
| 150 | `apps/nexora-ai/src/config/loader.rs` | 436 | Config struct + parse + validate + override |
| 151 | `apps/nexora-ai/src/server/billing_handlers.rs` | 430 | Plans + webhook + subscription + usage |
| 152 | `apps/nexora-ai/src/server/agent_handlers.rs` | 493 | 6 endpoint handler + oneshot pattern duplicate |
| 153 | `apps/nexora-ai/src/cli/commands.rs` | 521 | 18 subcommand definisi |
| 154 | `apps/nexora-ai/src/core/system.rs` | 529 | Monitor + health + metrics buffer |
| 155 | `apps/nexora-ai/src/core/chat.rs` | 508 | ChatEngine + analysis + types |
| 156 | `apps/nexora-ai/src/core/tier_router.rs` | 303 | Intent + routing + debate detection |
| 157 | `apps/nexora-ai/src/server/system_handlers.rs` | ~200 | 3 endpoint handler |
| 158 | `apps/nexora-ai/src/system.rs` | 159 | EventBus + memory + scheduler + optimizer + observability init |

### Root & Config — Multi-tanggung jawab

| # | File | Sections | Saran Split |
|---|------|----------|-------------|
| 159 | `nexora.toml` | 9 section (core, tokenizer, models, memory, utils, server, api, logging, isolation) | `server.toml`, `memory.toml`, `isolation.toml`, `core.toml` |
| 160 | `configs/inference.toml` | 6 section | `inference-engine.toml`, `decoding.toml`, `distributed.toml` |
| 161 | `configs/runtime.toml` | 6 section | `scheduler.toml`, `resource.toml`, `distributed-runtime.toml` |
| 162 | `configs/gnac.toml` | 8 section | per feature domain |
| 163 | `configs/logging.toml` | 4 section (masih kohesif) | optional: pisah tracing |
| 164 | `docker-compose.yml` | 9 services | `docker-compose.db.yml`, `observability.yml`, `storage.yml` |
| 165 | `Makefile` | 6 target groups, 260 lines | `Makefile.build`, `train`, `infra` |
| 166 | `setup.sh` | 4 concern (detect + config gen + build + verify) | `detect_env.sh`, `gen_config.sh`, `build.sh` |
| 167 | `deny.toml` | 4 section | `deny-advisories.toml`, `licenses.toml` |
| 168 | `nextest.toml` | 3 profile + per-package overrides | `nextest-default.toml`, `nextest-ci.toml` |
| 169 | `Cargo.toml` (root) | workspace members + deps + profile | pisah deps via `include` |
| 170 | `.github/workflows/ci.yml` | check + fmt + clippy + test + deny | `ci-check.yml`, `ci-lint.yml`, `ci-test.yml`, `ci-deny.yml` |

### Dead / Duplicate Files

| # | File | Status |
|---|------|--------|
| 171 | `crates/model-aether/src/agents_new.rs` (6 lines) | Stale duplicate `agents/mod.rs` |
| 172 | `crates/model-spectra/src/agents/spectral_analyzer.rs` | Potensi duplikasi `spectrum_analyzer.rs` |
| 173 | `apps/nexora-ai/src/core/processing.rs` | Test code inline tanpa `#[cfg(test)]` |

---

## Ringkasan Pattern Dominan

### 1. God Object Pattern (7 file)
`utils.rs` (4216), `cuda/context.rs` (3295), `wgsl.rs` (2569), `nn.rs` (2192), `engine.rs` (2199), `registry.rs` (2257), `gpu_sedc.rs` (1982)

Semua GPU compute, inference engine, model registry adalah god object.

### 2. Type Dump Pattern
`atqs/types.rs` (30+ types), `database/lib.rs` (20+ types), `inference/metrics.rs` (14 types), `model-axiom/config.rs` (100+ types), `model-nexum/capabilities.rs` (100+ specs)

Satu file jadi dumping ground semua type definitions untuk satu subsistem.

### 3. Agent Monolith Pattern (7 file di agent crate + 8 file di model agents)
Setiap agent file >1000 lines: types + config + trait impl + stats + logic.

### 4. Config Bukan Murni (11 file di models)
`config.rs` berisi validation logic, query methods, macro — bukan pure config.

### 5. Dead / Duplicate (3 file)

---

## Prioritas Refactor

### Gelombang 1 (High Impact — 10 file = ~50% kode)
| Prioritas | File | Lokasi |
|-----------|------|--------|
| P1 | `gpu/utils.rs` | `crates/autograd/src/` |
| P2 | `cuda/context.rs` | `crates/autograd/src/gpu/cuda/` |
| P3 | `engine.rs` | `crates/inference/src/` |
| P4 | `registry.rs` | `crates/transformer/src/model/` |
| P5 | `cli/training.rs` | `apps/nexora-ai/src/cli/` |
| P6 | `server/handlers.rs` | `apps/nexora-ai/src/server/` |
| P7 | `lib.rs` | `apps/nexora-ai/src/` |
| P8 | `arch_weaver.rs` | `crates/model-vortex/src/agents/` |
| P9 | `config.rs` (axiom) | `crates/model-axiom/src/` |
| P10 | `debate.rs` | `apps/nexora-ai/src/core/` |

### Gelombang 2 (Medium Impact — 30 file)
Agent crate, oracle crate, memory crate, star-x crate, inference crate sisanya.

### Gelombang 3 (Config & Root)
12 config files + setup.sh + Makefile + docker-compose + CI.

---

## Statistik Akhir

| Metrik | Nilai |
|--------|-------|
| Total file SRP violation | **241** |
| Kritis (god object) | 7 |
| Tinggi (>1000 lines) | 61 |
| Sedang (500-1000) | 51 |
| Model agents besar | 30 |
| Config multi-section | 12 |
| App (>300 lines) | 17 |
| Dead/duplicate | 3 |
| **Estimasi hemat kode setelah refactor** | ~30-40% |
| **Estimasi effort** | ~4-6 sprint |
