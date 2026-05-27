# Phase 5 — Production Intelligence Platform

**Target:** Survive production scale. Bukan "bisa jalan" tapi "bisa survive di 1.000+ GPU, 10K+ QPS, 99.9% uptime."

---

## 1. Distributed Inference

| Komponen | Deskripsi |
|----------|-----------|
| Multi-node inference | Model sharded across physical machines, single inference request served by N nodes |
| Tensor parallel | Per-layer sharding across GPUs/nodes (row/column split for QKV, FFN) |
| Pipeline parallel | Layer stages across GPUs (micro-batching, 1F1B scheduling) |
| Scheduler cluster | Central/dispatch scheduler untuk multi-node — request routing, load-aware dispatch |
| Heterogeneous GPU support | Mixed A100/H100/B200 cluster — weighted dispatch by GPU compute/memory capacity |

**Key challenge:** Communication overhead (NVLink vs RDMA vs TCP). Tensor parallelism membutuhkan all-reduce per layer — bottleneck di inter-node bandwidth.

---

## 2. Enterprise Observability

| Metrik | Instrumentasi |
|--------|---------------|
| Token/sec tracing | Per-node, per-GPU, per-model throughput tracing |
| GPU occupancy | Streaming Multiprocessor utilization, memory bandwidth utilization |
| KV fragmentation | Paged cache fragmentation ratio, # of page faults, compaction pressure |
| Scheduler pressure | Queue depth, dispatch latency, backlog per node, overload shedding |
| Thermal throttling | GPU temp → clock speed mapping, throttling events/s |
| Adaptive batching telemetry | Dynamic batch size, padding efficiency, utilization vs latency tradeoff |

---

## 3. Reliability Systems

| Sistem | Mekanisme |
|--------|-----------|
| Checkpoint recovery | Periodic weight/KV save → restore on crash (not full restart) |
| Circuit breaker | Per-node error rate threshold → automatic node drain + traffic reroute |
| Degraded mode | Partial capacity — inference with fewer GPUs/nodes, higher latency SLA |
| Self-healing worker | Watchdog detects frozen/errored GPU → kill + relaunch worker process |
| GPU crash recovery | GPU wedge detection (timeout polling) → mark GPU bad, migrate KV |
| Retry orchestration | Idempotent request retry with exponential backoff + jitter, max 3 attempts |

**Architecture:** Supervisor-worker pattern. Supervisor monitors heartbeat from each worker GPU. On N consecutive missed heartbeats → initiate recovery sequence.

---

## 4. Advanced Memory Architecture

| Komponen | Deskripsi |
|----------|-----------|
| True paged attention | Block-based KV allocation, not flat. Variable-length sequences without fragmentation |
| Cache dedup | Shared prefix KV antara requests (prefix caching engine already exists V1) |
| Prefix DAG | Tree-based prefix storage (not flat key-value) — branch dari arbitrary prefix, bukan hanya full prefix |
| Persistent memory | KV cache survives beyond request lifecycle — session-persistent, user-persistent |
| Shared KV pools | Cross-request KV pool: multiple requests with overlapping prefixes share KV blocks (copy-on-write) |

**Currently:** `PagedKVCache` exists but `#[deprecated]`. Engine uses `GpuKVCache` (flat). Phase 5 = remove deprecated, wire paged cache to engine, add DAG prefix + dedup.

---

## 5. Agent Ecosystem

| Kemampuan | Deskripsi |
|-----------|-----------|
| Interoperable agents | Agents can call each other — not just all delegate to foundation model |
| Persistent state | Agent memory survives conversation — user profile, context, learned preferences |
| Planner-worker hierarchy | High-level planner decomposes task → dispatches to specialized workers → synthesizes results |
| Long-running tasks | Async execution — submit task, poll status, collect results later (minutes/hours) |
| Asynchronous reasoning | Chain-of-thought without blocking request — stream intermediate steps |
| Cooperative execution | Multiple agents work on same task — peer review, consensus, vote |

**Phase 3 gave us 10 model crates with real MLP classifiers.**  
**Phase 4 wired those to real infrastructure (MoE, verifiers, multimodal).**  
**Phase 5 makes them actually communicate and cooperate.**

---

## 6. Performance War Mode

| Optimasi | Target |
|----------|--------|
| CUDA specialized kernel | Replace wgpu fallback with native CUDA kernels for critical ops (attention, FFN) |
| FlashAttention-class optimization | Fused attention: online softmax, no large attention matrix materialization |
| Fused RMSNorm | Single kernel: normalization + scaling + residual add — no intermediate reads/writes |
| Speculative decoding | Draft model → target model verification — 2-3x throughput for autoregressive generation |
| Async sampling | Overlap sampling with next forward pass — hide GPU-CPU sync latency |
| Graph capture | Capture CUDA graph for static inference shapes — bypass kernel launch overhead |
| Kernel fusion | Fuse elementwise ops (residual + norm + activation + dropout) into single kernel |

**Philosophy:** Setiap 3% speedup = uang listrik, latency SLA, dan jumlah GPU. "Clean architecture" kalah melawan "kernel fusion jam 3 pagi."

---

## Execution Strategy

| Phase | Focus | Timeline Estimate |
|-------|-------|-------------------|
| 5a | Memory architecture (paged attention, prefix DAG, dedup) | Most critical — fixes the `#[deprecated]` paged cache |
| 5b | Reliability systems (checkpoint, circuit breaker, self-healing) | Second — ops reliability before scale |
| 5c | Distributed inference (tensor/pipeline parallel, scheduler) | Third — needs reliability layer below |
| 5d | Observability (metrics, tracing, telemetry) | Parallel with 5a-5c |
| 5e | Agent ecosystem | Independent — can start in parallel |
| 5f | Performance war (CUDA, fusion, speculative decoding) | Continuous — runs alongside everything |

---

## Key Risks

1. **wgpu → CUDA migration:** wgpu is cross-platform but lacks CUDA-specific optimizations. Native CUDA kernels would require a `cuda/` directory with `build.rs` for `nvcc`.
2. **Multi-node communication:** Inter-node tensor parallelism needs RDMA (InfiniBand) for acceptable overhead. TCP is too slow for per-layer all-reduce.
3. **Agent ecosystem scope creep:** "Interoperable agents" is a full platform — risk of infinite scope. Need narrowest wedge (probably planner-worker with 3 agent types).
4. **Paged attention complexity:** True paged attention means rewriting `GpuKVCache` and `forward_gpu_batched` to handle block tables. This touches every hot path in inference.
