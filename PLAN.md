# Production Readiness Plan — Nexora AI

**Tujuan:** Memperbaiki 16 crate prioritas dari ~15-20% menjadi >80% production-ready dalam 4 tahap.

---

## Stage 1: Foundation & Infrastructure (4 crates)
**Dependency:** bottom of tree — semua crate lain bergantung pada ini
**Target readiness:** 60% → 85%

| # | Crate | Readiness | Fokus |
|---|---|---|---|
| 1 | `nexora-core` | ~60% | Cleanup intent detector, thread safety, error handling |
| 2 | `nexora-common` | ~60% | Logging init guard, error handling |
| 3 | `nexora-utils` | ~60% | SIMD ops module activation + safe dispatch, crypto deprecation |
| 4 | `nexora-deeplearning` (autograd, star-x, gnac, echo-net) | ~20% | Silent GPU fallback → warn, safe SIMD wrappers, gradient flow TODOs |

---

## Stage 2: Data & Storage Layer (4 crates)
**Dependency:** stage 1
**Target readiness:** 20-30% → 75%

| # | Crate | Readiness | Fokus |
|---|---|---|---|
| 1 | `nexora-database` | ~30% | Fix deprecated SQL methods, connection pool hardening |
| 2 | `nexora-datastream` | ~20% | Add ExecutionGraph unit tests, filter pipeline fixes |
| 3 | `nexora-memory` | ~20% | Memory management hardening |
| 4 | `nexora-tokenizer` | ~20% | Tokenizer validation edge cases |

---

## Stage 3: AI & Model Layer (4 crates)
**Dependency:** stage 1, 2
**Target readiness:** 5-15% → 70%

| # | Crate | Readiness | Fokus |
|---|---|---|---|
| 1 | `nexora-foundation` | ~15% | Byte-level tokenizer → BPE, streaming fix, clustering real impl, resource metrics |
| 2 | `nexora-models` | ~15% | Hapus 10 experimental flags, real model validation |
| 3 | `nexora-intelligence` | ~5% | OpenAI endpoint real impl, GPU feature matching source |
| 4 | `nexora-inference` | ~10% | Memory leak `Box::leak`, ContinuousBatching real batch, logits handling |

---

## Stage 4: Application & Specialized (4 crates)
**Dependency:** stage 3
**Target readiness:** 5-15% → 70%

| # | Crate | Readiness | Fokus |
|---|---|---|---|
| 1 | `nexora-cognition` | ~5% | Reasoning, planning, reflection, context — real LLM backend integration |
| 2 | `nexora-alignment` | ~15% | Hapus mock judge, SPARO RLAIF real implementation |
| 3 | `nexora-isolation` | ~10% | Fix LD_DEBUG exposure, L0-L6 hardening |
| 4 | `nexora-ai` (CLI) | ~15% | Chat/generation template → real inference, training stub |

---

## Metrik Keberhasilan per Stage

- ✅ `cargo check` — zero errors
- ✅ `cargo clippy` — zero warnings  
- ✅ `cargo nextest run` — all tests pass
- ✅ No `unwrap()` in new code
- ✅ No silent failures
- ✅ No placeholder/stub implementations
