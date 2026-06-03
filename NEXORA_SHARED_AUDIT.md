# Nexora Shared Component Audit — Complete

> Audit date: 3 Juni 2026
> Source: 853 file scan across 40 crate workspace
> Fokus: Identifikasi komponen shared/non-shared untuk 10 NXR model paralel

---

## DAFTAR PARAMETER PER MODEL (init.rs tier_config)

| Model | Tier | hidden | layers | heads | kv_heads | max_seq | intermediate | Est. Param | Est. VRAM (f32) |
|-------|------|--------|--------|-------|----------|---------|-------------|------------|-----------------|
| Omnis | Flagship | 512 | 16 | 8 | 4 | 2048 | 2048 | ~71M | ~284 MB |
| Axiom | High | 384 | 10 | 8 | 4 | 1024 | 1536 | ~34M | ~136 MB |
| Genesis | Mid | 256 | 6 | 8 | 4 | 1024 | 1024 | ~13M | ~52 MB |
| Nexum | Mid | 256 | 6 | 8 | 4 | 1024 | 1024 | ~13M | ~52 MB |
| Cipher | Low | 128 | 3 | 4 | 2 | 512 | 512 | ~2.5M | ~10 MB |
| Vortex | Low | 128 | 3 | 4 | 2 | 512 | 512 | ~2.5M | ~10 MB |
| Aether | Low | 128 | 3 | 4 | 2 | 512 | 512 | ~2.5M | ~10 MB |
| Spectra | Low | 128 | 3 | 4 | 2 | 512 | 512 | ~2.5M | ~10 MB |
| Swift | Low | 128 | 3 | 4 | 2 | 512 | 512 | ~2.5M | ~10 MB |
| Kronos | Low | 128 | 3 | 4 | 2 | 512 | 512 | ~2.5M | ~10 MB |
| **Total** | | | | | | | | **~146M** | **~584 MB** |

~146M parameters total, ~584 MB VRAM jika semua aktif (f32). Saat ini hanya **2 aktif** (Omnis + Axiom = ~105M / ~420 MB), 8 standby.

---

## BAGIAN 1: SHARED CANDIDATES

### 1.1 Transformer Backbone per Tier

**Lokasi File:** `crates/transformer/src/backbone_registry.rs` (REGISTRY OnceLock)
**Memory Saat Ini:** ~420 MB (2 model aktif: Omnis + Axiom)
**Potensi Penghematan:** **SUDAH IMPLEMENT** via `Arc<CausalLM>` sharing. Model dalam tier sama (Omnis, Axiom, Genesis = Ultra) secara teknis bisa share 1 backbone. Saat ini hanya Omnis + Axiom aktif — jika Genesis ikut aktif, Arc sharing menghemat ~52 MB.
**Risiko:** Rendah. Clone-on-write untuk model butuh modifikasi (EchoNet/SEDC). Zero-copy path aman.
**Alasan Bisa Shared:** `resolve_tier_backbone()` di `backbone_registry.rs:87` — semua model dalam tier yang sama dapat `Arc::clone()` dari `CausalLM` yang sama. Saat ini tier Ultra hanya dipakai Omnis+Axiom.

```
SHARED SCORE:    10/10 (sudah implement untuk model dalam tier sama)
ISOLATION SCORE: 3/10 (perlu copy-on-write jika ada modifikasi)
RECOMMENDATION:  TIER SHARED (sudah berjalan)
```

### 1.2 MiniTokenizer / BpeTokenizer

**Lokasi File:** 
- `crates/tokenizer/src/bpe_tokenizer.rs` — BPE tokenizer (FxHashMap-based, ~4.5 MB untuk 30K vocab)
- `crates/foundation/src/causal_lm_model/tokenizer.rs` — MiniTokenizer wrapper (50257 vocab)
- `crates/shared/src/foundation_components.rs` — `FoundationComponents.tokenizer` (Arc<parking_lot::RwLock<BpeTokenizer>>)
- `crates/shared/src/tokenizer_integration.rs` — `NxrTokenizerRef`

**Memory Saat Ini:** ~7 MB per instance (BpeTokenizer + MiniTokenizer + vocab builder)
**Potensi Penghematan:** ~63 MB (jika 10 model masing-masing punya instance sendiri → 1 shared instance)
**Risiko:** Sangat rendah. Tokenizer adalah fungsi deterministik murni — tidak ada state per-model. `FoundationComponents` sudah menyediakan shared tokenizer via `NxrTokenizerRef`.
**Alasan Bisa Shared:** Semua model share vocabulary 50257 (GPT-2 size). Tidak ada model-specific tokenization rules. Sudah ada `Arc<parking_lot::RwLock<BpeTokenizer>>` di `FoundationComponents`.

```
SHARED SCORE:    10/10
ISOLATION SCORE: 1/10
RECOMMENDATION:  FULL SHARED (wajib)
```

### 1.3 Token Embedding Table

**Lokasi File:** 
- `crates/transformer/src/model.rs` — CausalLM.token_embedding: `Option<Array2<f32>>` (vocab_size × hidden_size)
- `crates/models/src/classifier_util.rs:embed_average()` — membaca token_embedding untuk classifier

**Memory Saat Ini:** ~196 MB total jika 10 model punya embedding sendiri (50257 × rata-rata hidden_size ~390 × 4 bytes × 10)
**Potensi Penghematan:** **SUDAH SHARED** via backbone Arc sharing. Tapi classifier di setiap model crate melakukan `embed_table.clone()` — menyimpan snapshot embedding sendiri.
**Risiko:** Snapshot classifier hanya ~133 KB per model — negligible. Tapi setiap clone menambah memory.
**Alasan Bisa Shared:** Embedding adalah weight identik untuk model dalam tier yang sama. Classifier bisa baca dari backbone langsung tanpa clone.

```
SHARED SCORE:    9/10 (embedded di backbone — tapi classifier clone menambah overhead)
ISOLATION SCORE: 2/10
RECOMMENDATION:  FULL SHARED (hindari clone di classifier init)
```

### 1.4 Paged KV Cache Global

**Lokasi File:** `crates/inference/src/paged_cache.rs` — `GLOBAL_PAGED_CACHE: OnceLock<Mutex<PagedKVCache>>`
**Memory Saat Ini:** 4 GiB max (configurable via `DEFAULT_MAX_CACHE_MEMORY_BYTES`)
**Potensi Penghematan:** **SUDAH GLOBAL SHARED** — satu instance untuk semua sequence dan model. PrefixDAG sharing memungkinkan sequences dari model berbeda share blocks jika dimensi model sama.
**Risiko:** Jika 2 model memiliki dimensi berbeda (`num_layers`, `num_kv_heads`, `head_dim`), mereka TIDAK bisa share blocks — karena block layout tergantung dimensi.
**Alasan Bisa Shared:** Semua model low-tier (Cipher, Vortex, Aether, Spectra, Swift, Kronos) punya dimensi IDENTIK (`layers=3, kv_heads=2, head_dim=32`). Omnis+Axiom punya dimensi berbeda — perlu cache terpisah.

**Rekomendasi:** Multi-pool paged cache — satu pool per group dimensi model:
- Pool A: low-tier (6 model, layers=3, kv_heads=2, head_dim=32)  
- Pool B: mid-tier (Genesis, Nexum, layers=6, kv_heads=4, head_dim=32)
- Pool C: high-tier (Axiom, layers=10, kv_heads=4, head_dim=48)
- Pool D: flagship (Omnis, layers=16, kv_heads=4, head_dim=64)

```
SHARED SCORE:    8/10 (dalam dimensi yang sama — perlu multiple pools)
ISOLATION SCORE: 5/10 (beda dimensi = beda pool)
RECOMMENDATION:  PARTIAL SHARED (pool per dimensi group)
```

### 1.5 Attention Workspace Pool

**Lokasi File:** `crates/autograd/src/attention_workspace.rs` — `GLOBAL_POOL: OnceLock<WorkspacePool>`
**Memory Saat Ini:** 512 MiB budget, 8 buffer max
**Potensi Penghematan:** **SUDAH GLOBAL**. Semua model dan layer attention share pool yang sama.
**Risiko:** Konkurensi — 8 buffer mungkin tidak cukup untuk multi-model paralel. Jika 2 model running bersamaan, bisa starvation.
**Alasan Bisa Shared:** Buffer bersifat sementara (per forward pass) — reusable. Hanya perlu perhatikan thread safety.

```
SHARED SCORE:    9/10
ISOLATION SCORE: 3/10 (perlu lebih banyak buffer untuk multi-model)
RECOMMENDATION:  FULL SHARED (tambah max_buffers dari 8 → 16 untuk multi-model)
```

### 1.6 GPU Memory Pool

**Lokasi File:** `crates/autograd/src/gpu_memory.rs` — `GpuMemoryPool`
**Memory Saat Ini:** 26 buckets (1KB - 80GB), 512 buffer max, 30s TTL
**Potensi Penghematan:** **SUDAH GLOBAL** per device. Semua operasi GPU share pool yang sama.
**Risiko:** Sangat rendah. Bucket-based allocation aman untuk sharing.
**Alasan Bisa Shared:** Buffer reusable — ukuran yang sama dikembalikan ke bucket yang sama.

```
SHARED SCORE:    10/10
ISOLATION SCORE: 1/10
RECOMMENDATION:  FULL SHARED (sudah berjalan)
```

### 1.7 CUDA Kernels (JIT + NVRTC)

**Lokasi File:** 
- `crates/autograd/src/gpu/cuda/context.rs` — 18+ CUDA ops via NVRTC JIT
- `crates/autograd/src/gpu/gpu_types.rs` — `GpuBackend` auto-detection

**Memory Saat Ini:** Compiled kernel cache (PTX/CUBIN) — biasanya ~50-200 MB
**Potensi Penghematan:** **SUDAH SHARED** — kernels compiled sekali, digunakan semua model.
**Risiko:** Tidak ada. Kernel adalah kode — immutable.
**Alasan Bisa Shared:** Semua kernel (matmul, softmax, flash attention, gelu, dll) adalah fungsi deterministik tanpa state.

```
SHARED SCORE:    10/10
ISOLATION SCORE: 0/10
RECOMMENDATION:  FULL SHARED (sudah)
```

### 1.8 Caffeine Multimodal Processor

**Lokasi File:** `crates/multimodal/src/caffeine/` — `CaffeineProcessor` (5 encoders, Q-Former, tokenizer)
**Memory Saat Ini:** Encoder weights (image/audio/video/text/regional) — estimasi ~200-500 MB
**Potensi Penghematan:** Jika 10 model masing-masing punya Caffeine sendiri → duplikasi 2-5 GB. Saat ini di-share via `OnceLock` di integration layer.
**Risiko:** Rendah. CaffeineProcessor adalah stateless pipeline — tidak ada per-request state. Semua encoders read-only setelah init.
**Alasan Bisa Shared:** Multimodal processing adalah fungsi transformasi — model-agnostic.

```
SHARED SCORE:    9/10
ISOLATION SCORE: 2/10 (per-process singleton — aman)
RECOMMENDATION:  FULL SHARED (sudah — via OnceLock di foundation/multimodal/mod.rs)
```

### 1.9 SACA Reasoning Engine

**Lokasi File:** `crates/reasoning/src/saca/` — `SacaEngine` (6-phase pipeline)
**Memory Saat Ini:** ~1-5 MB (algoritma + config state)
**Potensi Penghematan:** Kecil — SACA adalah algoritma reasoning, bukan neural network.
**Risiko:** Rendah. SACA adalah stateless pipeline — per-call state di-clear setelah `reason()` selesai.
**Alasan Bisa Shared:** Semua model yang menggunakan SACA (Axiom, Genesis, Kronos, Nexum) call `SacaEngine::reason()` yang sama — tidak ada model-specific behavior.

```
SHARED SCORE:    10/10
ISOLATION SCORE: 1/10
RECOMMENDATION:  FULL SHARED (sudah — via OnceLock)
```

### 1.10 Oracle Code Verifiers

**Lokasi File:** `crates/oracle/src/linters/` — `CodeVerifierManager` (4 rule-based verifiers: correctness, performance, security, style) + `CodeLinterManager`
**Memory Saat Ini:** ~1-2 MB (rule-based pattern matching — tidak ada neural network)
**Potensi Penghematan:** Tidak signifikan secara memory, tapi menghindari duplikasi initialization.
**Risiko:** Sangat rendah. Verifiers adalah functions — tidak ada mutable state yang perlu isolasi.
**Alasan Bisa Shared:** Rule-based — identik untuk semua model. Digunakan oleh Vortex (code review), Cipher (security), Nexum (task scoring).

```
SHARED SCORE:    10/10
ISOLATION SCORE: 1/10
RECOMMENDATION:  FULL SHARED (sudah — via OnceLock)
```

### 1.11 FoundationComponents (Algorithmic Engines)

**Lokasi File:** `crates/shared/src/foundation_components.rs`
**Memory Saat Ini:** 
- ATQS Compression: ~10 MB (algorithm + cache)
- ERP Engine: ~5 MB (clustering state)
- VOGP Engine: ~5 MB (pruning state)
- GNAC Engine: ~3 MB
- DeepLearningEngine: ~2 MB
- **Total: ~25 MB**

**Potensi Penghematan:** ~225 MB jika 10 model punya instance masing-masing
**Risiko:** Rendah — semua adalah algoritma, bukan model neural. Bisa di-global-sharing.
**Alasan Bisa Shared:** Semua engine sudah dirancang sebagai shared utility — tidak ada model-specific configuration.

```
SHARED SCORE:    10/10
ISOLATION SCORE: 1/10
RECOMMENDATION:  FULL SHARED (sudah — via foundation_components.rs)
```

### 1.12 EchoNet APSS Injector

**Lokasi File:** `crates/foundation/src/echo_net_injector.rs` — `EchoNetInjector`
**Memory Saat Ini:** Ring buffer per-layer (~hidden_size × buffer_size × num_layers). Untuk Omnis (512×10×16): ~320 KB.
**Potensi Penghematan:** **SUDAH SHARED** — satu injector instance digunakan semua model via `init.rs:137`.
**Risiko:** Injector stateful (ring buffer per posisi). Jika 2 model running paralel dengan EchoNet, ring buffer akan corrupt.
**Alasan Bisa Shared:** Saat ini hanya 1 model aktif yang pakai EchoNet. Untuk multi-model paralel, perlu ring buffer per-sequence atau per-model.

**⚠️ PERINGATAN:** EchoNetInjector menggunakan ring buffer internal — **tidak thread-safe untuk multi-model paralel**. Perlu fork-on-write atau per-model instance.

```
SHARED SCORE:    5/10 (stateful — ring buffer conflict)
ISOLATION SCORE: 7/10 (perlu per-model atau per-sequence instance)
RECOMMENDATION:  PARTIAL SHARED (share code + weights, isolate state via cloning)
```

### 1.13 SafeTensor I/O

**Lokasi File:** `crates/foundation/src/safetensors/io.rs` + `crates/transformer/src/safetensors/`
**Memory Saat Ini:** 0 bytes (I/O utility — tidak ada persistent state)
**Potensi Penghematan:** Tidak ada duplikasi memory, tapi menghindari code duplication.
**Risiko:** Tidak ada.
**Alasan Bisa Shared:** Fungsi murni — deterministik.

```
SHARED SCORE:    10/10
ISOLATION SCORE: 0/10
RECOMMENDATION:  FULL SHARED (sudah)
```

### 1.14 Quantization Tables

**Lokasi File:** `crates/quantization/src/` — QFormat enum, quantize/dequantize, groupwise packing
**Memory Saat Ini:** ~1 MB (lookup tables + config)
**Potensi Penghematan:** Kecil, tapi menghindari duplikasi.
**Risiko:** Tidak ada.
**Alasan Bisa Shared:** Semua model pakai Q8 { group_size: 128 } — quantization parameters identik.

```
SHARED SCORE:    10/10
ISOLATION SCORE: 0/10
RECOMMENDATION:  FULL SHARED (sudah)
```

### 1.15 ATQS/ERP/VOGP Compression

**Lokasi File:** `crates/atqs/`, `crates/erp/`, `crates/vogp/`
**Memory Saat Ini:** Algoritma murni — ~0 MB untuk kode, ~10-50 MB untuk compression cache
**Potensi Penghematan:** Cache compression bisa dishare — menghindari recomputation.
**Risiko:** Cache compression key harus termasuk model_id agar tidak conflict.
**Alasan Bisa Shared:** Algoritma identik — hanya input (weights) yang berbeda per model.

```
SHARED SCORE:    8/10
ISOLATION SCORE: 4/10 (cache perlu key per model)
RECOMMENDATION:  PARTIAL SHARED (share kode, cache dengan model-specific key)
```

### 1.16 MoE Router Weights

**Lokasi File:** 
- `crates/has-moe-ffn/src/routing.rs` — `Router::forward()` dengan OnceLock weight caching
- `crates/has-moe-ffn/src/experts.rs` — Expert FFN dengan OnceLock per-expert

**Memory Saat Ini:** Per expert: `hidden_size × intermediate_size × 2` (fc1 + fc2). 8 experts × ~2M = ~16M parameters.
**Potensi Penghematan:** Jika semua model share MoE experts, bisa hemat ~144M parameters (9 model × 16M). TAPI expert harus di-train bersama.
**Risiko:** Tinggi. Expert weights adalah learned — jika setiap model butuh expert specialization berbeda, shared experts akan menurunkan kualitas.
**Alasan Bisa Shared:** Hanya jika semua model di-train bersama dengan shared expert pool. Saat ini `num_experts = 0` di semua tier config — MoE tidak aktif.

```
SHARED SCORE:    4/10 (secara teknis bisa, tapi trade-off quality)
ISOLATION SCORE: 8/10 (model-specific specialization mungkin diperlukan)
RECOMMENDATION:  TIER SHARED (model dalam tier sama bisa share experts)
```

---

## BAGIAN 2: NON-SHARED CANDIDATES

### 2.1 Per-Sequence KV Cache Runtime

**Lokasi File:** 
- `crates/inference/src/sequence_state.rs:Sequence` — per-sequence state (prompt, generated tokens)
- `crates/inference/src/kv_cache.rs:KVCache` — per-instance flat KV cache (1 GiB max)
- `crates/inference/src/paged_cache.rs` — per-sequence block table + block data
- `crates/runtime/src/kv_cache.rs` — runtime LRU cache (1 GiB max, 16 shards)

**Alasan Tidak Bisa Shared:** Setiap sequence memiliki konteks unik (prompt tokens, generation state). KV cache menyimpan key-value pairs yang hanya relevan untuk sequence spesifik.
**Konsekuensi Jika Dipaksa Shared:** Kontaminasi data — token dari sequence A akan tercampur dengan sequence B → garbage output.
**Dampak Kualitas:** Fatal — output tidak bermakna.
**Dampak Training:** Tidak relevan (KV cache hanya untuk inference).
**Dampak Inference:** Semua hasil inference akan corrupt.

```
SHARED SCORE:    0/10
ISOLATION SCORE: 10/10
RECOMMENDATION:  NOT SHARED
```

### 2.2 Optimizer State (AdamW: m + v)

**Lokasi File:** 
- `crates/autograd/src/gpu_adam.rs:GpuAdam` — `m: Vec<GpuTensor>`, `v: Vec<GpuTensor>` (2× parameter memory)
- `crates/training/src/lib.rs` — CPU Adam state, saved as `{base}.opt.safetensors`

**Alasan Tidak Bisa Shared:** Optimizer state = 2× parameters (momentum + variance). State ini menginkorporasi history training dari model spesifik. Berbeda model → berbeda trajectory → berbeda state.
**Konsekuensi Jika Dipaksa Shared:** Momentum dan variance akan mencerminkan gradient dari model lain — training crash atau konvergen ke local minimum salah.
**Dampak Kualitas:** Training gagal konvergen.
**Dampak Training:** Model loss tidak turun — bahkan bisa naik.
**Dampak Inference:** Tidak relevan (optimizer state hanya dipakai training).

**Estimasi Memory:**
- Omnis: ~71M params × 4 bytes × 2 (m+v) = ~568 MB optimizer state
- Axiom: ~34M × 4 × 2 = ~272 MB
- Total (Omnis + Axiom aktif): ~840 MB optimizer state

```
SHARED SCORE:    0/10
ISOLATION SCORE: 10/10
RECOMMENDATION:  NOT SHARED
```

### 2.3 Training Gradients

**Lokasi File:** `crates/training/src/lib.rs` — autograd tape gradients, gradient accumulation buffer
**Alasan Tidak Bisa Shared:** Gradient milik model spesifik — ukuran dan shape tergantung arsitektur model. Gradient accumulation juga per-model.
**Konsekuensi Jika Dipaksa Shared:** Shape mismatch — crash.
**Dampak Training:** Fatal — compilation/runtime error.
**Dampak Inference:** Tidak relevan.

```
SHARED SCORE:    0/10
ISOLATION SCORE: 10/10
RECOMMENDATION:  NOT SHARED
```

### 2.4 LoRA Adapters

**Lokasi File:** `crates/training/src/lora.rs` — `LoRALayer`, `LoRAConfig`
**Alasan Tidak Bisa Shared:** LoRA adapter di-train spesifik untuk task tertentu. Setiap model atau fine-tune job punya adapter berbeda.
**Konsekuensi Jika Dipaksa Shared:** Task interference — adapter dari task A akan merusak performa di task B.
**Dampak Kualitas:** Catastrophic forgetting — model kehilangan kemampuan original.
**Dampak Training:** Tidak bisa fine-tune multiple tasks dalam satu adapter.
**Dampak Inference:** Output akan campur aduk antara task.

**Estimasi Memory per LoRA:** rank=8, 4 target modules per layer:
- Omnis (16 layers): 16 × 4 × 8 × (512+512) = ~524K params = ~2 MB
- Axiom (10 layers): ~328K params = ~1.3 MB
- Semua model: ~2-5 MB total — negligible

```
SHARED SCORE:    2/10
ISOLATION SCORE: 9/10
RECOMMENDATION:  NOT SHARED
```

### 2.5 Model-Specific Classifier Weights

**Lokasi File:** Setiap `crates/models/src/{model}/classifier.rs` atau `router.rs` atau `analyzer.rs`
**Alasan Tidak Bisa Shared:** Setiap classifier punya output categories berbeda (7 domains, 8 emotions, 6 threats, dll) dan arsitektur MLP dengan input hidden_size berbeda per tier.
**Konsekuensi Jika Dipaksa Shared:** Output dimension mismatch.
**Dampak Kualitas:** Tidak relevan — classifier adalah komponen independen.
**Dampak Training:** Harus di-train terpisah untuk setiap model.
**Dampak Inference:** Fatal — output categories salah.

**Estimasi Memory per Classifier:**
| Model | Params | Memory |
|-------|--------|--------|
| Omnis | 5,376 | ~21 KB |
| Aether | 33,280 | ~133 KB |
| Swift | 4,256 | ~17 KB |
| Vortex | 33,152 | ~133 KB |
| Axiom | 24,768 | ~99 KB |
| Cipher | 12,480 | ~50 KB |
| Kronos | 8,352 | ~33 KB |
| Genesis | 24,768 | ~99 KB |
| Nexum | 16,512 | ~66 KB |
| Spectra | 12,480 | ~50 KB |
| **Total** | **175,424** | **~701 KB** |

Total hanya ~700 KB untuk SEMUA classifier — **tidak signifikan**.

```
SHARED SCORE:    1/10 (arsitektur berbeda + output categories berbeda)
ISOLATION SCORE: 9/10
RECOMMENDATION:  NOT SHARED (tapi memory impact negligible)
```

### 2.6 Agent Runtime State (Aether Agents)

**Lokasi File:** 
- `crates/models/src/aether/agents/` — ContextWeaveAgent (conversation history), SoulMirrorAgent (encounter_count, trust_level)
- `crates/models/src/omnis/agents/empathy_catalyst/` — config/capabilities state
- `crates/agent/src/worker_agent.rs` — WorkerAgent stats, step tracking
- `crates/agent/src/agent_manager.rs` — AgentManager state

**Alasan Tidak Bisa Shared:** Agent state bersifat per-instance — conversation history, trust level, encounter count adalah konteks spesifik user.
**Konsekuensi Jika Dipaksa Shared:** Cross-user data leak — user A melihat history user B.
**Dampak Kualitas:** Privacy violation — data leak.
**Dampak Training:** Tidak relevan.
**Dampak Inference:** Output menggunakan konteks dari user yang salah.

```
SHARED SCORE:    1/10
ISOLATION SCORE: 10/10
RECOMMENDATION:  NOT SHARED
```

### 2.7 Checkpoint Files

**Lokasi File:** 
- `crates/foundation/src/safetensors/io.rs` — save/load safetensors
- `crates/foundation/src/training/active_standby.rs` — swap directory (`/tmp/nexora_swap`)
- `crates/foundation/src/causal_lm_model/mod.rs` — `save_checkpoint()` / `load_checkpoint()`

**Alasan Tidak Bisa Shared:** Setiap model punya weight berbeda. Checkpoint adalah serialisasi spesifik model — tidak bisa dipertukarkan.
**Konsekuensi Jika Dipaksa Shared:** Weight shape mismatch → load error.
**Dampak Kualitas:** Tidak relevan.
**Dampak Training:** Tidak bisa restore training.
**Dampak Inference:** Model tidak bisa di-load.

**Estimasi Storage:** 
- Checkpoint Omnis (f16): ~71M × 2 bytes = ~142 MB
- Checkpoint Axiom (f16): ~68 MB
- Total checkpoint (10 model, f16): ~300 MB
- Swap dir plus optimizer state: ~600 MB peak

```
SHARED SCORE:    0/10
ISOLATION SCORE: 10/10
RECOMMENDATION:  NOT SHARED
```

### 2.8 Continuous Batching Sequence State

**Lokasi File:** `crates/inference/src/continuous_batching.rs` — `ContinuousBatchingEngine<M>` — `sequences: HashMap<u64, Sequence>`
**Alasan Tidak Bisa Shared:** Sequence state adalah runtime per-request — generated tokens, prompt_pos, sampling params.
**Konsekuensi Jika Dipaksa Shared:** Sequence state conflict — dua request pakai state yang sama.
**Dampak Kualitas:** Output campur aduk.
**Dampak Inference:** Semua request corrupt.

```
SHARED SCORE:    0/10
ISOLATION SCORE: 10/10
RECOMMENDATION:  NOT SHARED
```

### 2.9 Model Registry Entry State

**Lokasi File:** `crates/shared/src/model_registry.rs` — per-model `RegistryEntry` (model instance, metadata, capabilities, config)
**Alasan Tidak Bisa Shared:** Setiap entry milik model spesifik. Metadata, capabilities, dan config berbeda per model.
**Konsekuensi Jika Dipaksa Shared:** Registry lookup akan return instance yang salah.
**Dampak Keseluruhan:** Routing error — request dikirim ke model salah.

```
SHARED SCORE:    0/10
ISOLATION SCORE: 10/10
RECOMMENDATION:  NOT SHARED
```

---

## BAGIAN 3: HYBRID CANDIDATES

### 3.1 Active/Standby Scheduler

**Lokasi File:** `crates/foundation/src/training/active_standby.rs` — `ActiveStandbyScheduler`
**Saat Ini:** Shared scheduler mengelola 10 model (2 active, 8 standby). Logic rotation, checkpoint, loading/unloading.
**Bagian Shared:** Scheduler logic, rotation algorithm, config.
**Bagian Terpisah:** Per-model checkpoint path, per-model step counter, per-model status.
**Potensi Penghematan:** 5× peak memory reduction (10M → 2M).
**Rekomendasi:** **SUDAH OPTIMAL** — shared logic, isolated state per model.

### 3.2 NxrModelId / ModelIdentity

**Lokasi File:** `crates/shared/src/model_identity.rs` — `NxrModelId` enum, `ModelMeta`, `ModelTier`
**Saat Ini:** Semua model share enum yang sama.
**Bagian Shared:** Enum definition, tier mapping, fullname, description (fungsi murni).
**Bagian Terpisah:** `ModelMeta` instance per model (uuid, created_at, parameter_count).
**Rekomendasi:** **SUDAH OPTIMAL** — shared type definitions, isolated instances.

### 3.3 CapabilitySpec Vectors

**Lokasi File:** `crates/shared/src/capability_spec.rs`
**Saat Ini:** Predefined capability vectors per model, stored in RegistryEntry.
**Bagian Shared:** CapabilitySpec struct definition, predefined() factory functions.
**Bagian Terpisah:** Per-model CapabilityVector instance (immutable).
**Rekomendasi:** **SUDAH OPTIMAL**.

### 3.4 Memory Manager (4-Layer)

**Lokasi File:** `crates/memory/src/` — `MemoryManager` (layers, episodic, cache, compressor)
**Saat Ini:** Per-instance MemoryManager — bisa shared antar model atau per-model tergantung deployment.
**Bagian Shared:** Lock ordering protocol, MemoryLayers, LRUCache, ContextCompressor (algoritma).
**Bagian Terpisah:** Episodic memory isi per-user/session.
**Rekomendasi:** **SHARED PER-PROCESS** — satu MemoryManager untuk semua model dalam 1 process. Per-user isolation via session key.

### 3.5 DistributedScheduler / GossipProtocol

**Lokasi File:** 
- `crates/runtime/src/distributed.rs` — `DistributedScheduler`
- `crates/runtime/src/gossip.rs` — `GossipProtocol`
- `crates/runtime/src/cluster.rs` — `NodeRegistry`

**Saat Ini:** Per-instance, tapi bisa di-share.
**Bagian Shared:** Scheduler logic, gossip protocol, routing strategies.
**Bagian Terpisah:** Node-specific load metrics, per-request routing decisions.
**Rekomendasi:** **SHARED PER-CLUSTER** — satu scheduler untuk semua model dalam satu cluster node.

---

## BAGIAN 4: MATRIX SHARING KOMPLIT

| Komponen | Shared Score | Isolation Score | Rekomendasi | VRAM Impact | Notes |
|----------|:-----------:|:--------------:|:-----------:|:-----------:|-------|
| Transformer Backbone | 10 | 3 | TIER SHARED | 5× (via Arc) | **SUDAH** |
| Tokenizer | 10 | 1 | FULL SHARED | ~63 MB | **WAJIB** |
| Token Embedding | 9 | 2 | FULL SHARED | ~196 MB | Hindari clone |
| Paged KV Cache | 8 | 5 | PARTIAL (pool per dimensi) | 4 GiB pool | Perlu multi-pool |
| Attention Workspace | 9 | 3 | FULL SHARED | 512 MB | Tambah buffer count |
| GPU Memory Pool | 10 | 1 | FULL SHARED | Variable | **SUDAH** |
| CUDA Kernels | 10 | 0 | FULL SHARED | ~200 MB | **SUDAH** |
| Caffeine Multimodal | 9 | 2 | FULL SHARED | ~200-500 MB | **SUDAH** |
| SACA Reasoning | 10 | 1 | FULL SHARED | ~5 MB | **SUDAH** |
| Oracle Verifiers | 10 | 1 | FULL SHARED | ~2 MB | **SUDAH** |
| FoundationComponents | 10 | 1 | FULL SHARED | ~25 MB | **SUDAH** |
| EchoNet Injector | 5 | 7 | PARTIAL (fork-on-write) | ~320 KB | ⚠️ Stateful conflict |
| SafeTensor I/O | 10 | 0 | FULL SHARED | 0 | **SUDAH** |
| Quantization Tables | 10 | 0 | FULL SHARED | ~1 MB | **SUDAH** |
| ATQS/ERP/VOGP | 8 | 4 | PARTIAL | ~50 MB | Cache key per model |
| MoE Experts | 4 | 8 | TIER SHARED | ~144 MB | Trade-off quality |
| **KV Cache Runtime** | **0** | **10** | **NOT SHARED** | Per-sequence | Wajib terpisah |
| **Optimizer State** | **0** | **10** | **NOT SHARED** | ~840 MB | Wajib terpisah |
| **Training Gradients** | **0** | **10** | **NOT SHARED** | ~584 MB | Wajib terpisah |
| **LoRA Adapters** | **2** | **9** | **NOT SHARED** | ~5 MB | Wajib terpisah |
| **Classifiers (10×)** | **1** | **9** | **NOT SHARED** | ~701 KB | Kecil — no issue |
| **Agent State** | **1** | **10** | **NOT SHARED** | Per-session | Privacy critical |
| **Checkpoint Files** | **0** | **10** | **NOT SHARED** | ~600 MB | Wajib terpisah |
| **Sequence State** | **0** | **10** | **NOT SHARED** | Per-request | Wajib terpisah |
| **Registry Entry** | **0** | **10** | **NOT SHARED** | ~100 bytes | Wajib terpisah |

---

## BAGIAN 5: VRAM OPTIMIZATION

### 5.1 Sudah Terimplementasi

| Optimasi | Estimasi Penghematan | Status |
|----------|:-------------------:|--------|
| Tier Backbone Arc Sharing (3 model Ultra → 1 backbone) | ~52 MB (Genesis) | ✅ `resolve_tier_backbone()` |
| Active-Standby (2 dari 10 model aktif) | 5× peak (10M → 2M) | ✅ `ActiveStandbyScheduler` |
| KV Cache f16 storage | 2× vs f32 | ✅ Default ON |
| KV Cache Q4 storage | 8× vs f32 | ✅ Toggle ON (paged_cache_q4=true) |
| Hot→Warm→Cold tiering | 8× pada idle blocks | ✅ Default ON |
| Cold disk offload | VRAM → disk (10 GB) | ✅ Default ON |
| PrefixDAG (block sharing) | Eliminates redundant KV | ✅ `share_prefix_in_blocks()` |
| Prefix Trie cache | 1 GB KV reuse | ✅ Default |
| Attention workspace pool | 512 MB reusable buffer | ✅ `GLOBAL_POOL` |
| GPU memory pool (bucket) | Reuse wgpu buffers | ✅ `GpuMemoryPool` |
| Mixed precision (F16) training | 2× vs F32 | ✅ Available |
| AMP LossScaler | Mencegah underflow | ✅ Available |
| LoRA fine-tuning | 256× fewer trainable params | ✅ LoRA rank default 8 |

### 5.2 Potensi Optimasi Tambahan

| Optimasi | Estimasi Penghematan | Kompleksitas | Prioritas |
|----------|:-------------------:|:------------:|:---------:|
| **Multi-pool Paged Cache** (pool per dimensi group) | 4× pool isolation — menghindari cache conflict | Medium | HIGH |
| **Shared Expert Training** (MoE experts shared antar model) | ~144 MB (9 model × 16M) | High — butuh joint training | MEDIUM |
| **Embedding Table Read-only Access** (classifier baca langsung dari backbone, tanpa clone) | ~133 KB × 10 = ~1.3 MB | Low | LOW |
| **Classifier Quantization** (Q8 classifier weights) | ~87 KB penghematan dari ~701 KB | Low | LOW |
| **Weight Sharding** (distributed model parallelism) | VRAM linear dengan shard count | High | FUTURE |
| **CPU Offload untuk Standby** (simpan weight standby di RAM → GPU cuma saat aktif) | ~284 MB Omnis + ~136 MB Axiom = ~420 MB GPU → RAM | Medium | HIGH |
| **KV Cache Compression** (FP8/INT4 di warm/cold) | 2-4× lebih baik dari f16 | Medium | MEDIUM |
| **Gradient Checkpointing** (recompute forward activations) | ~50% pengurangan memory training | Low | MEDIUM |
| **Speculative Decoding** (small model draft → large model verify) | 2-3× throughput — tidak langsung VRAM | Medium | MEDIUM |

### 5.3 Estimasi VRAM Final (Skenario 10 Model Aktif)

| Skenario | Tanpa Optimasi | Dengan Optimasi Saat Ini | Dengan Optimasi Tambahan |
|----------|:-------------:|:----------------------:|:----------------------:|
| Model weights (f32) | ~5,840 MB | ~420 MB (2 aktif) | ~420 MB (2 aktif di GPU, 8 di CPU) |
| Optimizer state | ~8,400 MB | ~840 MB (2 model) | ~840 MB |
| KV Cache | ~4,096 MB | ~4,096 MB (shared) | ~512 MB (FP8) |
| Caffeine Multimodal | ~500 MB | ~500 MB (shared) | ~500 MB (shared) |
| Other (agents, etc) | ~200 MB | ~200 MB | ~200 MB |
| **Total** | **~19 GB** | **~6 GB** | **~2.5 GB** |

---

## BAGIAN 6: CRITICAL FINDINGS

### CF-1: Mismatched Config Tables 🔴 KRITIS

**Temuan:** Tiga tempat mendefinisikan ukuran model yang BERBEDA untuk model yang sama:

| Model | `init.rs` (foundation) | `foundation.rs` (models) | `transformer` presets |
|-------|:---------------------:|:------------------------:|:---------------------:|
| Omnis | hidden=512, layers=16 | hidden=768, layers=8 | Ultra: hidden=6144, layers=48 |
| Axiom | hidden=384, layers=10 | hidden=768, layers=8 | (sama Omnis: Ultra) |
| Genesis | hidden=256, layers=6 | hidden=768, layers=8 | (sama Omnis: Ultra) |

**Severity:** 🔴 KRITIS
**File:** 
- `crates/foundation/src/init.rs:18-109` (tier_config)
- `crates/models/src/foundation.rs:35-86` (transformer_config_for)
- `crates/transformer/src/config.rs:150-200` (preset)
- `crates/shared/src/model_identity.rs` (NxrModelId::tier — mapping tier vs model)

**Penjelasan:** Tiga config table menggunakan nilai hidden_size / layers yang sangat berbeda. `init.rs` (yang dipakai runtime) menggunakan hidden=512 untuk Omnis, sementara `foundation.rs` (yang dipakai classifier di model crate) menggunakan hidden=768 — selisih 256. Ini berarti classifier MLP di model crate menggunakan input dimension yang SALAH untuk backbone actual.

**Dampak:** Runtime error (shape mismatch) jika classifier benar-benar dijalankan — atau silent correctness bug (classifier membaca weight layout yang salah dari embedding table).

**Fix:** Sinkronkan ketiga config table ke satu source of truth (sebaiknya `transformer/src/config.rs` presets).

### CF-2: Tidak Ada Real SpecialistModel Implementasi 🔴 KRITIS

**Temuan:** `Arc<dyn SpecialistModel>` routing dari `CoreController` tidak pernah diimplementasi oleh 10 model crate. Hanya ada `DefaultSpecialistModel` (pass-through mock).

**Severity:** 🔴 KRITIS
**File:** 
- `crates/core/src/types.rs:498-540` (DefaultSpecialistModel mock)
- `crates/intelligence/src/model_registry/specialists.rs:10-98` (mock)
- `crates/core/src/controller.rs:284` (register_specialist_model — tidak dipanggil)
- `crates/agent/src/routing_agent.rs:28` (HashMap<String, Box<dyn SpecialistModel>> kosong)

**Penjelasan:** Arsitektur `SpecialistModel` adalah fondasi routing model. Tanpa implementasi, `CoreController` dan `RoutingAgent` hanya punya mock. 10 model crate (Omnis, Aether, dll) terisolasi dari sistem routing.

**Dampak:** Permintaan tidak bisa dirouting ke specialist model yang tepat. Semua fallback ke default.

### CF-3: QuarantineManager Tidak Terintegrasi dengan Agent System 🟠 HIGH

**Temuan:** `QuarantineManager` di `crates/isolation/src/quarantine.rs` memiliki kemampuan penuh (quarantine/resolve/escalate agent) tapi tidak di-wire ke `AgentManager` atau `WorkerAgent`.

**Severity:** 🟠 HIGH
**File:**
- `crates/isolation/src/quarantine.rs:50-158` (QuarantineManager — implementasi penuh)
- `crates/agent/src/lib.rs:100-106` (`_agent_isolation_check` — private, unused, create new orchestrator tiap call)
- `crates/agent/src/agent_manager.rs` — tidak ada reference ke quarantine

**Penjelasan:** Fungsi `_agent_isolation_check` ada tapi private, tidak dipanggil, dan membuat `IsolationOrchestrator` baru setiap kali. Tidak ada mekanisme quarantine agent behavior anomaly.

**Dampak:** Agent yang berperilaku abnormal tidak bisa di-isolate secara otomatis.

### CF-4: 8 dari 10 Model Tidak Punya Bobot Aktif 🟠 HIGH

**Temuan:** Hanya 2 model (Omnis, Axiom) yang aktif dengan random weights di startup. 8 model lainnya standby — tidak bisa digunakan tanpa checkpoint.

**Severity:** 🟠 HIGH
**File:** `crates/foundation/src/init.rs:112` — `ACTIVE_MODEL_IDS: [NxrModelId; 2]`

**Penjelasan:** `initialize_foundation_models()` hanya memberi bobot ke 2 model. 8 model lainnya dipanggil dengan `initialize_empty()`. `wire_model()` memberikan warning untuk standby model.

**Dampak:** Hanya Omnis dan Axiom yang bisa infer tanpa checkpoint. 80% model tidak fungsional di startup.

### CF-5: Semua Bobot Classifier Adalah Random 🟠 HIGH

**Temuan:** Semua 10 classifier MLP di model crate diinisialisasi dengan Xavier uniform random weights via `rand::thread_rng()`. Tidak ada training pipeline untuk classifier ini.

**Severity:** 🟠 HIGH
**File:** Setiap `crates/models/src/{model}/classifier.rs` — semua menggunakan `xavier_init()`

**Penjelasan:** Classifier weights adalah random — tidak di-train. Ketika model crate melakukan `classify(prompt)`, hasilnya adalah random noise, bukan klasifikasi bermakna.

**Dampak:** Semua delegation routing (expert routing, emotion detection, threat classification, dll) menghasilkan output acak — fitur utama Phase 3/4 tidak berfungsi.

### CF-6: EchoNet Injector Stateful — Tidak Thread-Safe 🟡 MEDIUM

**Temuan:** `EchoNetInjector` di `crates/foundation/src/echo_net_injector.rs:242` menggunakan ring buffer internal untuk tracking hidden states dan phase vectors. Jika 2 model paralel menggunakan EchoNet yang sama, buffer akan terkontaminasi.

**Severity:** 🟡 MEDIUM
**File:** `crates/foundation/src/echo_net_injector.rs` — ring buffer (`[Option<Array1<f32>>; 10]`)

**Penjelasan:** Ring buffer per position — jika model A dan B running paralel, posisi dari model B akan meng-overwrite buffer model A.

**Dampak:** Phase stabilization error, bisa menyebabkan output tidak stabil.

**Fix:** Fork-on-write — clone ring buffer untuk setiap sequence yang butuh EchoNet.

### CF-7: TokenizerCore Pakai std::HashMap 🟡 MEDIUM

**Temuan:** `TokenizerCore` di `crates/tokenizer/src/tokenizer_core.rs` menggunakan `std::collections::HashMap` (SipHash) untuk vocab lookup, sementara `BpeTokenizer` sudah pakai `FxHashMap`. Ini menyebabkan ~3× lookup lebih lambat dan ~600KB memory ekstra.

**Severity:** 🟡 MEDIUM
**File:** 
- `crates/tokenizer/src/tokenizer_core.rs` — `HashMap<String, u32>` (SipHash)
- `crates/tokenizer/src/bpe_tokenizer.rs` — `FxHashMap<String, TokenId>` (FxHasher)

**Penjelasan:** Tokenizer adalah hot path — dipanggil setiap inference request. Perbedaan 3× lookup speed signifikan untuk throughput.

**Dampak:** ~3× lebih lambat tokenization untuk code path yang menggunakan TokenizerCore.

### CF-8: GpuPageTable vs PagedKVCache — Dua GPU Mirror Paralel 🟡 MEDIUM

**Temuan:** Autograd crate memiliki `GpuPageTable` (`crates/autograd/src/gpu_kv_cache.rs`) yang merupakan GPU-side page table dengan WGSL gather/scatter shaders. Tapi inference crate juga memiliki `PagedKVCache.GpuKVCacheEntry` GPU mirror. Keduanya parallel — tidak terintegrasi.

**Severity:** 🟡 MEDIUM
**File:**
- `crates/autograd/src/gpu_kv_cache.rs` — GpuPageTable
- `crates/inference/src/paged_cache.rs:200-250` — GpuKVCacheEntry list

**Penjelasan:** Dua implementasi mirror KV cache ke GPU dengan mekanisme berbeda. Data path antara keduanya tidak jelas — mungkin duplikasi effort atau data inconsistency.

**Dampak:** Waste GPU memory untuk dua mirror. Potensi inconsistency jika satu diupdate tanpa update yang lain.

### CF-9: Paged Cache Global Singleton Tidak Support Multi-Dimensi 🟡 MEDIUM

**Temuan:** `GLOBAL_PAGED_CACHE: OnceLock<Mutex<PagedKVCache>>` adalah singleton dengan satu konfigurasi dimensi (`num_layers`, `num_kv_heads`, `head_dim`). Model dengan dimensi berbeda tidak bisa share.

**Severity:** 🟡 MEDIUM
**File:** `crates/inference/src/paged_cache.rs` — `PagedCacheConfig`

**Penjelasan:** Omnis (layers=16, kv_heads=4, head_dim=64) dan Swift (layers=3, kv_heads=2, head_dim=32) butuh block layout berbeda. Satu cache tidak bisa mengakomodasi keduanya.

**Fix:** Implement multi-pool paged cache — satu pool per group dimensi model.

### CF-10: Tidak Ada Integration Test untuk Delegasi Pipeline 🟢 LOW

**Temuan:** `crates/models/tests/` dan `crates/foundation/tests/` — direktori kosong. 236+ test adalah inline unit test.

**Severity:** 🟢 LOW
**File:** — (tidak ada)

**Penjelasan:** Tidak ada test yang memverifikasi bahwa seluruh 10 model ter-register, ter-wire dengan benar, dan delegation pipeline bekerja.

---

## BAGIAN 7: KESIMPULAN

### Arsitektur Saat Ini

Nexora sudah memiliki arsitektur sharing yang cukup matang:
- **Tier Backbone Sharing** via `Arc<CausalLM>` — ✅ optimal
- **Active-Standby** (2 dari 10 model) — ✅ 5× penghematan VRAM
- **Global Singletons** (Cache, Pool, CUDA kernels) — ✅ sudah shared
- **FoundationComponents** — ✅ single instance
- **Delegation Agents** — ✅ OnceLock per-model, tapi weights shared via Arc

### Gap Utama

1. ✅ **Config mismatch** — CF-1 resolved
2. ✅ **SpecialistModel** — CF-2 resolved (bridge trait + CoreController wiring)
3. ✅ **Classifier training pipeline** — CF-5 resolved (SGD trainer + save/load)
4. ✅ **Checkpoint loading** — CF-4 resolved (config-based standby model loading)
5. ✅ **Quarantine integration** — CF-3 resolved (AgentManager + dispatch)
6. 🟡 **EchoNet tidak thread-safe** — perlu fork-on-write
7. 🟡 **Paged cache single pool** — tidak support multi-dimensi model

### Potensi Penghematan Tambahan

| Area | Penghematan | Effort |
|------|:----------:|:------:|
| CPU offload untuk standby model | ~420 MB GPU → RAM | Medium |
| Multi-pool paged cache | Cache isolation + reuse | Medium |
| FP8/KV cache compression | 2-4× pada warm/cold tier | Medium |
| Gradient checkpointing | ~50% training memory | Low |
| Shared MoE experts training | ~144 MB | High |
| **Total potensi** | **~1 GB VRAM** | |

### Status Final

| Metrik | Nilai |
|--------|:-----:|
| Total crates | 40 |
| Total source files | 853 |
| Parameter total (semua model) | ~146M |
| VRAM aktif saat ini | ~420 MB (2 model) |
| VRAM maksimal (10 model) | ~584 MB (weights only) |
| VRAM dengan optimizer state | ~1.4 GB |
| Shared score rata-rata | 6.2/10 |
| Isolation score rata-rata | 5.1/10 |
| Critical issues | 0 (✅ CF-1, ✅ CF-2) |
| High issues | 0 (✅ CF-3, ✅ CF-4, ✅ CF-5) |
| Medium issues | 3 (CF-6, CF-7, CF-8) |
| Low issues | 1 (CF-10) |

---

## BAGIAN 8: FIX PROGRESS (Batch Fix 30)

### ✅ CF-1 — Config Mismatch Resolved (3 Juni 2026)

**Perubahan:**
- `crates/models/src/foundation.rs:transformer_config_for()` — synchronized with `init.rs` values:
  - Hidden: 512/384/256/128 (was 768/512/256/128)
  - Layers: 16/10/6/3 (was 24/12/8/4)
- `crates/foundation/src/init.rs:tier_config()` — now delegates to `transformer_config_for()` instead of duplicating config
- `crates/models/src/classifier_util.rs` — added `validate_embedding_dim()` runtime validation

**Verifikasi:** `cargo check -p nexora-models -p nexora-foundation` ✅ clean

### ✅ CF-2 — SpecialistModel Bridge Resolved (3 Juni 2026)

**Perubahan:**
- `crates/models/src/specialist.rs` (new) — `NxrSpecialist` trait + `NxrCoreSpecialistBridge`
- 10 `define_specialist!` structs (Omnis, Vortex, Aether, Spectra, Nexum, Axiom, Cipher, Swift, Kronos, Genesis)
- `HasSpecialistRegistry` trait + impl for `CoreController`
- `NxrCoreSpecialistBridge::register_all()` — registers all 10 specialists
- `crates/core/src/types.rs` — added `ModelId::Reasoning` variant for Omnis mapping
- `crates/core/src/execution/controller_models.rs` — added `process_reasoning()` dispatch
- `crates/core/src/coordination.rs` — added Reasoning dependency node

**Verifikasi:** `cargo check -p nexora-models -p nexora-core -p nexora-foundation -p nexora-inference` ✅ clean

### ✅ CF-3 — QuarantineManager Integration (3 Juni 2026)

**Perubahan:**
- `crates/agent/src/agent_manager.rs` — added `quarantine: Arc<RwLock<QuarantineManager>>` field
- `check_agent_quarantined(agent_id)` — quarantine check before every `send_message_internal`
- `dispatch_plan_internal` — skips quarantined workers during step dispatch
- Removed 5 dead `_*_isolation_check` stubs across agent, models, inference, api, runtime, intelligence crates

**Verifikasi:** `cargo check -p nexora-agent -p nexora-core -p nexora-models -p nexora-inference -p nexora-api -p nexora-runtime -p nexora-intelligence` ✅ clean

### ✅ CF-4 — Checkpoint Loading for Standby Models (3 Juni 2026)

**Perubahan:**
- `crates/foundation/src/init.rs` — `register_causal_lm()` now takes `checkpoints: &HashMap<NxrModelId, String>`
- `initialize_foundation_models_with_checkpoints()` — new entry point; loads weights from path
- `apps/nexora-ai/src/config/models.rs` — added `model_checkpoints: HashMap<String, String>` to `ModelsConfig`
- `ModelsConfig::resolved_checkpoints()` — string→NxrModelId conversion with validation

**Config example:**
```toml
[models]
model_checkpoints = { vortex = "./checkpoints/vtx.safetensors", aether = "./checkpoints/aeth.safetensors" }
```

**Verifikasi:** `cargo check -p nexora-foundation -p nexora-ai --lib` ✅ clean

### ✅ CF-5 — Classifier Training Pipeline (3 Juni 2026)

**Perubahan:**
- `crates/models/src/classifier_util.rs` — added `ClassifierWeights` (serializable), `save_classifier_weights()`, `load_classifier_weights()`, `train_classifier_sgd()`
- `train_classifier_sgd()` — manual backward SGD with cross-entropy loss, explicit gradient loops

**Usage:**
```rust
use crate::classifier_util::{train_classifier_sgd, save_classifier_weights};
train_classifier_sgd(&mut w1, &mut b1, &mut w2, &mut b2, &inputs, &labels, 0.01, 100);
save_classifier_weights("./emotion_clf.json", &w1, &b1, &w2, &b2);
```

**Verifikasi:** `cargo check -p nexora-models` ✅ clean

### Remaining Issues

| ID | Status | Severity | Description |
|----|--------|----------|-------------|
| CF-6 | 🟡 PENDING | Medium | EchoNet tidak thread-safe — perlu fork-on-write |
| CF-7 | 🟡 PENDING | Medium | TokenizerCore pakai std::HashMap (lambat + mahal) |
| CF-8 | 🟡 PENDING | Medium | Paged cache single pool — tidak support multi-dimensi |
| CF-10 | 🟢 PENDING | Low | Integration test untuk delegasi pipeline |
