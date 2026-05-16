# Adaptive Representation Clustering Architecture

> Bukan K-Means biasa. Arsitektur ini mencakup **seluruh spektrum pengelompokan representasi** — dari neuron-level graph clustering, layer-level sensitivity grouping, vector quantization, hierarchical abstraction, hingga similarity-based retrieval — agar AI bekerja lebih cepat, efisien, dan adaptif.

---

## Level 1: Neuron-Level Clustering (ERP)

Komponen clustering terdalam. Neuron dengan distribusi informasi mirip dideteksi dan dikelompokkan secara mandiri.

| Sub-Komponen | Metode | Output |
|---|---|---|
| **Signature Extraction** | Random Projection / PCA / Low-Rank Embedding | NeuronSignature (projection, fisher_info, gradient_norm) |
| **Two-Stage Filtering** | Stage 1: cosine similarity cepat. Stage 2: KL divergence eksak | Resonance pairs |
| **Graph Building** | Neuron = node, resonance = edge berbobot | ResonanceGraph |
| **Louvain Clustering** | Modularity-based community detection | ResonanceGroup (Conservative mode) |
| **Spectral Clustering** | Laplacian matrix + eigenvalue decomposition | 32 cluster (Balanced mode) |
| **Adaptive Modular** | BFS expansion + size constraints + auto-merge group kecil | ResonanceGroup (Aggressive mode) |
| **Pattern Cache** | Context similarity clustering (cosine + Euclidean) | PatternCluster untuk gating |

**Alur:** Weights → Neuron signatures → Filtering → Graph → Cluster → Group compression

---

## Level 2: Layer-Level Clustering (ATQS)

Layer dalam neural network dikelompokkan berdasarkan sensitivitas dan keterkaitan untuk strategi kompresi.

| Sub-Komponen | Metode | Fungsi |
|---|---|---|
| **Sensitivity Clustering** | Threshold-based (similarity > 0.8) | Layer dengan sensitivitas mirip → strategi kompresi seragam |
| **Entanglement Clustering** | Mutual information correlation | Layer redundan → pruning bareng |
| **Adaptive Quantization** | Sensitivity-based bit-width assignment | INT4 untuk low sensitivity, INT8 untuk high sensitivity |

**Output per layer:** CompressionStrategy — sparse / prune / quantize — berdasarkan cluster membership.

---

## Level 3: Embedding-Level Quantization (CAFFEINE VQ-VAE)

Vector quantization — bentuk clustering paling klasik (k-means-like).

| Komponen | Detail |
|---|---|
| **Codebook** | 8 codebooks × 1024 entries per codebook |
| **Assignment** | `find_closest_code()` — Euclidean distance ke centroid terdekat |
| **Update** | Exponential Moving Average (EMA) — centroid bergeser adaptif |
| **Fungsi** | Embedding kontinu → token diskrit untuk multimodal |

**Konsep:** Setiap codebook entry adalah centroid/representative. Input di-cluster ke centroid terdekat — persis seperti K-Means inference.

---

## Level 4: Hierarchical Representation Grouping (ECHO-Net RHC)

Kompresi representasi hierarkis — dari detail ke abstraksi.

| Level | Granularitas | Metode |
|---|---|---|
| Level 0 | Token detail | FFT-Pool |
| Level 1 | Frasa | FFT-Pool + ResSummary |
| Level 2 | Paragraf | FFT-Pool + ResSummary |
| Level 3 | Konsep | FFT-Pool + ResSummary |
| Level 4 | Abstraksi global | FFT-Pool + ResSummary |

**Formula:** `C(H) = FFT-Pool(H) + ResSummary(H)` — representasi dikelompokkan ke level abstraksi via hierarchical pooling. Semantic zooming.

Didukung oleh:
- **APSS (Adaptive Phase Separation):** Pairwise cosine similarity matrix antar token → fase dipisah berdasarkan semantic dissimilarity
- **TkRR (Top-K Resonance Routing):** Similarity threshold untuk diversity filtering

---

## Level 5: Memory-Level Clustering & Konsolidasi

Pola-pola berulang dikelompokkan dan dipromosikan dari memori jangka pendek ke panjang.

| Komponen | Metode | Lokasi |
|---|---|---|
| **k-NN Local Competition** | Cosine similarity + nearest neighbor | `memory/src/memory_model.rs` |
| **STAR-X EMR Consolidation** | Priority-based grouping + averaging | `deeplearning/src/star_x/emr.rs` |
| **Hebbian Memory** | Co-occurrence pattern grouping | `memory/src/memory_model.rs` |
| **LRU Temporal Clustering** | Akses temporal terdekat | `memory/src/memory_model.rs` |

**STAR-X EMR:** Episodic memory entries dikelompokkan berdasarkan priority range, lalu di-average untuk konsolidasi. `associative_retrieval()` mencari similaritas untuk retrieval dengan dedup.

---

## Level 6: Decoding-Level Grouping (Beam Search)

Hipotesis dalam beam search yang konvergen (text similarity > threshold) dikelompokkan dan di-collapse — hanya menyisakan hipotesis terbaik per cluster.

| Komponen | Metode | Lokasi |
|---|---|---|
| **Convergence Grouping** | Text similarity threshold | `inference/src/beam_search.rs` |
| **Beam Pruning** | Divergence penalty | `inference/src/beam_search.rs` |

---

## Level 7: Data-Level Clustering (Pipeline)

Dokumen serupa dikelompokkan untuk dibersihkan atau disaring.

| Komponen | Metode | Lokasi |
|---|---|---|
| **MinHash LSH** | Bands-based bucketing + Jaccard similarity | `data/src/deduplicator.rs` |
| **Semantic Dedup** | Cosine similarity + semantic embedding | `datastream/src/filter/semantic_dedup.rs` |
| **ExactHash Dedup** | Hash identik | `data/src/deduplicator.rs` |
| **Cosine Dedup** | TF-IDF cosine vectors | `data/src/deduplicator.rs` |

---

## Level 8: Routing-Level Grouping

Token/rute dikelompokkan berdasarkan modality atau expert assignment.

| Komponen | Metode | Lokasi |
|---|---|---|
| **HAS-MoE-FFN Routing** | Top-k expert assignment | `foundation/src/has_moe_ffn/routing.rs` |
| **CAFFEINE Modality Router** | Modality-based expert dispatch | `foundation/src/multimodal/caffeine/mod.rs` |
| **Grouped Query Attention** | num_heads / num_kv_heads | `foundation/src/models/transformer/gqa.rs` |

---

## Level 9: Similarity-Based Caching (Swift FastCache)

Embedding query dikelompokkan ke similarity buckets untuk cache hit detection.

| Komponen | Metode | Lokasi |
|---|---|---|
| **FastCache** | Cosine/Jaccard similarity + distribution bucketing (0.9+, 0.7+, etc.) | `foundation/src/models/swift/agents/fast_cache.rs` |

---

## Ringkasan 9 Level

| Level | Komponen | Unit yg Di-cluster | Metode |
|---|---|---|---|
| 1 | ERP | Neuron | Louvain / Spectral / Adaptive Modular |
| 2 | ATQS Profiling | Layer neural | Threshold similarity / Mutual info |
| 3 | CAFFEINE VQ-VAE | Embedding vectors | Nearest-centroid (EMA codebook) |
| 4 | ECHO-Net RHC | Representasi temporal | FFT-Pool hierarkis |
| 5 | Memory (k-NN + EMR + Hebbian) | Memori episodik | Cosine similarity / Priority grouping |
| 6 | Beam Search | Hipotesis decoding | Text similarity |
| 7 | Data Pipeline | Dokumen | MinHash LSH / Jaccard / Cosine |
| 8 | Routing (MoE + GQA + Modality) | Token / Query | Top-k / Modality rule |
| 9 | Swift FastCache | Cache entries | Cosine/Jaccard bucketing |

Semua level ini membentuk **Adaptive Representation Clustering Architecture** — bukan satu algoritma, tapi ekosistem clustering adaptif di seluruh stack AI: dari neuron hingga dokumen, dari training hingga inference.
