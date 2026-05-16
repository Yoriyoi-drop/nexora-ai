# K-Means Layer di Nexora-AI

> **Definisi:** K-Means pada layer AI adalah jembatan untuk menyederhanakan data yang kompleks, mengelompokkan informasi secara mandiri, atau mengompres model agar AI bekerja lebih cepat dan efisien.

---

## 1. ERP — Echo Resonance Processing (Clustering Neural)

**Lokasi:** `crates/foundation/src/erp/`

Komponen paling mendekati K-Means di project ini. Neuron dengan distribusi informasi mirip dideteksi dan dikelompokkan secara otomatis.

### Cara Kerja
| Tahap | Deskripsi |
|-------|-----------|
| **Signature Extraction** | Setiap neuron diekstrak information distribution-nya, lalu diproyeksi ke low-dimensional space (Random Projection / PCA / Low-Rank Embedding) |
| **Two-Stage Filtering** | Stage 1: Locality-Sensitive Hashing untuk kandidat cepat. Stage 2: Exact resonance dengan KL divergence |
| **Graph Building** | Neuron jadi node, resonance pairs jadi edge berbobot |
| **Clustering** | Graf di-cluster dengan 3 algoritma |

### 3 Algoritma Clustering

| Algoritma | Mode | Fungsi |
|-----------|------|--------|
| **Louvain Clustering** | Conservative | Modularity-based community detection. Neuron dalam komunitas sama punya distribusi informasi serupa |
| **Spectral Clustering** | Balanced | Pakai Laplacian matrix & eigenvalue decomposition. Jumlah cluster ditentukan (`n_clusters: 32`) |
| **Adaptive Modular** | Aggressive | BFS-based grouping dengan batas ukuran group (min-max). Group yang terlalu kecil di-merge ke group terdekat |

### Output
- **ResonanceGroup:** Kumpulan neuron yang "beresonansi" — bisa dikompres bareng karena redundan secara informasi
- Setiap group punya `stability_variance` (homogenitas internal) dan `importance_scores` (Fisher info + gradient norm)

---

## 2. ATQS — Adaptive Tensor Quantum Sparsification (Layer Grouping & Kompresi)

**Lokasi:** `crates/foundation/src/atqs/profiling/`

### Sensitivity Clustering
Layer-layer dalam neural network dikelompokkan berdasarkan **sensitivitas** terhadap kompresi:

| Kategori | Arti |
|----------|------|
| **High Sensitivity** | Layer penting — dikompres dengan hati-hati (low compression) |
| **Low Sensitivity** | Layer redundan — bisa dikompres agresif atau dipruning |
| **Sensitivity Clusters** | Layer dengan sensitivitas mirip dikelompokkan untuk strategi kompresi seragam |

### Entanglement Clustering
Layer-layer yang **terjerat (redundant)** dideteksi dengan korelasi mutual information:

| Kategori | Arti |
|----------|------|
| **Redundancy Clusters** | Layer yang saling tergantung — jika satu dipruning, yang lain harus ikut |
| **Independent Layers** | Layer bisa diproses sendiri tanpa efek samping |

### Hasil Akhir
- Setiap layer dapat strategi kompresi berbeda:
  - **High sensitivity** → sparse augmentation ringan
  - **Low sensitivity** → kuantisasi agresif atau pruning
  - **Redundancy cluster** → semuanya dikompres bareng dengan format sama

---

## 3. Data Pipeline — Deduplikasi & Filtering

**Lokasi:** `crates/data/src/deduplicator.rs`, `crates/datastream/src/filter/`

### MinHash LSH (Locality-Sensitive Hashing)
Dokumen yang mirip dikelompokkan (clustering) untuk di-dedup:
- **ExactHash:** Dedup exact (hash identik)
- **MinHashLSH:** Dedup fuzzy — dokumen mirip tapi tidak identik tetap terdeteksi
- **Jaccard Similarity:** Ukur kemiripan antar dokumen dalam cluster
- **Cosine Similarity:** Embedding-based similarity
- **Semantic Dedup:** Pakai makna/konten, bukan token literal

### 17 Filter Types
Data kompleks disederhanakan via filter berlapis:
- Length, Language, Regex — filter dasar
- Quality, Toxicity, PromptInjection — filter keamanan & kualitas
- Perplexity, Entropy — filter kebisingan
- SemanticDedup, CurriculumRanker — filter cerdas

---

## 4. Memory — Konsolidasi Pola

**Lokasi:** `crates/memory/src/memory_model.rs`

### Clustering Memory
- **Hebbian memory:** Pola yang sering muncul bareng dikonsolidasi jadi satu
- **Short → Long term:** Episode serupa dikelompokkan, pola berulang dipromosikan ke long-term memory
- **LRU Cache:** Akses terbaru di-cluster berdasarkan temporal locality

### Cognitive Dynamics
- **Field intensity:** Interaksi linear + cubic + regularization antar memori
- **Entropy:** Neural noise + interference + forgetting — ukur seberapa "berantakan" cluster memori
- **Emergence:** Cluster baru muncul secara mandiri via sigmoid-gated coherence

---

## 5. Deep Learning — Quantization & Kompresi Model

**Lokasi:** `crates/deeplearning/src/star_x/quantization.rs`

### Weight Quantization
Bobot model dikelompokkan ke dalam representasi presisi lebih rendah:
- **FP32 → INT8:** Weight value di-cluster ke 256 kemungkinan nilai
- **Group-wise quantization:** Weight per layer dikelompokkan, tiap group punya scale & zero-point sendiri
- **Result:** Model lebih kecil, inference lebih cepat

---

## Ringkasan Konseptual

Komponen project yang menjalankan fungsi K-Means layer:

| Komponen | Simplifikasi Data | Clustering Mandiri | Kompresi Model |
|----------|:-:|:-:|:-:|
| ERP Resonance Clustering | ✅ Neuron signature → group | ✅ Otomatis via graph clustering | ✅ Group dikompres bareng |
| ATQS Sensitivity Clustering | ✅ Layer → sensitivity group | ✅ Threshold-based | ✅ Strategi kompresi per group |
| ATQS Entanglement Clustering | ✅ Layer → redundancy group | ✅ Korelasi mutual info | ✅ Pruning bareng |
| MinHash LSH Dedup | ✅ Dokumen → similarity cluster | ✅ Jaccard/Cosine | ✅ Penyimpanan lebih efisien |
| Hebbian Memory | ✅ Pola → memori terkonsolidasi | ✅ Pola berulang | ✅ Memori jangka panjang |
| Weight Quantization | ✅ Bobot → INT8 cluster | ✅ Group-wise | ✅ Model lebih kecil |

Inti dari K-Means layer di Nexora-AI adalah: **mengelompokkan unit-unit yang mirip (neuron, layer, dokumen, memori, bobot) agar bisa diproses lebih efisien — baik via kompresi, deduplikasi, pruning, atau konsolidasi.**
