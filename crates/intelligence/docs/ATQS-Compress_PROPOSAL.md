# ATQS-Compress: Adaptive Tensor Quantum-Sparse Compression for Foundation Models

## PROPOSAL PENELITIAN

### Metode Baru Kompresi Foundation Model Berbasis Tensor Decomposition, Quantum-Inspired Network, dan Sparse Augmentation

**Bidang:** Artificial Intelligence | Foundation Models | Efficient Deep Learning  
**Tahun:** 2025 - 2026

---

## IDENTITAS PENELITIAN

### Judul Penelitian
ATQS-Compress: Adaptive Tensor Quantum-Sparse Compression for Foundation Models

### Bidang Ilmu
Artificial Intelligence, Deep Learning, Model Compression

### Kata Kunci
Foundation Model, Tensor Decomposition, Quantum-Inspired, Sparse Augmentation, LLM Compression

### Durasi
12 Bulan (Januari 2026 - Desember 2026)

### Referensi Utama
5 Jurnal Internasional (arXiv, IEEE)

---

## ABSTRAK

Foundation model seperti Large Language Models (LLM) dan Large Multimodal Models (LMM) telah merevolusi bidang kecerdasan buatan. Namun, ukurannya yang masif — seringkali mencapai miliaran parameter — menciptakan hambatan komputasi dan memori yang serius bagi deployment di lingkungan terbatas. Penelitian ini mengusulkan metode baru bernama ATQS-Compress (Adaptive Tensor Quantum-Sparse Compression), sebuah framework kompresi foundation model yang menggabungkan lima pendekatan mutakhir dari jurnal internasional terkini.

ATQS-Compress mengintegrasikan: (1) attention-aware joint tensor decomposition, (2) quantum-inspired tensor network berbasis MPO/iPEPS, (3) sparse augmentation pada layer redundan, (4) adaptive rank selection berbasis layer sensitivity profiling, dan (5) post-training calibration dengan LoRA residual. Target metode ini adalah reduksi memori lebih dari 85% dan pengurangan parameter lebih dari 75%, dengan penurunan akurasi di bawah 2% pada model LLaMA-2 7B dan 13B serta BERT-Large.

**Kata Kunci:** Foundation Model, Tensor Decomposition, Quantum-Inspired Network, Sparse Augmentation, LLM Compression, Attention-Aware, Tucker/TT Format, Layer Profiling

---

## BAB I — LATAR BELAKANG

### 1.1 Perkembangan Foundation Model
Foundation model merupakan terobosan terbesar dalam AI modern. Model-model seperti GPT-4, LLaMA, BERT, dan Stable Diffusion dilatih pada dataset raksasa dan dapat diaplikasikan lintas domain. Namun, skala parameter mereka tumbuh secara eksponensial: GPT-3 memiliki 175 miliar parameter, sementara model generasi terbaru diperkirakan melampaui angka triliun.

Konsekuensinya sangat besar: pelatihan ChatGPT-3 saja menghabiskan biaya listrik sekitar 100 juta dolar AS. Biaya ini diperkirakan berlipat ganda setiap sepuluh bulan. Di sisi inference, kebutuhan GPU kelas atas membuat deployment di perangkat edge — smartphone, IoT, sistem embedded — menjadi hampir mustahil tanpa kompresi model yang signifikan.

### 1.2 Permasalahan Utama
- Konsumsi memori GPU yang sangat tinggi (>80GB VRAM untuk LLM 70B+)
- Biaya inferensi yang mahal dan tidak berkelanjutan secara energi
- Ketidakmampuan deployment pada perangkat edge dan resource-constrained
- Metode kompresi existing (pruning, quantization, distillation) tidak mampu mencapai rasio kompresi tinggi tanpa degradasi akurasi signifikan
- Kurangnya framework kompresi yang mempertimbangkan struktur global antar-layer

### 1.3 Celah Penelitian (Research Gap)
Meskipun terdapat berbagai metode kompresi individual, belum ada framework yang secara holistik menggabungkan: analisis entanglement quantum antar-layer, adaptasi rank tensor berbasis sensitivitas layer, sparse augmentation, dan refinement global berbasis attention. Masing-masing paper terdahulu hanya mengoptimalkan satu aspek, sehingga masih ada ruang besar untuk meningkatkan rasio kompresi sambil mempertahankan akurasi.

---

## BAB II — TINJAUAN PUSTAKA

### 2.1 Lima Jurnal Acuan Utama
Metode ATQS-Compress dibangun di atas fondasi lima paper berikut:

| No | Judul Paper | Penulis/Institusi | Tahun | Kontribusi ke Metode |
|----|-------------|-------------------|-------|---------------------|
| 1 | LatentLLM: Attention-Aware Joint Tensor Compression | MERL (Koike-Akino et al.) | 2025 | Attention-aware global tensor decomposition, training-free |
| 2 | CompactifAI: Extreme Compression of LLMs using Quantum-Inspired Tensor Networks | Tomut et al. (Multiverse Computing) | 2024 | Quantum-inspired MPO tensor network, 93% memory reduction |
| 3 | Saten: Sparse Augmented Tensor Networks for Post-Training Compression of LLMs | arXiv 2505.14871 | 2025 | Low-rank + sparse augmentation, post-training pipeline |
| 4 | Tensor Decomposition for Model Reduction in Neural Networks: A Review | Liu & Parhi, Univ. Minnesota | 2023 | Landasan teoritis 6 metode dekomposisi tensor |
| 5 | KARIPAP: Quantum-Inspired Tensor Network Compression Using iPEPS-TRG | Nazri, Univ. Putra Malaysia | 2025 | Entanglement profiling, TRG coarse-graining, layer sensitivity |

### 2.2 Sintesis Literatur

#### 2.2.1 LatentLLM (2025) — Attention-Aware Joint Tensor Compression
Paper ini dari Mitsubishi Electric Research Laboratories mengusulkan kompresi LLM dengan mengubah dekomposisi tensor lokal (per-layer) menjadi dekomposisi tensor global yang dipandu oleh sinyal attention. Pendekatan ini memungkinkan kompresi multi-layer secara bersama tanpa memerlukan fine-tuning ulang. Kontribusi utamanya terhadap ATQS-Compress adalah mekanisme attention-aware sebagai panduan untuk memilih dimensi mana yang paling kritikal dipertahankan.

#### 2.2.2 CompactifAI (2024) — Quantum-Inspired MPO Compression
CompactifAI memperkenalkan paradigma baru: alih-alih melihat LLM sebagai kumpulan matriks bobot, ia memperlakukan model sebagai sistem kuantum dengan ruang korelasi yang dapat didekomposisi menggunakan Matrix Product Operators (MPO). Hasilnya: 93% reduksi memori pada LLaMA 7B dengan hanya 2-3% penurunan akurasi. Pendekatan ini memberi ATQS-Compress landasan untuk representasi tensor berbasis quantum entanglement.

#### 2.2.3 Saten (2025) — Sparse Augmented Tensor Networks
Saten mengatasi masalah kritis dalam tensorisasi post-training: LLM open-source tidak dilatih dengan struktur low-rank, sehingga dekomposisi langsung menghasilkan degradasi besar. Solusinya adalah menambahkan sparse residual mask pada tensor yang terdekomposisi. Pendekatan ini diadopsi ATQS-Compress sebagai komponen sparse augmentation untuk menstabilkan kualitas representasi setelah kompresi.

#### 2.2.4 Tensor Decomposition Review (2023) — Landasan Teoritis
Review komprehensif dari University of Minnesota ini mengevaluasi enam format dekomposisi tensor: Canonical Polyadic (CP), Tucker, Tensor-Train (TT), Tensor-Ring (TR), Block-Term, dan Hierarchical Tucker. Evaluasi pada CNN, RNN, dan Transformer memberikan panduan empiris untuk memilih format yang tepat sesuai struktur layer. ATQS-Compress menggunakan Tucker untuk layer attention dan TT untuk layer feed-forward berdasarkan rekomendasi review ini.

#### 2.2.5 KARIPAP (2025) — Entanglement Profiling dengan iPEPS-TRG
KARIPAP dari Universiti Putra Malaysia mengintegrasikan Infinite Projected Entangled Pair States (iPEPS) dengan Tensor Renormalization Group (TRG) untuk kompresi LLM. Temuan krusialnya: layer lebih dalam transformer menunjukkan pola entanglement redundan dan lebih cocok untuk tensorisasi. Layer-wise entanglement profiling ini menjadi komponen pertama dan paling mendasar dalam pipeline ATQS-Compress.

---

## BAB III — METODOLOGI PENELITIAN

### 3.1 Gambaran Umum Metode ATQS-Compress
ATQS-Compress adalah framework kompresi lima fase yang dapat diterapkan pada foundation model yang sudah dilatih (post-training), tanpa memerlukan akses ke data pelatihan penuh. Setiap fase memanfaatkan teknik dari salah satu paper acuan, dan keseluruhan pipeline didesain agar modular dan dapat dikonfigurasi sesuai target rasio kompresi.

### 3.2 Pipeline ATQS-Compress

| Step | Fase | Teknik / Proses | Sumber Paper |
|------|------|----------------|--------------|
| 1 | Layer Entanglement Profiling | Analisis quantum entanglement tiap layer transformer untuk menentukan redundansi parameter | KARIPAP (#5) |
| 2 | Adaptive Rank Selection | Pilih rank tensor secara adaptif per-layer berdasarkan sensitivitas (Tucker / TT format) | Paper #4 Review + KARIPAP (#5) |
| 3 | Quantum-Sparse Tensorization | Dekomposisi bobot dengan MPO/iPEPS dan tambahkan sparse mask pada layer redundan | CompactifAI (#2) + Saten (#3) |
| 4 | Global Attention-Aware Refinement | Optimasi joint compression antar-layer dipandu sinyal attention untuk menjaga representasi semantik | LatentLLM (#1) |
| 5 | Post-Training Calibration | Fine-tuning ringan dengan LoRA residual untuk memulihkan akurasi setelah kompresi | Saten (#3) + CompactifAI (#2) |

### 3.3 Detail Teknis Per Fase

#### Fase 1: Layer Entanglement Profiling
Input: Foundation model pretrained (misalnya LLaMA-2 7B). Setiap layer transformer dievaluasi menggunakan metrik quantum entanglement entropy. Layer dengan entanglement entropy rendah diklasifikasikan sebagai 'highly compressible'. Output fase ini adalah peta sensitivitas layer (layer sensitivity map) yang menjadi panduan seluruh proses kompresi berikutnya.

#### Fase 2: Adaptive Rank Selection
Berdasarkan sensitivity map, setiap layer diberikan rank tensor yang berbeda. Layer kritis (early layers, layer attention utama) mendapat rank tinggi untuk mempertahankan representasi. Layer redundan (deep layers, layer MLP tertentu) mendapat rank rendah untuk kompresi maksimal. Format dekomposisi dipilih adaptif: Tucker untuk weight matrix attention, Tensor-Train untuk feed-forward network.

#### Fase 3: Quantum-Sparse Tensorization
Bobot setiap layer didekomposisi menggunakan MPO (Matrix Product Operator) terinspirasi quantum. Setelah dekomposisi, sparse mask ditambahkan untuk mengkompensasi error dekomposisi pada layer yang tidak sepenuhnya low-rank. Sparse mask hanya mempertahankan top-k% elemen residual terbesar, memberikan representasi Low-Rank + Sparse yang jauh lebih ekspresif.

#### Fase 4: Global Attention-Aware Refinement
Setelah kompresi per-layer selesai, dilakukan optimasi joint antar-layer. Sinyal attention dari layer-layer kritis digunakan sebagai panduan untuk menyesuaikan rank dan sparse mask secara global, memastikan konsistensi representasi semantik lintas layer. Fase ini unik karena sebagian besar metode sebelumnya hanya mengoptimalkan kompresi secara lokal.

#### Fase 5: Post-Training Calibration
Fine-tuning ringan dilakukan menggunakan LoRA (Low-Rank Adaptation) sebagai residual adapter pada subset layer terkritis. Kalibrasi hanya memerlukan sebagian kecil data (calibration set ~1000 sampel), bukan dataset pelatihan penuh. Fase ini memulihkan akurasi tanpa mengembalikan ukuran model ke kondisi semula.

---

## BAB IV — KONTRIBUSI BARU & KEUNGGULAN

### 4.1 Inovasi Utama ATQS-Compress
ATQS-Compress membawa beberapa kontribusi baru yang belum ada pada metode sebelumnya:

1. **Pertama kalinya quantum entanglement profiling diintegrasikan secara langsung ke dalam pipeline pemilihan rank tensor adaptif untuk foundation model skala produksi.**
2. **Kombinasi Low-Rank Decomposition + Sparse Augmentation + Global Attention Refinement dalam satu framework terpadu** — tidak ada paper sebelumnya yang menggabungkan ketiga komponen ini.
3. **Pipeline yang sepenuhnya training-free pada 4 dari 5 fase, dengan kalibrasi minimal hanya pada fase terakhir, menjadikannya praktis untuk deployment industri.**
4. **Dukungan multi-format dekomposisi (Tucker dan TT) yang dipilih secara adaptif per jenis layer, berbeda dari pendekatan one-size-fits-all pada paper sebelumnya.**

### 4.2 Perbandingan dengan State-of-the-Art

| Metrik | Target ATQS-Compress | Baseline Terbaik |
|--------|---------------------|------------------|
| Reduksi Memori | > 85% | 93% (KARIPAP, single method) |
| Pengurangan Parameter | > 75% | 70% (CompactifAI) |
| Penurunan Akurasi | < 2% | 2-3% (CompactifAI) |
| Speedup Training | > 50% | 50% (KARIPAP) |
| Speedup Inference | > 30% | 25% (CompactifAI) |
| Target Model | LLaMA-2 7B, 13B; BERT-Large | LLaMA-2 7B |

### 4.3 Novelty Statement
Tidak ada paper yang saat ini menggabungkan quantum entanglement profiling (KARIPAP), attention-aware joint tensor decomposition (LatentLLM), quantum-inspired MPO tensorization (CompactifAI), sparse augmentation post-training (Saten), dan adaptive multi-format rank selection (Tensor Decomposition Review) dalam satu metode terintegrasi. ATQS-Compress adalah yang pertama melakukan hal ini.

---

## BAB V — RENCANA PENELITIAN

### 5.1 Dataset dan Model Eksperimen
- **LLaMA-2 7B dan 13B** (Meta AI) — model bahasa generatif utama
- **BERT-Large** (Google) — model understanding bidireksional
- **OPT-6.7B** (Meta AI) — baseline kompresi
- **Benchmark:** WikiText-2, C4, MMLU, HellaSwag, ARC

### 5.2 Lingkungan Eksperimen
- **Hardware:** 4x NVIDIA A100 80GB atau H100 80GB
- **Framework:** PyTorch 2.x, Hugging Face Transformers, custom tensor decomposition library
- **Bahasa:** Python 3.11+

### 5.3 Metrik Evaluasi
- **Compression ratio** (rasio kompresi parameter)
- **Memory footprint reduction** (reduksi footprint memori GPU)
- **Perplexity** (untuk language modeling tasks)
- **Accuracy pada benchmark downstream** (MMLU, HellaSwag, ARC)
- **Inference latency** (ms/token)
- **Training speedup** (x faktor akselerasi)

### 5.4 Timeline Penelitian

| Bulan 1-2 | Bulan 3-4 | Bulan 5-6 | Bulan 7-8 | Bulan 9-10 | Bulan 11-12 |
|-----------|-----------|-----------|-----------|------------|-------------|
| Studi Literatur & Reproduksi Baseline | Implementasi Layer Profiling | Pengembangan Quantum-Sparse Tensorization | Integrasi Attention-Aware Refinement | Eksperimen & Evaluasi Benchmark | Penulisan Paper & Diseminasi |

---

## BAB VI — DAFTAR REFERENSI

1. Koike-Akino, T., Chen, X., Liu, J., Wang, Y., Wang, P., & Brand, M. (2025). **LatentLLM: Attention-Aware Joint Tensor Compression**. arXiv:2505.18413. Mitsubishi Electric Research Laboratories.
2. Tomut, A., et al. (2024). **CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks**. arXiv:2401.14109. Multiverse Computing.
3. arXiv:2505.14871. (2025). **Saten: Sparse Augmented Tensor Networks for Post-Training Compression of Large Language Models**. arXiv preprint.
4. Liu, X., & Parhi, K. K. (2023). **Tensor Decomposition for Model Reduction in Neural Networks: A Review**. arXiv:2304.13539. University of Minnesota.
5. Nazri, A. (2025). **KARIPAP: Quantum-Inspired Tensor Network Compression of Large Language Models Using Infinite Projected Entangled Pair States and Tensor Renormalization Group**. arXiv:2510.21844. University Putra Malaysia.
6. Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. P. (2015). **Tensorizing Neural Networks**. NeurIPS 2015.
7. Hu, E. J., et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.
8. Vaswani, A., et al. (2017). **Attention Is All You Need**. NeurIPS 2017.
9. Touvron, H., et al. (2023). **LLaMA 2: Open Foundation and Fine-Tuned Chat Models**. Meta AI Research.
10. Ma, X., et al. (2019). **A Tensorized Transformer for Language Modeling**. NeurIPS 2019.

---

## LAMPIRAN — DIAGRAM ARSITEKTUR ATQS-Compress

```
[ ALUR PIPELINE ATQS-Compress ]

INPUT
Foundation Model Pretrained
(LLaMA-2 7B/13B, BERT-Large, OPT)
  ▼
FASE 1
Layer Entanglement Profiling
→ Quantum entanglement entropy per layer
→ Output: Layer Sensitivity Map
  ▼
FASE 2
Adaptive Rank Selection
→ Rank tinggi: layer kritis (attention utama)
→ Rank rendah: layer redundan (deep MLP)
→ Format: Tucker (attention) | TT (FFN)
  ▼
FASE 3
Quantum-Sparse Tensorization
→ MPO/iPEPS decomposition
→ Sparse residual mask (top-k%)
→ Representasi: Low-Rank + Sparse
  ▼
FASE 4
Global Attention-Aware Refinement
→ Joint optimization antar-layer
→ Dipandu sinyal attention layer kritis
→ Konsistensi semantik lintas layer
  ▼
FASE 5
Post-Training Calibration
→ LoRA residual adapter
→ ~1000 sampel kalibrasi
→ Pemulihan akurasi minimal
  ▼
OUTPUT
Model Terkompresi
> 85% reduksi memori | > 75% reduksi parameter
< 2% penurunan akurasi | Siap edge deployment
```

---

**ATQS-Compress** - Making Foundation Models Accessible for Edge Deployment
