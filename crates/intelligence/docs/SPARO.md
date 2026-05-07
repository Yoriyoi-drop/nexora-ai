Tentu, berikut adalah penggabungan keenam metode tersebut menjadi satu kerangka kerja baru yang koheren.

---

### **Metode Baru: *Self-Play Aligned Reasoning via Prospect-Theoretic Stepwise Optimization* (SPARO)**

#### **Inti Konsep**
**SPARO** adalah sebuah paradigma alignment yang memungkinkan model bahasa untuk meningkatkan dirinya secara iteratif tanpa data manusia tambahan, dengan menggabungkan **umpan balik langkah-demi-langkah** dari AI, **optimasi langsung preferensi** dalam dua moda (berpasangan dan independen), **regularisasi anti-overfitting**, serta **mekanisme self-play**.

Metode ini menggabungkan enam inovasi utama:
1. **DPO**: Menghilangkan reward model terpisah, mengubah pelatihan preferensi menjadi klasifikasi biner.
2. **KTO**: Menerima label “baik/buruk” secara independen tanpa data berpasangan, terinspirasi *Prospect Theory*.
3. **IPO**: Menyuntikkan regularisasi agar model tidak sekadar menghafal data preferensi.
4. **RLVF**: Mengevaluasi kualitas **setiap langkah penalaran**, bukan hanya jawaban akhir.
5. **SPIN**: Model berlatih tanding melawan dirinya sendiri untuk menjadi lebih kuat secara iteratif.
6. **RLAIF**: Menggunakan AI cerdas sebagai pemberi umpan balik, menggantikan peran manusia yang mahal.

#### **Cara Kerja**
1. **Pengumpulan Data dengan AI (RLAIF)**
   Pada setiap iterasi, model *student* menghasilkan beberapa *trace* lengkap (langkah demi langkah) untuk sebuah *prompt* penalaran. AI *judge* (misalnya model yang lebih besar atau model *teacher* yang sudah kuat) mengevaluasi setiap langkah:
   - Baik atau buruknya **langkah individu** (mengimplementasikan RLVF).
   - Bisa juga memberikan preferensi antar langkah (*pairwise*) **atau** hanya label independen (*KTO-compatible*).

2. **Format Umpan Balik Ganda**
   Data yang dihasilkan tersimpan dalam dua jenis:
   - **Pasangan preferensi (DPO-ready)**: “Langkah A lebih baik dari Langkah B” (untuk pasangan langkah atau respons lengkap).
   - **Label independen (KTO-ready)**: “Langkah ini baik / buruk” tanpa perlu pembanding.
   Keduanya bisa dicampur dalam satu batch; KTO menangani contoh yang tidak berpasangan, DPO menangani yang berpasangan.

3. **Fungsi Kerugian Terintegrasi**
   Fungsi loss SPARO adalah kombinasi tertimbang dari:
   - **L_DPO**: Kerugian klasifikasi biner ala DPO pada data berpasangan.
   - **L_KTO**: Kerugian *prospect-theoretic* ala KTO pada data independen, yang membandingkan log-probabilitas terhadap suatu nilai referensi.
   - **L_IPO_regularizer**: Istilah regularisasi kontras antara *policy* saat ini dan *reference policy* (model awal), yang menjaga generalisasi dan mencegah overfitting ala IPO.
   Total loss: \( \mathcal{L}_{\text{SPARO}} = \alpha \mathcal{L}_{\text{DPO}} + \beta \mathcal{L}_{\text{KTO}} + \gamma \mathcal{L}_{\text{IPO}} \)

4. **Pelatihan Self-Play Iteratif (SPIN)**
   - **Tahap 1**: *Student* (model yang akan dilatih) menghasilkan banyak solusi langkah-demi-langkah.
   - **Tahap 2**: *Teacher* bisa berupa (a) model *student* dari iterasi sebelumnya yang telah diperkuat dengan RLAIF, atau (b) AI *judge* eksternal. *Teacher* menghasilkan langkah-langkah “ideal” atau sekadar memilih langkah terbaik.
   - **Tahap 3**: *Student* dilatih dengan loss SPARO untuk membedakan langkah-langkah buatan *teacher* (baik) dari langkah-langkah buatan *dirinya sendiri* (buruk, kecuali sudah baik). Ini persis dengan kerangka SPIN: model belajar dengan kontras antara distribusinya sendiri dan distribusi yang lebih kuat.
   - Iterasi berlanjut: *student* yang sudah ditingkatkan menjadi *teacher* baru di putaran berikutnya, sehingga performa meningkat tanpa henti.

5. **Keunggulan Sinergis**
   - Karena umpan balik diberikan per langkah (RLVF), model dapat memperbaiki kesalahan kecil di tengah penalaran, bukan hanya menebak jawaban akhir.
   - DPO memungkinkan pemanfaatan preferensi kuat dari AI *judge*, sementara KTO mengizinkan pembelajaran bahkan ketika AI *judge* hanya sanggup memberi label mutlak (“langkah ini salah”) tanpa perbandingan, mencerminkan kenyataan data dunia nyata yang berantakan.
   - Regularisasi IPO mencegah *student* menjauh terlalu jauh dari kebijakan awalnya hanya untuk menyenangkan *judge*, sehingga pengetahuan generik bahasa tidak menguap.
   - Seluruh siklus didorong oleh AI *judge* (RLAIF) dan tidak membutuhkan anotasi manusia mahal, sementara iterasi SPIN memastikan perbaikan berkelanjutan.

#### **Kesimpulan**
**SPARO** bukan sekadar penjumlahan teknik, melainkan sebuah sintesis: ia mewarisi efisiensi DPO dan fleksibilitas data KTO, ketahanan regularisasi IPO, presisi langkah-demi-langkah RLVF, kemandirian data SPIN, dan skalabilitas RLAIF. Hasilnya adalah sebuah pipeline alignment yang mampu bekerja dengan sinyal pengawasan minimal dari manusia, terus memperbaiki diri sendiri, dan sangat cocok untuk tugas yang memerlukan penalaran bertahap seperti matematika, pemrograman, dan logika.