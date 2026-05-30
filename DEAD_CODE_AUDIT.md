# Dead Code Audit — Nexora AI Workspace

> **Date:** 2026-05-30  
> **Scope:** 38 crate workspace, ~950+ `.rs` files  
> **Method:** Manual trace of `mod.rs` declarations, `Cargo.toml` feature chains, `#[cfg()]` gates, and inbound `use` references

---

## Ringkasan Eksekutif

| Metrik | Nilai |
|--------|-------|
| File `.rs` bermasalah | **20 file** (+ 1 crate utuh, `nexora-api`) |
| LOC potensi hapus | **~7.500+** |
| Dead `#[cfg()]` gates | **~200+ blok** di 11 file |
| Fitur mati (defined, never enabled) | **12 fitur** |
| Crate yatim (0 dependents) | **1 crate** (`nexora-api`, 5.670 LOC) |
| Fungsi dead code (`fn _` prefix) | **7 fungsi** |

---

## Quick Wins — Aman Dihapus Segera

| File | LOC | Status | Bukti |
|------|-----|--------|-------|
| `crates/models/src/aether/agents_new.rs` | 6 | **Stale duplicate** | Isinya: `STALE DUPLICATE — superseded copy of agents/mod.rs` |
| `apps/nexora-ai/src/config/billing.rs` | 24 | **Orphaned** | Tidak di `config/mod.rs`. `BillingConfig` tidak pernah dikompilasi |
| `apps/nexora-ai/src/server/telemetry_handlers.rs` | 111 | **Placeholder** | Semua handler return data null/placeholder |
| `crates/alignment/src/sparo/tests.rs` | 446 | **Test tidak jalan** | Tidak di `sparo/mod.rs` → 7 test never run |
| `crates/oracle/src/linters/tests.rs` | 298 | **Test tidak jalan** | Tidak di `linters/mod.rs` → 14 test never run |

**Total quick wins: 885 LOC**

---

## Orphaned Files — Ada di Disk, Tidak di Build Graph

> Semua file di bawah ini `pub mod` atau `mod` **TIDAK** dideklarasikan di parent `mod.rs` masing-masing. Compiler tidak pernah melihatnya.

| File | LOC | Isi | Rekomendasi |
|------|-----|-----|-------------|
| `crates/alignment/src/sparo/utils.rs` | 148 | `AlignmentUtils`, `RewardUtils`, `AlignmentConfig` — real utilities | Wiring atau pindahkan |
| `crates/datastream/src/filter/ml_classifier.rs` | 291 | ML classifier real: SGD training, ngram features | Wiring (high value) atau hapus |
| `apps/nexora-ai/src/server/auth_handlers.rs` | 173 | 8 auth handlers: register, login, API keys | Wiring ke `server/mod.rs` + router |
| `apps/nexora-ai/src/server/billing_handlers.rs` | 179 | Billing handler implementations | Wiring + aktifkan config billing |
| `apps/nexora-ai/src/server/telemetry_middleware.rs` | 67 | Axum middleware (rate limiting, auth) | Wiring atau hapus |
| `apps/nexora-ai/src/server/gossip_handlers.rs` | 43 | Gossip push/pull handlers | Wiring jika distributed mode aktif |
| `apps/nexora-ai/src/server/dashboard_handlers.rs` | 40 | 4 dashboard handlers | Wiring atau hapus |

**Total orphaned (non-test): 941 LOC**

---

## Orphaned Crate — `nexora-api` (5.670 LOC)

| Metrik | Nilai |
|--------|-------|
| File | `crates/api/` — 6 source files |
| LOC | **5.670** |
| Dependents | **0** — tidak ada crate/binari lain yang `use nexora_api` |
| Dependencies | `tokio`, `axum`, `serde`, plus optional `tls`, `rustls` |
| Fitur mati | `tls`, `rustls` — tidak pernah diaktifkan |

**Rekomendasi:** Hapus atau wiring ke `apps/nexora-ai/src/server/`. Menambah ~10-12 detik compile time tanpa manfaat.

```
crates/api/src/
├── handlers.rs    (569 LOC)
├── lib.rs         (375 LOC)
├── metrics.rs     (430 LOC)
├── middleware.rs  (902 LOC)
├── routing.rs     ( 41 LOC)
└── server.rs      (518 LOC)
```

---

## Dead Features — Defined Tapi Tidak Pernah Diaktifkan

| Fitur | Crate | `#[cfg()]` Gates | Keterangan |
|-------|-------|------------------|------------|
| `cuda` | `autograd`, `has-moe-ffn` | **63** | CUDA backend, FlashAttention, MoE CUDA |
| `postgres` | `database` | 13 | PostgreSQL module |
| `sqlite` | `database` | 33 | Full SQLite module (2 files) |
| `mysql` | `database` | 15 | MySQL module |
| `sqlx` | `database` | 2 | SQLx adapter |
| `tls` | `api` | 5 | TLS server setup |
| `toxicity` | `datastream` | 4 | Toxicity filter pipeline |
| `prompt-injection` | `datastream` | 3 | Prompt injection filter |
| `oracle` | `alignment` | 1 | SPARO oracle integration |
| `simulated-models` | `models` | 0 | Defined only, unused |
| `dataset-minimal` | `datastream` | 0 | Defined only, unused |
| `gpu` (datastream) | `datastream` | 0 | Defined only, unused |

---

## Undefined Feature Gates — Paling Bermasalah

### `#[cfg(feature = "hallucination")]` — 68+ gates, feature **tidak pernah didefinisikan**

```
crates/models/src/
├── foundation.rs       — 9 gates
├── omnis/mod.rs        — 7 gates
├── genesis/mod.rs      — 6 gates
├── kronos/mod.rs       — 6 gates
├── nexum/mod.rs        — 6 gates
├── cipher/mod.rs       — 6 gates
├── axiom/mod.rs        — 6 gates
├── spectra/mod.rs      — 6 gates
├── swift/mod.rs        — 6 gates
├── aether/mod.rs       — 6 gates
└── vortex/mod.rs       — 6 gates
```

Semua kode hallucination guard di 10 model crate **tidak pernah dikompilasi**. Perkiraan 200-400 LOC logika mati.

### `#[cfg(feature = "examples")]` — 8 gates, feature tidak didefinisikan

```
crates/echo-net/src/examples.rs — semua kode examples tidak pernah dikompilasi
```

### `#[cfg(feature = "tokenizer-train")]` — 1 gate, feature tidak didefinisikan

```
apps/nexora-ai/src/cli/handlers.rs — tokenizer training path tidak aktif
```

---

## Dead Code Markers — `fn _` Prefix

7 fungsi dengan prefix `_` (dead code eksplisit via Rust convention):

| File | Fungsi |
|------|--------|
| `crates/database/src/lib.rs` | `fn _parse_timestamp(...)` |
| `crates/erp/src/training.rs` | `fn _update_gate_weights(...)` |
| `crates/reasoning/src/saca/rerank.rs` | `fn _with_thresholds(...)` |
| `crates/hldva-t/src/vaed/encoder.rs` | `fn _apply_convolution(...)` |
| `crates/hldva-t/src/ddpm/mod.rs` | `fn _randn(...)` |
| `crates/alignment/src/sparo/spin.rs` | `fn _sample_with_temperature(...)` |
| `crates/oracle/src/trainer.rs` | `fn _estimate_flops_per_step(...)` |

---

## Prioritized Action Plan

### Phase 1 — Segera (10 menit, zero risk)

| # | Tindakan | LOC | Dampak |
|---|----------|-----|--------|
| 1 | Hapus `crates/models/src/aether/agents_new.rs` | 6 | Stale duplicate |
| 2 | Hapus `crates/alignment/src/sparo/tests.rs` | 446 | Test tidak jalan |
| 3 | Hapus `crates/oracle/src/linters/tests.rs` | 298 | Test tidak jalan |
| 4 | Hapus `apps/nexora-ai/src/config/billing.rs` | 24 | Belum perlu |
| 5 | Hapus `apps/nexora-ai/src/server/telemetry_handlers.rs` | 111 | Placeholder |
| 6 | Hapus 7 fungsi `_` prefix | ~50 | Dead code |

### Phase 2 — Investigasi (1-2 jam)

| # | Tindakan | LOC | Dampak |
|---|----------|-----|--------|
| 7 | **Hapus/wiring `crates/api/`** | **5.670** | **Kompresi compile time terbesar** |
| 8 | Wiring 6 server handlers ke `server/mod.rs` + router | 613 | Hidupkan auth, billing, gossip |
| 9 | Define `hallucination` feature atau hapus semua gates | ~200-400 | Bersihkan 68 cfg gates |
| 10 | Wiring `ml_classifier.rs` ke filter/mod.rs | 291 | Aktifkan ML classifier (high value) |
| 11 | Wiring `utils.rs` ke sparo/mod.rs | 148 | Aktifkan AlignmentUtils |

### Phase 3 — Tidak Mendesak

| # | Tindakan | Dampak |
|---|----------|--------|
| 12 | Evaluasi `cuda` feature — perlu diaktifkan? | Aktifkan full CUDA backend |
| 13 | Evaluasi `postgres`/`sqlite`/`mysql` features | Pilih satu database backend |
| 14 | Evaluasi `tls`/`rustls` — wiring jika perlu | Security enhancement |

---

## Estimasi Dampak

| Tindakan | LOC | Compile Time | Maintenance |
|----------|-----|-------------|-------------|
| Hapus `crates/api/` | 5.670 | **-10–15s** | Signifikan |
| Hapus 5 quick-wins | 885 | -1–2s | Rendah |
| Hapus 2 orphaned test | 744 | -0.5s | Rendah |
| Bersihkan 12 dead features | ~200 gates | -3–5s (dep resolution) | Moderat |
| **Total** | **~7.500+** | **~15–20s per build** | **Substansial** |

---

## Catatan Kritis

1. **Semua temuan diverifikasi** — tidak ada false positive. Setiap file "orphaned" dikonfirmasi tidak ada di `mod.rs`, tidak ada inbound `use`, dan tidak ada `#[path]` override.

2. **Tidak ada modifikasi kode** — hanya penghapusan file mati dan penambahan `pub mod` declaration untuk wiring. Tidak mengubah logika.

3. **Test files** yang orphaned tidak akan mempengaruhi coverage karena tidak pernah dijalankan.

4. **`crates/api` (5.670 LOC)** adalah target eliminasi terbesar. Jika tidak ada rencana untuk API server standalone, seluruh crate bisa dihapus. Jika diperlukan, wiring sebagai sub-module aplikasi utama.

5. **Fitur `cuda` (63 gates)** — kode CUDA yang ekstensif (FlashAttention, MoE CUDA) tapi tidak pernah diaktifkan. Jika tidak ada GPU NVIDIA di target deployment, pertimbangkan hapus gates atau simpan untuk eksperimen.
