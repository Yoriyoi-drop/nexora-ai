# Multi-Model Debate & Voting System — Flow Diagram (v2)

## Arsitektur

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT (prompt)                              │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           IntentRouter                                    │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                         route(prompt)                            │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐  │   │
│  │  │ IntentKind   │───▶│ NxrModelId   │───▶│ confidence: f32    │  │   │
│  │  │ classify()   │    │ target_model │    │                    │  │   │
│  │  └──────────────┘    └──────────────┘    └────────────────────┘  │   │
│  └───────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│  ┌───────────────────────────▼──────────────────────────────────────┐   │
│  │                     requires_debate()                            │   │
│  │                                                                  │   │
│  │  Deteksi keyword:                                                │   │
│  │    • "debat", "diskusi", "brainstorm"  → DEBATE                 │   │
│  │    • "bandingkan", "vs", "pro kontra"  → DEBATE                 │   │
│  │    • "analisis", "evaluasi", "dampak"   → DEBATE                │   │
│  │    • "keputusan", "strategi"           → DEBATE                 │   │
│  │    • "kompleks", "kontroversial"       → DEBATE                 │   │
│  │    • Intent Strategy / Reasoning       → DEBATE                 │   │
│  │    • else                             → SINGLE MODEL             │   │
│  └───────────────────┬──────────────────────────────────────────────┘   │
└───────────────────────┼──────────────────────────────────────────────────┘
                        │
           ┌────────────┼────────────┐
           │            │            │
      false│            │true        │
           │            │            │
           ▼            │            ▼
 ┌──────────────────┐   │  ┌─────────────────────────────────────────────────┐
 │ SINGLE MODEL     │   │  │            DebateOrchestrator v2                │
 │                  │   │  │                                                  │
 │ delegate_for     │   │  │  ┌─── 1. CostController ───────────────────┐   │
 │ _model(          │   │  │  │  analyze(prompt) → ComplexityScore      │   │
 │   model_id,      │   │  │  │    0-30  → SingleModel (skip debate)    │   │
 │   prompt)        │   │  │  │   31-60  → DualModel                    │   │
 │                  │   │  │  │   61-80  → ThreeModel                    │   │
 │   → String       │   │  │  │   81-100 → FullDebate                   │   │
 │   (langsung)     │   │  │  └─────────────────────────────────────────┘   │
 │                  │   │  │                                                  │
 └────────┬─────────┘   │  │  ┌─── 2. CapabilityScorer ─────────────────┐   │
          │             │  │  │  analyze_requirements(prompt)            │   │
          │             │  │  │    → capability vector (8 axes)          │   │
          │             │  │  │  select_participants(required, models)   │   │
          │             │  │  │    → score + sort + top-N                │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 3. Hub-and-Spoke ────────────────────┐   │
          │             │  │  │  Moderator (Nexum) sebagai hub           │   │
          │             │  │  │  ContextCompressor: each round           │   │
          │             │  │  │    key_claims | agreements | disputes    │   │
          │             │  │  │    evidence | open_questions             │   │
          │             │  │  │  → ~300 token compressed summary         │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 4. Round 1: Initial Responses ───────┐   │
          │             │  │  │  Model A ──► delegate_for_model(prompt) │   │
          │             │  │  │  Model B ──► delegate_for_model(prompt) │   │
          │             │  │  │  Model C ──► delegate_for_model(prompt) │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 5. Round 2..N: Compressed Debat ─────┐   │
          │             │  │  │  Moderator (Nexum):                       │   │
          │             │  │  │    compress(messages) → ~300 token       │   │
          │             │  │  │                                           │   │
          │             │  │  │  Participant A lihat ringkasan:          │   │
          │             │  │  │    "Key claim: monolith lebih cepat      │   │
          │             │  │  │     Disagreement: tim size threshold...  │   │
          │             │  │  │     Evidence: Yegor Bugayenko blog..."   │   │
          │             │  │  │    → 5-15 tokens untuk baca              │   │
          │             │  │  │     (vs 500+ full transcript)            │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 6. ConfidenceEngine ────────────────┐   │
          │             │  │  │  calibrate(                             │   │
          │             │  │  │    classifier_score: f32,               │   │
          │             │  │  │    verifier_score: Option<f32>,         │   │
          │             │  │  │    consensus_score: Option<f32>,        │   │
          │             │  │  │    historical_accuracy: Option<f32>     │   │
          │             │  │  │  ) → calibrated confidence 0.0-1.0     │   │
          │             │  │  │     w0*classifier + w1*verifier        │   │
          │             │  │  │     + w2*consensus + w3*history        │   │
          │             │  │  │     defaults: w0=0.5, w1=0.2,          │   │
          │             │  │  │              w2=0.2, w3=0.1            │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 7. Weighted Voting ──────────────────┐   │
          │             │  │  │  WeightedVote {                          │   │
          │             │  │  │    voter: NxrModelId,                    │   │
          │             │  │  │    vote_for: NxrModelId,                 │   │
          │             │  │  │    weight: f32,     // tier-based        │   │
          │             │  │  │    confidence: f32, // calibrated        │   │
          │             │  │  │  }                                       │   │
          │             │  │  │                                           │   │
          │             │  │  │  Tier → weight mapping:                  │   │
          │             │  │  │    Ultra=1.0, Master=0.9, Apex=0.8      │   │
          │             │  │  │    Pro=0.6, Core=0.5, Edge=0.3          │   │
          │             │  │  │                                           │   │
          │             │  │  │  Score = Σ(weight_i × confidence_i)     │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 8. DebateVerifier ───────────────────┐   │
          │             │  │  │  verify(votes, messages):               │   │
          │             │  │  │    ✓ Check contradictions               │   │
          │             │  │  │    ✓ Check logic consistency            │   │
          │             │  │  │    ⚠ Check echo chamber                 │   │
          │             │  │  │       (all votes same → warning)        │   │
          │             │  │  │    → VerificationReport                 │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 9. Top-K Synthesis ──────────────────┐   │
          │             │  │  │  rank by weighted score                  │   │
          │             │  │  │  take top-K (default: 2)                 │   │
          │             │  │  │  extract 2 kalimat insight per model     │   │
          │             │  │  │  synthesize → final answer               │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  │                                                  │
          │             │  │  ┌─── 10. Failure Mode ────────────────────┐   │
          │             │  │  │  call_model_with_timeout()               │   │
          │             │  │  │    tokio::time::timeout(30s, future)     │   │
          │             │  │  │  if participants < 2: fallback single    │   │
          │             │  │  │  model failures recorded in ModelHistory │   │
          │             │  │  └─────────────────────────────────────────┘   │
          │             │  └─────────────────────────────────────────────────┘
          │             │                         │
          └─────────────┼─────────────────────────┼───────────────────┘
                        │                         │
                        ▼                         ▼
            ┌──────────────────┐  ┌────────────────────────────────────────┐
            │ Response         │  │ Response                               │
            │ (single model)   │  │ (debate: DebateResult {                │
            └────────┬─────────┘  │   winner, final_response,              │
                      │           │   depth: DebateDepth,                  │
                      │           │   complexity_score: f32,               │
                      │           │   verification: VerificationReport,    │
                      │           │   top_insights: Vec<String>,           │
                      │           │   votes, all_responses,                │
                      │           │   round_count, consensus,              │
                      │           │   participants })                      │
                      │           └──────────────────┬─────────────────────┘
                      │                              │
                      └──────────┬───────────────────┘
                                 │
                                 ▼
                     ┌──────────────────────────┐
                     │      TAMPILAN USER        │
                     └──────────────────────────┘
```

---

## Struktur Data (v2)

```rust
// ── Enums ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DebateDepth {
    SingleModel,    // complexity 0-30: skip debate entirely
    DualModel,      // complexity 31-60: 2 participants
    ThreeModel,     // complexity 61-80: 3 participants
    FullDebate,     // complexity 81-100: up to config max
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NxrModelTier {
    Ultra,   // weight = 1.0
    Master,  // weight = 0.9
    Apex,    // weight = 0.8
    Pro,     // weight = 0.6
    Core,    // weight = 0.5
    Edge,    // weight = 0.3
}

// ── Konfigurasi ───────────────────────────────────────────────────

pub struct DebateConfig {
    pub max_rounds: usize,                // Maks putaran (default: 3)
    pub max_participants: usize,          // Maks model (default: 4)
    pub voting_threshold: f32,            // Threshold konsensus (default: 0.6)
    pub enable_discussion: bool,          // Multi-round debate
    pub min_participants: usize,          // Minimum (default: 2)
    pub enable_hub_and_spoke: bool,       // Nexum sebagai moderator (default: true)
    pub compression_target_tokens: usize, // Target token kompresi (default: 300)
    pub model_timeout_ms: u64,            // Timeout per model (default: 30000)
    pub max_retries: usize,               // Retry gagal (default: 2)
    pub top_k: usize,                     // Synthesis top-K (default: 2)
}

// ── Cost Controller ───────────────────────────────────────────────

pub struct ComplexityScore {
    pub overall: f32,           // 0.0 - 1.0
    pub depth: DebateDepth,     // Hasil mapping
    pub domain_count: usize,    // Banyak domain terdeteksi
    pub uncertainty: f32,       // Skor ambiguitas
}

// ── Capability Scoring ────────────────────────────────────────────

pub struct CapabilityProfile {
    pub model_id: NxrModelId,
    pub tier: NxrModelTier,
    pub capabilities: HashMap<String, f32>,
    // reasoning, code, security, creative, emotional,
    // knowledge, strategy, orchestration → 0.0 - 1.0
}

pub struct RequiredCapabilities {
    pub axes: Vec<String>,          // Axes yang relevan
    pub min_scores: Vec<f32>,       // Minimum per axis
}

// ── Compressed Context ────────────────────────────────────────────

pub struct CompressedContext {
    pub key_claims: Vec<String>,        // Klaim utama per model
    pub agreements: Vec<String>,        // Poin kesepakatan
    pub disagreements: Vec<String>,     // Poin perdebatan
    pub evidence: Vec<String>,          // Data/bukti disebut
    pub open_questions: Vec<String>,    // Pertanyaan belum terjawab
}

// ── Confidence ────────────────────────────────────────────────────

pub struct ConfidenceEngine {
    pub weight_classifier: f32,     // default 0.5
    pub weight_verifier: f32,       // default 0.2
    pub weight_consensus: f32,      // default 0.2
    pub weight_history: f32,        // default 0.1
}

// ── Verifier ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub passed: bool,
    pub factual_issues: Vec<String>,
    pub logical_issues: Vec<String>,
    pub contradictions: Vec<String>,
    pub overall_score: f32,         // 0.0 - 1.0
}

// ── Weighted Voting ───────────────────────────────────────────────

pub struct WeightedVote {
    pub voter: NxrModelId,
    pub vote_for: NxrModelId,
    pub weight: f32,                // Berdasarkan tier
    pub confidence: f32,            // Calibrated confidence
    pub reasoning: String,
}

// ── Messages & State ──────────────────────────────────────────────

pub struct DebateMessage {
    pub model_id: NxrModelId,
    pub content: String,
    pub round: usize,
    pub intent: IntentKind,
    pub calibrated_confidence: f32,
    pub tier: NxrModelTier,
}

pub struct DebateResult {
    pub winner: NxrModelId,
    pub final_response: String,
    pub votes: Vec<WeightedVote>,
    pub all_responses: HashMap<NxrModelId, String>,
    pub round_count: usize,
    pub consensus: bool,
    pub participants: Vec<NxrModelId>,
    pub depth: DebateDepth,
    pub verification: VerificationReport,
    pub top_insights: Vec<String>,
    pub complexity_score: f32,
}

pub struct ModelHistory {
    pub accuracy: f32,
    pub total_calls: usize,
    pub failures: usize,
    pub avg_latency_ms: f64,
}
```

---

## Intent → Participant Mapping (v2 — Dynamic)

Tidak ada mapping statis. `CapabilityScorer` menganalisis prompt secara real-time:

| Langkah | Proses |
|---------|--------|
| 1. `analyze_requirements(prompt)` | Deteksi domain keywords → buat `RequiredCapabilities` |
| 2. Score semua model | Dot product antara required axes dan `CapabilityProfile` |
| 3. Sort by score | Highest first |
| 4. Top-N by depth | `DualModel`=2, `ThreeModel`=3, `FullDebate`=config.max |

### Contoh scoring untuk prompt bisnis-strategi

| Model | Reasoning | Strategy | Knowledge | Score |
|-------|-----------|----------|-----------|-------|
| Axiom | 0.95 | 0.90 | 0.80 | **2.65** |
| Nexum | 0.70 | 0.95 | 0.70 | **2.35** |
| Kronos | 0.75 | 0.85 | 0.95 | **2.55** |
| Omnis | 0.85 | 0.75 | 0.85 | **2.45** |

---

## Contoh Alur Lengkap (v2)

### Input: *"Analisis dampak AI automation terhadap lapangan kerja di Indonesia 2025-2030"*

```
Step 1 — IntentRouter
───────────────────────────────────────────────
  route("Analisis dampak AI automation...")
    → intent:   Reasoning
    → model:    Omnis
    → requires_debate() = true

Step 2 — DebateOrchestrator.orchestrate("Analisis...", Omnis)
───────────────────────────────────────────────

  Phase A — CostController
    analyze("Analisis dampak AI automation...")
      length: 10 words (normal)
      questions: 0
      domains: tech + business + human = 3
      uncertainty keywords: "dampak" = yes
      stake keywords: "lapangan kerja" = high
    → ComplexityScore { overall: 0.78, depth: ThreeModel }

  Phase B — CapabilityScorer
    Required: reasoning(0.3), strategy(0.2), knowledge(0.4), emotional(0.1)
    Top 3: Axiom(2.65), Kronos(2.55), Omnis(2.45)
    → Participants: [Omnis (primary), Axiom, Kronos]

  Phase C — Hub-and-Spoke Round 1
    Omnis   → "AI automation diperkirakan menggantikan 30% pekerjaan
               repetitif di manufaktur dan admin..."
    Axiom   → "Dari sisi logika dampak: ada 3 skenario — optimis,
               moderat, pesimis..."
    Kronos  → "Data historis revolusi industri sebelumnya: setiap
               gelombang automation ciptakan job baru dalam 5-10 tahun..."

    [Nexum sebagai moderator]
    ContextCompressor() → CompressedContext {
      key_claims: [
        "30% pekerjaan repetitif tergantikan",
        "3 skenario dampak (optimis/moderat/pesimis)",
        "Historis: automation cipta job baru 5-10 tahun"
      ],
      agreements: ["reskilling critical", "timeline 5-10 tahun"],
      disagreements: ["30% vs 15% estimasi"],
      evidence: ["WEF Future of Jobs Report", "McKinsey 2023"],
      open_questions: ["Peran regulasi pemerintah?"],
    }  // ~300 token

  Phase D — Hub-and-Spoke Round 2
    Setiap model lihat ringkasan ~300 token (bukan 1500+ token full)

    Axiom:
      "[CompressedContext]... Menambahkan: regulasi pemerintah bisa
       mempercepat atau memperlambat transisi. Singapura punya
       contoh reskilling grant yang efektif..."
    Kronos:
      "Saya update proyeksi: dengan perkembangan AI 2025-2026,
       timeline mungkin 3-7 tahun, bukan 5-10..."
    Omnis:
      "Mensintesis: Poin kunci — 1) Timing dipercepat, 2) Kontroversi
       angka 30%, 3) Regulasi sebagai variabel kunci..."

  Phase E — ConfidenceEngine
    Omnis:   calibrate(0.85, Some(0.80), Some(0.70), Some(0.90))
             → 0.835
    Axiom:   calibrate(0.80, None,     Some(0.60), Some(0.85))
             → 0.775
    Kronos:  calibrate(0.90, Some(0.85), Some(0.70), Some(0.75))
             → 0.855

  Phase F — Weighted Voting
    Omnis  → vote Kronos | weight=0.9 | conf=0.835 | score=0.7515
    Axiom  → vote Omnis  | weight=0.8 | conf=0.775 | score=0.6200
    Kronos → vote Omnis  | weight=0.5 | conf=0.855 | score=0.4275

    Final scores:
      Omnis  = 0.6200 + 0.4275 = 1.0475
      Kronos = 0.7515

    → Winner: Omnis

  Phase G — DebateVerifier
    verify(votes, messages):
      ✓ No contradictions found
      ✓ Logic consistent
      ⚠ Echo chamber warning: Axiom+Kronos vote same
      → VerificationReport { passed: true, overall_score: 0.85 }

  Phase H — Top-K Synthesis
    Top 2 winner: Omnis, Kronos
    Insight Omnis:  "3 faktor kunci — timing dipercepat, kontroversi
                     angka, regulasi sebagai variabel"
    Insight Kronos: "Dengan perkembangan AI 2025-2026 timeline
                     terkontraksi dari 5-10 tahun ke 3-7 tahun"

    Synthesis → final answer

Step 3 — User mendapat respons final
───────────────────────────────────────────────
  ✅ Respons dari sintesis insight 2 model terbaik
  ✅ Ringkas, berbasis data, tanpa mention proses debat
```

---

## Perbandingan v1 vs v2

| Aspek | v1 | v2 |
|-------|----|----|
| Participant selection | Static mapping by intent | Dynamic capability scoring |
| Debate format | All-to-all broadcast | Hub-and-Spoke (Nexum moderator) |
| Context management | Full transcript (O(n²) token) | CompressedContext (~300 token) |
| Confidence | Raw dari model | Calibrated (4 weighted sources) |
| Voting | Simple majority | Weighted by tier + confidence |
| Verification | Tidak ada | Contradiction + echo chamber check |
| Synthesis | Winner's full response | Top-K insight extraction |
| Cost control | Tidak ada | ComplexityScorer → DebateDepth |
| Failure mode | Panic on timeout | 30s timeout + graceful fallback |
| Orchestrate signature | `(prompt, model, participants)` | `(prompt, model)` — internal selection |
```

---

## Metrik Token Savings (v2 vs v1)

| Skenario | v1 (full transcript) | v2 (compressed) | Savings |
|----------|---------------------|-----------------|---------|
| 3 models × 2 rounds | ~1500 tokens | ~600 tokens | **60%** |
| 4 models × 3 rounds | ~3500 tokens | ~900 tokens | **74%** |
| 5 models × 3 rounds | ~5500 tokens | ~1200 tokens | **78%** |
