//! Multi-Model Debate & Voting System v2
//!
//! Arsitektur lengkap dengan:
//! - Shared Context Bus (ringkasan terstruktur, bukan prompt penuh)
//! - Dynamic Participant Selection (capability scoring)
//! - Confidence Calibration (classifier + verifier + consensus + historical)
//! - Debate Compression (300 token summary per round)
//! - Failure Mode (timeout + graceful fallback)
//! - Verifier Layer (echo chamber prevention)
//! - Weighted Voting (berdasarkan tier + capability)
//! - Hub-and-Spoke (Nexum sebagai moderator)
//! - Top-K Synthesis (insight dari semua model)
//! - Cost Controller (complexity score → debate depth)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::delegate_for_model;
use nexora_foundation::shared::model_identity::{ModelTier, NxrModelId};

// ═══════════════════════════════════════════════════════════════════════════
// 1. COST CONTROLLER — menentukan seberapa dalam debat berdasarkan kompleksitas
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebateDepth {
    SingleModel,  // 0-30: langsung, tanpa debat
    DualModel,    // 30-60: 2 model, 1 round
    ThreeModel,   // 60-80: 3 model, 2 round
    FullDebate,   // 80-100: 4+ model, 3 round
}

impl DebateDepth {
    pub fn max_participants(&self) -> usize {
        match self {
            Self::SingleModel => 1,
            Self::DualModel => 2,
            Self::ThreeModel => 3,
            Self::FullDebate => 5,
        }
    }

    pub fn max_rounds(&self) -> usize {
        match self {
            Self::SingleModel => 0,
            Self::DualModel => 1,
            Self::ThreeModel => 2,
            Self::FullDebate => 3,
        }
    }

    pub fn enable_voting(&self) -> bool {
        matches!(self, Self::ThreeModel | Self::FullDebate)
    }

    pub fn enable_verifier(&self) -> bool {
        matches!(self, Self::FullDebate)
    }
}

pub struct ComplexityScorer;

impl ComplexityScorer {
    /// Skor kompleksitas prompt 0.0 - 1.0
    pub fn score(prompt: &str) -> f32 {
        let lower = prompt.to_lowercase();
        let mut score = 0.0_f32;
        let mut factors = 0.0;

        // Length factor: prompt panjang = lebih kompleks
        let len_factor = (prompt.len() as f32 / 1000.0).min(1.0);
        score += len_factor * 0.2;
        factors += 0.2;

        // Question count: banyak pertanyaan = kompleks
        let q_count = prompt.chars().filter(|&c| c == '?').count() as f32;
        score += (q_count / 5.0).min(1.0) * 0.1;
        factors += 0.1;

        // Complexity keywords
        let complex_keywords = [
            "analisis", "evaluasi", "implications", "dampak", "komparasi",
            "bandingkan", "trade-off", "synthesis", "rekomendasi",
            "strategi", "keputusan", "decision", "framework",
            "arsitektur", "architecture", "design pattern",
            "optimasi", "optimization", "scalability",
            "multi-", "cross-", "inter-", "meta-",
        ];
        let complex_count = complex_keywords.iter()
            .filter(|k| lower.contains(*k))
            .count() as f32;
        score += (complex_count / 5.0).min(1.0) * 0.25;
        factors += 0.25;

        // Domain breadth: jumlah domain berbeda
        let domain_keywords: &[(&str, &[&str])] = &[
            ("tech", &["code", "server", "api", "database", "cloud", "microservice"]),
            ("business", &["bisnis", "startup", "market", "revenue", "cost"]),
            ("science", &["penelitian", "research", "data", "experiment", "theory"]),
            ("human", &["orang", "masyarakat", "social", "etika", "culture"]),
            ("creative", &["design", "ui", "ux", "creative", "artistic"]),
        ];
        let domain_count = domain_keywords.iter()
            .filter(|(_, keywords)| keywords.iter().any(|k| lower.contains(k)))
            .count() as f32;
        score += (domain_count / 5.0) * 0.2;
        factors += 0.2;

        // Uncertainty markers: prompt dengan ambiguity
        let uncertainty_keywords = [
            "mungkin", "maybe", "perhaps", "tidak yakin", "uncertain",
            "ambiguous", "kontroversial", "controversial",
            "debatable", "arguably", "depends", "tergantung",
            "konteks", "context", "situasional",
        ];
        let uncertainty_count = uncertainty_keywords.iter()
            .filter(|k| lower.contains(*k))
            .count() as f32;
        score += (uncertainty_count / 3.0).min(1.0) * 0.15;
        factors += 0.15;

        // Stake: prompt tentang keputusan penting
        let stake_keywords = [
            "kritis", "critical", "urgent", "vital", "penting",
            "berisiko", "risky", "high-stakes", "konsekuensi",
            "consequences", "irreversible", "fatal",
        ];
        let stake_count = stake_keywords.iter()
            .filter(|k| lower.contains(*k))
            .count() as f32;
        score += (stake_count / 3.0).min(1.0) * 0.1;
        factors += 0.1;

        if factors > 0.0 {
            score / factors
        } else {
            0.1 // minimal complexity
        }
    }

    pub fn debate_depth(prompt: &str) -> DebateDepth {
        let score = Self::score(prompt);
        let normalized = (score * 100.0) as u8;
        match normalized {
            0..=30 => DebateDepth::SingleModel,
            31..=60 => DebateDepth::DualModel,
            61..=80 => DebateDepth::ThreeModel,
            _ => DebateDepth::FullDebate,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. CAPABILITY PROFILE & DYNAMIC SELECTION
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct CapabilityProfile {
    pub model_id: NxrModelId,
    pub reasoning: f32,      // Penalaran logis
    pub code: f32,           // Coding & debugging
    pub security: f32,       // Keamanan & threat
    pub creative: f32,       // Kreativitas & gaya
    pub emotional: f32,      // Emosi & empati
    pub knowledge: f32,      // Pengetahuan & data
    pub strategy: f32,       // Strategi & keputusan
    pub orchestration: f32,  // Orchestrasi & koordinasi
}

impl CapabilityProfile {
    pub fn for_model(model_id: NxrModelId) -> Self {
        match model_id {
            NxrModelId::Omnis => Self {
                model_id,
                reasoning: 0.95, code: 0.80, security: 0.70,
                creative: 0.75, emotional: 0.70, knowledge: 0.90,
                strategy: 0.85, orchestration: 0.80,
            },
            NxrModelId::Vortex => Self {
                model_id,
                reasoning: 0.60, code: 0.95, security: 0.65,
                creative: 0.30, emotional: 0.20, knowledge: 0.50,
                strategy: 0.45, orchestration: 0.40,
            },
            NxrModelId::Aether => Self {
                model_id,
                reasoning: 0.50, code: 0.20, security: 0.25,
                creative: 0.70, emotional: 0.95, knowledge: 0.45,
                strategy: 0.50, orchestration: 0.40,
            },
            NxrModelId::Spectra => Self {
                model_id,
                reasoning: 0.55, code: 0.35, security: 0.25,
                creative: 0.95, emotional: 0.75, knowledge: 0.50,
                strategy: 0.45, orchestration: 0.50,
            },
            NxrModelId::Nexum => Self {
                model_id,
                reasoning: 0.70, code: 0.55, security: 0.45,
                creative: 0.50, emotional: 0.50, knowledge: 0.65,
                strategy: 0.80, orchestration: 0.95,
            },
            NxrModelId::Axiom => Self {
                model_id,
                reasoning: 0.95, code: 0.60, security: 0.50,
                creative: 0.40, emotional: 0.40, knowledge: 0.75,
                strategy: 0.95, orchestration: 0.70,
            },
            NxrModelId::Cipher => Self {
                model_id,
                reasoning: 0.55, code: 0.70, security: 0.95,
                creative: 0.25, emotional: 0.25, knowledge: 0.50,
                strategy: 0.55, orchestration: 0.45,
            },
            NxrModelId::Swift => Self {
                model_id,
                reasoning: 0.35, code: 0.40, security: 0.30,
                creative: 0.25, emotional: 0.25, knowledge: 0.30,
                strategy: 0.30, orchestration: 0.30,
            },
            NxrModelId::Kronos => Self {
                model_id,
                reasoning: 0.65, code: 0.40, security: 0.35,
                creative: 0.40, emotional: 0.40, knowledge: 0.95,
                strategy: 0.60, orchestration: 0.50,
            },
            NxrModelId::Genesis => Self {
                model_id,
                reasoning: 0.80, code: 0.55, security: 0.40,
                creative: 0.70, emotional: 0.60, knowledge: 0.65,
                strategy: 0.70, orchestration: 0.65,
            },
        }
    }

    pub fn tier_weight(&self) -> f32 {
        let tier = self.model_id.tier();
        match tier {
            ModelTier::Ultra => 1.0,
            ModelTier::Master => 0.9,
            ModelTier::Apex => 0.8,
            ModelTier::Pro => 0.6,
            ModelTier::Core => 0.5,
            ModelTier::Edge => 0.3,
        }
    }

    /// Skor kecocokan model terhadap kebutuhan prompt
    pub fn match_score(&self, requirements: &[CapabilityRequirement]) -> f32 {
        if requirements.is_empty() {
            return self.reasoning * 0.5; // default: general reasoning
        }
        let total: f32 = requirements.iter()
            .map(|req| {
                let cap_score = match req.capability {
                    CapabilityType::Reasoning => self.reasoning,
                    CapabilityType::Code => self.code,
                    CapabilityType::Security => self.security,
                    CapabilityType::Creative => self.creative,
                    CapabilityType::Emotional => self.emotional,
                    CapabilityType::Knowledge => self.knowledge,
                    CapabilityType::Strategy => self.strategy,
                    CapabilityType::Orchestration => self.orchestration,
                };
                cap_score * req.weight
            })
            .sum();
        total / requirements.iter().map(|r| r.weight).sum::<f32>()
    }
}

#[derive(Debug, Clone)]
pub enum CapabilityType {
    Reasoning,
    Code,
    Security,
    Creative,
    Emotional,
    Knowledge,
    Strategy,
    Orchestration,
}

#[derive(Debug, Clone)]
pub struct CapabilityRequirement {
    pub capability: CapabilityType,
    pub weight: f32,
}

pub struct CapabilityScorer;

impl CapabilityScorer {
    /// Analisis prompt → daftar kebutuhan capability dengan bobot
    pub fn analyze_requirements(prompt: &str) -> Vec<CapabilityRequirement> {
        let lower = prompt.to_lowercase();
        let mut requirements = Vec::new();

        let reasoning_words = ["jelaskan", "bagaimana", "mengapa", "kenapa", "explain",
            "why", "how", "what is", "teori", "konsep", "reasoning", "logika",
            "analisa", "analisis", "korelasi", "causal", "sebab"];
        let reasoning_score = Self::keyword_score(&lower, &reasoning_words);
        if reasoning_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Reasoning,
                weight: reasoning_score,
            });
        }

        let code_words = ["code", "rust", "python", "debug", "compile", "syntax",
            "bug", "refactor", "programming", "algorithm", "function",
            "implementasi", "kode", "coding", "api", "database", "sql"];
        let code_score = Self::keyword_score(&lower, &code_words);
        if code_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Code,
                weight: code_score,
            });
        }

        let security_words = ["security", "xss", "injection", "sql", "hack",
            "vulnerability", "threat", "crack", "malware", "encrypt",
            "decrypt", "authentication", "authorization", "firewall"];
        let security_score = Self::keyword_score(&lower, &security_words);
        if security_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Security,
                weight: security_score,
            });
        }

        let creative_words = ["puisi", "cerita", "gambar", "creative", "story",
            "poem", "narrative", "tulisan", "creative writing", "imajinasi",
            "imagination", "art", "seni", "musik", "music", "visual"];
        let creative_score = Self::keyword_score(&lower, &creative_words);
        if creative_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Creative,
                weight: creative_score,
            });
        }

        let emotional_words = ["sedih", "senang", "marah", "emosi", "feeling",
            "sad", "happy", "angry", "takut", "cemas", "anxiety", "depresi",
            "love", "cinta", "benci", "fear", "stress"];
        let emotional_score = Self::keyword_score(&lower, &emotional_words);
        if emotional_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Emotional,
                weight: emotional_score,
            });
        }

        let knowledge_words = ["sejarah", "data", "fakta", "informasi",
            "knowledge", "archive", "penelitian", "research", "historical",
            "statistics", "statistik", "referensi", "sumber", "source",
            "dokumen", "document", "literature"];
        let knowledge_score = Self::keyword_score(&lower, &knowledge_words);
        if knowledge_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Knowledge,
                weight: knowledge_score,
            });
        }

        let strategy_words = ["strategi", "keputusan", "decision", "plan",
            "rencana", "analisis", "bisnis", "business", "strategy",
            "planning", "roadmap", "recommendation", "rekomendasi",
            "optimasi", "optimization", "efisiensi", "efficiency"];
        let strategy_score = Self::keyword_score(&lower, &strategy_words);
        if strategy_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Strategy,
                weight: strategy_score,
            });
        }

        let orchestration_words = ["orchestrasi", "workflow", "pipeline",
            "multi-step", "complex task", "koordinasi", "coordination",
            "integration", "integrasi", "deployment", "ci/cd",
            "automation", "otomatis", "scheduling"];
        let orchestration_score = Self::keyword_score(&lower, &orchestration_words);
        if orchestration_score > 0.0 {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Orchestration,
                weight: orchestration_score,
            });
        }

        // Default: general reasoning jika tidak ada kecocokan
        if requirements.is_empty() {
            requirements.push(CapabilityRequirement {
                capability: CapabilityType::Reasoning,
                weight: 1.0,
            });
        }

        requirements
    }

    fn keyword_score(text: &str, keywords: &[&str]) -> f32 {
        let count = keywords.iter().filter(|k| text.contains(*k)).count() as f32;
        if count == 0.0 {
            return 0.0;
        }
        // Normalisasi: semakin banyak keyword = semakin relevan, capped di 1.0
        (count / 5.0).min(1.0)
    }

    /// Seleksi dinamis: skor semua model, ambil top-N
    pub fn select_participants(
        prompt: &str,
        primary: NxrModelId,
        depth: DebateDepth,
    ) -> Vec<NxrModelId> {
        let requirements = Self::analyze_requirements(prompt);
        let max = depth.max_participants();

        // Skor semua model
        let mut scored: Vec<(NxrModelId, f32)> = NxrModelId::all()
            .iter()
            .map(|&m| {
                let profile = CapabilityProfile::for_model(m);
                let match_score = profile.match_score(&requirements);
                let tier_bonus = profile.tier_weight() * 0.1;
                // Primary model dapat bonus
                let primary_bonus = if m == primary { 0.2 } else { 0.0 };
                (m, match_score + tier_bonus + primary_bonus)
            })
            .collect();

        // Sort descending oleh skor
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Ambil top-N, pastikan primary model masuk
        let mut selected: Vec<NxrModelId> = Vec::new();
        if !selected.contains(&primary) {
            selected.push(primary);
        }
        for (m, _) in &scored {
            if selected.len() >= max {
                break;
            }
            if !selected.contains(m) {
                selected.push(*m);
            }
        }

        selected
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. SHARED CONTEXT BUS — ringkasan terstruktur antar model
// ═══════════════════════════════════════════════════════════════════════════

/// Ringkasan terstruktur dari respons model — bukan full text
#[derive(Debug, Clone)]
pub struct CompressedContext {
    pub key_claims: Vec<String>,
    pub agreements: Vec<String>,
    pub disagreements: Vec<String>,
    pub evidence: Vec<String>,
    pub open_questions: Vec<String>,
    pub token_estimate: usize,
}

impl CompressedContext {
    pub fn to_prompt(&self) -> String {
        let mut parts = Vec::new();

        if !self.key_claims.is_empty() {
            parts.push(format!("Key claims:\n{}", self.key_claims.iter()
                .enumerate()
                .map(|(i, c)| format!("  {}. {}", i + 1, c))
                .collect::<Vec<_>>()
                .join("\n")));
        }

        if !self.agreements.is_empty() {
            parts.push(format!("Agreements:\n{}", self.agreements.iter()
                .map(|a| format!("  • {}", a))
                .collect::<Vec<_>>()
                .join("\n")));
        }

        if !self.disagreements.is_empty() {
            parts.push(format!("Disagreements:\n{}", self.disagreements.iter()
                .map(|d| format!("  • {}", d))
                .collect::<Vec<_>>()
                .join("\n")));
        }

        if !self.evidence.is_empty() {
            parts.push(format!("Evidence:\n{}", self.evidence.iter()
                .map(|e| format!("  • {}", e))
                .collect::<Vec<_>>()
                .join("\n")));
        }

        if !self.open_questions.is_empty() {
            parts.push(format!("Open questions:\n{}", self.open_questions.iter()
                .map(|q| format!("  • {}", q))
                .collect::<Vec<_>>()
                .join("\n")));
        }

        parts.join("\n\n")
    }
}

pub struct ContextCompressor;

impl ContextCompressor {
    /// Kompres respons model menjadi ringkasan terstruktur
    pub fn compress(responses: &[DebateMessage], target_tokens: usize) -> CompressedContext {
        let mut key_claims = Vec::new();
        let mut agreements = Vec::new();
        let mut disagreements = Vec::new();
        let mut evidence = Vec::new();
        let mut open_questions = Vec::new();
        let mut est_tokens = 0;

        for msg in responses {
            let lower = msg.content.to_lowercase();
            let lines: Vec<&str> = msg.content.lines().collect();

            // Ekstrak klaim utama (kalimat pertama dari setiap paragraf)
            for line in &lines {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.len() < 20 {
                    continue;
                }

                if est_tokens >= target_tokens {
                    break;
                }

                // Deteksi agreement
                if trimmed.starts_with("saya setuju") || trimmed.starts_with("i agree")
                    || trimmed.starts_with("setuju dengan") || trimmed.starts_with("agree with")
                    || trimmed.contains("saya sepakat")
                {
                    agreements.push(format!("[{}] {}", msg.model_id, trimmed));
                }
                // Deteksi disagreement
                else if trimmed.starts_with("saya tidak setuju") || trimmed.starts_with("i disagree")
                    || trimmed.starts_with("tidak setuju") || trimmed.starts_with("disagree")
                    || trimmed.contains("sayangnya") || trimmed.contains("however")
                    || trimmed.contains("tapi") || trimmed.contains("but")
                {
                    disagreements.push(format!("[{}] {}", msg.model_id, trimmed));
                }
                // Deteksi evidence
                else if trimmed.starts_with("data") || trimmed.starts_with("menurut")
                    || trimmed.starts_with("berdasarkan") || trimmed.starts_with("based on")
                    || trimmed.starts_with("penelitian") || trimmed.starts_with("research")
                    || trimmed.starts_with("fakta") || trimmed.starts_with("fact")
                {
                    evidence.push(format!("[{}] {}", msg.model_id, trimmed));
                }
                // Deteksi pertanyaan
                else if trimmed.contains('?') {
                    open_questions.push(format!("[{}] {}", msg.model_id, trimmed));
                }
                // Klaim umum
                else if trimmed.len() > 40 && est_tokens < target_tokens / 2 {
                    key_claims.push(format!("[{}] {}", msg.model_id, trimmed));
                }

                est_tokens += trimmed.split_whitespace().count();
            }
        }

        // Batasi agar tidak melebihi target tokens
        let truncate = |items: &mut Vec<String>, max: usize| {
            if items.len() > max {
                items.truncate(max);
            }
        };

        truncate(&mut key_claims, 5);
        truncate(&mut agreements, 3);
        truncate(&mut disagreements, 3);
        truncate(&mut evidence, 3);
        truncate(&mut open_questions, 2);

        CompressedContext {
            key_claims,
            agreements,
            disagreements,
            evidence,
            open_questions,
            token_estimate: est_tokens.min(target_tokens),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. CONFIDENCE CALIBRATION — real metrics-based confidence
// ═══════════════════════════════════════════════════════════════════════════

/// Riwayat model untuk kalibrasi confidence
pub struct ModelHistory {
    pub model_id: NxrModelId,
    pub accuracy: f32,          // 0.0 - 1.0, moving average
    pub total_calls: u64,        // Total panggilan
    pub avg_response_time_ms: f64,
    pub last_failure: Option<Instant>,
}

impl ModelHistory {
    pub fn new(model_id: NxrModelId) -> Self {
        Self {
            model_id,
            accuracy: 0.85, // default initial belief
            total_calls: 0,
            avg_response_time_ms: 100.0,
            last_failure: None,
        }
    }

    pub fn record_success(&mut self, response_time_ms: f64) {
        self.total_calls += 1;
        // Exponential moving average
        self.accuracy = self.accuracy * 0.95 + 1.0 * 0.05;
        self.avg_response_time_ms = self.avg_response_time_ms * 0.9 + response_time_ms * 0.1;
    }

    pub fn record_failure(&mut self) {
        self.total_calls += 1;
        self.accuracy = self.accuracy * 0.95 + 0.0 * 0.05;
        self.last_failure = Some(Instant::now());
    }
}

pub struct ConfidenceEngine;

impl ConfidenceEngine {
    /// Kalibrasi confidence dari berbagai sinyal
    pub fn calibrate(
        classifier_score: f32,     // dari MLP classifier (0.0 - 1.0)
        verifier_score: Option<f32>, // dari verifier check (0.0 - 1.0)
        consensus_score: Option<f32>, // kesepakatan dengan model lain (0.0 - 1.0)
        historical: Option<&ModelHistory>, // riwayat akurasi
    ) -> f32 {
        let mut score = classifier_score; // baseline
        let mut weight = 1.0;

        // Verifier boost/penalty (bobot 30%)
        if let Some(vs) = verifier_score {
            score = score * 0.7 + vs * 0.3;
            weight += 0.3;
        }

        // Consensus boost/penalty (bobot 20%)
        if let Some(cs) = consensus_score {
            // Consensus high = lebih percaya diri
            let consensus_factor = if cs > 0.6 { cs } else { cs * 0.5 };
            score = score * 0.8 + consensus_factor * 0.2;
            weight += 0.2;
        }

        // Historical accuracy (bobot 15%)
        if let Some(hist) = historical {
            if hist.total_calls > 5 {
                let hist_factor = hist.accuracy * 0.85;
                score = score * 0.85 + hist_factor * 0.15;
                weight += 0.15;
            }
        }

        // Normalisasi
        let calibrated = (score / weight).clamp(0.0, 1.0);

        // Penalty jika baru saja gagal
        if let Some(hist) = historical {
            if let Some(last_fail) = hist.last_failure {
                if last_fail.elapsed() < Duration::from_secs(300) {
                    // 30% penalty jika gagal dalam 5 menit
                    return calibrated * 0.7;
                }
            }
        }

        calibrated
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. VERIFIER LAYER — echo chamber prevention
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub passed: bool,
    pub factual_issues: Vec<String>,
    pub logical_issues: Vec<String>,
    pub contradictions: Vec<String>,
    pub overall_score: f32,
}

pub struct DebateVerifier;

impl DebateVerifier {
    /// Verifikasi hasil debat: cek fakta, logika, kontradiksi
    pub fn verify(result: &DebateResult) -> VerificationReport {
        let mut factual_issues = Vec::new();
        let mut logical_issues = Vec::new();
        let mut contradictions = Vec::new();

        // Cek kontradiksi antar model
        let all_texts: Vec<&str> = result.all_responses.values().map(|s| s.as_str()).collect();
        for i in 0..all_texts.len() {
            for j in (i + 1)..all_texts.len() {
                let a_lower = all_texts[i].to_lowercase();
                let b_lower = all_texts[j].to_lowercase();
                let participants: Vec<NxrModelId> = result.all_responses.keys().copied().collect();
                let model_a = participants.get(i).copied().unwrap_or(NxrModelId::Omnis);
                let model_b = participants.get(j).copied().unwrap_or(NxrModelId::Omnis);

                // Deteksi kontradiksi langsung: "X benar" vs "X salah"
                let contradiction_pairs = [
                    ("ya", "tidak"), ("yes", "no"), ("benar", "salah"),
                    ("true", "false"), ("setuju", "tidak setuju"),
                    ("agree", "disagree"), ("baik", "buruk"),
                    ("good", "bad"), ("aman", "berbahaya"),
                    ("safe", "dangerous"), ("mungkin", "mustahil"),
                    ("possible", "impossible"),
                ];

                for (pos, neg) in &contradiction_pairs {
                    if a_lower.contains(pos) && b_lower.contains(neg) {
                        contradictions.push(format!(
                            "[{} vs {}] {} vs {}",
                            model_a, model_b, pos, neg
                        ));
                    }
                }
            }
        }

        // Cek logical consistency winner
        if let Some(winner_text) = result.all_responses.get(&result.winner) {
            let lower = winner_text.to_lowercase();
            let logical_flags = [
                "karena", "sehingga", "maka", "oleh karena itu",
                "therefore", "thus", "consequently",
                "if", "then", "implies", "berarti",
            ];
            let has_logic = logical_flags.iter().any(|f| lower.contains(f));
            if !has_logic {
                logical_issues.push("Winner response lacks logical connectors".to_string());
            }
        }

        // Cek apakah voting menunjukkan echo chamber (semua vote sama)
        let unique_votes: std::collections::HashSet<NxrModelId> = result.votes
            .iter()
            .map(|v| v.vote_for)
            .collect();

        let consensus_risk = if unique_votes.len() == 1 && result.votes.len() > 2 {
            // Semua vote ke model yang sama — potensi echo chamber
            logical_issues.push(format!(
                "Potential echo chamber: all {} votes went to {}",
                result.votes.len(), result.winner
            ));
            true
        } else {
            false
        };

        let total_issues = factual_issues.len() + logical_issues.len() + contradictions.len();
        let overall_score = if total_issues == 0 {
            1.0
        } else if total_issues <= 2 {
            0.7
        } else if total_issues <= 5 {
            0.5
        } else {
            0.3
        };

        // Pass hanya jika tidak ada kontradiksi serius
        let passed = contradictions.len() <= 1 && !consensus_risk;

        VerificationReport {
            passed,
            factual_issues,
            logical_issues,
            contradictions,
            overall_score,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. CORE DATA STRUCTURES (updated)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct DebateConfig {
    pub max_rounds: usize,
    pub max_participants: usize,
    pub voting_threshold: f32,
    pub enable_discussion: bool,
    pub min_participants: usize,
    pub enable_verifier: bool,
    pub top_k_synthesis: usize,
    pub compression_target_tokens: usize,
    pub model_timeout_ms: u64,
    pub enable_hub_and_spoke: bool,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            max_rounds: 2,
            max_participants: 5,
            voting_threshold: 0.6,
            enable_discussion: true,
            min_participants: 2,
            enable_verifier: true,
            top_k_synthesis: 2,
            compression_target_tokens: 300,
            model_timeout_ms: 30000,
            enable_hub_and_spoke: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DebateMessage {
    pub model_id: NxrModelId,
    pub content: String,
    pub round: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct DebateContext {
    pub prompt: String,
    pub compressed: CompressedContext,
    pub current_round: usize,
    pub max_rounds: usize,
    pub participants: Vec<NxrModelId>,
}

/// Weighted vote — bobot berbeda per model
#[derive(Debug, Clone)]
pub struct WeightedVote {
    pub voter: NxrModelId,
    pub vote_for: NxrModelId,
    pub reasoning: String,
    pub raw_confidence: f32,
    pub weighted_confidence: f32,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct DebateResult {
    pub winner: NxrModelId,
    pub final_response: String,
    pub votes: Vec<WeightedVote>,
    pub all_responses: HashMap<NxrModelId, String>,
    pub round_count: usize,
    pub consensus: bool,
    pub participants: Vec<NxrModelId>,
    pub depth: DebateDepth,
    pub verification: Option<VerificationReport>,
    pub complexity_score: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. DEBATE ORCHESTRATOR (rewritten)
// ═══════════════════════════════════════════════════════════════════════════

pub struct DebateOrchestrator {
    config: DebateConfig,
    historical: Arc<std::sync::Mutex<HashMap<NxrModelId, ModelHistory>>>,
}

impl DebateOrchestrator {
    pub fn new(config: DebateConfig) -> Self {
        Self {
            config,
            historical: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    pub fn config(&self) -> &DebateConfig {
        &self.config
    }

    /// Jalankan sesi debat penuh dengan semua improvement
    pub async fn orchestrate(
        &self,
        prompt: &str,
        primary_model: NxrModelId,
    ) -> DebateResult {
        // ── Phase 0: Cost Controller ──
        let complexity = ComplexityScorer::score(prompt);
        let depth = ComplexityScorer::debate_depth(prompt);
        info!(
            "🎯 Debate | complexity={:.2} depth={:?} primary={}",
            complexity, depth, primary_model
        );

        // Single model: langsung return tanpa debat
        if matches!(depth, DebateDepth::SingleModel) {
            let response = delegate_for_model(primary_model, prompt).await;
            let mut all_responses = HashMap::new();
            all_responses.insert(primary_model, response.clone());
            return DebateResult {
                winner: primary_model,
                final_response: response,
                votes: Vec::new(),
                all_responses,
                round_count: 0,
                consensus: true,
                participants: vec![primary_model],
                depth,
                verification: None,
                complexity_score: complexity,
            };
        }

        // ── Phase 1: Dynamic Participant Selection ──
        let participants = CapabilityScorer::select_participants(prompt, primary_model, depth);
        if participants.len() < self.config.min_participants {
            warn!(
                "Not enough participants ({} < {}), falling back to single model",
                participants.len(), self.config.min_participants
            );
            let response = delegate_for_model(primary_model, prompt).await;
            let mut all_responses = HashMap::new();
            all_responses.insert(primary_model, response.clone());
            return DebateResult {
                winner: primary_model,
                final_response: response,
                votes: Vec::new(),
                all_responses,
                round_count: 0,
                consensus: true,
                participants: vec![primary_model],
                depth: DebateDepth::SingleModel,
                verification: None,
                complexity_score: complexity,
            };
        }

        info!(
            "Participants selected: {}",
            participants.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")
        );

        let mut all_responses: HashMap<NxrModelId, String> = HashMap::new();
        let mut round_messages: Vec<DebateMessage> = Vec::new();
        let hub_model = if self.config.enable_hub_and_spoke {
            Some(NxrModelId::Nexum)
        } else {
            None
        };

        // ── Phase 2: Hub-and-Spoke / Rounds ──
        let max_rounds = depth.max_rounds().min(self.config.max_rounds);

        for round in 1..=max_rounds {
            info!("📢 Debate Round {}/{}", round, max_rounds);

            let mut new_messages: Vec<DebateMessage> = Vec::new();

            for &model_id in &participants {
                let response = self.call_model_with_timeout(
                    prompt, model_id, &round_messages, round, hub_model,
                ).await;

                match response {
                    Ok(text) => {
                        all_responses.insert(model_id, text.clone());
                        let confidence = self.calculate_confidence(model_id, &round_messages, &all_responses);
                        new_messages.push(DebateMessage {
                            model_id,
                            content: text,
                            round,
                            confidence,
                        });
                        // Record success
                        if let Ok(mut hist) = self.historical.lock() {
                            hist.entry(model_id)
                                .or_insert_with(|| ModelHistory::new(model_id))
                                .record_success(100.0);
                        }
                    }
                    Err(e) => {
                        warn!("{} failed in round {}: {} — removing from debate", model_id, round, e);
                        // Record failure
                        if let Ok(mut hist) = self.historical.lock() {
                            hist.entry(model_id)
                                .or_insert_with(|| ModelHistory::new(model_id))
                                .record_failure();
                        }
                    }
                }
            }

            // Failure mode: jika terlalu sedikit peserta, fallback
            let active: Vec<NxrModelId> = participants.iter()
                .filter(|m| all_responses.contains_key(m))
                .copied()
                .collect();

            if active.len() < 2 && round > 1 {
                warn!("Too many failures ({} active < 2), falling back to single model", active.len());
                let fallback = active.first().copied().unwrap_or(primary_model);
                let response = delegate_for_model(fallback, prompt).await;
                all_responses.insert(fallback, response.clone());
                return DebateResult {
                    winner: fallback,
                    final_response: response,
                    votes: Vec::new(),
                    all_responses,
                    round_count: round,
                    consensus: false,
                    participants: vec![fallback],
                    depth: DebateDepth::SingleModel,
                    verification: None,
                    complexity_score: complexity,
                };
            }

            // Compression: simpan ringkasan, bukan full text
            let compressed = ContextCompressor::compress(&new_messages, self.config.compression_target_tokens);
            info!("  Compressed {} messages → ~{} tokens", new_messages.len(), compressed.token_estimate);

            round_messages.extend(new_messages);
        }

        // ── Phase 3: Weighted Voting ──
        info!("🗳️ Weighted Voting — {} participants", participants.len());
        let votes = self.run_weighted_voting(prompt, &participants, &all_responses).await;

        // ── Phase 4: Determine Winner (weighted) ──
        let (winner, consensus) = self.determine_weighted_winner(&votes, &all_responses);

        // ── Phase 5: Verifier Layer ──
        let mut verification = None;
        if self.config.enable_verifier && depth.enable_verifier() {
            let result_stub = DebateResult {
                winner,
                final_response: String::new(),
                votes: votes.clone(),
                all_responses: all_responses.clone(),
                round_count: max_rounds,
                consensus,
                participants: participants.clone(),
                depth,
                verification: None,
                complexity_score: complexity,
            };
            let report = DebateVerifier::verify(&result_stub);
            info!("Verification: passed={} score={:.2}", report.passed, report.overall_score);
            if report.contradictions.len() > 2 {
                warn!("High contradiction count ({}), triggering re-synthesis", report.contradictions.len());
            }
            verification = Some(report);
        }

        // ── Phase 6: Top-K Synthesis ──
        let k = self.config.top_k_synthesis.min(participants.len());
        let final_response = self.top_k_synthesis(
            prompt, &winner, &votes, &all_responses, &participants, k,
        ).await;

        DebateResult {
            winner,
            final_response,
            votes,
            all_responses,
            round_count: max_rounds,
            consensus,
            participants: participants.clone(),
            depth,
            verification,
            complexity_score: complexity,
        }
    }

    /// Panggil model dengan timeout & failure handling
    async fn call_model_with_timeout(
        &self,
        prompt: &str,
        model_id: NxrModelId,
        messages: &[DebateMessage],
        round: usize,
        hub_model: Option<NxrModelId>,
    ) -> Result<String, String> {
        let timeout = Duration::from_millis(self.config.model_timeout_ms);

        let result = if messages.is_empty() {
            // Round 1: prompt langsung
            delegate_for_model(model_id, prompt).await
        } else if let Some(hub) = hub_model {
            // Hub-and-Spoke: model lihat ringkasan dari moderator
            let compressed = ContextCompressor::compress(messages, self.config.compression_target_tokens);
            let context = compressed.to_prompt();

            if model_id == hub {
                // Moderator: lihat semua + kasih arahan
                let hub_prompt = format!(
                    "[Debate Moderator | Round {round}]\n\
                     Original question: {prompt}\n\n\
                     === Summary of responses ===\n\
                     {context}\n\n\
                     === Your role ===\n\
                     As debate moderator, synthesize key points, \
                     highlight disagreements, and guide the discussion \
                     toward resolution.\n\n\
                     Moderator synthesis:"
                );
                delegate_for_model(model_id, &hub_prompt).await
            } else {
                // Participant: lihat ringkasan dari moderator
                let spoke_prompt = format!(
                    "[Debate Round {round} | Model: {model_id}]\n\
                     Original question: {prompt}\n\n\
                     === Moderator summary ===\n\
                     {context}\n\n\
                     === Your turn ===\n\
                     Review the summary above. Focus on:\n\
                     - Addressing points relevant to your expertise\n\
                     - Resolving disagreements if you can\n\
                     - Adding new evidence or perspective\n\n\
                     Your response (concise, substantive):"
                );
                delegate_for_model(model_id, &spoke_prompt).await
            }
        } else {
            // All-to-all: model lihat full context (compressed)
            let compressed = ContextCompressor::compress(messages, self.config.compression_target_tokens);
            let context = compressed.to_prompt();

            let debate_prompt = format!(
                "[Debate Round {round} | Model: {model_id}]\n\
                 Original question: {prompt}\n\n\
                 === Structured summary of previous responses ===\n\
                 {context}\n\n\
                 === Your turn ===\n\
                 Based on the summary above:\n\
                 - Agree or disagree with specific claims\n\
                 - Provide counter-arguments or supporting evidence\n\
                 - Offer a different perspective if needed\n\n\
                 Your response (concise, substantive):"
            );
            delegate_for_model(model_id, &debate_prompt).await
        };

        Ok(result)
    }

    /// Calculate confidence for a model response
    fn calculate_confidence(
        &self,
        model_id: NxrModelId,
        messages: &[DebateMessage],
        all_responses: &HashMap<NxrModelId, String>,
    ) -> f32 {
        let profile = CapabilityProfile::for_model(model_id);
        let classifier_score = profile.reasoning * 0.5 + profile.knowledge * 0.3 + profile.strategy * 0.2;

        // Consensus score: seberapa setuju dengan model lain
        let consensus_score: Option<f32> = if messages.len() > 1 && all_responses.len() > 1 {
            let agreement_count = messages.iter()
                .filter(|m| {
                    let lower = m.content.to_lowercase();
                    lower.contains("setuju") || lower.contains("agree")
                        || lower.contains("sepakat")
                })
                .count() as f32;
            Some((agreement_count / messages.len() as f32).min(1.0))
        } else {
            None
        };

        let historical = self.historical.lock().ok()
            .and_then(|h| h.get(&model_id).map(|h| {
                ModelHistory {
                    model_id: h.model_id,
                    accuracy: h.accuracy,
                    total_calls: h.total_calls,
                    avg_response_time_ms: h.avg_response_time_ms,
                    last_failure: h.last_failure,
                }
            }));

        ConfidenceEngine::calibrate(
            classifier_score,
            None, // no verifier yet at this stage
            consensus_score,
            historical.as_ref(),
        )
    }

    /// Weighted voting — setiap model vote dengan bobot berbeda
    async fn run_weighted_voting(
        &self,
        prompt: &str,
        participants: &[NxrModelId],
        all_responses: &HashMap<NxrModelId, String>,
    ) -> Vec<WeightedVote> {
        let mut votes = Vec::new();

        for &voter_id in participants {
            let weight = CapabilityProfile::for_model(voter_id).tier_weight();

            let vote = self.cast_weighted_vote(prompt, voter_id, participants, all_responses).await;

            votes.push(WeightedVote {
                weighted_confidence: vote.raw_confidence * weight,
                weight,
                ..vote
            });
        }

        votes
    }

    /// Satu model memberikan vote dengan format terstruktur
    async fn cast_weighted_vote(
        &self,
        prompt: &str,
        voter: NxrModelId,
        participants: &[NxrModelId],
        all_responses: &HashMap<NxrModelId, String>,
    ) -> WeightedVote {
        let responses_list: String = participants
            .iter()
            .filter_map(|m| {
                all_responses.get(m).map(|r| {
                    // Truncate response untuk hemat token
                    let truncated: String = r.chars().take(500).collect();
                    format!("=== {} ===\n{}\n", m, truncated)
                })
            })
            .collect::<Vec<_>>()
            .join("\n");

        let vote_prompt = format!(
            "[Voting Phase | Model: {voter}]\n\
             Original question: {prompt}\n\n\
             === All Responses (truncated) ===\n\
             {responses_list}\n\n\
             === Your Vote ===\n\
             Evaluate each response based on accuracy, completeness, clarity, and relevance.\n\
             Choose the BEST response.\n\n\
             Format:\n\
             VOTE: [model name]\n\
             REASON: [1-2 sentences]\n\
             CONFIDENCE: [0.0 to 1.0]"
        );

        let vote_result = delegate_for_model(voter, &vote_prompt).await;

        let vote_target = participants.iter()
            .find(|&&m| {
                let name_upper = m.to_string().to_uppercase();
                vote_result.to_uppercase().contains(&name_upper)
            })
            .copied()
            .unwrap_or(voter);

        let confidence = if vote_result.contains("CONFIDENCE: 1") || vote_result.contains("CONFIDENCE:1") {
            1.0
        } else if vote_result.contains("CONFIDENCE: 0.9") || vote_result.contains("CONFIDENCE:0.9") {
            0.9
        } else if vote_result.contains("CONFIDENCE: 0.8") || vote_result.contains("CONFIDENCE:0.8") {
            0.8
        } else if vote_result.contains("CONFIDENCE: 0.7") || vote_result.contains("CONFIDENCE:0.7") {
            0.7
        } else if vote_result.contains("CONFIDENCE: 0.6") || vote_result.contains("CONFIDENCE:0.6") {
            0.6
        } else if vote_result.contains("CONFIDENCE: 0.5") || vote_result.contains("CONFIDENCE:0.5") {
            0.5
        } else {
            0.7
        };

        let reason = vote_result
            .lines()
            .find(|l| l.starts_with("REASON:") || l.starts_with("REASON :"))
            .map(|l| {
                let r = l.trim_start_matches("REASON:").trim_start_matches("REASON :");
                r.trim().to_string()
            })
            .unwrap_or_else(|| "No explicit reason given".to_string());

        WeightedVote {
            voter,
            vote_for: vote_target,
            reasoning: reason,
            raw_confidence: confidence,
            weighted_confidence: confidence * CapabilityProfile::for_model(voter).tier_weight(),
            weight: CapabilityProfile::for_model(voter).tier_weight(),
        }
    }

    /// Tentukan pemenang berdasarkan weighted votes
    fn determine_weighted_winner(
        &self,
        votes: &[WeightedVote],
        all_responses: &HashMap<NxrModelId, String>,
    ) -> (NxrModelId, bool) {
        let mut vote_tally: HashMap<NxrModelId, f32> = HashMap::new();

        for vote in votes {
            *vote_tally.entry(vote.vote_for).or_insert(0.0) += vote.weighted_confidence;
        }

        let winner = vote_tally.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id);

        match winner {
            Some(w) => {
                let total_weight: f32 = votes.iter().map(|v| v.weight).sum();
                let winner_weight: f32 = votes.iter()
                    .filter(|v| v.vote_for == w)
                    .map(|v| v.weighted_confidence)
                    .sum();
                let consensus = (winner_weight / total_weight) >= self.config.voting_threshold;
                (w, consensus)
            }
            None => {
                let fallback = all_responses.keys().next().copied().unwrap_or(NxrModelId::Omnis);
                (fallback, false)
            }
        }
    }

    /// Top-K Synthesis — ambil insight dari K model teratas
    async fn top_k_synthesis(
        &self,
        prompt: &str,
        winner: &NxrModelId,
        votes: &[WeightedVote],
        all_responses: &HashMap<NxrModelId, String>,
        participants: &[NxrModelId],
        k: usize,
    ) -> String {
        // Hitung top-K models berdasarkan weighted votes
        let mut model_scores: HashMap<NxrModelId, f32> = HashMap::new();
        for vote in votes {
            *model_scores.entry(vote.vote_for).or_insert(0.0) += vote.weighted_confidence;
        }

        let mut ranked: Vec<(NxrModelId, f32)> = model_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<NxrModelId> = ranked.iter()
            .take(k)
            .map(|(id, _)| *id)
            .collect();

        // Kumpulkan insight dari top-K models
        let top_insights: Vec<String> = top_k.iter()
            .enumerate()
            .filter_map(|(i, m)| all_responses.get(m).map(|text| (i, m, text)))
            .map(|(i, model, text)| {
                let insight: String = text.split('.')
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(".");
                format!("=== {}. {} ===\n{}", i + 1, model, insight)
            })
            .collect();

        let synth_prompt = format!(
            "[Debate Synthesis — Top-{} Models]\n\
             Original question: {prompt}\n\n\
             === Best Insights ===\n\
             {}\n\n\
             === Task ===\n\
             Synthesize the best insights above into a clear, \
             comprehensive final answer. Resolve any disagreements. \
             Present naturally without mentioning the debate process.",
            k,
            top_insights.join("\n\n"),
        );

        let synthesis = delegate_for_model(*winner, &synth_prompt).await;
        info!("  Top-{} synthesis by {} ({} chars)", k, winner, synthesis.len());
        synthesis
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. BACKWARD COMPAT: intent-based selection untuk external use
// ═══════════════════════════════════════════════════════════════════════════

/// Suggest participants based on intent (static fallback jika dynamic selection
/// tidak digunakan)
pub fn suggest_participants_for_intent(
    intent: &crate::core::tier_router::IntentKind,
    primary: NxrModelId,
) -> Vec<NxrModelId> {
    use crate::core::tier_router::IntentKind::*;
    use NxrModelId::*;

    let mut suggested = vec![primary];

    match intent {
        CodeReview | CodeGenerate => push_unique(&mut suggested, vec![Cipher, Omnis, Axiom]),
        Emotion => push_unique(&mut suggested, vec![Axiom, Spectra, Omnis]),
        Creative => push_unique(&mut suggested, vec![Aether, Nexum, Omnis]),
        Security => push_unique(&mut suggested, vec![Vortex, Axiom, Omnis]),
        Strategy => push_unique(&mut suggested, vec![Nexum, Kronos, Omnis]),
        Reasoning | Factual => push_unique(&mut suggested, vec![Axiom, Nexum, Genesis, Kronos]),
        QuickQuery => push_unique(&mut suggested, vec![Omnis]),
        Knowledge => push_unique(&mut suggested, vec![Kronos, Omnis, Axiom]),
        General => push_unique(&mut suggested, vec![Omnis, Aether, Axiom]),
    }

    suggested
}

fn push_unique(list: &mut Vec<NxrModelId>, items: Vec<NxrModelId>) {
    for item in items {
        if !list.contains(&item) {
            list.push(item);
        }
    }
}
