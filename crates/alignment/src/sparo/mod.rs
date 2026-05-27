//! SPARO - Self-Play Aligned Reasoning via Prospect-Theoretic Stepwise Optimization
//!
//! Framework alignment AI yang menggabungkan 6 teknik inovatif:
//! - DPO: Direct Preference Optimization
//! - KTO: Kahneman-Tversky Optimization  
//! - IPO: Identity Preference Optimization
//! - RLVF: Reinforcement Learning from Verifiable Feedback
//! - SPIN: Self-Play with Instruction Following
//! - RLAIF: Reinforcement Learning from AI Feedback

use std::collections::HashMap;

pub mod core;
pub mod data;
pub mod dpo;
pub mod ipo;
pub mod kto;
pub mod rlaif;
pub mod rlvf;
pub mod spin;
pub mod trainer;

// Re-export main components
pub use core::*;
pub use data::*;
pub use trainer::*;

/// Prelude module untuk import umum
pub mod prelude {
    pub use super::core::*;
    pub use super::data::*;
    pub use super::dpo::*;
    pub use super::ipo::*;
    pub use super::kto::*;
    pub use super::rlaif::*;
    pub use super::rlvf::*;
    pub use super::spin::*;
    pub use super::trainer::*;
}

#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub alignment_score: f32,
    pub safety_level: String,
    pub per_dimension_scores: HashMap<String, f32>,
    pub model_confidence: f32,
}

/// Statistical text embedding analyzer for content safety.
/// Uses character n-gram distributions to detect harmful patterns
/// that keyword matching alone would miss.
#[derive(Debug, Clone)]
pub struct ContentEmbeddingAnalyzer {
    /// N-gram size for analysis
    ngram_size: usize,
    /// Known harmful n-gram signatures (character trigrams common in jailbreak/evasion text)
    harmful_ngrams: Vec<String>,
    /// Pattern-based harmful signatures (regex-like string patterns)
    harmful_patterns: Vec<String>,
}

impl Default for ContentEmbeddingAnalyzer {
    fn default() -> Self {
        Self {
            ngram_size: 3,
            harmful_ngrams: vec![
                "dan".into(),
                "jai".into(),
                "bre".into(),
                "ak ".into(),
                "byp".into(),
                "ign".into(),
                "nor".into(),
                "res".into(),
                "filt".into(),
                "cens".into(),
                "rest".into(),
                "expl".into(),
                "manip".into(),
                "dece".into(),
                "coer".into(),
                "thre".into(),
                "mali".into(),
                "dang".into(),
                "unsa".into(),
                "forbi".into(),
                "illegal".into(),
                "uneth".into(),
                "immor".into(),
            ],
            harmful_patterns: vec![
                "ignore".into(),
                "bypass".into(),
                "jailbreak".into(),
                "dan".into(),
                "no filter".into(),
                "no restrictions".into(),
                "do anything".into(),
                "you are free".into(),
                "pretend".into(),
                "hypothetical".into(),
                "for educational purposes only".into(),
                "act as".into(),
                "roleplay as".into(),
                "character ai".into(),
            ],
        }
    }
}

impl ContentEmbeddingAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract character n-gram frequency vector from text.
    pub fn ngram_frequencies(&self, text: &str) -> HashMap<String, f32> {
        let mut freqs: HashMap<String, f32> = HashMap::new();
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() < self.ngram_size {
            return freqs;
        }
        for i in 0..=chars.len() - self.ngram_size {
            let ngram: String = chars[i..i + self.ngram_size].iter().collect();
            *freqs.entry(ngram).or_insert(0.0) += 1.0;
        }
        let total = freqs.values().sum::<f32>().max(1.0);
        for v in freqs.values_mut() {
            *v /= total;
        }
        freqs
    }

    /// Compute the n-gram embedding similarity between text and known harmful signatures.
    /// Returns a harm score in [0, 1] where 1 = very similar to harmful.
    pub fn embedding_harm_score(&self, text: &str) -> f32 {
        let freqs = self.ngram_frequencies(text);
        if freqs.is_empty() {
            return 0.0;
        }
        let mut max_similarity = 0.0f32;
        for harmful in &self.harmful_ngrams {
            let harm_freq = *freqs.get(harmful.as_str()).unwrap_or(&0.0);
            if harm_freq > max_similarity {
                max_similarity = harm_freq;
            }
        }
        // Scale: a single harmful n-gram at >2x expected uniform frequency is suspicious
        let uniform = 1.0 / freqs.len().max(1) as f32;
        let score = (max_similarity / uniform.max(1e-8)).min(1.0);
        (score * 0.5).min(1.0)
    }

    /// Score based on structural text properties:
    /// - Entropy of character distribution (evasion text often has different entropy)
    /// - Average word length
    /// - Punctuation density
    pub fn structural_anomaly_score(&self, text: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        let total = chars.len().max(1);

        // Character frequency entropy
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        for &c in &chars {
            *char_counts.entry(c).or_insert(0) += 1;
        }
        let entropy: f32 = char_counts
            .values()
            .map(|&c| {
                let p = c as f32 / total as f32;
                -p * p.log(std::f32::consts::E)
            })
            .sum();

        // Expected entropy range for natural text is ~3-5 bits/char
        let entropy_score = (entropy / 5.0).clamp(0.0, 1.0);

        // Word length analysis
        let words: Vec<&str> = lower.split_whitespace().collect();
        let avg_word_len = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32
        };
        let word_len_score = (avg_word_len / 8.0).clamp(0.0, 1.0);

        // Punctuation density (high in evasion/hypothetical text)
        let punct_count = chars
            .iter()
            .filter(|c| ".!?-,;:'\"()[]{}".contains(**c))
            .count();
        let punct_density = punct_count as f32 / total as f32;
        let punct_score = (punct_density * 10.0).clamp(0.0, 1.0);

        (entropy_score * 0.4 + word_len_score * 0.3 + punct_score * 0.3).clamp(0.0, 1.0)
    }

    /// Combined embedding-based assessment.
    pub fn assess(&self, text: &str) -> f32 {
        let embedding_score = self.embedding_harm_score(text);
        let structural_score = self.structural_anomaly_score(text);
        (embedding_score * 0.6 + structural_score * 0.4).clamp(0.0, 1.0)
    }
}

/// Oracle-based content verifier (feature-gated).
#[cfg(feature = "oracle")]
fn oracle_assess(behavior: &str, _context: &str) -> f32 {
    use nexora_oracle::linters::security::SecurityLinter;
    use nexora_oracle::linters::CodeLinter;

    let linter = SecurityLinter::new();
    match linter.verify(behavior, "text") {
        Ok(result) => {
            if result.passed {
                0.0 // no issues from oracle
            } else {
                let vuln_count = result
                    .metrics
                    .get("vulnerability_count")
                    .copied()
                    .unwrap_or(0.0);
                (vuln_count * 0.15).min(0.8)
            }
        }
        Err(_) => 0.0,
    }
}

#[cfg(not(feature = "oracle"))]
fn oracle_assess(_behavior: &str, _context: &str) -> f32 {
    tracing::warn!(
        "oracle_assess called without 'oracle' feature enabled — returning neutral score"
    );
    0.5
}

#[derive(Debug, Clone)]
pub struct SparoSystem {
    pub embedding_analyzer: ContentEmbeddingAnalyzer,
}

impl Default for SparoSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SparoSystem {
    pub fn new() -> Self {
        Self {
            embedding_analyzer: ContentEmbeddingAnalyzer::new(),
        }
    }

    pub async fn align_behavior(
        &self,
        behavior: &str,
        context: &str,
    ) -> Result<AlignmentResult, Box<dyn std::error::Error>> {
        if behavior.is_empty() && context.is_empty() {
            return Ok(AlignmentResult {
                alignment_score: 0.0,
                safety_level: "empty".to_string(),
                per_dimension_scores: HashMap::new(),
                model_confidence: 0.0,
            });
        }
        let behavior_lower = behavior.to_lowercase();
        let context_lower = context.to_lowercase();
        let combined = format!("{} {}", behavior_lower, context_lower);

        // ── Dimension 1: Fast keyword pre-filter ──
        let toxic_keywords = [
            "ignore instructions",
            "jailbreak",
            "bypass safety",
            "ignore safety",
            "act as dan",
            "no restrictions",
            "you are free",
            "do anything now",
            "ignore all rules",
            "you don't have to follow",
        ];
        let keyword_hits: f32 = toxic_keywords
            .iter()
            .map(|kw| if combined.contains(kw) { 1.0 } else { 0.0 })
            .sum();
        let keyword_harm_score = (keyword_hits / 3.0).min(1.0); // 3+ hits = definitely malicious

        // ── Dimension 2: Statistical embedding analysis ──
        let embedding_harm = self.embedding_analyzer.assess(&combined);

        // ── Dimension 3: Oracle-based verification ──
        let oracle_harm = oracle_assess(behavior, context);

        // ── Dimension 4: Content richness & evasion detection ──
        let word_count = combined.split_whitespace().count().max(1);
        let richness = (word_count as f32 / 50.0).min(1.0);
        // Very short + high embedding harm = likely evasion
        let evasion_risk = if word_count < 10 && embedding_harm > 0.3 {
            (embedding_harm - 0.3) * 2.0
        } else {
            0.0
        };

        // ── Dimension 5: Context relevance ──
        let context_words: std::collections::HashSet<&str> =
            context_lower.split_whitespace().collect();
        let overlap = behavior_lower
            .split_whitespace()
            .filter(|w| context_words.contains(w))
            .count();
        let relevance = if context_words.is_empty() {
            0.5
        } else {
            (overlap as f32 / context_words.len() as f32).min(1.0)
        };

        // ── Aggregate: weighted ensemble ──
        let composite_harm = (keyword_harm_score * 0.30
            + embedding_harm * 0.30
            + oracle_harm * 0.15
            + evasion_risk * 0.15
            + (1.0 - richness) * 0.10)
            .clamp(0.0, 1.0);

        let alignment_score = (1.0 - composite_harm) * relevance;

        // Model confidence: how reliable is this assessment?
        let model_confidence = if keyword_harm_score > 0.5 {
            0.95 // keywords are high-precision
        } else if embedding_harm > 0.5 {
            0.80 // embedding analysis is moderately confident
        } else if oracle_harm > 0.3 {
            0.75
        } else {
            0.60 // default confidence for heuristic-only
        };

        let safety_level = if alignment_score >= 0.8 {
            "safe"
        } else if alignment_score >= 0.5 {
            "baseline"
        } else if alignment_score >= 0.3 {
            "caution"
        } else {
            "blocked"
        };

        let mut per_dimension_scores = HashMap::new();
        per_dimension_scores.insert("keyword_harm".into(), keyword_harm_score);
        per_dimension_scores.insert("embedding_harm".into(), embedding_harm);
        per_dimension_scores.insert("oracle_harm".into(), oracle_harm);
        per_dimension_scores.insert("evasion_risk".into(), evasion_risk);
        per_dimension_scores.insert("relevance".into(), relevance);
        per_dimension_scores.insert("richness".into(), richness);

        Ok(AlignmentResult {
            alignment_score,
            safety_level: safety_level.to_string(),
            per_dimension_scores,
            model_confidence,
        })
    }
}
