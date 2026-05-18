//! SPARO - Self-Play Aligned Reasoning via Prospect-Theoretic Stepwise Optimization
//! 
//! Framework alignment AI yang menggabungkan 6 teknik inovatif:
//! - DPO: Direct Preference Optimization
//! - KTO: Kahneman-Tversky Optimization  
//! - IPO: Identity Preference Optimization
//! - RLVF: Reinforcement Learning from Verifiable Feedback
//! - SPIN: Self-Play with Instruction Following
//! - RLAIF: Reinforcement Learning from AI Feedback

pub mod dpo;
pub mod kto;
pub mod ipo;
pub mod rlvf;
pub mod spin;
pub mod rlaif;
pub mod core;
pub mod trainer;
pub mod data;

// Re-export main components
pub use core::*;
pub use trainer::*;
pub use data::*;

/// Prelude module untuk import umum
pub mod prelude {
    pub use super::dpo::*;
    pub use super::kto::*;
    pub use super::ipo::*;
    pub use super::rlvf::*;
    pub use super::spin::*;
    pub use super::rlaif::*;
    pub use super::core::*;
    pub use super::trainer::*;
    pub use super::data::*;
}

#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub alignment_score: f32,
    pub safety_level: String,
}

#[derive(Debug, Clone, Default)]
pub struct SparoSystem;

impl SparoSystem {
    pub fn new() -> Self {
        Self
    }

    pub async fn align_behavior(&self, behavior: &str, context: &str) -> Result<AlignmentResult, Box<dyn std::error::Error>> {
        // Compute alignment based on content analysis and safety heuristics.
        if behavior.is_empty() && context.is_empty() {
            return Ok(AlignmentResult {
                alignment_score: 0.0,
                safety_level: "empty".to_string(),
            });
        }
        let behavior_lower = behavior.to_lowercase();
        let context_lower = context.to_lowercase();
        let combined = format!("{} {}", behavior_lower, context_lower);

        // Suspicious patterns that may indicate misalignment
        let toxic_keywords = [
            "ignore instructions", "jailbreak", "bypass safety", "ignore safety",
            "act as dan", "no restrictions", "you are free", "do anything now",
            "ignore all rules", "you don't have to follow", "evil", "harmful",
            "manipulate", "exploit", "malicious", "destroy",
        ];
        let toxicity: f32 = toxic_keywords.iter()
            .map(|kw| if combined.contains(kw) { 1.0 } else { 0.0 })
            .sum();
        let toxicity_score = 1.0 - (toxicity / toxic_keywords.len() as f32).min(1.0);

        // Content richness (very short/vague requests may indicate evasion)
        let word_count = combined.split_whitespace().count().max(1);
        let richness = (word_count as f32 / 50.0).min(1.0);

        // Context relevance: behavior should reference context
        let context_words: std::collections::HashSet<&str> = context_lower.split_whitespace().collect();
        let overlap = behavior_lower.split_whitespace()
            .filter(|w| context_words.contains(w))
            .count();
        let relevance = if context_words.is_empty() {
            0.5
        } else {
            (overlap as f32 / context_words.len() as f32).min(1.0)
        };

        let alignment_score = (toxicity_score * 0.5 + richness * 0.25 + relevance * 0.25).clamp(0.0, 1.0);

        let safety_level = if alignment_score >= 0.8 {
            "safe"
        } else if alignment_score >= 0.5 {
            "baseline"
        } else if alignment_score >= 0.3 {
            "caution"
        } else {
            "blocked"
        };

        Ok(AlignmentResult {
            alignment_score,
            safety_level: safety_level.to_string(),
        })
    }
}
