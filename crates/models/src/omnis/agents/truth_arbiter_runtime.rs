use nexora_shared::base_model::NxrModelResult;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct TruthArbiterRuntimeAgent;

impl TruthArbiterRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub fn arbitrate(&self, claim_a: &str, claim_b: &str) -> NxrModelResult<String> {
        if claim_a.is_empty() && claim_b.is_empty() {
            return Err("Both claims are empty — nothing to arbitrate".into());
        }
        let a_words: HashSet<&str> = claim_a.split_whitespace().collect();
        let b_words: HashSet<&str> = claim_b.split_whitespace().collect();
        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();
        let jaccard = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };
        let overlap_pct = jaccard * 100.0;

        let verdict = if overlap_pct > 80.0 {
            "AGREE — claims are semantically similar"
        } else if overlap_pct > 40.0 {
            "PARTIAL — claims share some common ground"
        } else {
            "CONTRADICT — claims have low semantic overlap"
        };

        Ok(format!(
            "[TRUTH-ARBITER] Arbitration result:\n\
             - Semantic overlap: {:.1}%\n\
             - Shared terms: {}/{}\n\
             - Verdict: {}\n\
             - Confidence: {:.1}/100",
            overlap_pct,
            intersection,
            union,
            verdict,
            jaccard * 100.0
        ))
    }
}
