use crate::types::InGenCheckResult;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct InGenConfig {
    pub uncertainty_threshold: f32,
    pub cot_threshold: f32,
    pub max_specific_claims_per_response: usize,
}

impl Default for InGenConfig {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.3,
            cot_threshold: 0.5,
            max_specific_claims_per_response: 3,
        }
    }
}

pub struct InGenerationGuard {
    config: InGenConfig,
    recency_keywords: Vec<Regex>,
    specific_claim_patterns: Vec<Regex>,
}

impl InGenerationGuard {
    pub fn new(config: InGenConfig) -> Self {
        Self {
            recency_keywords: vec![
                Regex::new(r"(?i)\b(202[4-9]|2030)\b").unwrap(),
                Regex::new(r"(?i)\b(terbaru|latest|recently|baru-baru)\b").unwrap(),
            ],
            specific_claim_patterns: vec![
                Regex::new(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b").unwrap(),
                Regex::new(r#""[^"]{15,}""#).unwrap(),
                Regex::new(r"(?i)\b(menurut|according to|research shows|studies show)\b").unwrap(),
            ],
            config,
        }
    }

    pub fn compute_uncertainty(&self, _input: &str, _context: Option<&str>) -> f32 {
        let mut score = 0.0f32;
        if let Some(input) = Some(_input) {
            for p in &self.recency_keywords {
                if p.is_match(input) {
                    score += 0.3;
                }
            }
            let specific = self.specific_claim_patterns.iter()
                .filter(|p| p.is_match(input))
                .count();
            score += specific as f32 * 0.15;
        }
        score.min(1.0)
    }

    pub fn enhance_with_uncertainty(&self, text: &str, uncertainty: f32) -> String {
        if uncertainty > 0.7 {
            format!(
                "[TIDAK YAKIN — uncertainty: {:.1}] {}\n\nNote: I am not fully confident about this. Please verify.",
                uncertainty, text
            )
        } else if uncertainty > 0.4 {
            format!("{} [perlu verifikasi — please verify specific claims]", text)
        } else {
            text.to_string()
        }
    }

    pub fn extract_claims(&self, text: &str) -> Vec<String> {
        let mut claims = Vec::new();
        for sentence in text.split(|c: char| c == '.' || c == '!' || c == '?') {
            let trimmed = sentence.trim();
            if trimmed.len() < 20 { continue; }
            if self.specific_claim_patterns.iter().any(|p| p.is_match(trimmed)) {
                claims.push(trimmed.to_string());
            }
        }
        claims
    }

    pub fn count_specific_claims(&self, text: &str) -> usize {
        self.specific_claim_patterns.iter()
            .map(|p| p.find_iter(text).count())
            .sum()
    }
}
