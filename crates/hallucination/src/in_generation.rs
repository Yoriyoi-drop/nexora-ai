use regex::Regex;
use std::sync::LazyLock;

static RE_YEAR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(202[4-9]|2030)\b").expect("valid year pattern regex")
});
static RE_RECENCY_WORDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(terbaru|latest|recently|baru-baru)\b").expect("valid recency keyword regex")
});
static RE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b").expect("valid number pattern regex")
});
static RE_LONG_QUOTE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#""[^"]{15,}""#).expect("valid quote pattern regex")
});
static RE_CITATION_KEYWORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(menurut|according to|research shows|studies show)\b").expect("valid citation keyword regex")
});

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
    recency_keywords: [&'static Regex; 2],
    specific_claim_patterns: [&'static Regex; 3],
}

impl InGenerationGuard {
    pub fn new(config: InGenConfig) -> Self {
        Self {
            recency_keywords: [
                &RE_YEAR_PATTERN,
                &RE_RECENCY_WORDS,
            ],
            specific_claim_patterns: [
                &RE_NUMBER,
                &RE_LONG_QUOTE,
                &RE_CITATION_KEYWORD,
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
            let specific = self
                .specific_claim_patterns
                .iter()
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
            format!(
                "{} [perlu verifikasi — please verify specific claims]",
                text
            )
        } else {
            text.to_string()
        }
    }

    pub fn extract_claims(&self, text: &str) -> Vec<String> {
        let mut claims = Vec::new();
        for sentence in text.split(|c: char| c == '.' || c == '!' || c == '?') {
            let trimmed = sentence.trim();
            if trimmed.len() < 20 {
                continue;
            }
            if self
                .specific_claim_patterns
                .iter()
                .any(|p| p.is_match(trimmed))
            {
                claims.push(trimmed.to_string());
            }
        }
        claims
    }

    pub fn count_specific_claims(&self, text: &str) -> usize {
        self.specific_claim_patterns
            .iter()
            .map(|p| p.find_iter(text).count())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> InGenerationGuard {
        InGenerationGuard::new(InGenConfig::default())
    }

    #[test]
    fn test_uncertainty_clean_input() {
        let g = guard();
        let score = g.compute_uncertainty("What is Rust?", None);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_uncertainty_recency_keyword() {
        let g = guard();
        let score = g.compute_uncertainty("What happened in 2025?", None);
        assert!(score > 0.0);
    }

    #[test]
    fn test_uncertainty_recency_latest() {
        let g = guard();
        let score = g.compute_uncertainty("What is the latest news?", None);
        assert!(score > 0.0);
    }

    #[test]
    fn test_uncertainty_specific_claim() {
        let g = guard();
        let score = g.compute_uncertainty("Menurut penelitian, 1,234 people agree", None);
        assert!(score >= 0.15);
    }

    #[test]
    fn test_uncertainty_multiple_triggers_clamped() {
        let g = guard();
        let score = g.compute_uncertainty(
            "In 2025, according to research, 1,234,567 people say the latest news",
            None,
        );
        assert!(score <= 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_enhance_with_uncertainty_low() {
        let g = guard();
        let result = g.enhance_with_uncertainty("Hello", 0.2);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_enhance_with_uncertainty_medium() {
        let g = guard();
        let result = g.enhance_with_uncertainty("Hello", 0.5);
        assert!(result.contains("verifikasi"));
    }

    #[test]
    fn test_enhance_with_uncertainty_high() {
        let g = guard();
        let result = g.enhance_with_uncertainty("Hello", 0.8);
        assert!(result.contains("TIDAK YAKIN"));
    }

    #[test]
    fn test_extract_claims_short_sentences_ignored() {
        let g = guard();
        let claims = g.extract_claims("Hi. OK. Bye.");
        assert!(claims.is_empty());
    }

    #[test]
    fn test_extract_claims_with_numbers() {
        let g = guard();
        let claims = g.extract_claims("The population is 1,234,567 according to research.");
        assert_eq!(claims.len(), 1);
    }

    #[test]
    fn test_extract_claims_with_quotes() {
        let g = guard();
        let claims = g.extract_claims("He said \"this is a very long quote indeed\".");
        assert_eq!(claims.len(), 1);
    }

    #[test]
    fn test_count_specific_claims_none() {
        let g = guard();
        assert_eq!(g.count_specific_claims("hello world"), 0);
    }

    #[test]
    fn test_count_specific_claims_numbers() {
        let g = guard();
        let count = g.count_specific_claims("There are 1,234 items and 5,678 people");
        assert!(count > 0);
    }
}
