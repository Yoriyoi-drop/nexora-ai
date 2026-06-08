use crate::types::PostGenCheckResult;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

#[cfg(feature = "gpu")]
use crate::gpu_verifier::gpu_batch_verify;

static RE_CITATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[(\d+|citation|sumber|source)\]").expect("valid citation pattern regex")
});
static RE_LARGE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b\d{2,}(\.\d+)?%?\b").expect("valid large number pattern regex")
});
static RE_CONTRADICTION_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(namun|however|but|on the other hand|sebaliknya|di sisi lain)\b")
        .expect("valid contradiction marker regex")
});
static RE_CONTRADICTION_KEYWORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(bertentangan|contradict|contrary|sebaliknya)\b")
        .expect("valid contradiction keyword regex")
});

#[derive(Debug, Clone)]
pub struct PostGenConfig {
    pub consistency_threshold: f32,
    pub source_grounding_threshold: f32,
    pub enable_self_consistency: bool,
    pub num_consistency_samples: usize,
}

impl Default for PostGenConfig {
    fn default() -> Self {
        Self {
            consistency_threshold: 0.7,
            source_grounding_threshold: 0.5,
            enable_self_consistency: true,
            num_consistency_samples: 3,
        }
    }
}

pub struct PostGenerationVerifier {
    config: PostGenConfig,
    contradiction_markers: [&'static Regex; 2],
}

impl PostGenerationVerifier {
    pub fn new(config: PostGenConfig) -> Self {
        Self {
            contradiction_markers: [&RE_CONTRADICTION_MARKER, &RE_CONTRADICTION_KEYWORD],
            config,
        }
    }

    pub async fn verify(
        &self,
        text: &str,
        sources: Option<Vec<String>>,
    ) -> Result<PostGenCheckResult, crate::HallucinationError> {
        let sentences: Vec<&str> = text
            .split(|c: char| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| s.len() > 15)
            .collect();

        // Try GPU-accelerated batch verification when sources are available
        let sent_owned: Vec<String> = sentences.iter().map(|s| s.to_string()).collect();
        if sources.is_some() {
            if let Some(gpu_result) = self.try_gpu_verify(&sent_owned, &sources) {
                return Ok(gpu_result);
            }
        }

        let total_claims = sentences.len();
        let high_risk_sentences = self.identify_high_risk(&sentences);
        let contradiction_count = self.detect_contradictions(text);

        let internal_consistency = if total_claims > 0 {
            (1.0 - contradiction_count as f32 / total_claims.max(1) as f32).max(0.0)
        } else {
            1.0
        };

        let source_grounding = self.check_source_grounding(text, &sources);

        let mut verified_claims = 0;
        if let Some(ref src) = sources {
            for sent in &sentences {
                for word in sent.split_whitespace().take(10) {
                    if src.iter().any(|s| s.contains(word)) {
                        verified_claims += 1;
                        break;
                    }
                }
            }
        }

        Ok(PostGenCheckResult {
            internal_consistency,
            source_grounding,
            high_risk_sentences,
            contradiction_count,
            total_claims,
            verified_claims,
        })
    }

    fn try_gpu_verify(
        &self,
        sentences: &[String],
        sources: &Option<Vec<String>>,
    ) -> Option<PostGenCheckResult> {
        let src = sources.as_ref()?;
        if src.is_empty() || sentences.is_empty() {
            return None;
        }
        #[cfg(feature = "gpu")]
        {
            match gpu_batch_verify(sentences, src) {
                Ok(result) => {
                    let contradictions = self.detect_contradictions(&sentences.join(". "));
                    let internal_consistency = if result.total_claims > 0 {
                        (1.0 - contradictions as f32 / result.total_claims.max(1) as f32).max(0.0)
                    } else {
                        1.0
                    };
                    return Some(PostGenCheckResult {
                        internal_consistency,
                        ..result
                    });
                }
                Err(_) => {}
            }
        }
        None
    }

    fn identify_high_risk(&self, sentences: &[&str]) -> Vec<String> {
        let mut risky = Vec::new();
        for s in sentences {
            let mut risk = 0.0;
            if RE_LARGE_NUMBER.find_iter(s).count() > 2 {
                risk += 0.4;
            }
            if !RE_CITATION.is_match(s) && RE_LARGE_NUMBER.is_match(s) {
                risk += 0.3;
            }
            if s.to_lowercase().contains("menurut") && !s.to_lowercase().contains("saya") {
                risk += 0.3;
            }
            if risk > 0.5 {
                risky.push(s.to_string());
            }
        }
        risky
    }

    fn detect_contradictions(&self, text: &str) -> usize {
        let lower = text.to_lowercase();
        let mut contradictions = 0;
        let yes_no_pairs = vec![
            ("ya", "tidak"),
            ("yes", "no"),
            ("benar", "salah"),
            ("true", "false"),
            ("setuju", "menolak"),
            (" always", " never"),
            ("semua", "tidak ada"),
        ];

        for (a, b) in &yes_no_pairs {
            if lower.contains(a) && lower.contains(b) {
                contradictions += 1;
            }
        }

        for marker in &self.contradiction_markers {
            if marker.find_iter(&lower).count() >= 2 {
                contradictions += 1;
            }
        }

        contradictions
    }

    fn check_source_grounding(&self, _text: &str, sources: &Option<Vec<String>>) -> f32 {
        match sources {
            Some(src) if !src.is_empty() => {
                let grounded: f32 = src
                    .iter()
                    .map(|s| s.split_whitespace().count() as f32)
                    .sum();
                (grounded / 100.0).min(1.0)
            }
            _ => 0.0,
        }
    }

    pub fn check_self_consistency(&self, samples: &[String]) -> HashMap<String, f32> {
        let mut key_claims: HashMap<String, Vec<bool>> = HashMap::new();

        for sample in samples {
            let lower = sample.to_lowercase();
            let sentences: Vec<&str> = lower.split('.').collect();
            for sent in &sentences {
                let sent = sent.trim();
                if sent.len() < 20 {
                    continue;
                }
                let words: Vec<&str> = sent.split_whitespace().collect();
                if words.len() < 5 {
                    continue;
                }
                let key = words[..5.min(words.len())].join(" ");
                let entry = key_claims.entry(key).or_default();
                entry.push(true);
            }
        }

        let mut consistency: HashMap<String, f32> = HashMap::new();
        for (claim, occurrences) in &key_claims {
            if occurrences.len() < 2 {
                consistency.insert(claim.clone(), 1.0);
                continue;
            }
            let total = occurrences.len() as f32;
            let agreed = occurrences.iter().filter(|&&x| x).count() as f32;
            consistency.insert(claim.clone(), agreed / total);
        }

        consistency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verifier() -> PostGenerationVerifier {
        PostGenerationVerifier::new(PostGenConfig::default())
    }

    #[tokio::test]
    async fn test_verify_empty_text() {
        let v = verifier();
        let result = v.verify("", None).await.unwrap();
        assert_eq!(result.total_claims, 0);
        assert_eq!(result.internal_consistency, 1.0);
        assert_eq!(result.source_grounding, 0.0);
    }

    #[tokio::test]
    async fn test_verify_short_sentences_ignored() {
        let v = verifier();
        let result = v.verify("Hi. OK. Bye.", None).await.unwrap();
        assert_eq!(result.total_claims, 0);
    }

    #[tokio::test]
    async fn test_verify_with_claims() {
        let v = verifier();
        let text = "Paris is the capital of France. It has a population of 2,161,000 people.";
        let result = v.verify(text, None).await.unwrap();
        assert_eq!(result.total_claims, 2);
    }

    #[tokio::test]
    async fn test_verify_with_sources_grounding() {
        let v = verifier();
        let text = "Rust is a systems programming language.";
        let sources = Some(vec!["Rust programming language".into()]);
        let result = v.verify(text, sources).await.unwrap();
        assert!(result.verified_claims > 0);
    }

    #[tokio::test]
    async fn test_verify_contradiction_detected() {
        let v = verifier();
        let text = "The answer is yes. Actually the answer is no.";
        let result = v.verify(text, None).await.unwrap();
        assert!(result.contradiction_count > 0);
    }

    #[tokio::test]
    async fn test_verify_high_risk_numbers_without_citation() {
        let v = verifier();
        let text = "The population is 12,500,000 and the area covers 99,000 square kilometers.";
        let result = v.verify(text, None).await.unwrap();
        assert!(!result.high_risk_sentences.is_empty());
    }

    #[tokio::test]
    async fn test_verify_high_risk_menurut_without_saya() {
        let v = verifier();
        let text = "Menurut penelitian, the GDP grew by 12,500,000 last year.";
        let result = v.verify(text, None).await.unwrap();
        assert!(!result.high_risk_sentences.is_empty());
    }

    #[tokio::test]
    async fn test_verify_not_high_risk_with_citation() {
        let v = verifier();
        let text = "According to [1], the GDP grew by 4.5 percent.";
        let result = v.verify(text, None).await.unwrap();
        assert!(result.high_risk_sentences.is_empty());
    }

    #[test]
    fn test_detect_contradictions_yes_no() {
        let v = verifier();
        let count = v.detect_contradictions("The answer is yes. Actually no.");
        assert!(count > 0);
    }

    #[test]
    fn test_detect_contradictions_true_false() {
        let v = verifier();
        let count = v.detect_contradictions("It is true. It is false.");
        assert!(count > 0);
    }

    #[test]
    fn test_detect_contradictions_marker_repeated() {
        let v = verifier();
        let count = v.detect_contradictions("However this. However that.");
        assert!(count > 0);
    }

    #[test]
    fn test_detect_contradictions_none() {
        let v = verifier();
        let count = v.detect_contradictions("Everything is consistent and clear.");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_check_source_grounding_no_sources() {
        let v = verifier();
        assert_eq!(v.check_source_grounding("text", &None), 0.0);
    }

    #[test]
    fn test_check_source_grounding_with_sources() {
        let v = verifier();
        let src = Some(vec!["a".repeat(100)]);
        let score = v.check_source_grounding("text", &src);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_self_consistency_single_sample() {
        let v = verifier();
        let samples = vec!["This is a very long sample sentence that should be captured.".into()];
        let result = v.check_self_consistency(&samples);
        assert!(!result.is_empty());
        for (_, score) in &result {
            assert_eq!(*score, 1.0);
        }
    }

    #[test]
    fn test_self_consistency_multiple_samples() {
        let v = verifier();
        let samples = vec![
            "Population of Tokyo is 37 million people according to data.".into(),
            "Population of Tokyo is 37 million people according to research.".into(),
        ];
        let result = v.check_self_consistency(&samples);
        assert!(!result.is_empty());
    }
}
