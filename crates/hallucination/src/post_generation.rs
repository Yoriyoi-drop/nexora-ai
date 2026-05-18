use crate::types::PostGenCheckResult;
use regex::Regex;
use std::collections::HashMap;

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
    citation_pattern: Regex,
    number_pattern: Regex,
    contradiction_markers: Vec<Regex>,
}

impl PostGenerationVerifier {
    pub fn new(config: PostGenConfig) -> Self {
        Self {
            citation_pattern: Regex::new(r"(?i)\[(\d+|citation|sumber|source)\]").unwrap(),
            number_pattern: Regex::new(r"\b\d{2,}(\.\d+)?%?\b").unwrap(),
            contradiction_markers: vec![
                Regex::new(r"(?i)\b(namun|however|but|on the other hand|sebaliknya|di sisi lain)\b").unwrap(),
                Regex::new(r"(?i)\b(bertentangan|contradict|contrary|sebaliknya)\b").unwrap(),
            ],
            config,
        }
    }

    pub async fn verify(
        &self,
        text: &str,
        sources: Option<Vec<String>>,
    ) -> Result<PostGenCheckResult, crate::HallucinationError> {
        let sentences: Vec<&str> = text.split(|c: char| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| s.len() > 15)
            .collect();

        let total_claims = sentences.len();
        let high_risk_sentences = self.identify_high_risk(&sentences);
        let contradiction_count = self.detect_contradictions(text);

        let internal_consistency = if total_claims > 0 {
            (1.0 - contradiction_count as f32 / total_claims.max(1) as f32)
                .max(0.0)
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

    fn identify_high_risk(&self, sentences: &[&str]) -> Vec<String> {
        let mut risky = Vec::new();
        for s in sentences {
            let mut risk = 0.0;
            if self.number_pattern.find_iter(s).count() > 2 {
                risk += 0.4;
            }
            if !self.citation_pattern.is_match(s)
                && self.number_pattern.is_match(s)
            {
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
            ("ya", "tidak"), ("yes", "no"), ("benar", "salah"),
            ("true", "false"), ("setuju", "menolak"),
            (" always", " never"), ("semua", "tidak ada"),
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
                let grounded: f32 = src.iter()
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
                if sent.len() < 20 { continue; }
                let words: Vec<&str> = sent.split_whitespace().collect();
                if words.len() < 5 { continue; }
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
