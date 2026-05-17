use async_trait::async_trait;
use regex::Regex;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction, Domain};

#[derive(Debug, Clone)]
pub struct DomainClassifier {
    pub code_patterns: Vec<Regex>,
    pub reasoning_patterns: Vec<Regex>,
    pub knowledge_patterns: Vec<Regex>,
    pub math_patterns: Vec<Regex>,
    pub instruction_patterns: Vec<Regex>,
}

impl Default for DomainClassifier {
    fn default() -> Self {
        Self {
            code_patterns: vec![
                Regex::new(r"(?m)^(fn|def|function|class|impl|struct|enum|pub|use|import|from)\s").expect("valid regex pattern"),
                Regex::new(r"\{[\s\S]*\}").expect("valid regex pattern"),
                Regex::new(r"(?m)^\s*#\s*(include|define|pragma)").expect("valid regex pattern"),
                Regex::new(r"->\s*[A-Za-z_][A-Za-z0-9_<>]*").expect("valid regex pattern"),
                Regex::new(r"(?m)^(for|while|if|match|switch)\s*\(").expect("valid regex pattern"),
            ],
            reasoning_patterns: vec![
                Regex::new(r"(?i)\b(therefore|because|since|hence|thus|consequently)\b").expect("valid regex pattern"),
                Regex::new(r"(?i)\b(step\s+\d+|firstly|secondly|finally)\b").expect("valid regex pattern"),
                Regex::new(r"(?i)\b(conclusion|analysis|reasoning|logically)\b").expect("valid regex pattern"),
                Regex::new(r"(?i)\b(if.*then|implies|iff|whenever)\b").expect("valid regex pattern"),
            ],
            knowledge_patterns: vec![
                Regex::new(r"(?i)\b(wikipedia|according\s+to|reference|source)\b").expect("valid regex pattern"),
                Regex::new(r"(?i)\b(century|decade|era|period|historical)\b").expect("valid regex pattern"),
                Regex::new(r"\b\d{4}\b").expect("valid regex pattern"),
            ],
            math_patterns: vec![
                Regex::new(r"\\[\(\[].*?\\[\)\]]").expect("valid regex pattern"),
                Regex::new(r"\b(\d+[\+\-\*\/]\d+|\d+=\d+)\b").expect("valid regex pattern"),
                Regex::new(r"\b(equation|theorem|lemma|proof|axiom)\b").expect("valid regex pattern"),
                Regex::new(r"\b(sin|cos|tan|log|ln|sqrt|integral|derivative)\b").expect("valid regex pattern"),
            ],
            instruction_patterns: vec![
                Regex::new(r"(?i)\b(please|could\s+you|can\s+you|would\s+you)\s+(explain|help|tell|show|write|create)").expect("valid regex pattern"),
                Regex::new(r"(?i)^(write|create|generate|make|build|design|implement)\s").expect("valid regex pattern"),
                Regex::new(r"(?i)^(what|how|why|when|where|who|which)\s").expect("valid regex pattern"),
            ],
        }
    }
}

impl DomainClassifier {
    pub fn classify(&self, text: &str) -> Vec<(Domain, f64)> {
        let mut scores = Vec::with_capacity(6);

        let code_score = self.score_patterns(text, &self.code_patterns);
        if code_score > 0.0 {
            scores.push((Domain::Code, code_score));
        }

        let reasoning_score = self.score_patterns(text, &self.reasoning_patterns);
        if reasoning_score > 0.0 {
            scores.push((Domain::Reasoning, reasoning_score));
        }

        let knowledge_score = self.score_patterns(text, &self.knowledge_patterns);
        if knowledge_score > 0.0 {
            scores.push((Domain::Knowledge, knowledge_score));
        }

        let math_score = self.score_patterns(text, &self.math_patterns);
        if math_score > 0.0 {
            scores.push((Domain::Math, math_score));
        }

        let instruction_score = self.score_patterns(text, &self.instruction_patterns);
        if instruction_score > 0.0 {
            scores.push((Domain::Instruction, instruction_score));
        }

        if scores.is_empty() {
            let word_count = text.split_whitespace().count();
            if word_count > 50 {
                scores.push((Domain::General, 0.6));
            } else if word_count > 10 {
                scores.push((Domain::Conversation, 0.7));
            } else {
                scores.push((Domain::General, 0.3));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    fn score_patterns(&self, text: &str, patterns: &[Regex]) -> f64 {
        let mut score = 0.0;
        for pattern in patterns {
            let matches = pattern.find_iter(text).count() as f64;
            score += matches * 0.2;
        }
        score.min(1.0)
    }
}

#[async_trait]
impl Filter for DomainClassifier {
    fn name(&self) -> &str {
        "domain_classifier"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let domains = self.classify(&sample.text);
        FilterResult {
            passed: true,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason: Some(format!("domains: {:?}", domains.iter().map(|(d, s)| (d, s)).collect::<Vec<_>>())),
            score_delta: domains.first().map(|(_, s)| s / 2.0).unwrap_or(0.0),
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reroute(Domain::General)
    }
}
