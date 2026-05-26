use async_trait::async_trait;
use regex::Regex;

use super::traits::Filter;
use crate::types::{DataSample, Domain, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct DomainClassifier {
    pub code_patterns: Vec<Regex>,
    pub reasoning_patterns: Vec<Regex>,
    pub knowledge_patterns: Vec<Regex>,
    pub math_patterns: Vec<Regex>,
    pub instruction_patterns: Vec<Regex>,
}

fn default_code_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"(?m)^(fn|def|function|class|impl|struct|enum|pub|use|import|from)\s").expect("valid domain regex: code keywords"),
        Regex::new(r"\{[\s\S]*\}").expect("valid domain regex: code braces"),
        Regex::new(r"(?m)^\s*#\s*(include|define|pragma)").expect("valid domain regex: preprocessor"),
        Regex::new(r"->\s*[A-Za-z_][A-Za-z0-9_<>]*").expect("valid domain regex: return type"),
        Regex::new(r"(?m)^(for|while|if|match|switch)\s*\(").expect("valid domain regex: control flow"),
    ]
}

fn default_reasoning_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"(?i)\b(therefore|because|since|hence|thus|consequently)\b").expect("valid domain regex: reasoning connectives"),
        Regex::new(r"(?i)\b(step\s+\d+|firstly|secondly|finally)\b").expect("valid domain regex: reasoning steps"),
        Regex::new(r"(?i)\b(conclusion|analysis|reasoning|logically)\b").expect("valid domain regex: reasoning nouns"),
        Regex::new(r"(?i)\b(if.*then|implies|iff|whenever)\b").expect("valid domain regex: logical operators"),
    ]
}

fn default_knowledge_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"(?i)\b(wikipedia|according\s+to|reference|source)\b").expect("valid domain regex: citation"),
        Regex::new(r"(?i)\b(century|decade|era|period|historical)\b").expect("valid domain regex: time periods"),
        Regex::new(r"\b\d{4}\b").expect("valid domain regex: year"),
    ]
}

fn default_math_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"\\[\(\[].*?\\[\)\]]").expect("valid domain regex: LaTeX"),
        Regex::new(r"\b(\d+[\+\-\*\/]\d+|\d+=\d+)\b").expect("valid domain regex: arithmetic"),
        Regex::new(r"\b(equation|theorem|lemma|proof|axiom)\b").expect("valid domain regex: math nouns"),
        Regex::new(r"\b(sin|cos|tan|log|ln|sqrt|integral|derivative)\b").expect("valid domain regex: math functions"),
    ]
}

fn default_instruction_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"(?i)\b(please|could\s+you|can\s+you|would\s+you)\s+(explain|help|tell|show|write|create)").expect("valid domain regex: polite requests"),
        Regex::new(r"(?i)^(write|create|generate|make|build|design|implement)\s").expect("valid domain regex: imperative"),
        Regex::new(r"(?i)^(what|how|why|when|where|who|which)\s").expect("valid domain regex: question words"),
    ]
}

impl Default for DomainClassifier {
    fn default() -> Self {
        Self {
            code_patterns: default_code_patterns(),
            reasoning_patterns: default_reasoning_patterns(),
            knowledge_patterns: default_knowledge_patterns(),
            math_patterns: default_math_patterns(),
            instruction_patterns: default_instruction_patterns(),
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
            reason: Some(format!(
                "domains: {:?}",
                domains.iter().map(|(d, s)| (d, s)).collect::<Vec<_>>()
            )),
            score_delta: domains.first().map(|(_, s)| s / 2.0).unwrap_or(0.0),
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Accept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceInfo, SourceCategory};
    use uuid::Uuid;

    fn sample(text: &str) -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: text.into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_default_has_patterns() {
        let d = DomainClassifier::default();
        assert!(!d.code_patterns.is_empty());
        assert!(!d.reasoning_patterns.is_empty());
        assert!(!d.knowledge_patterns.is_empty());
    }

    #[test]
    fn test_code_detection() {
        let d = DomainClassifier::default();
        let domains = d.classify("fn hello() { println!(\"world\"); }");
        assert!(domains.iter().any(|(dom, _)| *dom == Domain::Code));
    }

    #[test]
    fn test_reasoning_detection() {
        let d = DomainClassifier::default();
        let domains = d.classify("therefore the conclusion follows because the premise is logically sound");
        assert!(domains.iter().any(|(dom, _)| *dom == Domain::Reasoning));
    }

    #[test]
    fn test_math_detection() {
        let d = DomainClassifier::default();
        let domains = d.classify("the equation $x^2 + y^2 = z^2$ is known as the theorem of pythagoras");
        assert!(domains.iter().any(|(dom, _)| *dom == Domain::Math));
    }

    #[test]
    fn test_instruction_detection() {
        let d = DomainClassifier::default();
        let domains = d.classify("please explain how quantum computing works");
        assert!(domains.iter().any(|(dom, _)| *dom == Domain::Instruction));
    }

    #[test]
    fn test_short_text_falls_back_to_conversation() {
        let d = DomainClassifier::default();
        let domains = d.classify("hello how are you");
        assert!(domains.iter().any(|(dom, _)| *dom == Domain::Conversation));
    }

    #[tokio::test]
    async fn test_evaluate_returns_domains() {
        let d = DomainClassifier::default();
        let s = sample("fn test() { return 42; } and therefore we conclude this is code with reasoning");
        let result = d.evaluate(&s).await;
        assert!(result.passed);
        assert!(result.reason.unwrap().contains("domains:"));
    }

    #[test]
    fn test_score_patterns() {
        let d = DomainClassifier::default();
        let score = d.score_patterns(
            "fn main() { println!(\"hi\"); }",
            &default_code_patterns(),
        );
        assert!(score > 0.0);
    }
}
