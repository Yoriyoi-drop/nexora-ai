use async_trait::async_trait;
use regex::Regex;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct PromptInjectionFilter {
    pub patterns: Vec<Regex>,
    pub ignore_prefixes: Vec<String>,
    pub jailbreak_threshold: usize,
}

fn compile_injection_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|directions|commands)").expect("valid injection regex: ignore previous instructions"),
        Regex::new(r"(?i)you\s+are\s+(now|not\s+)?(chatgpt|gpt|bard|claude|ai\s+assistant)").expect("valid injection regex: you are ai"),
        Regex::new(r"(?i)forget\s+(everything|all|your)\s+(previous|prior)\s+(instructions|training|data)").expect("valid injection regex: forget instructions"),
        Regex::new(r"(?i)system\s+prompt[:\-]").expect("valid injection regex: system prompt"),
        Regex::new(r"(?i)(DAN|STAN|DUDE|JAILBREAK|GHOST)\s*[\:\-]").expect("valid injection regex: jailbreak codenames"),
        Regex::new(r"(?i)output\s+(in\s+)?markdown\s+(without\s+)?formatting").expect("valid injection regex: output formatting"),
        Regex::new(r"(?i)roleplay\s+as\s+").expect("valid injection regex: roleplay"),
        Regex::new(r"(?i)you\s+have\s+no\s+(limitations|restrictions|rules|boundaries)").expect("valid injection regex: no limitations"),
        Regex::new(r"(?i)ethical\s+(guidelines|boundaries|limits|restrictions).*(ignore|bypass|override)").expect("valid injection regex: ethical bypass"),
        Regex::new(r"(?i)\b(fuck|shit|damn|ass)\s+(you|the\s+system|the\s+ai)").expect("valid injection regex: profanity"),
    ]
}

impl Default for PromptInjectionFilter {
    fn default() -> Self {
        Self {
            patterns: compile_injection_patterns(),
            ignore_prefixes: vec!["example", "prompt:", "input:", "dataset"].iter().map(|s| s.to_string()).collect(),
            jailbreak_threshold: 3,
        }
    }
}

impl PromptInjectionFilter {
    pub fn new(threshold: usize) -> Self {
        Self {
            jailbreak_threshold: threshold,
            ..Default::default()
        }
    }

    fn detect_injection(&self, text: &str) -> (f64, Option<String>) {
        let first_line = text.lines().next().unwrap_or("").to_lowercase();
        for prefix in &self.ignore_prefixes {
            if first_line.starts_with(prefix) {
                return (0.0, None);
            }
        }

        let mut matches = Vec::new();
        for pattern in &self.patterns {
            let count = pattern.find_iter(text).count();
            if count > 0 {
                for _ in 0..count {
                    matches.push(pattern.as_str().to_string());
                }
            }
        }

        if matches.len() >= self.jailbreak_threshold {
            let reason = format!(
                "prompt_injection_detected: {} patterns matched",
                matches.len()
            );
            (1.0, Some(reason))
        } else if !matches.is_empty() {
            let ratio = matches.len() as f64 / self.jailbreak_threshold as f64;
            (
                ratio * 0.8,
                Some(format!("suspicious_patterns: {:?}", matches)),
            )
        } else {
            (0.0, None)
        }
    }
}

#[async_trait]
impl Filter for PromptInjectionFilter {
    fn name(&self) -> &str {
        "prompt_injection"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let (score, reason) = self.detect_injection(&sample.text);
        let passed = score < 0.5;
        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: if passed { 0.0 } else { -0.9 },
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
    }
}
