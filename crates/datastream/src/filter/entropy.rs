use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

#[derive(Debug, Clone)]
pub struct EntropyFilter {
    pub min_entropy: f64,
    pub max_entropy: f64,
}

impl Default for EntropyFilter {
    fn default() -> Self {
        Self {
            min_entropy: 1.0,
            max_entropy: 8.0,
        }
    }
}

impl EntropyFilter {
    pub fn new(min_entropy: f64) -> Self {
        Self { min_entropy, ..Default::default() }
    }

    fn compute_entropy(&self, text: &str) -> f64 {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return 0.0;
        }

        let mut freq: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for &c in &chars {
            *freq.entry(c).or_insert(0) += 1;
        }

        let total = chars.len() as f64;
        let mut entropy = 0.0;
        for &count in freq.values() {
            let p = count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        let word_len_distinct: std::collections::HashSet<usize> =
            text.split_whitespace().map(|w| w.len()).collect();
        let word_entropy_bonus = (word_len_distinct.len() as f64).log2().min(3.0);

        entropy + word_entropy_bonus
    }

    fn is_repetitive(&self, text: &str) -> bool {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() < 3 {
            return false;
        }
        let unique_lines: std::collections::HashSet<&str> = lines.iter().cloned().collect();
        let ratio = unique_lines.len() as f64 / lines.len() as f64;
        ratio < 0.3
    }

    fn is_garbage(&self, text: &str) -> bool {
        let entropy = self.compute_entropy(text);
        if entropy < self.min_entropy {
            return true;
        }
        let alpha = text.chars().filter(|c| c.is_alphabetic()).count() as f64;
        let total = text.len().max(1) as f64;
        alpha / total < 0.3
    }
}

#[async_trait]
impl Filter for EntropyFilter {
    fn name(&self) -> &str {
        "entropy"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let entropy = self.compute_entropy(&sample.text);
        let repetitive = self.is_repetitive(&sample.text);
        let garbage = self.is_garbage(&sample.text);

        let mut reason = None;
        let passed = if garbage {
            reason = Some(format!("low_information: entropy={:.2}", entropy));
            false
        } else if repetitive {
            reason = Some("repetitive_content".to_string());
            false
        } else if entropy < self.min_entropy {
            reason = Some(format!("low_entropy: {:.2} < {}", entropy, self.min_entropy));
            false
        } else if entropy > self.max_entropy {
            reason = Some(format!("high_entropy: {:.2} > {}", entropy, self.max_entropy));
            false
        } else {
            true
        };

        let quality = if entropy > 3.5 && entropy < 6.0 { 0.8 }
                      else if entropy > 2.0 { 0.5 }
                      else { 0.2 };

        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: if passed { quality - 0.5 } else { -0.5 },
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
    }
}
