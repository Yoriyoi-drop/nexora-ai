use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

#[derive(Debug, Clone)]
pub struct TokenFilter {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub max_token_ratio: f64,
    pub block_tokens: Vec<String>,
}

impl Default for TokenFilter {
    fn default() -> Self {
        Self {
            min_tokens: 10,
            max_tokens: 2048,
            max_token_ratio: 0.8,
            block_tokens: vec![
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<s".to_string(),
                "</s>".to_string(),
            ],
        }
    }
}

impl TokenFilter {
    fn estimate_tokens(&self, text: &str) -> usize {
        let words = text.split_whitespace().count() as f64;
        let chars = text.len() as f64;
        (words * 1.3 + chars * 0.04) as usize
    }

    fn has_blocked_tokens(&self, text: &str) -> Option<String> {
        for token in &self.block_tokens {
            if text.contains(token) {
                return Some(format!("blocked_token: {}", token));
            }
        }
        None
    }
}

#[async_trait]
impl Filter for TokenFilter {
    fn name(&self) -> &str {
        "token"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        if let Some(reason) = self.has_blocked_tokens(&sample.text) {
            return FilterResult {
                passed: false,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: Some(reason),
                score_delta: -0.5,
            };
        }

        let token_count = self.estimate_tokens(&sample.text);
        let mut reason = None;
        let passed = if token_count < self.min_tokens {
            reason = Some(format!("too_few_tokens: {} < {}", token_count, self.min_tokens));
            false
        } else if token_count > self.max_tokens {
            reason = Some(format!("too_many_tokens: {} > {}", token_count, self.max_tokens));
            false
        } else {
            true
        };

        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: if passed { 0.0 } else { -0.2 },
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
    }
}
