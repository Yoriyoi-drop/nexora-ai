use async_trait::async_trait;
use parking_lot::RwLock;
use std::sync::Arc;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct TokenFilter {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub max_token_ratio: f64,
    pub block_tokens: Vec<String>,
    pub tokenizer: Option<Arc<RwLock<nexora_foundation::tokenizer::BpeTokenizer>>>,
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
            tokenizer: None,
        }
    }
}

impl TokenFilter {
    pub fn with_tokenizer(t: Arc<RwLock<nexora_foundation::tokenizer::BpeTokenizer>>) -> Self {
        Self {
            tokenizer: Some(t),
            ..Default::default()
        }
    }

    fn count_tokens(&self, text: &str) -> usize {
        if let Some(tk) = &self.tokenizer {
            tk.read().encode(text).len()
        } else {
            let words = text.split_whitespace().count() as f64;
            let chars = text.len() as f64;
            ((words * 1.3 + chars * 0.04) as usize).max(1)
        }
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

        let token_count = self.count_tokens(&sample.text);
        let mut reason = None;
        let passed = if token_count < self.min_tokens {
            reason = Some(format!(
                "too_few_tokens: {} < {}",
                token_count, self.min_tokens
            ));
            false
        } else if token_count > self.max_tokens {
            reason = Some(format!(
                "too_many_tokens: {} > {}",
                token_count, self.max_tokens
            ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceCategory, SourceInfo};
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
    fn test_defaults() {
        let f = TokenFilter::default();
        assert_eq!(f.min_tokens, 10);
        assert_eq!(f.max_tokens, 2048);
        assert!(f.tokenizer.is_none());
    }

    #[test]
    fn test_estimates_tokens_without_tokenizer() {
        let f = TokenFilter::default();
        let count = f.count_tokens("hello world this is a test");
        assert!(count >= 1);
    }

    #[test]
    fn test_blocked_tokens_detected() {
        let f = TokenFilter::default();
        assert!(f
            .has_blocked_tokens("some text with <|endoftext|>")
            .is_some());
    }

    #[test]
    fn test_no_blocked_tokens() {
        let f = TokenFilter::default();
        assert!(f
            .has_blocked_tokens("clean text without special tokens")
            .is_none());
    }

    #[tokio::test]
    async fn test_blocked_token_rejects() {
        let f = TokenFilter::default();
        let s = sample("contains <|endoftext|> token");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
        assert!(result.reason.unwrap().contains("blocked_token"));
    }

    #[tokio::test]
    async fn test_sufficient_tokens_passes() {
        let f = TokenFilter::default();
        let text = (0..50)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let s = sample(&text);
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_too_few_tokens_fails() {
        let f = TokenFilter::default();
        let s = sample("hello");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
        assert!(result.reason.unwrap().contains("too_few_tokens"));
    }
}
