use async_trait::async_trait;
use unicode_segmentation::UnicodeSegmentation;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct LengthFilter {
    pub min_chars: usize,
    pub max_chars: usize,
    pub min_words: usize,
    pub max_words: usize,
    pub min_lines: usize,
}

impl Default for LengthFilter {
    fn default() -> Self {
        Self {
            min_chars: 50,
            max_chars: 1_000_000,
            min_words: 10,
            max_words: 200_000,
            min_lines: 1,
        }
    }
}

#[async_trait]
impl Filter for LengthFilter {
    fn name(&self) -> &str {
        "length"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let char_count = sample.text.chars().count();
        let word_count = sample.text.unicode_words().count();
        let line_count = sample.text.lines().count();

        let mut reason = None;
        let passed = if char_count < self.min_chars {
            reason = Some(format!(
                "too_short: {} chars < {}",
                char_count, self.min_chars
            ));
            false
        } else if char_count > self.max_chars {
            reason = Some(format!(
                "too_long: {} chars > {}",
                char_count, self.max_chars
            ));
            false
        } else if word_count < self.min_words {
            reason = Some(format!(
                "too_few_words: {} < {}",
                word_count, self.min_words
            ));
            false
        } else if word_count > self.max_words {
            reason = Some(format!(
                "too_many_words: {} > {}",
                word_count, self.max_words
            ));
            false
        } else if line_count < self.min_lines {
            reason = Some(format!(
                "too_few_lines: {} < {}",
                line_count, self.min_lines
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
            score_delta: if passed { 0.0 } else { -0.3 },
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
        let f = LengthFilter::default();
        assert_eq!(f.min_chars, 50);
        assert_eq!(f.max_chars, 1_000_000);
        assert_eq!(f.min_words, 10);
    }

    #[tokio::test]
    async fn test_short_text_rejected() {
        let f = LengthFilter::default();
        let s = sample("Hi");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
        assert!(result.reason.unwrap().contains("too_short"));
    }

    #[tokio::test]
    async fn test_long_text_accepted() {
        let f = LengthFilter::default();
        let text = "this sentence has enough words to pass the minimum length filter without any problem whatsoever";
        let s = sample(text);
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_too_few_words() {
        let f = LengthFilter::default();
        let text = "a".repeat(100);
        let s = sample(&text);
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
        assert!(result.reason.unwrap().contains("too_few_words"));
    }

    #[tokio::test]
    async fn test_exact_boundary_passes() {
        let f = LengthFilter::default();
        let words = (0..10)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let text = words + &" ".repeat(50);
        let s = sample(&text);
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }
}
