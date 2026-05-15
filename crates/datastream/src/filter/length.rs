use async_trait::async_trait;
use unicode_segmentation::UnicodeSegmentation;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

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
            reason = Some(format!("too_short: {} chars < {}", char_count, self.min_chars));
            false
        } else if char_count > self.max_chars {
            reason = Some(format!("too_long: {} chars > {}", char_count, self.max_chars));
            false
        } else if word_count < self.min_words {
            reason = Some(format!("too_few_words: {} < {}", word_count, self.min_words));
            false
        } else if word_count > self.max_words {
            reason = Some(format!("too_many_words: {} > {}", word_count, self.max_words));
            false
        } else if line_count < self.min_lines {
            reason = Some(format!("too_few_lines: {} < {}", line_count, self.min_lines));
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
