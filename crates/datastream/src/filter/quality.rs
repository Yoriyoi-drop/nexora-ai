use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct QualityFilter {
    pub min_quality_score: f64,
    pub max_repetition_ratio: f64,
    pub max_punctuation_ratio: f64,
    pub max_uppercase_ratio: f64,
    pub min_unique_word_ratio: f64,
}

impl Default for QualityFilter {
    fn default() -> Self {
        Self {
            min_quality_score: 0.3,
            max_repetition_ratio: 0.4,
            max_punctuation_ratio: 0.3,
            max_uppercase_ratio: 0.5,
            min_unique_word_ratio: 0.2,
        }
    }
}

impl QualityFilter {
    fn compute_quality(&self, sample: &DataSample) -> (f64, Option<String>) {
        let text = &sample.text;
        let char_count = text.len().max(1);
        let word_count = text.split_whitespace().count().max(1);

        let uppercase = text.chars().filter(|c| c.is_uppercase()).count() as f64;
        let uppercase_ratio = uppercase / char_count as f64;
        if uppercase_ratio > self.max_uppercase_ratio {
            return (
                0.0,
                Some(format!("too_much_uppercase: {:.2}", uppercase_ratio)),
            );
        }

        let punctuation = text.chars().filter(|c| c.is_ascii_punctuation()).count() as f64;
        let punct_ratio = punctuation / char_count as f64;
        if punct_ratio > self.max_punctuation_ratio {
            return (
                0.0,
                Some(format!("too_much_punctuation: {:.2}", punct_ratio)),
            );
        }

        let mut word_freq: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for w in text.split_whitespace() {
            *word_freq.entry(w).or_insert(0) += 1;
        }
        let unique_ratio = word_freq.len() as f64 / word_count as f64;
        if unique_ratio < self.min_unique_word_ratio {
            return (
                0.0,
                Some(format!(
                    "too_much_repetition: unique_ratio={:.2}",
                    unique_ratio
                )),
            );
        }

        let max_freq = *word_freq.values().max().unwrap_or(&0) as f64;
        let repetition_ratio = max_freq / word_count as f64;
        if repetition_ratio > self.max_repetition_ratio {
            return (
                0.0,
                Some(format!("high_repetition: {:.2}", repetition_ratio)),
            );
        }

        let avg_word_len: f64 =
            text.split_whitespace().map(|w| w.len() as f64).sum::<f64>() / word_count as f64;
        let has_mixed_case =
            text.chars().any(|c| c.is_lowercase()) && text.chars().any(|c| c.is_uppercase());
        let has_variety = word_freq.len() > 5;

        let score = if has_variety && has_mixed_case && avg_word_len > 3.0 && avg_word_len < 12.0 {
            0.8
        } else if has_variety {
            0.5
        } else {
            0.2
        };

        (score, None)
    }
}

#[async_trait]
impl Filter for QualityFilter {
    fn name(&self) -> &str {
        "quality"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let (score, reason) = self.compute_quality(sample);
        let passed = score >= self.min_quality_score;
        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: score - 0.5,
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
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
    fn test_default_thresholds() {
        let f = QualityFilter::default();
        assert_eq!(f.min_quality_score, 0.3);
        assert_eq!(f.max_repetition_ratio, 0.4);
        assert_eq!(f.min_unique_word_ratio, 0.2);
    }

    #[test]
    fn test_high_quality_text() {
        let f = QualityFilter::default();
        let text = "The quick brown fox jumps over the lazy dog near the riverbank. This sentence has varied vocabulary and mixed case. Quality content should pass the filter with good scores.";
        let s = sample(text);
        let (score, reason) = f.compute_quality(&s);
        assert!(score >= f.min_quality_score);
        assert!(reason.is_none());
    }

    #[test]
    fn test_too_much_uppercase() {
        let f = QualityFilter::default();
        let text = "THIS IS ALL UPPERCASE AND SHOULD BE REJECTED BECAUSE IT HAS NO LOWERCASE CONTENT WHATSOEVER";
        let s = sample(text);
        let (score, reason) = f.compute_quality(&s);
        assert_eq!(score, 0.0);
        assert!(reason.unwrap().contains("uppercase"));
    }

    #[test]
    fn test_too_repetitive() {
        let f = QualityFilter::default();
        let text = "word word word word word word word word word word word word";
        let s = sample(text);
        let (score, reason) = f.compute_quality(&s);
        assert_eq!(score, 0.0);
        assert!(reason.unwrap().contains("repetition"));
    }

    #[tokio::test]
    async fn test_filter_passes_good_quality() {
        let f = QualityFilter::default();
        let text = "The quick brown fox jumps over the lazy dog. This sentence has varied vocabulary and good structure for a quality assessment test.";
        let s = sample(text);
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_filter_rejects_low_quality() {
        let f = QualityFilter::default();
        let s = sample("AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
    }
}
