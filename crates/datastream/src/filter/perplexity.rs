use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct PerplexityFilter {
    pub min_perplexity: f64,
    pub max_perplexity: f64,
}

impl Default for PerplexityFilter {
    fn default() -> Self {
        Self {
            min_perplexity: 1.5,
            max_perplexity: 5000.0,
        }
    }
}

impl PerplexityFilter {
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            min_perplexity: min,
            max_perplexity: max,
        }
    }

    fn estimate_perplexity(&self, text: &str) -> f64 {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 5 {
            return 100.0;
        }

        let mut char_freq: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();
        for &c in &chars {
            *char_freq.entry(c).or_insert(0) += 1;
        }

        let total = chars.len() as f64;
        let mut log_prob_sum = 0.0;
        let mut ngram_count = 0;

        let mut trigram_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            *trigram_counts.entry(trigram).or_insert(0) += 1;
        }

        for i in 0..chars.len().saturating_sub(2) {
            let trigram: String = chars[i..=i + 2].iter().collect();
            let count = *trigram_counts.get(&trigram).unwrap_or(&0) as f64;
            let prob = count / (total - 2.0).max(1.0);
            if prob > 0.0 {
                log_prob_sum -= prob.ln();
                ngram_count += 1;
            }
        }

        if ngram_count == 0 {
            return 1000.0;
        }

        let avg_neg_log_prob = log_prob_sum / ngram_count as f64;
        avg_neg_log_prob.exp().min(10000.0).max(1.0)
    }
}

#[async_trait]
impl Filter for PerplexityFilter {
    fn name(&self) -> &str {
        "perplexity"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let ppl = self.estimate_perplexity(&sample.text);
        let passed = ppl >= self.min_perplexity && ppl <= self.max_perplexity;
        let reason = if !passed {
            Some(format!("perplexity_out_of_range: {:.2}", ppl))
        } else {
            None
        };
        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: if passed {
                0.1 - ((ppl - 50.0) / 500.0).clamp(-0.1, 0.1)
            } else {
                -0.3
            },
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
    fn test_defaults() {
        let f = PerplexityFilter::default();
        assert_eq!(f.min_perplexity, 1.5);
        assert_eq!(f.max_perplexity, 5000.0);
    }

    #[test]
    fn test_short_text_fallback() {
        let f = PerplexityFilter::default();
        let ppl = f.estimate_perplexity("hi");
        assert_eq!(ppl, 100.0);
    }

    #[test]
    fn test_longer_text_has_perplexity() {
        let f = PerplexityFilter::default();
        let text = "the quick brown fox jumps over the lazy dog near the river bank and then runs away into the forest where many animals live together in harmony";
        let ppl = f.estimate_perplexity(text);
        assert!(ppl >= 1.0);
        assert!(ppl <= 10000.0);
    }

    #[tokio::test]
    async fn test_normal_text_passes() {
        let f = PerplexityFilter::default();
        let text = "the cat sat on the mat and then the dog came along to play with them both outside in the garden where the flowers are blooming beautifully today";
        let s = sample(text);
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_extreme_perplexity_fails() {
        let f = PerplexityFilter::new(1000.0, 2000.0);
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo pppp";
        let s = sample(text);
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
    }

    #[test]
    fn test_new_constructor() {
        let f = PerplexityFilter::new(2.0, 100.0);
        assert_eq!(f.min_perplexity, 2.0);
        assert_eq!(f.max_perplexity, 100.0);
    }
}
