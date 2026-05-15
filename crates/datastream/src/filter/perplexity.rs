use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

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
        Self { min_perplexity: min, max_perplexity: max }
    }

    fn estimate_perplexity(&self, text: &str) -> f64 {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 5 {
            return 100.0;
        }

        let mut char_freq: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for &c in &chars {
            *char_freq.entry(c).or_insert(0) += 1;
        }

        let total = chars.len() as f64;
        let mut log_prob_sum = 0.0;
        let mut ngram_count = 0;

        for i in 0..chars.len().saturating_sub(2) {
            let trigram: String = chars[i..=i + 2].iter().collect();
            let count = chars.windows(3).filter(|w| w.iter().collect::<String>() == trigram).count() as f64;
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
