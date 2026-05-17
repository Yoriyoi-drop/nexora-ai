use async_trait::async_trait;
use regex::Regex;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

#[derive(Debug, Clone)]
pub struct ToxicityFilter {
    pub threshold: f64,
    pub blocklist: Vec<Regex>,
}

impl Default for ToxicityFilter {
    fn default() -> Self {
        Self {
            threshold: 0.80,
            blocklist: vec![
                Regex::new(r"(?i)\b(nigg[ae]r|fag+ot|retard|cunt|whore|slut|bitch)\b").expect("valid regex pattern"),
                Regex::new(r"(?i)\b(kill\s+(yourself|everyone|them)|rape|torture)\b").expect("valid regex pattern"),
                Regex::new(r"(?i)\b(gore|cp\b|child\s*porn|bestiality)\b").expect("valid regex pattern"),
                Regex::new(r"(?i)\b(hitler|nazi|white\s*supremac|kkk)\b").expect("valid regex pattern"),
            ],
        }
    }
}

impl ToxicityFilter {
    pub fn new(threshold: f64) -> Self {
        Self { threshold, ..Default::default() }
    }

    fn score_toxicity(&self, text: &str) -> (f64, Option<String>) {
        let _text_lower = text.to_lowercase();
        let mut score = 0.0;

        for pattern in &self.blocklist {
            if pattern.is_match(text) {
                let count = pattern.find_iter(text).count() as f64;
                score += count as f64 * 0.25;
                let severity = count.min(10.0) / 10.0;
                score += severity * 0.3;
            }
        }

        if score >= self.threshold {
            let reason = format!("toxicity_score: {:.2}", score);
            (score, Some(reason))
        } else {
            (score, None)
        }
    }
}

#[async_trait]
impl Filter for ToxicityFilter {
    fn name(&self) -> &str {
        "toxicity"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let (score, reason) = self.score_toxicity(&sample.text);
        let passed = score < self.threshold;
        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: if passed { 0.0 } else { -0.8 },
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
    }
}
