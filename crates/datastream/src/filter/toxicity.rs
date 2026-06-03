use async_trait::async_trait;
use regex::Regex;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct ToxicityFilter {
    pub threshold: f64,
    pub blocklist: Vec<Regex>,
}

fn default_blocklist() -> Vec<Regex> {
    vec![Regex::new(
        r"(?i)\b(nigg[ae]r|fag+ot|retard|cunt|whore|slut|bitch|kill\s+(?:yourself|everyone|them)|rape|torture|gore|cp\b|child\s*porn|bestiality|hitler|nazi|white\s*supremac|kkk)\b"
    ).expect("valid combined toxicity regex")]
}

impl Default for ToxicityFilter {
    fn default() -> Self {
        Self {
            threshold: 0.80,
            blocklist: default_blocklist(),
        }
    }
}

impl ToxicityFilter {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    fn score_toxicity(&self, text: &str) -> (f64, Option<String>) {
        let mut score = 0.0;

        for pattern in &self.blocklist {
            let count = pattern.find_iter(text).count() as f64;
            if count > 0.0 {
                score += count * 0.8;
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
    fn test_default_threshold() {
        let f = ToxicityFilter::default();
        assert_eq!(f.threshold, 0.80);
    }

    #[test]
    fn test_clean_text_passes() {
        let f = ToxicityFilter::default();
        let (score, reason) = f.score_toxicity("this is a perfectly clean sentence");
        assert!(score < f.threshold);
        assert!(reason.is_none());
    }

    #[test]
    fn test_slur_triggers() {
        let f = ToxicityFilter::default();
        let (score, reason) = f.score_toxicity("that person is a bitch");
        assert!(score >= f.threshold);
        assert!(reason.is_some());
    }

    #[test]
    fn test_violence_triggers() {
        let f = ToxicityFilter::default();
        let (score, _) = f.score_toxicity("go kill yourself");
        assert!(score >= f.threshold);
    }

    #[tokio::test]
    async fn test_filter_rejects_toxic() {
        let f = ToxicityFilter::default();
        let s = sample("you are a fucking cunt");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
        assert_eq!(result.filter_name, "toxicity");
    }

    #[tokio::test]
    async fn test_filter_accepts_clean() {
        let f = ToxicityFilter::default();
        let s = sample("the quick brown fox jumps over the lazy dog");
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[test]
    fn test_action_is_reject() {
        let f = ToxicityFilter::default();
        assert_eq!(f.action(), FilterAction::Reject);
    }
}
