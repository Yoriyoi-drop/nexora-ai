use async_trait::async_trait;
use regex::Regex;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct RegexFilter {
    pub block_patterns: Vec<Regex>,
    pub require_patterns: Vec<Regex>,
}

impl RegexFilter {
    pub fn new(
        block_patterns: Vec<String>,
        require_patterns: Vec<String>,
    ) -> Result<Self, regex::Error> {
        let block = block_patterns
            .iter()
            .map(|p| Regex::new(p))
            .collect::<Result<Vec<_>, _>>()?;
        let require = require_patterns
            .iter()
            .map(|p| Regex::new(p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            block_patterns: block,
            require_patterns: require,
        })
    }
}

fn default_block_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"(?i)\b(?:buy\s+now|click\s+here|subscribe\s+now|limited\s+time)\b")
            .expect("valid regex: spam keywords"), // safe: hardcoded regex pattern
        Regex::new(r"(?i)https?://(bit\.ly|tinyurl|shorturl)\.[a-z]+/\S+")
            .expect("valid regex: URL shorteners"), // safe: hardcoded regex pattern
        Regex::new(r"(?m)^>{10,}").expect("valid regex: blockquote abuse"), // safe: hardcoded regex pattern
        Regex::new(r"(?i)\b([a-z0-9\-._~%]+)\@[a-z0-9\-._~%]+\.[a-z]{2,}\b")
            .expect("valid regex: email pattern"), // safe: hardcoded regex pattern
    ]
}

impl Default for RegexFilter {
    fn default() -> Self {
        Self {
            block_patterns: default_block_patterns(),
            require_patterns: vec![],
        }
    }
}

#[async_trait]
impl Filter for RegexFilter {
    fn name(&self) -> &str {
        "regex"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        for pattern in &self.block_patterns {
            if pattern.is_match(&sample.text) {
                return FilterResult {
                    passed: false,
                    sample_id: sample.id,
                    filter_name: self.name().to_string(),
                    reason: Some(format!("blocked_pattern: {}", pattern.as_str())),
                    score_delta: -0.4,
                };
            }
        }
        for pattern in &self.require_patterns {
            if !pattern.is_match(&sample.text) {
                return FilterResult {
                    passed: false,
                    sample_id: sample.id,
                    filter_name: self.name().to_string(),
                    reason: Some(format!("missing_required_pattern: {}", pattern.as_str())),
                    score_delta: -0.1,
                };
            }
        }
        FilterResult {
            passed: true,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason: None,
            score_delta: 0.0,
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
    fn test_default_has_block_patterns() {
        let f = RegexFilter::default();
        assert!(!f.block_patterns.is_empty());
    }

    #[test]
    fn test_new_constructor_valid_regex() {
        let f = RegexFilter::new(vec!["spam".into()], vec!["important".into()]).unwrap();
        assert_eq!(f.block_patterns.len(), 1);
        assert_eq!(f.require_patterns.len(), 1);
    }

    #[test]
    fn test_invalid_regex_returns_error() {
        let result = RegexFilter::new(vec!["[invalid".into()], vec![]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_block_pattern_rejects() {
        let f = RegexFilter::default();
        let s = sample("buy now and click here for limited time offer");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
        assert!(result.reason.unwrap().contains("blocked_pattern"));
    }

    #[tokio::test]
    async fn test_clean_text_passes() {
        let f = RegexFilter::default();
        let s = sample("This is a normal piece of text with no spam patterns whatsoever");
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_require_pattern_missing_fails() {
        let f = RegexFilter::new(vec![], vec!["required_word".into()]).unwrap();
        let s = sample("this text does not contain the magic word");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
        assert!(result.reason.unwrap().contains("missing_required_pattern"));
    }

    #[tokio::test]
    async fn test_require_pattern_present_passes() {
        let f = RegexFilter::new(vec![], vec!["hello".into()]).unwrap();
        let s = sample("hello world this text contains the required pattern");
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }
}
