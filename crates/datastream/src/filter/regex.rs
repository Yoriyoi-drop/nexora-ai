use async_trait::async_trait;
use regex::Regex;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

#[derive(Debug, Clone)]
pub struct RegexFilter {
    pub block_patterns: Vec<Regex>,
    pub require_patterns: Vec<Regex>,
}

impl RegexFilter {
    pub fn new(block_patterns: Vec<String>, require_patterns: Vec<String>) -> Result<Self, regex::Error> {
        let block = block_patterns.iter().map(|p| Regex::new(p)).collect::<Result<Vec<_>, _>>()?;
        let require = require_patterns.iter().map(|p| Regex::new(p)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { block_patterns: block, require_patterns: require })
    }
}

impl Default for RegexFilter {
    fn default() -> Self {
        Self {
            block_patterns: vec![
                Regex::new(r"(?i)\b(?:buy\s+now|click\s+here|subscribe\s+now|limited\s+time)\b").unwrap(),
                Regex::new(r"(?i)https?://(bit\.ly|tinyurl|shorturl)\.[a-z]+/\S+").unwrap(),
                Regex::new(r"(?m)^>{10,}").unwrap(),
                Regex::new(r"(?i)\b([a-z0-9\-._~%]+)\@[a-z0-9\-._~%]+\.[a-z]{2,}\b").unwrap(),
            ],
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
