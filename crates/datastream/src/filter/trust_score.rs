use async_trait::async_trait;
use url::Url;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction, TrustScoreMap, SourceCategory};

#[derive(Debug, Clone)]
pub struct TrustScoreFilter {
    pub default_trust: f64,
    pub min_trust: f64,
    pub sources: TrustScoreMap,
}

impl Default for TrustScoreFilter {
    fn default() -> Self {
        Self {
            default_trust: 0.5,
            min_trust: 0.2,
            sources: TrustScoreMap::default(),
        }
    }
}

impl TrustScoreFilter {
    pub fn new(min_trust: f64) -> Self {
        Self { min_trust, ..Default::default() }
    }

    fn source_trust(&self, source: &str) -> f64 {
        let source_lower = source.to_lowercase();
        if let Some(score) = self.sources.0.get(&source_lower) {
            return *score;
        }
        if let Ok(url) = Url::parse(source) {
            if let Some(host) = url.host_str() {
                if let Some(score) = self.sources.0.get(host) {
                    return *score;
                }
                if let Some(score) = self.sources.0.iter()
                    .find(|(k, _)| host.ends_with(k.as_str()))
                    .map(|(_, v)| v)
                {
                    return *score;
                }
            }
        }
        match source {
            s if s.contains("wikipedia") => 0.95,
            s if s.contains("arxiv") => 0.97,
            s if s.contains("github") => 0.85,
            s if s.contains(".edu") => 0.90,
            s if s.contains(".gov") => 0.85,
            s if s.contains("stackoverflow") => 0.80,
            s if s.contains("reddit") => 0.55,
            s if s.contains("blog") => 0.45,
            s if s.contains("forum") => 0.41,
            _ => self.default_trust,
        }
    }

    fn category_trust(&self, category: &SourceCategory) -> f64 {
        match category {
            SourceCategory::Wikipedia => 0.95,
            SourceCategory::Arxiv => 0.97,
            SourceCategory::GitHub => 0.85,
            SourceCategory::Academic => 0.95,
            SourceCategory::Books => 0.90,
            SourceCategory::WebCrawl => 0.50,
            SourceCategory::CommonCrawl => 0.45,
            SourceCategory::Telemetry => 0.70,
            SourceCategory::Synthetic => 0.60,
            SourceCategory::Forum => 0.41,
            SourceCategory::SocialMedia => 0.35,
            SourceCategory::SEOFarm => 0.12,
            SourceCategory::Other => self.default_trust,
        }
    }
}

#[async_trait]
impl Filter for TrustScoreFilter {
    fn name(&self) -> &str {
        "trust_score"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let source_trust = self.source_trust(&sample.source.name);
        let category_trust = self.category_trust(&sample.source.category);
        let trust_score = source_trust * 0.6 + category_trust * 0.4;

        let passed = trust_score >= self.min_trust;
        let reason = if !passed {
            Some(format!("low_trust: {:.2} < min={:.2} (source={}, category={:?})",
                trust_score, self.min_trust, sample.source.name, sample.source.category))
        } else {
            Some(format!("trust_score={:.2}", trust_score))
        };

        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: trust_score - 0.5,
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
    }
}
