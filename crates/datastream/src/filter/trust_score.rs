use async_trait::async_trait;
use url::Url;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult, SourceCategory, TrustScoreMap};

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
        Self {
            min_trust,
            ..Default::default()
        }
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
                if let Some(score) = self
                    .sources
                    .0
                    .iter()
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
            SourceCategory::Documentation => 0.85,
            SourceCategory::News => 0.70,
            SourceCategory::StackOverflow => 0.82,
            SourceCategory::Reddit => 0.45,
            SourceCategory::YouTube => 0.50,
            SourceCategory::Patents => 0.92,
            SourceCategory::Government => 0.88,
            SourceCategory::Medical => 0.95,
            SourceCategory::Legal => 0.85,
            SourceCategory::Education => 0.88,
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
            Some(format!(
                "low_trust: {:.2} < min={:.2} (source={}, category={:?})",
                trust_score, self.min_trust, sample.source.name, sample.source.category
            ))
        } else {
            None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceInfo, SourceCategory};
    use uuid::Uuid;

    fn sample(name: &str, category: SourceCategory) -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: "some content".into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: name.into(),
                url: Some(format!("https://{}", name)),
                trust_score: 0.5,
                category,
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
        let f = TrustScoreFilter::default();
        assert_eq!(f.default_trust, 0.5);
        assert_eq!(f.min_trust, 0.2);
    }

    #[test]
    fn test_wikipedia_high_trust() {
        let f = TrustScoreFilter::default();
        let score = f.source_trust("wikipedia.org");
        assert_eq!(score, 0.95);
    }

    #[test]
    fn test_unknown_source_uses_default() {
        let f = TrustScoreFilter::default();
        let score = f.source_trust("unknown-site.example.com");
        assert_eq!(score, 0.5);
    }

    #[test]
    fn test_category_trust() {
        let f = TrustScoreFilter::default();
        assert_eq!(f.category_trust(&SourceCategory::Arxiv), 0.97);
        assert_eq!(f.category_trust(&SourceCategory::SEOFarm), 0.12);
        assert_eq!(f.category_trust(&SourceCategory::SocialMedia), 0.35);
    }

    #[tokio::test]
    async fn test_high_trust_source_passes() {
        let f = TrustScoreFilter::default();
        let s = sample("wikipedia.org", SourceCategory::Wikipedia);
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_low_trust_source_rejected() {
        let f = TrustScoreFilter::new(0.8);
        let s = sample("some-spam-blog.com", SourceCategory::WebCrawl);
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
    }

    #[test]
    fn test_url_based_trust() {
        let f = TrustScoreFilter::default();
        let score = f.source_trust("https://en.wikipedia.org/wiki/Main_Page");
        assert_eq!(score, 0.95);
    }

    #[tokio::test]
    async fn test_score_delta_reflects_trust() {
        let f = TrustScoreFilter::default();
        let s = sample("arxiv.org", SourceCategory::Arxiv);
        let result = f.evaluate(&s).await;
        assert!(result.score_delta > 0.0);
    }
}
