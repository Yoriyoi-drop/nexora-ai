use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use tracing::info;

use crate::types::{DataSample, SampleStats, SourceCategory, SourceInfo};
use uuid::Uuid;

#[async_trait]
pub trait SourceProvider: Send + Sync {
    fn name(&self) -> &str;
    fn url(&self) -> &str;
    fn category(&self) -> SourceCategory;
    fn default_trust_score(&self) -> f64;
    fn description(&self) -> &str;

    fn source_info(&self) -> SourceInfo {
        SourceInfo {
            name: self.name().to_string(),
            url: Some(self.url().to_string()),
            trust_score: self.default_trust_score(),
            category: self.category(),
            fetch_timestamp: Utc::now().timestamp(),
        }
    }

    fn sample_data(&self) -> Vec<String>;

    async fn fetch_samples(&self) -> Vec<DataSample> {
        let source = self.source_info();
        self.sample_data()
            .into_iter()
            .map(|text| DataSample {
                id: Uuid::new_v4(),
                text,
                token_ids: None,
                metadata: HashMap::new(),
                source: source.clone(),
                stats: SampleStats::default(),
                domains: vec![],
                score: None,
                curriculum_level: None,
            })
            .collect()
    }
}

pub mod fetcher;

pub struct SourceRegistry {
    providers: HashMap<String, Box<dyn SourceProvider>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn register(&mut self, provider: Box<dyn SourceProvider>) {
        self.providers.insert(provider.name().to_string(), provider);
    }

    pub fn get(&self, name: &str) -> Option<&dyn SourceProvider> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    pub fn all(&self) -> Vec<&dyn SourceProvider> {
        self.providers.values().map(|p| p.as_ref()).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    pub fn resolve(&self, names: &[String]) -> Vec<&dyn SourceProvider> {
        names.iter().filter_map(|n| self.get(n)).collect()
    }

    pub async fn collect_all(
        &self,
        names: &[String],
        max_samples: Option<usize>,
    ) -> Vec<DataSample> {
        let providers = if names.is_empty() {
            self.all()
        } else {
            self.resolve(names)
        };

        let mut all = Vec::new();
        for provider in &providers {
            info!("Collecting from {}...", provider.name());
            let samples = provider.fetch_samples().await;
            info!("  {}: {} samples", provider.name(), samples.len());
            all.extend(samples);
            if let Some(max) = max_samples {
                if all.len() >= max {
                    all.truncate(max);
                    break;
                }
            }
        }

        all
    }

    pub fn build_default() -> Self {
        fetcher::build_registry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_builds() {
        let reg = SourceRegistry::build_default();
        assert_eq!(reg.all().len(), 3);
    }

    #[test]
    fn test_registry_names() {
        let reg = SourceRegistry::build_default();
        let names = reg.names();
        assert!(names.contains(&"hackernews"));
        assert!(names.contains(&"wikipedia"));
        assert!(names.contains(&"reddit"));
    }

    #[test]
    fn test_registry_resolve() {
        let reg = SourceRegistry::build_default();
        let names = vec!["hackernews".to_string(), "wikipedia".to_string()];
        let resolved = reg.resolve(&names);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_source_info_fields() {
        let reg = SourceRegistry::build_default();
        let provider = reg.get("hackernews").expect("hackernews should be registered in default registry");
        let info = provider.source_info();
        assert_eq!(info.category, SourceCategory::News);
        assert!((info.trust_score - 0.70).abs() < 0.01);
        assert!(info.url.is_some());
    }
}
