use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

type CustomFilterFn = Arc<dyn Fn(&DataSample) -> FilterResult + Send + Sync>;

use std::sync::Arc;

#[derive(Clone)]
pub struct CustomFilter {
    pub name: String,
    pub filter_fn: Option<CustomFilterFn>,
    pub action_on_fail: FilterAction,
}

impl std::fmt::Debug for CustomFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomFilter")
            .field("name", &self.name)
            .field("filter_fn", &self.filter_fn.as_ref().map(|_| ".."))
            .field("action_on_fail", &self.action_on_fail)
            .finish()
    }
}

impl CustomFilter {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            filter_fn: None,
            action_on_fail: FilterAction::Reject,
        }
    }

    pub fn with_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&DataSample) -> FilterResult + Send + Sync + 'static,
    {
        self.filter_fn = Some(Arc::new(f));
        self
    }

    pub fn with_action(mut self, action: FilterAction) -> Self {
        self.action_on_fail = action;
        self
    }
}

#[async_trait]
impl Filter for CustomFilter {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        match &self.filter_fn {
            Some(f) => f(sample),
            None => FilterResult {
                passed: true,
                sample_id: sample.id,
                filter_name: self.name.clone(),
                reason: None,
                score_delta: 0.0,
            },
        }
    }

    fn action(&self) -> FilterAction {
        self.action_on_fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceInfo, SourceCategory};
    use uuid::Uuid;

    fn sample() -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: "test".into(),
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
    fn test_new_with_name() {
        let f = CustomFilter::new("my_filter");
        assert_eq!(f.name, "my_filter");
        assert!(f.filter_fn.is_none());
        assert_eq!(f.action_on_fail, FilterAction::Reject);
    }

    #[test]
    fn test_with_fn_overrides_behavior() {
        let f = CustomFilter::new("always_pass")
            .with_fn(|_| FilterResult {
                passed: true,
                sample_id: Uuid::nil(),
                filter_name: "always_pass".into(),
                reason: None,
                score_delta: 0.0,
            });
        assert!(f.filter_fn.is_some());
    }

    #[test]
    fn test_with_action_changes_action() {
        let f = CustomFilter::new("flag_it")
            .with_action(FilterAction::Flag);
        assert_eq!(f.action_on_fail, FilterAction::Flag);
    }

    #[tokio::test]
    async fn test_no_fn_returns_passed() {
        let f = CustomFilter::new("noop");
        let result = f.evaluate(&sample()).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_custom_fn_executed() {
        let f = CustomFilter::new("check_len")
            .with_fn(|s| FilterResult {
                passed: s.text.len() > 3,
                sample_id: s.id,
                filter_name: "check_len".into(),
                reason: None,
                score_delta: 0.0,
            });
        let result = f.evaluate(&sample()).await;
        assert!(result.passed);
    }

    #[test]
    fn test_debug_format() {
        let f = CustomFilter::new("test")
            .with_action(FilterAction::Flag);
        let debug = format!("{:?}", f);
        assert!(debug.contains("test"));
        assert!(debug.contains("Flag"));
    }

    #[test]
    fn test_clone_works() {
        let f = CustomFilter::new("a").with_action(FilterAction::Accept);
        let g = f.clone();
        assert_eq!(g.name, "a");
    }
}
