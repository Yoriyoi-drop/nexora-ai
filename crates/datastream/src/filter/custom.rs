use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

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
