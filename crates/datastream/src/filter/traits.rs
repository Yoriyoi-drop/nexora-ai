use async_trait::async_trait;
use crate::types::{DataSample, FilterResult, FilterAction};
use std::fmt::Debug;

#[async_trait]
pub trait Filter: Debug + Send + Sync {
    fn name(&self) -> &str;

    async fn filter(&self, sample: &DataSample) -> FilterResult {
        let start = std::time::Instant::now();
        let result = self.evaluate(sample).await;
        let _elapsed = start.elapsed();
        result
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult;

    fn action(&self) -> FilterAction {
        FilterAction::Accept
    }
}

#[async_trait]
pub trait ParallelFilter: Filter {
    fn partition_key(&self, sample: &DataSample) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        sample.id.hash(&mut hasher);
        hasher.finish()
    }
}
