use crate::types::{DataSample, FilterAction, FilterResult};
use async_trait::async_trait;
use std::fmt::Debug;

#[async_trait]
pub trait Filter: Debug + Send + Sync {
    fn name(&self) -> &str;

    async fn filter(&self, sample: &DataSample) -> FilterResult {
        self.evaluate(sample).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceInfo, SourceCategory};
    use uuid::Uuid;

    fn dummy_sample() -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: "hello world".into(),
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

    #[tokio::test]
    async fn test_filter_trait_default_action() {
        let _ = dummy_sample();
        assert_eq!(FilterAction::Accept, FilterAction::Accept);
    }

    #[tokio::test]
    async fn test_parallel_filter_partition_key() {
        #[derive(Debug)]
        struct DummyFilter;
        #[async_trait]
        impl Filter for DummyFilter {
            fn name(&self) -> &str { "dummy" }
            async fn evaluate(&self, _: &DataSample) -> FilterResult {
                FilterResult {
                    passed: true,
                    sample_id: Uuid::nil(),
                    filter_name: "dummy".into(),
                    reason: None,
                    score_delta: 0.0,
                }
            }
        }
        #[async_trait]
        impl ParallelFilter for DummyFilter {}

        let f = DummyFilter;
        let s = dummy_sample();
        let key = f.partition_key(&s);
        // Same sample should produce same key
        assert_eq!(f.partition_key(&s), key);
    }
}
