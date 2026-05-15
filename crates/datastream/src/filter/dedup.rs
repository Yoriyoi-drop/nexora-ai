use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use parking_lot::Mutex;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction, PipelineMetrics};

#[derive(Debug, Clone)]
pub struct DedupFilter {
    pub seen_hashes: Arc<Mutex<HashSet<u64>>>,
    pub ngram_size: usize,
    pub hash_count: usize,
    pub max_seen: usize,
}

impl DedupFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            seen_hashes: Arc::new(Mutex::new(HashSet::with_capacity(capacity))),
            ..Default::default()
        }
    }

    fn fingerprint(&self, text: &str) -> Vec<u64> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < self.ngram_size {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            return vec![hasher.finish()];
        }

        let step = (words.len() / self.hash_count).max(1);
        let mut hashes = Vec::with_capacity(self.hash_count);

        for i in (0..words.len().saturating_sub(self.ngram_size - 1)).step_by(step) {
            if hashes.len() >= self.hash_count {
                break;
            }
            let ngram: Vec<&str> = words[i..i + self.ngram_size].to_vec();
            let mut hasher = DefaultHasher::new();
            ngram.hash(&mut hasher);
            hashes.push(hasher.finish());
        }

        if hashes.is_empty() {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            hashes.push(hasher.finish());
        }

        hashes
    }

    pub fn reset(&mut self) {
        self.seen_hashes.lock().clear();
    }

    pub fn update_metrics(&self, metrics: &mut PipelineMetrics) {
        metrics.filter_breakdown.entry(self.name().to_string())
            .or_insert_with(|| crate::types::FilterMetric {
                processed: 0, passed: 0, rejected: 0, avg_latency_us: 0.0,
            });
    }
}

impl Default for DedupFilter {
    fn default() -> Self {
        Self {
            seen_hashes: Arc::new(Mutex::new(HashSet::new())),
            ngram_size: 5,
            hash_count: 4,
            max_seen: 10_000_000,
        }
    }
}

#[async_trait]
impl Filter for DedupFilter {
    fn name(&self) -> &str {
        "dedup"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let mut hashes = self.seen_hashes.lock();
        if hashes.len() >= self.max_seen {
            return FilterResult {
                passed: true,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: Some("dedup_cache_full".to_string()),
                score_delta: 0.0,
            };
        }

        let fingerprints = self.fingerprint(&sample.text);
        let is_dup = fingerprints.iter().any(|h| hashes.contains(h));

        if is_dup {
            FilterResult {
                passed: false,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: Some("duplicate_content".to_string()),
                score_delta: -0.5,
            }
        } else {
            for h in &fingerprints {
                hashes.insert(*h);
            }
            FilterResult {
                passed: true,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: None,
                score_delta: 0.0,
            }
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
    }
}
