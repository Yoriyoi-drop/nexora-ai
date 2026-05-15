use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

#[derive(Debug, Clone)]
pub struct SemanticDedupFilter {
    pub similarity_threshold: f64,
    pub embeddings: Vec<(u64, Vec<f32>)>,
    pub max_embeddings: usize,
    pub min_hash_permutations: usize,
}

impl Default for SemanticDedupFilter {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            embeddings: Vec::new(),
            max_embeddings: 100_000,
            min_hash_permutations: 128,
        }
    }
}

impl SemanticDedupFilter {
    pub fn new(threshold: f64) -> Self {
        Self { similarity_threshold: threshold, ..Default::default() }
    }

    fn minhash_signature(&self, text: &str) -> Vec<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let shingles: Vec<&str> = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(3)
            .map(|w| w.join(" "))
            .map(|s| Box::leak(s.into_boxed_str()) as &str)
            .collect();

        if shingles.is_empty() {
            return vec![0; self.min_hash_permutations.min(4)];
        }

        let mut sig = vec![u64::MAX; self.min_hash_permutations.min(16)];
        for shingle in &shingles {
            let mut hasher = DefaultHasher::new();
            shingle.hash(&mut hasher);
            let hash = hasher.finish();
            for (i, s) in sig.iter_mut().enumerate() {
                let perm_hash = hash.wrapping_mul(i as u64 + 1);
                if perm_hash < *s {
                    *s = perm_hash;
                }
            }
        }
        sig
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)) as f64
    }

    fn simple_embedding(&self, text: &str) -> Vec<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut counts = vec![0.0f32; 256];
        for w in words {
            let idx = (w.len().min(255)) as usize;
            counts[idx] += 1.0;
        }
        let sum: f32 = counts.iter().sum();
        if sum > 0.0 {
            for c in &mut counts {
                *c /= sum;
            }
        }
        counts
    }
}

#[async_trait]
impl Filter for SemanticDedupFilter {
    fn name(&self) -> &str {
        "semantic_dedup"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        if self.embeddings.len() >= self.max_embeddings {
            return FilterResult {
                passed: true,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: Some("embedding_cache_full".to_string()),
                score_delta: 0.0,
            };
        }

        let sig = self.minhash_signature(&sample.text);

        for (_sig_id, _emb) in &self.embeddings {
            let shared = sig.iter().filter(|&s| s == _sig_id).count();
            let similarity = shared as f64 / sig.len().max(1) as f64;
            if similarity >= self.similarity_threshold {
                return FilterResult {
                    passed: false,
                    sample_id: sample.id,
                    filter_name: self.name().to_string(),
                    reason: Some(format!("semantic_duplicate: similarity={:.4}", similarity)),
                    score_delta: -0.4,
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
