use std::sync::Mutex;
use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

pub struct SemanticDedupFilter {
    pub similarity_threshold: f64,
    pub signatures: Mutex<Vec<Vec<u64>>>,
    pub max_signatures: usize,
    pub min_hash_permutations: usize,
}

impl std::fmt::Debug for SemanticDedupFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.signatures.lock().unwrap().len();
        f.debug_struct("SemanticDedupFilter")
            .field("similarity_threshold", &self.similarity_threshold)
            .field("signatures_count", &count)
            .field("max_signatures", &self.max_signatures)
            .field("min_hash_permutations", &self.min_hash_permutations)
            .finish()
    }
}

impl Clone for SemanticDedupFilter {
    fn clone(&self) -> Self {
        Self {
            similarity_threshold: self.similarity_threshold,
            signatures: Mutex::new(self.signatures.lock().unwrap().clone()),
            max_signatures: self.max_signatures,
            min_hash_permutations: self.min_hash_permutations,
        }
    }
}

impl Default for SemanticDedupFilter {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            signatures: Mutex::new(Vec::new()),
            max_signatures: 100_000,
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

        let shingles: Vec<String> = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(3)
            .map(|w| w.join(" "))
            .collect();

        if shingles.is_empty() {
            return vec![0; self.min_hash_permutations.min(4)];
        }

        let num_perms = self.min_hash_permutations.min(16);
        let mut sig = vec![u64::MAX; num_perms];
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

    fn jaccard_similarity(a: &[u64], b: &[u64]) -> f64 {
        let shared = a.iter().filter(|&x| b.contains(x)).count();
        let total = a.len().max(b.len());
        if total == 0 { return 0.0; }
        shared as f64 / total as f64
    }
}

#[async_trait]
impl Filter for SemanticDedupFilter {
    fn name(&self) -> &str {
        "semantic_dedup"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let sig = self.minhash_signature(&sample.text);
        let mut signatures = self.signatures.lock().unwrap();

        for stored in signatures.iter() {
            let similarity = Self::jaccard_similarity(&sig, stored);
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

        if signatures.len() < self.max_signatures {
            signatures.push(sig);
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
