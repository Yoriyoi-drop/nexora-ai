use async_trait::async_trait;
use tokio::sync::Mutex;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

pub struct SemanticDedupFilter {
    pub similarity_threshold: f64,
    pub signatures: Mutex<Vec<Vec<u64>>>,
    pub max_signatures: usize,
    pub min_hash_permutations: usize,
}

impl std::fmt::Debug for SemanticDedupFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.signatures.try_lock().map(|g| g.len()).unwrap_or(0);
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
        let signatures = self
            .signatures
            .try_lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        Self {
            similarity_threshold: self.similarity_threshold,
            signatures: Mutex::new(signatures),
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
        Self {
            similarity_threshold: threshold,
            ..Default::default()
        }
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
                let mut h = hash;
                h = h.wrapping_mul(0x9e3779b97f4a7c15);
                h ^= h >> 33;
                h = h.wrapping_mul(0xff51afd7ed558ccd).wrapping_add(i as u64);
                h ^= h >> 33;
                h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
                h ^= h >> 33;
                let perm_hash = h;
                if perm_hash < *s {
                    *s = perm_hash;
                }
            }
        }
        sig
    }

    fn jaccard_similarity(a: &[u64], b: &[u64]) -> f64 {
        use std::collections::HashSet;
        let set_b: HashSet<&u64> = b.iter().collect();
        let shared = a.iter().filter(|&x| set_b.contains(x)).count();
        let total = a.len() + b.len() - shared;
        if total == 0 {
            return 0.0;
        }
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
        let mut signatures = self.signatures.lock().await;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceCategory, SourceInfo};
    use uuid::Uuid;

    fn sample(text: &str) -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: text.into(),
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
    fn test_defaults() {
        let f = SemanticDedupFilter::default();
        assert_eq!(f.similarity_threshold, 0.92);
        assert_eq!(f.max_signatures, 100_000);
        assert_eq!(f.min_hash_permutations, 128);
    }

    #[test]
    fn test_minhash_signature_short_text() {
        let f = SemanticDedupFilter::default();
        let sig = f.minhash_signature("hello");
        assert_eq!(sig.len(), 4);
        assert_eq!(sig, vec![0; 4]);
    }

    #[test]
    fn test_minhash_signature_normal_text() {
        let f = SemanticDedupFilter::default();
        let sig = f.minhash_signature("the quick brown fox jumps over the lazy dog near the river");
        assert!(!sig.is_empty());
        assert!(!sig.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_jaccard_identical() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 4];
        assert!((SemanticDedupFilter::jaccard_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(SemanticDedupFilter::jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_jaccard_empty() {
        assert_eq!(SemanticDedupFilter::jaccard_similarity(&[], &[]), 0.0);
    }

    #[tokio::test]
    async fn test_first_sample_passes() {
        let f = SemanticDedupFilter::default();
        let s = sample("this is a completely unique piece of text for testing purposes");
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_similar_sample_detected() {
        let f = SemanticDedupFilter::new(0.2);
        let s1 = sample("the quick brown fox jumps over the lazy dog near the river bank");
        let s2 = sample("the quick brown fox jumps over the lazy dog near the river bank");
        assert!(f.evaluate(&s1).await.passed);
        let result = f.evaluate(&s2).await;
        assert!(!result.passed);
        assert!(result.reason.unwrap().contains("semantic_duplicate"));
    }

    #[test]
    fn test_new_with_threshold() {
        let f = SemanticDedupFilter::new(0.5);
        assert_eq!(f.similarity_threshold, 0.5);
    }

    #[test]
    fn test_debug_format() {
        let f = SemanticDedupFilter::default();
        let debug = format!("{:?}", f);
        assert!(debug.contains("SemanticDedupFilter"));
        assert!(debug.contains("similarity_threshold"));
    }
}
