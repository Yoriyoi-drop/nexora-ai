use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

const LSH_BANDS: usize = 16;

pub struct SemanticDedupFilter {
    pub similarity_threshold: f64,
    pub signatures: Mutex<Vec<Vec<u64>>>,
    pub max_signatures: usize,
    pub min_hash_permutations: usize,
    /// LSH index: band → bucket hash → list of signature indices
    lsh_index: Mutex<HashMap<u64, Vec<usize>>>,
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
        let lsh_index = self
            .lsh_index
            .try_lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        Self {
            similarity_threshold: self.similarity_threshold,
            signatures: Mutex::new(signatures),
            max_signatures: self.max_signatures,
            min_hash_permutations: self.min_hash_permutations,
            lsh_index: Mutex::new(lsh_index),
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
            lsh_index: Mutex::new(HashMap::new()),
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

        let num_perms = self.min_hash_permutations.min(128);
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

    /// LSH banding: split signature into bands, hash each band to a bucket
    fn lsh_bands(sig: &[u64]) -> Vec<u64> {
        let rows = sig.len() / LSH_BANDS;
        if rows == 0 {
            return vec![0];
        }
        let mut band_hashes = Vec::with_capacity(LSH_BANDS);
        for b in 0..LSH_BANDS {
            let start = b * rows;
            let end = (start + rows).min(sig.len());
            let mut h: u64 = 0xcbf29ce484222325;
            for &v in &sig[start..end] {
                h ^= v;
                h = h.wrapping_mul(0x100000001b3);
            }
            band_hashes.push(h);
        }
        band_hashes
    }

    fn jaccard_similarity(a: &[u64], b: &[u64]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let mut sorted_a = a.to_vec();
        let mut sorted_b = b.to_vec();
        sorted_a.sort_unstable();
        sorted_b.sort_unstable();

        let mut i = 0;
        let mut j = 0;
        let mut shared = 0usize;
        while i < sorted_a.len() && j < sorted_b.len() {
            match sorted_a[i].cmp(&sorted_b[j]) {
                std::cmp::Ordering::Equal => { shared += 1; i += 1; j += 1; }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        let total = a.len() + b.len() - shared;
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
        let Ok(mut signatures) = self.signatures.try_lock() else {
            tracing::warn!("SemanticDedupFilter::evaluate: signatures mutex poisoned, passing through");
            return FilterResult {
                passed: true,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: None,
                score_delta: 0.0,
            };
        };
        let Ok(mut lsh) = self.lsh_index.try_lock() else {
            tracing::warn!("SemanticDedupFilter::evaluate: lsh_index mutex poisoned, passing through");
            return FilterResult {
                passed: true,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: None,
                score_delta: 0.0,
            };
        };

        // Fast candidate selection via LSH bands with HashSet dedup
        let band_hashes = Self::lsh_bands(&sig);
        let mut candidate_set = HashSet::new();
        for bh in &band_hashes {
            if let Some(indices) = lsh.get(bh) {
                for &idx in indices {
                    candidate_set.insert(idx);
                }
            }
        }
        // Cap candidates to prevent worst-case O(n²)
        const MAX_CANDIDATES: usize = 256;
        let candidates: Vec<usize> = candidate_set.into_iter().take(MAX_CANDIDATES).collect();

        // Only compare against candidates (not all stored signatures)
        for &candidate_idx in &candidates {
            if candidate_idx >= signatures.len() { continue; }
            let similarity = Self::jaccard_similarity(&sig, &signatures[candidate_idx]);
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

        // Store new signature with LSH index
        let idx = signatures.len();
        if signatures.len() < self.max_signatures {
            signatures.push(sig);
            for bh in &band_hashes {
                lsh.entry(*bh).or_insert_with(Vec::new).push(idx);
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
