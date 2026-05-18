use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use parking_lot::Mutex;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

/// Fuzzy dedup filter using MinHash-style similarity estimation.
///
/// Computes `hash_count` hash signatures per document using `ngram_size`-word ngrams.
/// A new document is considered a duplicate when the proportion of its hashes that
/// collide with previously seen hashes meets or exceeds `similarity_threshold`.
///
/// References (large model training pipelines):
///   - LLaMA 2: 13-gram dedup for contamination detection (Touvron et al., 2023)
///   - LLaMA 3: 8-gram dedup + line-level exact dedup (Grattafiori et al., 2024)
///   - FineWeb: MinHash, 5-gram, 112 hash functions, 14 bands × 8 hashes, 75% similarity threshold
///   - RedPajama-v2: MinHash at multiple similarity levels
///   - DCLM (DataComp-LM): MinHash + Bloom filter exact dedup
///   - GPT-3: MinHash fuzzy dedup (Brown et al., 2020)
///   - Gopher / MassiveText: URL + exact + paragraph dedup (Rae et al., 2022)
#[derive(Debug, Clone)]
pub struct DedupFilter {
    pub seen_hashes: Arc<Mutex<HashSet<u64>>>,
    pub ngram_size: usize,
    pub hash_count: usize,
    pub max_seen: usize,
    /// MinHash similarity threshold (0.0–1.0).
    /// A document is rejected when `matched_hashes / total_hashes >= similarity_threshold`.
    /// Default: 0.5 — requires at least half of sampled ngrams to collide.
    pub similarity_threshold: f64,
    /// Short texts (fewer words than ngram_size) get a single exact hash.
    /// When `exact_reject_on_seen` is true, those exact-match collisions are also
    /// subject to `similarity_threshold` (= reject if the single hash is seen).
    /// When false, short texts always pass (less aggressive).
    pub exact_reject_on_seen: bool,
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

    /// Compute MinHash-style signatures for a text.
    ///
    /// For texts with >= `ngram_size` words: samples `hash_count` evenly-spaced
    /// ngrams and hashes each with a different RNG seed (simulating independent
    /// hash functions à la MinHash).
    ///
    /// For short texts: returns a single hash of the full text.
    fn fingerprint(&self, text: &str) -> Vec<u64> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < self.ngram_size {
            return vec![self::hash_text(text)];
        }

        let max_ngrams = words.len().saturating_sub(self.ngram_size - 1);
        let step = (max_ngrams / self.hash_count).max(1);

        let mut hashes = Vec::with_capacity(self.hash_count);
        let mut pos = 0;
        // Use slot-index-based seeds for independent hash functions (MinHash convention)
        for slot in 0..self.hash_count {
            if slot >= hashes.capacity() || pos >= max_ngrams {
                break;
            }
            let end = (pos + self.ngram_size).min(words.len());
            let ngram: Vec<&str> = words[pos..end].to_vec();
            let mut hasher = DefaultHasher::new();
            (slot as u64).hash(&mut hasher);
            ngram.hash(&mut hasher);
            hashes.push(hasher.finish());
            pos = pos.saturating_add(step);
        }

        if hashes.is_empty() {
            hashes.push(self::hash_text(text));
        }

        hashes
    }

    pub fn reset(&mut self) {
        self.seen_hashes.lock().clear();
    }
}

fn hash_text(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

impl Default for DedupFilter {
    fn default() -> Self {
        Self {
            seen_hashes: Arc::new(Mutex::new(HashSet::new())),
            ngram_size: 13,
            hash_count: 13,
            max_seen: 50_000_000,
            similarity_threshold: 0.5,
            exact_reject_on_seen: true,
        }
    }
}

#[async_trait]
impl Filter for DedupFilter {
    fn name(&self) -> &str {
        "dedup"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let fingerprints = self.fingerprint(&sample.text);
        let total_hashes = fingerprints.len();

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

        // Short texts (single hash) — use exact match check
        if total_hashes == 1 {
            let seen = hashes.contains(&fingerprints[0]);
            if seen && self.exact_reject_on_seen {
                return FilterResult {
                    passed: false,
                    sample_id: sample.id,
                    filter_name: self.name().to_string(),
                    reason: Some("exact_duplicate".to_string()),
                    score_delta: -0.5,
                };
            }
            hashes.insert(fingerprints[0]);
            return FilterResult {
                passed: true,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: None,
                score_delta: 0.0,
            };
        }

        // Multi-hash: use MinHash-style Jaccard similarity estimate
        let match_count = fingerprints
            .iter()
            .filter(|h| hashes.contains(h))
            .count();
        let match_ratio = match_count as f64 / total_hashes as f64;

        if match_ratio >= self.similarity_threshold {
            FilterResult {
                passed: false,
                sample_id: sample.id,
                filter_name: self.name().to_string(),
                reason: Some(format!(
                    "fuzzy_duplicate: {:.2} ({}/{})",
                    match_ratio, match_count, total_hashes
                )),
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
