//! ERP Cache-Aware Resonant Inference
//! 
//! Implementasi dari cache-aware resonant inference untuk
//! reuse activation patterns dan reduce recomputation.

use crate::erp::{ERPConfig, ERPError, GatePattern};
use ndarray::Array1;
use std::collections::HashMap;
use lru::LruCache;
use std::hash::{Hash, Hasher};
use std::num::Wrapping;

/// Cache-aware inference engine untuk ERP
pub struct InferenceCache {
    cache_size: usize,
    cache: LruCache<ContextHash, CachedGatePattern>,
    cache_stats: CacheStats,
    hash_function: ContextHasher,
}

/// Cache statistics untuk monitoring
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub total_contexts: usize,
}

impl InferenceCache {
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            cache: LruCache::new(std::num::NonZeroUsize::new(cache_size).unwrap_or_else(|| std::num::NonZeroUsize::MIN)),
            cache_stats: CacheStats::default(),
            hash_function: ContextHasher::new(),
        }
    }

    /// Get cached gate pattern untuk context
    pub fn get(&mut self, context_hash: ContextHash) -> Option<&CachedGatePattern> {
        match self.cache.get(&context_hash) {
            Some(cached_pattern) => {
                self.cache_stats.hits += 1;
                Some(cached_pattern)
            }
            None => {
                self.cache_stats.misses += 1;
                None
            }
        }
    }

    /// Insert gate pattern ke cache
    pub fn insert(&mut self, context_hash: ContextHash, gates: Vec<GatePattern>) {
        let cached_pattern = CachedGatePattern {
            gates,
            timestamp: std::time::Instant::now(),
            access_count: 0,
        };

        // Check jika cache penuh
        if self.cache.len() >= self.cache_size {
            self.cache_stats.evictions += 1;
        }

        self.cache.put(context_hash, cached_pattern);
        self.cache_stats.total_contexts += 1;
    }

    /// Compute context hash dari input
    pub fn hash_context(&self, input: &Array1<f32>) -> ContextHash {
        self.hash_function.hash_input(input)
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> &CacheStats {
        &self.cache_stats
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f32 {
        let total_requests = self.cache_stats.hits + self.cache_stats.misses;
        if total_requests > 0 {
            self.cache_stats.hits as f32 / total_requests as f32
        } else {
            0.0
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.cache_stats = CacheStats::default();
    }

    /// Precompute cache untuk frequent patterns
    pub fn precompute_patterns(&mut self, frequent_contexts: &[Array1<f32>], reconstructor: &crate::erp::reconstruction::ContextReconstructor, compressed_layers: &[crate::erp::compression::CompressedLayer]) -> Result<(), ERPError> {
        for context in frequent_contexts {
            let context_hash = self.hash_context(context);
            
            // Skip jika sudah ada di cache
            if self.cache.contains(&context_hash) {
                continue;
            }

            // Compute gates untuk context
            let gates = reconstructor.compute_gates(compressed_layers, context)?;
            self.insert(context_hash, gates);
        }

        Ok(())
    }

    /// Update cache access statistics
    pub fn update_access_stats(&mut self) {
        // Update access count untuk semua cached items
        for (_, cached_pattern) in self.cache.iter_mut() {
            cached_pattern.access_count += 1;
        }
    }

    /// Evict least frequently used patterns
    pub fn evict_lfu(&mut self, num_to_evict: usize) {
        let mut patterns_with_counts: Vec<_> = self.cache.iter()
            .map(|(hash, pattern)| (*hash, pattern.access_count))
            .collect();
        
        patterns_with_counts.sort_by_key(|(_, count)| *count);
        
        for (hash, _) in patterns_with_counts.iter().take(num_to_evict) {
            self.cache.pop(hash);
            self.cache_stats.evictions += 1;
        }
    }

    /// Get cache utilization
    pub fn utilization(&self) -> f32 {
        self.cache.len() as f32 / self.cache_size as f32
    }
}

/// Context hash untuk cache key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextHash(u64);

impl ContextHash {
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Context hasher untuk generating consistent hash dari input
pub struct ContextHasher {
    seed: u64,
    bucket_size: usize,
}

impl ContextHasher {
    pub fn new() -> Self {
        Self {
            seed: 0x9e3779b97f4a7c15, // Golden ratio
            bucket_size: 64,
        }
    }

    /// Hash input array ke context hash
    pub fn hash_input(&self, input: &Array1<f32>) -> ContextHash {
        let mut hasher = FnvHasher::with_seed(self.seed);
        
        // Hash input values dengan locality-sensitive hashing
        for (i, &value) in input.iter().enumerate() {
            if i < self.bucket_size {
                let quantized = (value * 1000.0) as i32;
                quantized.hash(&mut hasher);
            }
        }
        
        // Add input statistics untuk robustness
        let mean = input.mean().unwrap_or(0.0);
        let std_dev = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>().sqrt();
        
        ((mean * 100.0) as i32).hash(&mut hasher);
        ((std_dev * 100.0) as i32).hash(&mut hasher);
        input.len().hash(&mut hasher);
        
        ContextHash(hasher.finish())
    }

    /// Hash dengan sliding window untuk temporal patterns
    pub fn hash_sliding_window(&self, input: &Array1<f32>, window_size: usize) -> Vec<ContextHash> {
        let mut hashes = Vec::new();
        
        for start in 0..=input.len().saturating_sub(window_size) {
            let end = (start + window_size).min(input.len());
            let window = input.slice(ndarray::s![start..end]);
            hashes.push(self.hash_input(&window.to_owned()));
        }
        
        hashes
    }

    /// Combine multiple hashes
    pub fn combine_hashes(&self, hashes: &[ContextHash]) -> ContextHash {
        let mut combined = Wrapping(0u64);
        
        for hash in hashes {
            combined ^= Wrapping(hash.0);
            combined = combined * Wrapping(0x100000001b3); // FNV prime
        }
        
        ContextHash(combined.0)
    }
}

/// Cached gate pattern dengan metadata
#[derive(Debug, Clone)]
pub struct CachedGatePattern {
    pub gates: Vec<GatePattern>,
    pub timestamp: std::time::Instant,
    pub access_count: usize,
}

impl CachedGatePattern {
    /// Check jika cached pattern masih valid (age-based eviction)
    pub fn is_valid(&self, max_age: std::time::Duration) -> bool {
        self.timestamp.elapsed() < max_age
    }

    /// Get pattern age
    pub fn age(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }

    /// Update access timestamp
    pub fn touch(&mut self) {
        self.timestamp = std::time::Instant::now();
        self.access_count += 1;
    }
}

/// Advanced cache dengan LRU + LFU hybrid strategy
pub struct HybridCache {
    lru_cache: LruCache<ContextHash, CachedGatePattern>,
    lfu_tracker: HashMap<ContextHash, usize>,
    max_size: usize,
    lru_weight: f32,
    lfu_weight: f32,
}

impl HybridCache {
    pub fn new(max_size: usize, lru_weight: f32, lfu_weight: f32) -> Self {
        Self {
            lru_cache: LruCache::new(std::num::NonZeroUsize::new(max_size).unwrap_or_else(|| std::num::NonZeroUsize::MIN)),
            lfu_tracker: HashMap::new(),
            max_size,
            lru_weight,
            lfu_weight,
        }
    }

    /// Get item dengan hybrid scoring
    pub fn get(&mut self, key: ContextHash) -> Option<&CachedGatePattern> {
        if let Some(cached_pattern) = self.lru_cache.get(&key) {
            // Update LFU count
            *self.lfu_tracker.entry(key).or_insert(0) += 1;
            Some(cached_pattern)
        } else {
            None
        }
    }

    /// Insert item dengan hybrid eviction
    pub fn insert(&mut self, key: ContextHash, value: CachedGatePattern) {
        if self.lru_cache.len() >= self.max_size {
            self.hybrid_eviction();
        }

        self.lru_cache.put(key, value);
        self.lfu_tracker.insert(key, 1);
    }

    /// Hybrid eviction strategy
    fn hybrid_eviction(&mut self) {
        if self.lru_cache.len() == 0 {
            return;
        }

        let mut eviction_candidates: Vec<_> = self.lru_cache.iter()
            .map(|(key, _)| {
                let lru_score = self.lru_cache.cap().get() - self.lru_cache.len(); // Simplified LRU score
                let lfu_score = self.lfu_tracker.get(key).copied().unwrap_or(0);
                let hybrid_score = self.lru_weight * lru_score as f32 + self.lfu_weight * lfu_score as f32;
                (*key, hybrid_score)
            })
            .collect();

        eviction_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Evict lowest scoring item
        if let Some((key, _)) = eviction_candidates.first() {
            self.lru_cache.pop(key);
            self.lfu_tracker.remove(key);
        }
    }
}

/// Pattern-based cache untuk similar context patterns
pub struct PatternCache {
    pattern_clusters: HashMap<PatternCluster, Vec<ContextHash>>,
    cluster_cache: HashMap<PatternCluster, GatePattern>,
    context_cache: HashMap<ContextHash, Array1<f32>>,
    similarity_threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PatternCluster {
    pub cluster_id: usize,
    pub pattern_type: PatternType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternType {
    HighActivation,
    LowActivation,
    Sparse,
    Dense,
    Periodic,
}

impl PatternCache {
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            pattern_clusters: HashMap::new(),
            cluster_cache: HashMap::new(),
            context_cache: HashMap::new(),
            similarity_threshold,
        }
    }

    /// Find pattern cluster untuk context
    pub fn find_cluster(&mut self, context: &Array1<f32>, gates: &[GatePattern]) -> Option<PatternCluster> {
        let pattern_type = self.classify_pattern(gates);
        
        // Check existing clusters
        for (cluster, context_hashes) in &self.pattern_clusters {
            if cluster.pattern_type == pattern_type {
                // Check similarity dengan existing contexts in cluster
                for &context_hash in context_hashes {
                    if self.contexts_similar(context, context_hash) {
                        return Some(cluster.clone());
                    }
                }
            }
        }

        None
    }

    /// Classify pattern type dari gates
    fn classify_pattern(&self, gates: &[GatePattern]) -> PatternType {
        let total_neurons: usize = gates.iter().map(|g| g.gates.len()).sum();
        let active_neurons: usize = gates.iter().map(|g| g.active_neurons.len()).sum();
        let activation_ratio = active_neurons as f32 / total_neurons as f32;

        if activation_ratio > 0.8 {
            PatternType::Dense
        } else if activation_ratio < 0.2 {
            PatternType::Sparse
        } else if activation_ratio > 0.6 {
            PatternType::HighActivation
        } else {
            PatternType::LowActivation
        }
    }

    /// Check similarity antara contexts
    fn contexts_similar(&self, context: &Array1<f32>, context_hash: ContextHash) -> bool {
        // Get cached context untuk comparison
        if let Some(cached_context) = self.context_cache.get(&context_hash) {
            // Compute cosine similarity antara contexts
            let similarity = self.compute_cosine_similarity(context, cached_context);
            
            // Threshold untuk similarity (adjustable based on requirements)
            let similarity_threshold = 0.8;
            
            similarity >= similarity_threshold
        } else {
            // If no cached context found, assume not similar
            false
        }
    }
    
    /// Compute cosine similarity antara dua context vectors
    fn compute_cosine_similarity(&self, context1: &Array1<f32>, context2: &Array1<f32>) -> f32 {
        if context1.len() != context2.len() {
            return 0.0; // Different dimensions, not similar
        }
        
        // Compute dot product
        let dot_product: f32 = context1.iter().zip(context2.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        // Compute norms
        let norm1: f32 = context1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = context2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Handle zero norms
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        // Return cosine similarity
        dot_product / (norm1 * norm2)
    }
    
    /// Compute Euclidean distance antara contexts (alternative similarity metric)
    fn compute_euclidean_distance(&self, context1: &Array1<f32>, context2: &Array1<f32>) -> f32 {
        if context1.len() != context2.len() {
            return f32::INFINITY; // Different dimensions
        }
        
        let sum_sq_diff: f32 = context1.iter().zip(context2.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum();
        
        sum_sq_diff.sqrt()
    }
    
    /// Check similarity menggunakan multiple metrics untuk robust decision
    fn contexts_similar_robust(&self, context: &Array1<f32>, context_hash: ContextHash) -> bool {
        if let Some(cached_context) = self.context_cache.get(&context_hash) {
            // Compute multiple similarity metrics
            let cosine_sim = self.compute_cosine_similarity(context, cached_context);
            let euclidean_dist = self.compute_euclidean_distance(context, cached_context);
            
            // Normalize Euclidean distance to similarity (closer = more similar)
            let max_possible_dist = (context.len() as f32).sqrt() * 2.0; // Maximum possible distance
            let euclidean_sim = if max_possible_dist > 0.0 {
                1.0 - (euclidean_dist / max_possible_dist)
            } else {
                1.0
            };
            
            // Weighted combination of similarity metrics
            let combined_similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim;
            
            // Threshold untuk robust similarity
            combined_similarity >= 0.75
        } else {
            false
        }
    }

    /// Add context ke pattern cluster
    pub fn add_to_cluster(&mut self, cluster: PatternCluster, context_hash: ContextHash, gates: GatePattern) {
        self.pattern_clusters.entry(cluster.clone()).or_insert_with(Vec::new).push(context_hash);
        self.cluster_cache.insert(cluster, gates);
    }
}

/// FNV-1a hash implementation
struct FnvHasher {
    hash: Wrapping<u64>,
}

impl FnvHasher {
    fn with_seed(seed: u64) -> Self {
        Self {
            hash: Wrapping(14695981039346656037u64) ^ Wrapping(seed),
        }
    }

    fn finish(&self) -> u64 {
        self.hash.0
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.hash.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.hash;
        for &byte in bytes {
            hash ^= Wrapping(byte as u64);
            hash *= Wrapping(0x100000001b3);
        }
        self.hash = hash;
    }
}
