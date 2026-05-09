//! Fast Cache Agent
//! 
//! Intelligent caching system to avoid redundant inference

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Fast Cache Agent - Intelligent caching system
#[derive(Debug, Clone)]
pub struct FastCacheAgent {
    pub config: FastCacheConfig,
    pub cache_engine: CacheEngine,
    pub similarity_matcher: SimilarityMatcher,
    pub eviction_policy: EvictionPolicy,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastCacheConfig {
    pub base_config: BaseAgentConfig,
    pub cache_size_mb: u32,
    pub similarity_threshold: f32,
    pub cache_policy: CachePolicy,
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
    /// Similarity-based caching
    Semantic,
    /// Adaptive based on usage patterns
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEngine {
    pub cache_storage: HashMap<String, CacheEntry>,
    pub cache_size_bytes: usize,
    pub max_cache_size_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<f32>,
    pub embedding: Vec<f32>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub access_count: u64,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub ttl_seconds: Option<u64>,
    pub similarity_hash: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatcher {
    pub embedding_model: String,
    pub similarity_algorithm: SimilarityAlgorithm,
    pub similarity_threshold: f32,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityAlgorithm {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Dot product
    DotProduct,
    /// Jaccard similarity
    Jaccard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionPolicy {
    pub policy: EvictionStrategy,
    pub max_entries: usize,
    pub cleanup_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Remove least recently used entries
    LRU,
    /// Remove least frequently used entries
    LFU,
    /// Remove entries with lowest similarity scores
    LowSimilarity,
    /// Remove expired entries
    TTLExpiry,
    /// Combination of strategies
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastCacheTaskInput {
    pub query: String,
    pub query_embedding: Option<Vec<f32>>,
    pub cache_policy_override: Option<CachePolicy>,
    pub similarity_threshold_override: Option<f32>,
    pub force_refresh: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastCacheTaskOutput {
    pub cache_result: CacheResult,
    pub similarity_analysis: SimilarityAnalysis,
    pub performance_metrics: CacheMetrics,
    pub cache_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheResult {
    Hit {
        key: String,
        value: Vec<f32>,
        similarity_score: f32,
        entry_age_seconds: u64,
    },
    Miss {
        reason: MissReason,
        suggested_cache_key: String,
    },
    Stale {
        key: String,
        stale_value: Vec<f32>,
        freshness_score: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissReason {
    NoSimilarEntries,
    BelowThreshold,
    CacheEmpty,
    Evicted,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityAnalysis {
    pub total_entries_compared: usize,
    pub best_match_score: f32,
    pub average_similarity: f32,
    pub similarity_distribution: HashMap<String, u32>,
    pub processing_time_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hit_rate: f32,
    pub miss_rate: f32,
    pub total_queries: u64,
    pub cache_size_mb: f32,
    pub avg_lookup_time_ms: f32,
    pub eviction_count: u64,
}

impl Default for FastCacheConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            cache_size_mb: 256,
            similarity_threshold: 0.85,
            cache_policy: CachePolicy::Semantic,
            ttl_seconds: 3600, // 1 hour
        }
    }
}

impl Default for CacheEngine {
    fn default() -> Self {
        Self {
            cache_storage: HashMap::new(),
            cache_size_bytes: 0,
            max_cache_size_bytes: 256 * 1024 * 1024, // 256 MB
            hit_count: 0,
            miss_count: 0,
        }
    }
}

impl Default for SimilarityMatcher {
    fn default() -> Self {
        Self {
            embedding_model: "tiny_bert".to_string(),
            similarity_algorithm: SimilarityAlgorithm::Cosine,
            similarity_threshold: 0.85,
            embedding_dim: 384,
        }
    }
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            policy: EvictionStrategy::Hybrid,
            max_entries: 10000,
            cleanup_interval_seconds: 300, // 5 minutes
        }
    }
}

impl Default for FastCacheAgent {
    fn default() -> Self {
        Self {
            config: FastCacheConfig::default(),
            cache_engine: CacheEngine::default(),
            similarity_matcher: SimilarityMatcher::default(),
            eviction_policy: EvictionPolicy::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for FastCacheAgent {
    type Config = FastCacheConfig;
    type Input = FastCacheTaskInput;
    type Output = FastCacheTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Generate query embedding if not provided
        let query_embedding = match input.query_embedding {
            Some(ref embedding) => embedding.clone(),
            None => self.generate_embedding(&input.query).await?,
        };

        // Search cache for similar entries
        let cache_result = self.search_cache(&query_embedding, &input).await?;
        
        // Analyze similarity patterns
        let similarity_analysis = self.analyze_similarity(&query_embedding).await?;
        
        // Calculate performance metrics
        let performance_metrics = self.calculate_cache_metrics().await?;
        
        // Generate cache recommendations
        let cache_recommendations = self.generate_cache_recommendations(&cache_result, &similarity_analysis).await?;

        Ok(FastCacheTaskOutput {
            cache_result,
            similarity_analysis,
            performance_metrics,
            cache_recommendations,
        })
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability {
                name: "fast_cache".to_string(),
                description: "Intelligent caching system with semantic similarity matching".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["query".to_string(), "query_embedding".to_string()],
                output_types: vec!["cache_result".to_string(), "similarity_analysis".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.96,
                    avg_latency: 0.2,
                    resource_usage: 0.4,
                    reliability: 0.99,
                },
            },
        ]
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }

    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()> {
        self.config = config;
        self.status = AgentStatus::Idle;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        self.status = AgentStatus::Disabled;
        Ok(())
    }
}

impl FastCacheAgent {
    pub fn new(config: FastCacheConfig) -> Self {
        Self {
            config,
            cache_engine: CacheEngine::default(),
            similarity_matcher: SimilarityMatcher::default(),
            eviction_policy: EvictionPolicy::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    async fn generate_embedding(&self, query: &str) -> AgentResult<Vec<f32>> {
        // Simple embedding generation simulation
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut embedding = vec![0.0; self.similarity_matcher.embedding_dim];
        
        for (i, word) in words.iter().enumerate() {
            if i < embedding.len() {
                // Simple hash-based embedding
                let hash = word.chars().map(|c| c as u32).sum::<u32>() as f32;
                embedding[i] = (hash % 1000) as f32 / 1000.0;
            }
        }

        // Normalize embedding
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding = embedding.into_iter().map(|x| x / norm).collect();
        }

        Ok(embedding)
    }

    async fn search_cache(&self, query_embedding: &[f32], input: &FastCacheTaskInput) -> AgentResult<CacheResult> {
        if self.cache_engine.cache_storage.is_empty() {
            return Ok(CacheResult::Miss {
                reason: MissReason::CacheEmpty,
                suggested_cache_key: self.generate_cache_key(&input.query),
            });
        }

        let mut best_match: Option<(&String, &CacheEntry, f32)> = None;
        let similarity_threshold = input.similarity_threshold_override.unwrap_or(self.config.similarity_threshold);

        // Search for similar entries
        for (key, entry) in &self.cache_engine.cache_storage {
            let similarity = self.calculate_similarity(query_embedding, &entry.embedding);
            
            if similarity > similarity_threshold {
                match &best_match {
                    None => best_match = Some((key, entry, similarity)),
                    Some((_, _, current_similarity)) => if similarity > *current_similarity {
                        best_match = Some((key, entry, similarity));
                    }
                }
            }
        }

        match best_match {
            Some((key, entry, similarity)) => {
                let now = chrono::Utc::now();
                let entry_age = (now - entry.timestamp).num_seconds() as u64;
                
                // Check if entry is stale
                if let Some(ttl) = entry.ttl_seconds {
                    if entry_age > ttl {
                        return Ok(CacheResult::Stale {
                            key: key.clone(),
                            stale_value: entry.value.clone(),
                            freshness_score: 1.0 - (entry_age as f32 / ttl as f32),
                        });
                    }
                }

                Ok(CacheResult::Hit {
                    key: key.clone(),
                    value: entry.value.clone(),
                    similarity_score: similarity,
                    entry_age_seconds: entry_age,
                })
            },
            None => Ok(CacheResult::Miss {
                reason: MissReason::BelowThreshold,
                suggested_cache_key: self.generate_cache_key(&input.query),
            }),
        }
    }

    fn calculate_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }

        match self.similarity_matcher.similarity_algorithm {
            SimilarityAlgorithm::Cosine => {
                let dot_product = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum::<f32>();
                let norm1 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm2 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
                
                if norm1 == 0.0 || norm2 == 0.0 {
                    0.0
                } else {
                    dot_product / (norm1 * norm2)
                }
            },
            SimilarityAlgorithm::Euclidean => {
                let distance = embedding1.iter().zip(embedding2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                1.0 / (1.0 + distance) // Convert distance to similarity
            },
            SimilarityAlgorithm::DotProduct => {
                embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum::<f32>()
            },
            _ => 0.5, // Default similarity
        }
    }

    fn generate_cache_key(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("cache_{:x}", hasher.finish())
    }

    async fn analyze_similarity(&self, query_embedding: &[f32]) -> AgentResult<SimilarityAnalysis> {
        let start_time = std::time::Instant::now();
        
        let mut similarities = Vec::new();
        for entry in self.cache_engine.cache_storage.values() {
            let similarity = self.calculate_similarity(query_embedding, &entry.embedding);
            similarities.push(similarity);
        }

        let total_entries = similarities.len();
        let best_match_score = similarities.iter().fold(0.0, |a, &b| a.max(b));
        let average_similarity = if total_entries > 0 {
            similarities.iter().sum::<f32>() / total_entries as f32
        } else {
            0.0
        };

        // Create similarity distribution
        let mut distribution = HashMap::new();
        for similarity in similarities {
            let bucket = if similarity >= 0.9 {
                "very_high".to_string()
            } else if similarity >= 0.7 {
                "high".to_string()
            } else if similarity >= 0.5 {
                "medium".to_string()
            } else if similarity >= 0.3 {
                "low".to_string()
            } else {
                "very_low".to_string()
            };
            *distribution.entry(bucket).or_insert(0) += 1;
        }

        Ok(SimilarityAnalysis {
            total_entries_compared: total_entries,
            best_match_score,
            average_similarity,
            similarity_distribution: distribution,
            processing_time_ms: start_time.elapsed().as_millis() as f32,
        })
    }

    async fn calculate_cache_metrics(&self) -> AgentResult<CacheMetrics> {
        let total_queries = self.cache_engine.hit_count + self.cache_engine.miss_count;
        let hit_rate = if total_queries > 0 {
            self.cache_engine.hit_count as f32 / total_queries as f32
        } else {
            0.0
        };
        let miss_rate = 1.0 - hit_rate;
        
        let cache_size_mb = self.cache_engine.cache_size_bytes as f32 / (1024.0 * 1024.0);
        let avg_lookup_time_ms = 0.1; // Simulated average lookup time

        Ok(CacheMetrics {
            hit_rate,
            miss_rate,
            total_queries,
            cache_size_mb,
            avg_lookup_time_ms,
            eviction_count: 0, // Track evictions in real implementation
        })
    }

    async fn generate_cache_recommendations(&self, cache_result: &CacheResult, similarity_analysis: &SimilarityAnalysis) -> AgentResult<Vec<String>> {
        let mut recommendations = Vec::new();

        match cache_result {
            CacheResult::Hit { similarity_score, .. } => {
                if *similarity_score > 0.95 {
                    recommendations.push("Excellent cache hit - consider preloading similar queries".to_string());
                } else if *similarity_score < 0.7 {
                    recommendations.push("Low similarity hit - consider adjusting threshold".to_string());
                }
            },
            CacheResult::Miss { reason, .. } => {
                match reason {
                    MissReason::NoSimilarEntries => {
                        recommendations.push("No similar entries found - consider expanding cache".to_string());
                    },
                    MissReason::BelowThreshold => {
                        recommendations.push("All entries below threshold - consider lowering similarity threshold".to_string());
                    },
                    MissReason::CacheEmpty => {
                        recommendations.push("Cache is empty - start building cache with common queries".to_string());
                    },
                    _ => {}
                }
            },
            CacheResult::Stale { freshness_score, .. } => {
                if *freshness_score < 0.5 {
                    recommendations.push("Many stale entries - consider reducing TTL".to_string());
                }
            }
        }

        if similarity_analysis.average_similarity < 0.5 {
            recommendations.push("Low average similarity - consider improving embedding quality".to_string());
        }

        if similarity_analysis.total_entries_compared < 100 {
            recommendations.push("Small cache size - consider increasing cache capacity".to_string());
        }

        Ok(recommendations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_cache_agent_creation() {
        let agent = FastCacheAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_cache_search() {
        let agent = FastCacheAgent::default();
        let input = FastCacheTaskInput {
            query: "test query".to_string(),
            query_embedding: None,
            cache_policy_override: None,
            similarity_threshold_override: None,
            force_refresh: false,
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(matches!(output.cache_result, CacheResult::Miss { .. }));
        assert!(output.similarity_analysis.total_entries_compared == 0);
    }

    #[tokio::test]
    async fn test_embedding_generation() {
        let agent = FastCacheAgent::default();
        let embedding = agent.generate_embedding("hello world").await.unwrap();
        assert_eq!(embedding.len(), agent.similarity_matcher.embedding_dim);
    }

    #[test]
    fn test_similarity_calculation() {
        let agent = FastCacheAgent::default();
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![1.0, 0.0, 0.0];
        let embedding3 = vec![0.0, 1.0, 0.0];

        let sim12 = agent.calculate_similarity(&embedding1, &embedding2);
        let sim13 = agent.calculate_similarity(&embedding1, &embedding3);

        assert!((sim12 - 1.0).abs() < 0.001); // Identical vectors
        assert!(sim13 < sim12); // Different vectors
    }
}
