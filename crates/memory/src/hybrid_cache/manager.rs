use super::cache_layer::CacheStats;
use super::embedding_cache::EmbeddingCache;
use super::http_cache::HttpCache;
use super::model_cache::ModelCache;
use super::prompt_cache::PromptCache;
use super::retrieval_cache::RetrievalCache;
use super::token_cache::TokenCache;
use super::tool_cache::ToolCache;

/// HybridCacheManager — 7-layer cache terintegrasi
pub struct HybridCacheManager {
    pub prompt: PromptCache,
    pub embedding: EmbeddingCache,
    pub retrieval: RetrievalCache,
    pub tool: ToolCache,
    pub http: HttpCache,
    pub token: TokenCache,
    pub model: ModelCache,
}

impl HybridCacheManager {
    pub fn new() -> Self {
        Self {
            prompt: PromptCache::new(512),
            embedding: EmbeddingCache::new(100000),
            retrieval: RetrievalCache::new(50000),
            tool: ToolCache::new(10000),
            http: HttpCache::new(5000, 300),
            token: TokenCache::new(50000),
            model: ModelCache::new(10),
        }
    }

    pub fn stats(&self) -> Vec<CacheStats> {
        vec![
            self.prompt.stats(),
            self.embedding.stats(),
            self.retrieval.stats(),
            self.tool.stats(),
            self.http.stats(),
            self.token.stats(),
            self.model.stats(),
        ]
    }

    pub fn total_hits(&self) -> u64 {
        self.stats().iter().map(|s| s.hits).sum()
    }

    pub fn total_misses(&self) -> u64 {
        self.stats().iter().map(|s| s.misses).sum()
    }

    pub fn overall_hit_rate(&self) -> f64 {
        let hits = self.total_hits() as f64;
        let total = hits + self.total_misses() as f64;
        if total > 0.0 { hits / total } else { 0.0 }
    }

    pub fn clear_all(&self) {
        self.prompt.clear();
    }
}

impl Default for HybridCacheManager {
    fn default() -> Self {
        Self::new()
    }
}
