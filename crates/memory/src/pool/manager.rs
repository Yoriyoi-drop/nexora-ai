use std::sync::Arc;

use super::buffer_pool::{BufferPool, BufferPoolStats};
use super::embedding_pool::EmbeddingPool;
use super::kv_cache_pool::GlobalKVCachePool;
use super::memory_pool::{GlobalMemoryPools, MemoryBlock, MemoryPool, MemoryPoolConfig};
use super::tensor_pool::GlobalTensorPools;

/// Unified Memory Manager — semua alokasi melalui sini, no raw alloc/free
pub struct UnifiedMemoryManager {
    pub memory_pools: GlobalMemoryPools,
    pub tensor_pools: GlobalTensorPools,
    pub kv_cache_pool: GlobalKVCachePool,
    pub buffer_pool: BufferPool,
    embedding_pools: std::collections::HashMap<String, Arc<EmbeddingPool>>,
}

impl UnifiedMemoryManager {
    pub fn new() -> Self {
        let mut mgr = Self {
            memory_pools: GlobalMemoryPools::new(),
            tensor_pools: GlobalTensorPools::new(),
            kv_cache_pool: GlobalKVCachePool::default(),
            buffer_pool: BufferPool::new(256),
            embedding_pools: std::collections::HashMap::new(),
        };
        mgr.init_default_pools();
        mgr
    }

    fn init_default_pools(&mut self) {
        // Memory pools untuk berbagai ukuran blok
        let sizes = [
            ("tiny", 4096, 4096),
            ("small", 65536, 2048),
            ("medium", 524288, 512),     // 512KB
            ("large", 4194304, 128),      // 4MB
            ("huge", 33554432, 32),       // 32MB
            ("giant", 268435456, 8),       // 256MB
        ];
        for (name, block_size, max_blocks) in &sizes {
            let config = MemoryPoolConfig {
                name: name.to_string(),
                block_size: *block_size,
                max_blocks: *max_blocks,
                alignment: 64,
            };
            let pool = MemoryPool::new(config);
            pool.preallocate((*max_blocks / 4).max(8));
            self.memory_pools.register(*name, pool);
        }

        // Tensor pools untuk berbagai ukuran
        self.tensor_pools.register("small_tensor", 1024);
        self.tensor_pools.register("medium_tensor", 128);
        self.tensor_pools.register("large_tensor", 32);
    }

    pub fn create_embedding_pool(&mut self, name: impl Into<String>, dim: usize, capacity: usize) -> Arc<EmbeddingPool> {
        let name: String = name.into();
        let pool = Arc::new(EmbeddingPool::new(dim, capacity));
        self.embedding_pools.insert(name, pool.clone());
        pool
    }

    pub fn get_embedding_pool(&self, name: &str) -> Option<&Arc<EmbeddingPool>> {
        self.embedding_pools.get(name)
    }

    /// Auto-pilih pool berdasarkan ukuran
    pub fn acquire_block(&self, size: usize) -> Option<MemoryBlock> {
        let pool_name = if size <= 4096 {
            "tiny"
        } else if size <= 65536 {
            "small"
        } else if size <= 524288 {
            "medium"
        } else if size <= 4194304 {
            "large"
        } else if size <= 33554432 {
            "huge"
        } else {
            "giant"
        };
        self.memory_pools
            .get(pool_name)
            .and_then(|pool| pool.acquire())
    }

    pub fn memory_stats(&self) -> UnifiedMemoryStats {
        let mut pools = Vec::new();
        if let Some(p) = self.memory_pools.get("small") {
            pools.push(MemoryPoolInfo {
                name: "small".into(),
                block_size: p.block_size(),
                free: p.free_count(),
                allocated: p.allocated_count(),
                usage: p.usage_ratio(),
            });
        }
        UnifiedMemoryStats {
            memory_pools: pools,
            buffer_pool: self.buffer_pool.stats(),
            kv_cache_blocks: self.kv_cache_pool.pool().active_count(),
        }
    }
}

impl Default for UnifiedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryPoolInfo {
    pub name: String,
    pub block_size: usize,
    pub free: usize,
    pub allocated: usize,
    pub usage: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UnifiedMemoryStats {
    pub memory_pools: Vec<MemoryPoolInfo>,
    pub buffer_pool: BufferPoolStats,
    pub kv_cache_blocks: usize,
}
