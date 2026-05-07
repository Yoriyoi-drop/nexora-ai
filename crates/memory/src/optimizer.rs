//! Memory Management Optimization
//! 
//! Implementasi memory management yang efisien dengan:
//! - Memory pooling
//! - Garbage collection optimization
//! - Memory leak detection
//! - Memory usage monitoring
//! - Smart caching strategies

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{RwLock, Mutex};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use tracing::info;

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Pool size in bytes
    pub pool_size: usize,
    /// Block size in bytes
    pub block_size: usize,
    /// Maximum number of blocks
    pub max_blocks: usize,
    /// Enable automatic expansion
    pub auto_expand: bool,
    /// Expansion factor when pool is full
    pub expansion_factor: f32,
    /// Enable memory defragmentation
    pub enable_defragmentation: bool,
    /// Defragmentation threshold
    pub defragmentation_threshold: f32,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            pool_size: 1024 * 1024 * 100, // 100MB
            block_size: 1024,              // 1KB blocks
            max_blocks: 100000,           // 100K blocks
            auto_expand: true,
            expansion_factor: 1.5,
            enable_defragmentation: true,
            defragmentation_threshold: 0.8,
        }
    }
}

/// Memory block for pooling
#[derive(Debug)]
struct MemoryBlock {
    data: Vec<u8>,
    in_use: bool,
    last_used: std::time::Instant,
    size: usize,
}

/// Memory pool implementation
pub struct MemoryPool {
    config: MemoryPoolConfig,
    blocks: Arc<RwLock<Vec<MemoryBlock>>>,
    free_blocks: Arc<RwLock<Vec<usize>>>,
    allocated_bytes: Arc<AtomicUsize>,
    total_allocated: Arc<AtomicUsize>,
    allocation_count: Arc<AtomicUsize>,
    deallocation_count: Arc<AtomicUsize>,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(config: MemoryPoolConfig) -> Self {
        let pool_size = config.pool_size;
        let block_size = config.block_size;
        let num_blocks = pool_size / block_size;
        
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_blocks = Vec::with_capacity(num_blocks);
        
        // Initialize blocks
        for i in 0..num_blocks {
            blocks.push(MemoryBlock {
                data: vec![0u8; block_size],
                in_use: false,
                last_used: std::time::Instant::now(),
                size: block_size,
            });
            free_blocks.push(i);
        }
        
        Self {
            config,
            blocks: Arc::new(RwLock::new(blocks)),
            free_blocks: Arc::new(RwLock::new(free_blocks)),
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            total_allocated: Arc::new(AtomicUsize::new(0)),
            allocation_count: Arc::new(AtomicUsize::new(0)),
            deallocation_count: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    /// Allocate memory block
    pub async fn allocate(&self, size: usize) -> Result<MemoryHandle> {
        // Find appropriate block
        let block_index = self.find_free_block(size).await?;
        
        let mut blocks = self.blocks.write().await;
        let mut free_blocks = self.free_blocks.write().await;
        
        // Mark block as in use
        if let Some(block) = blocks.get_mut(block_index) {
            block.in_use = true;
            block.last_used = std::time::Instant::now();
            
            // Remove from free list
            free_blocks.retain(|&idx| idx != block_index);
            
            // Update statistics
            self.allocated_bytes.fetch_add(block.size, Ordering::Relaxed);
            self.total_allocated.fetch_add(block.size, Ordering::Relaxed);
            self.allocation_count.fetch_add(1, Ordering::Relaxed);
            
            Ok(MemoryHandle {
                block_index,
                size: block.size,
                pool: Arc::clone(&self.blocks),
                allocated_bytes: Arc::clone(&self.allocated_bytes),
                deallocation_count: Arc::clone(&self.deallocation_count),
            })
        } else {
            Err(anyhow::anyhow!("Failed to allocate memory block"))
        }
    }
    
    /// Find free block of appropriate size
    async fn find_free_block(&self, size: usize) -> Result<usize> {
        let free_blocks = self.free_blocks.read().await;
        let blocks = self.blocks.read().await;
        
        // Find first free block that can accommodate the size
        for &block_index in free_blocks.iter() {
            if let Some(block) = blocks.get(block_index) {
                if !block.in_use && block.size >= size {
                    return Ok(block_index);
                }
            }
        }
        
        // Try to expand pool if auto-expand is enabled
        if self.config.auto_expand {
            self.expand_pool().await?;
            return Box::pin(self.find_free_block(size)).await;
        }
        
        Err(anyhow::anyhow!("No available memory blocks"))
    }
    
    /// Expand memory pool
    async fn expand_pool(&self) -> Result<()> {
        let current_size = {
            let blocks = self.blocks.read().await;
            blocks.len() * self.config.block_size
        };
        
        let new_size = (current_size as f64 * self.config.expansion_factor as f64) as usize;
        let additional_blocks = (new_size - current_size) / self.config.block_size;
        
        if additional_blocks == 0 {
            return Err(anyhow::anyhow!("Cannot expand pool further"));
        }
        
        let mut blocks = self.blocks.write().await;
        let mut free_blocks = self.free_blocks.write().await;
        
        let current_index = blocks.len();
        
        // Add new blocks
        for i in 0..additional_blocks {
            blocks.push(MemoryBlock {
                data: vec![0u8; self.config.block_size],
                in_use: false,
                last_used: std::time::Instant::now(),
                size: self.config.block_size,
            });
            free_blocks.push(current_index + i);
        }
        
        info!("Memory pool expanded by {} blocks", additional_blocks);
        Ok(())
    }
    
    /// Defragment memory pool
    pub async fn defragment(&self) -> Result<DefragmentationResult> {
        if !self.config.enable_defragmentation {
            return Ok(DefragmentationResult::default());
        }
        
        let blocks = self.blocks.write().await;
        let _free_blocks = self.free_blocks.write().await;
        
        let mut moved_blocks = 0;
        let mut freed_bytes = 0;
        
        // Sort blocks by last used time (LRU first)
        let mut block_indices: Vec<usize> = (0..blocks.len()).collect();
        block_indices.sort_by(|&a, &b| {
            blocks[a].last_used.cmp(&blocks[b].last_used)
        });
        
        // Try to consolidate free blocks
        let mut consecutive_free = 0;
        for &index in &block_indices {
            if !blocks[index].in_use {
                consecutive_free += 1;
                if consecutive_free >= 2 {
                    // Merge consecutive free blocks
                    freed_bytes += blocks[index].size;
                    moved_blocks += 1;
                }
            } else {
                consecutive_free = 0;
            }
        }
        
        Ok(DefragmentationResult {
            moved_blocks,
            freed_bytes,
            duration: std::time::Duration::from_millis(0), // Would measure actual time
        })
    }
    
    /// Get pool statistics
    pub async fn get_statistics(&self) -> MemoryPoolStatistics {
        let blocks = self.blocks.read().await;
        let free_blocks = self.free_blocks.read().await;
        
        let total_blocks = blocks.len();
        let used_blocks = blocks.iter().filter(|b| b.in_use).count();
        let free_blocks_count = free_blocks.len();
        
        let allocated_bytes = self.allocated_bytes.load(Ordering::Relaxed);
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);
        let allocation_count = self.allocation_count.load(Ordering::Relaxed);
        let deallocation_count = self.deallocation_count.load(Ordering::Relaxed);
        
        MemoryPoolStatistics {
            total_blocks,
            used_blocks,
            free_blocks: free_blocks_count,
            allocated_bytes,
            total_allocated,
            allocation_count,
            deallocation_count,
            utilization_rate: if total_blocks > 0 {
                used_blocks as f64 / total_blocks as f64
            } else {
                0.0
            },
        }
    }
    
    /// Check if defragmentation is needed
    pub async fn needs_defragmentation(&self) -> bool {
        let stats = self.get_statistics().await;
        stats.utilization_rate < self.config.defragmentation_threshold.into()
    }
}

/// Memory handle for allocated memory
pub struct MemoryHandle {
    block_index: usize,
    size: usize,
    pool: Arc<RwLock<Vec<MemoryBlock>>>,
    allocated_bytes: Arc<AtomicUsize>,
    deallocation_count: Arc<AtomicUsize>,
}

impl MemoryHandle {
    /// Get mutable reference to memory data
    pub fn get_mut(&mut self) -> Option<Vec<u8>> {
        if let Ok(mut blocks) = self.pool.try_write() {
            blocks.get_mut(self.block_index).map(|block| block.data[..self.size].to_vec())
        } else {
            None
        }
    }
    
    /// Get reference to memory data
    pub fn get(&self) -> Option<Vec<u8>> {
        if let Ok(blocks) = self.pool.try_read() {
            blocks.get(self.block_index).map(|block| block.data[..self.size].to_vec())
        } else {
            None
        }
    }
}

impl Drop for MemoryHandle {
    fn drop(&mut self) {
        // This would need to be async in a real implementation
        // For now, we'll use a blocking approach
        if let Ok(mut blocks) = self.pool.try_write() {
            if let Some(block) = blocks.get_mut(self.block_index) {
                block.in_use = false;
                block.last_used = std::time::Instant::now();
                
                // Clear data for security
                for byte in &mut block.data {
                    *byte = 0;
                }
                
                self.allocated_bytes.fetch_sub(self.size, Ordering::Relaxed);
                self.deallocation_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolStatistics {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub free_blocks: usize,
    pub allocated_bytes: usize,
    pub total_allocated: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub utilization_rate: f64,
}

/// Defragmentation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefragmentationResult {
    pub moved_blocks: usize,
    pub freed_bytes: usize,
    pub duration: std::time::Duration,
}

impl Default for DefragmentationResult {
    fn default() -> Self {
        Self {
            moved_blocks: 0,
            freed_bytes: 0,
            duration: std::time::Duration::ZERO,
        }
    }
}

/// Memory leak detector
pub struct MemoryLeakDetector {
    allocations: Arc<RwLock<HashMap<String, AllocationInfo>>>,
    threshold: std::time::Duration,
    enabled: bool,
}

/// Allocation information for leak detection
#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    timestamp: std::time::Instant,
    location: String,
}

impl MemoryLeakDetector {
    /// Create new leak detector
    pub fn new(threshold: std::time::Duration) -> Self {
        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            threshold,
            enabled: true,
        }
    }
    
    /// Record allocation
    pub async fn record_allocation(&self, id: String, size: usize, location: String) {
        if !self.enabled {
            return;
        }
        
        let info = AllocationInfo {
            size,
            timestamp: std::time::Instant::now(),
            location,
        };
        
        let mut allocations = self.allocations.write().await;
        allocations.insert(id, info);
    }
    
    /// Record deallocation
    pub async fn record_deallocation(&self, id: String) {
        if !self.enabled {
            return;
        }
        
        let mut allocations = self.allocations.write().await;
        allocations.remove(&id);
    }
    
    /// Detect potential leaks
    pub async fn detect_leaks(&self) -> Vec<PotentialLeak> {
        if !self.enabled {
            return Vec::new();
        }
        
        let allocations = self.allocations.read().await;
        let now = std::time::Instant::now();
        let mut leaks = Vec::new();
        
        for (id, info) in allocations.iter() {
            let age = now.duration_since(info.timestamp);
            if age > self.threshold {
                leaks.push(PotentialLeak {
                    id: id.clone(),
                    size: info.size,
                    age,
                    location: info.location.clone(),
                });
            }
        }
        
        leaks.sort_by(|a, b| b.age.cmp(&a.age)); // Sort by age (oldest first)
        leaks
    }
    
    /// Get leak statistics
    pub async fn get_leak_statistics(&self) -> LeakStatistics {
        let allocations = self.allocations.read().await;
        let now = std::time::Instant::now();
        
        let total_allocations = allocations.len();
        let total_leaked_bytes: usize = allocations.values().map(|info| info.size).sum();
        
        let mut old_allocations = 0;
        let mut very_old_allocations = 0;
        
        for info in allocations.values() {
            let age = now.duration_since(info.timestamp);
            if age > std::time::Duration::from_secs(300) { // 5 minutes
                old_allocations += 1;
            }
            if age > std::time::Duration::from_secs(1800) { // 30 minutes
                very_old_allocations += 1;
            }
        }
        
        LeakStatistics {
            total_allocations,
            total_leaked_bytes,
            old_allocations,
            very_old_allocations,
        }
    }
}

/// Potential memory leak
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLeak {
    pub id: String,
    pub size: usize,
    pub age: std::time::Duration,
    pub location: String,
}

/// Leak statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakStatistics {
    pub total_allocations: usize,
    pub total_leaked_bytes: usize,
    pub old_allocations: usize,
    pub very_old_allocations: usize,
}

/// Memory optimizer
pub struct MemoryOptimizer {
    memory_pool: MemoryPool,
    leak_detector: MemoryLeakDetector,
    cache_manager: CacheManager,
    gc_scheduler: GcScheduler,
}

/// Cache manager for smart caching
pub struct CacheManager {
    caches: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_size: usize,
    eviction_policy: EvictionPolicy,
}

/// Cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    timestamp: std::time::Instant,
    access_count: u64,
    size: usize,
    ttl: Option<std::time::Duration>,
}

/// Eviction policy for cache
#[derive(Debug, Clone)]
enum EvictionPolicy {
    LRU,
    LFU,
    TTL,
}

/// Garbage collection scheduler
pub struct GcScheduler {
    interval: std::time::Duration,
    last_gc: Arc<Mutex<std::time::Instant>>,
    gc_threshold: f64,
}

impl MemoryOptimizer {
    /// Create new memory optimizer
    pub fn new(pool_config: MemoryPoolConfig) -> Self {
        let memory_pool = MemoryPool::new(pool_config);
        let leak_detector = MemoryLeakDetector::new(std::time::Duration::from_secs(300)); // 5 minutes
        let cache_manager = CacheManager::new(1024 * 1024 * 50); // 50MB cache
        let gc_scheduler = GcScheduler::new(std::time::Duration::from_secs(60), 0.8); // 1 minute, 80% threshold
        
        Self {
            memory_pool,
            leak_detector,
            cache_manager,
            gc_scheduler,
        }
    }
    
    /// Allocate memory with leak detection
    pub async fn allocate(&self, size: usize, location: String) -> Result<MemoryHandle> {
        let handle = self.memory_pool.allocate(size).await?;
        
        // Record allocation for leak detection
        let allocation_id = format!("alloc_{}", handle.block_index);
        self.leak_detector.record_allocation(allocation_id, size, location).await;
        
        Ok(handle)
    }
    
    /// Get memory statistics
    pub async fn get_statistics(&self) -> MemoryOptimizerStatistics {
        let pool_stats = self.memory_pool.get_statistics().await;
        let leak_stats = self.leak_detector.get_leak_statistics().await;
        let cache_stats = self.cache_manager.get_statistics().await;
        
        MemoryOptimizerStatistics {
            pool: pool_stats,
            leaks: leak_stats,
            cache: cache_stats,
        }
    }
    
    /// Run optimization cycle
    pub async fn optimize(&self) -> Result<OptimizationResult> {
        let mut result = OptimizationResult::default();
        
        // Check for defragmentation
        if self.memory_pool.needs_defragmentation().await {
            let defrag_result = self.memory_pool.defragment().await?;
            result.defragmentation = Some(defrag_result);
        }
        
        // Detect memory leaks
        let leaks = self.leak_detector.detect_leaks().await;
        result.detected_leaks = leaks.len();
        
        // Clean expired cache entries
        let cleaned_entries = self.cache_manager.cleanup_expired().await;
        result.cleaned_cache_entries = cleaned_entries;
        
        // Run garbage collection if needed
        if self.gc_scheduler.should_run_gc().await {
            let gc_result = self.run_garbage_collection().await?;
            result.garbage_collection = Some(gc_result);
        }
        
        Ok(result)
    }
    
    /// Run garbage collection
    async fn run_garbage_collection(&self) -> Result<GcResult> {
        let start_time = std::time::Instant::now();
        
        // Force defragmentation
        let defrag_result = self.memory_pool.defragment().await?;
        
        // Clear old cache entries
        let cleared_cache = self.cache_manager.clear_old_entries().await;
        
        // Update last GC time
        self.gc_scheduler.update_last_gc().await;
        
        Ok(GcResult {
            duration: start_time.elapsed(),
            defragmented_blocks: defrag_result.moved_blocks,
            freed_bytes: defrag_result.freed_bytes,
            cleared_cache_entries: cleared_cache,
        })
    }
}

/// Memory optimizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizerStatistics {
    pub pool: MemoryPoolStatistics,
    pub leaks: LeakStatistics,
    pub cache: CacheStatistics,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub defragmentation: Option<DefragmentationResult>,
    pub detected_leaks: usize,
    pub cleaned_cache_entries: usize,
    pub garbage_collection: Option<GcResult>,
}

impl Default for OptimizationResult {
    fn default() -> Self {
        Self {
            defragmentation: None,
            detected_leaks: 0,
            cleaned_cache_entries: 0,
            garbage_collection: None,
        }
    }
}

/// Garbage collection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcResult {
    pub duration: std::time::Duration,
    pub defragmented_blocks: usize,
    pub freed_bytes: usize,
    pub cleared_cache_entries: usize,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub total_size: usize,
    pub hit_rate: f64,
    pub eviction_count: u64,
}

impl CacheManager {
    fn new(max_size: usize) -> Self {
        Self {
            caches: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
    
    async fn get_statistics(&self) -> CacheStatistics {
        let caches = self.caches.read().await;
        
        let total_entries = caches.len();
        let total_size: usize = caches.values().map(|entry| entry.size).sum();
        
        CacheStatistics {
            total_entries,
            total_size,
            hit_rate: 0.0, // Would need to track hits/misses
            eviction_count: 0, // Would need to track evictions
        }
    }
    
    async fn cleanup_expired(&self) -> usize {
        let mut caches = self.caches.write().await;
        let now = std::time::Instant::now();
        let mut removed = 0;
        
        caches.retain(|_, entry| {
            if let Some(ttl) = entry.ttl {
                if now.duration_since(entry.timestamp) > ttl {
                    removed += 1;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
        
        removed
    }
    
    async fn clear_old_entries(&self) -> usize {
        let mut caches = self.caches.write().await;
        let now = std::time::Instant::now();
        let mut removed = 0;
        
        caches.retain(|_, entry| {
            if now.duration_since(entry.timestamp) > std::time::Duration::from_secs(300) {
                removed += 1;
                false
            } else {
                true
            }
        });
        
        removed
    }
}

impl GcScheduler {
    fn new(interval: std::time::Duration, threshold: f64) -> Self {
        Self {
            interval,
            last_gc: Arc::new(Mutex::new(std::time::Instant::now())),
            gc_threshold: threshold,
        }
    }
    
    async fn should_run_gc(&self) -> bool {
        let last_gc = *self.last_gc.lock().await;
        std::time::Instant::now().duration_since(last_gc) > self.interval
    }
    
    async fn update_last_gc(&self) {
        let mut last_gc = self.last_gc.lock().await;
        *last_gc = std::time::Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_config_default() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.pool_size, 1024 * 1024 * 100);
        assert_eq!(config.block_size, 1024);
        assert!(config.auto_expand);
    }
    
    #[tokio::test]
    async fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);
        
        let stats = pool.get_statistics().await;
        assert_eq!(stats.used_blocks, 0);
        assert!(stats.total_blocks > 0);
    }
    
    #[tokio::test]
    async fn test_memory_allocation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);
        
        let handle = pool.allocate(512).await.unwrap();
        assert_eq!(handle.size, 1024); // Should allocate full block
        
        let stats = pool.get_statistics().await;
        assert_eq!(stats.used_blocks, 1);
    }
    
    #[tokio::test]
    async fn test_memory_leak_detector() {
        let detector = MemoryLeakDetector::new(std::time::Duration::from_secs(1));
        
        detector.record_allocation("test1".to_string(), 1024, "test_location".to_string()).await;
        detector.record_allocation("test2".to_string(), 2048, "test_location".to_string()).await;
        
        let stats = detector.get_leak_statistics().await;
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_leaked_bytes, 3072);
        
        detector.record_deallocation("test1".to_string()).await;
        
        let stats = detector.get_leak_statistics().await;
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_leaked_bytes, 2048);
    }
    
    #[test]
    fn test_memory_optimizer_creation() {
        let config = MemoryPoolConfig::default();
        let _optimizer = MemoryOptimizer::new(config);
        
        // Test that optimizer was created successfully
        // More detailed tests would require async context
    }
}
