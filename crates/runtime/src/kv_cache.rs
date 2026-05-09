//! KV Cache
//! 
//! Transformer key-value cache untuk efficient inference.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use chrono::{DateTime, Utc};

use crate::{Result, InferenceError};

/// Configuration untuk KV cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size (bytes)
    pub max_size_bytes: usize,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression level (1-9)
    pub compression_level: u32,
    /// Cache shard count untuk parallel access
    pub shard_count: usize,
    /// TTL untuk cache entries (seconds)
    pub ttl_seconds: Option<u64>,
    /// Enable statistics
    pub enable_stats: bool,
}

/// Eviction policy untuk cache
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First-In-First-Out
    FIFO,
    /// Random eviction
    Random,
    /// No eviction (fail when full)
    None,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            eviction_policy: EvictionPolicy::LRU,
            enable_compression: false,
            compression_level: 6,
            shard_count: 16,
            ttl_seconds: None,
            enable_stats: true,
        }
    }
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Entry key
    pub key: String,
    /// Key tensor data
    pub key_tensor: Vec<f32>,
    /// Value tensor data
    pub value_tensor: Vec<f32>,
    /// Sequence length
    pub sequence_length: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
    /// Entry size (bytes)
    pub size_bytes: usize,
    /// TTL expiration (if any)
    pub expires_at: Option<DateTime<Utc>>,
    /// Entry metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CacheEntry {
    /// Calculate entry size
    fn calculate_size(&self) -> usize {
        let key_size = self.key_tensor.len() * std::mem::size_of::<f32>();
        let value_size = self.value_tensor.len() * std::mem::size_of::<f32>();
        key_size + value_size + self.key.len()
    }
    
    /// Check if entry is expired
    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }
    
    /// Update access information
    fn update_access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total entries
    pub total_entries: usize,
    /// Current cache size (bytes)
    pub current_size_bytes: usize,
    /// Maximum cache size (bytes)
    pub max_size_bytes: usize,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Cache evictions
    pub evictions: u64,
    /// Hit rate
    pub hit_rate: f64,
    /// Average access time (microseconds)
    pub avg_access_time_us: f64,
    /// Total access time (microseconds)
    pub total_access_time_us: u64,
    /// Expired entries cleaned
    pub expired_cleaned: u64,
    /// Compression ratio (if enabled)
    pub compression_ratio: Option<f64>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// KV Cache implementation
pub struct KVCache {
    /// Cache configuration
    config: CacheConfig,
    /// Cache shards for parallel access
    shards: Vec<Arc<RwLock<CacheShard>>>,
    /// Global statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Cache state
    state: Arc<RwLock<CacheState>>,
}

/// Individual cache shard
#[derive(Debug)]
struct CacheShard {
    /// Cache entries
    entries: HashMap<String, CacheEntry>,
    /// LRU tracking (for LRU eviction)
    lru_order: Vec<String>,
    /// Current shard size
    current_size_bytes: usize,
    /// Shard statistics
    stats: ShardStats,
}

/// Shard statistics
#[derive(Debug, Clone, Default)]
struct ShardStats {
    entries: usize,
    size_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Cache state
#[derive(Debug, Clone, PartialEq)]
pub enum CacheState {
    /// Cache tidak diinisialisasi
    Uninitialized,
    /// Cache sedang diinisialisasi
    Initializing,
    /// Cache siap
    Ready,
    /// Cache sedang shutdown
    ShuttingDown,
    /// Cache sudah shutdown
    Shutdown,
}

impl KVCache {
    /// Create new KV cache
    pub fn new(max_size_bytes: usize) -> Self {
        let config = CacheConfig {
            max_size_bytes,
            ..Default::default()
        };
        Self::with_config(config)
    }
    
    /// Create KV cache with configuration
    pub fn with_config(config: CacheConfig) -> Self {
        let mut shards = Vec::with_capacity(config.shard_count);
        for _ in 0..config.shard_count {
            shards.push(Arc::new(RwLock::new(CacheShard {
                entries: HashMap::new(),
                lru_order: Vec::new(),
                current_size_bytes: 0,
                stats: ShardStats::default(),
            })));
        }
        
        Self {
            config,
            shards,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            state: Arc::new(RwLock::new(CacheState::Uninitialized)),
        }
    }
    
    /// Initialize cache
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing KV cache");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = CacheState::Initializing;
        }
        
        // Initialize statistics
        {
            let mut stats = self.stats.write().await;
            stats.max_size_bytes = self.config.max_size_bytes;
            stats.last_updated = Utc::now();
        }
        
        // Update state to ready
        {
            let mut state = self.state.write().await;
            *state = CacheState::Ready;
        }
        
        info!("KV cache initialized successfully");
        Ok(())
    }
    
    /// Get cache entry
    pub async fn get(&self, key: &str) -> Result<Option<CacheEntry>> {
        debug!("Getting cache entry: {}", key);
        
        let start_time = std::time::Instant::now();
        
        // Check cache state
        {
            let state = self.state.read().await;
            if *state != CacheState::Ready {
                return Err(InferenceError::CacheError("Cache not ready".to_string()).into());
            }
        }
        
        // Get shard for this key
        let shard_index = self.get_shard_index(key);
        let mut shard = self.shards[shard_index].write().await;
        
        let result = if let Some(entry) = shard.entries.get(key) {
            // Check if expired
            if entry.is_expired() {
                let entry_size = entry.size_bytes;
                let key_owned = key.to_string(); // Clone key for removal
                drop(entry); // Drop the borrow before modifying shard
                shard.entries.remove(&key_owned);
                shard.lru_order.retain(|k| k != &key_owned);
                shard.current_size_bytes -= entry_size;
                
                // Update statistics
                {
                    let mut stats = self.stats.write().await;
                    stats.misses += 1;
                    stats.expired_cleaned += 1;
                    self.update_hit_rate(&mut stats);
                }
                
                return Ok(None);
            }
            
            // Cache hit - update access info and LRU order
            let mut entry = entry.clone();
            entry.update_access();
            
            // Update LRU order - move key to front (more efficient)
            shard.lru_order.retain(|k| k != key);
            shard.lru_order.insert(0, key.to_string());
            
            // Update statistics
            shard.stats.hits += 1;
            {
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                self.update_hit_rate(&mut stats);
            }
            
            Some(entry)
        } else {
            // Cache miss
            shard.stats.misses += 1;
            {
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                self.update_hit_rate(&mut stats);
            }
            
            None
        };
        
        // Update access time statistics
        let access_time_us = start_time.elapsed().as_micros() as u64;
        {
            let mut stats = self.stats.write().await;
            stats.total_access_time_us += access_time_us;
            let total_accesses = stats.hits + stats.misses;
            if total_accesses > 0 {
                stats.avg_access_time_us = stats.total_access_time_us as f64 / total_accesses as f64;
            }
        }
        
        debug!("Cache {} for key: {}", if result.is_some() { "hit" } else { "miss" }, key);
        Ok(result)
    }
    
    /// Put cache entry
    pub async fn put(&self, key: String, entry: CacheEntry) -> Result<()> {
        debug!("Putting cache entry: {}", key);
        
        // Check cache state
        {
            let state = self.state.read().await;
            if *state != CacheState::Ready {
                return Err(InferenceError::CacheError("Cache not ready".to_string()).into());
            }
        }
        
        // Calculate entry size
        let size_bytes = entry.calculate_size();
        
        // Get shard for this key
        let shard_index = self.get_shard_index(&key);
        let mut shard = self.shards[shard_index].write().await;
        
        // Check if entry already exists
        if shard.entries.contains_key(&key) {
            // Remove old entry
            if let Some(old_entry) = shard.entries.remove(&key) {
                shard.current_size_bytes -= old_entry.size_bytes;
                shard.lru_order.retain(|k| k != &key);
            }
        }
        
        // Check if we need to evict entries
        while shard.current_size_bytes + size_bytes > self.config.max_size_bytes / self.config.shard_count {
            if !self.evict_from_shard(&mut shard).await? {
                break; // Cannot evict more entries
            }
        }
        
        // Add new entry
        let mut new_entry = entry;
        new_entry.size_bytes = size_bytes;
        new_entry.key = key.clone();
        
        // Set expiration if TTL is configured
        if let Some(ttl_seconds) = self.config.ttl_seconds {
            new_entry.expires_at = Some(Utc::now() + chrono::Duration::seconds(ttl_seconds as i64));
        }
        
        shard.entries.insert(key.clone(), new_entry.clone());
        shard.lru_order.push(key.clone());
        shard.current_size_bytes += size_bytes;
        shard.stats.entries += 1;
        
        // Update global statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_entries += 1;
            stats.current_size_bytes += size_bytes;
            stats.last_updated = Utc::now();
        }
        
        debug!("Cache entry stored successfully: {}", key);
        Ok(())
    }
    
    /// Remove cache entry
    pub async fn remove(&self, key: &str) -> Result<bool> {
        debug!("Removing cache entry: {}", key);
        
        let shard_index = self.get_shard_index(key);
        let mut shard = self.shards[shard_index].write().await;
        
        if let Some(entry) = shard.entries.remove(key) {
            shard.lru_order.retain(|k| k != &key);
            shard.current_size_bytes -= entry.size_bytes;
            shard.stats.entries -= 1;
            
            // Update global statistics
            {
                let mut stats = self.stats.write().await;
                stats.total_entries -= 1;
                stats.current_size_bytes -= entry.size_bytes;
            }
            
            debug!("Cache entry removed successfully: {}", key);
            Ok(true)
        } else {
            debug!("Cache entry not found for removal: {}", key);
            Ok(false)
        }
    }
    
    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        info!("Clearing all cache entries");
        
        for shard in &self.shards {
            let mut shard_guard = shard.write().await;
            shard_guard.entries.clear();
            shard_guard.lru_order.clear();
            shard_guard.current_size_bytes = 0;
            shard_guard.stats = ShardStats::default();
        }
        
        // Reset global statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_entries = 0;
            stats.current_size_bytes = 0;
            stats.hits = 0;
            stats.misses = 0;
            stats.evictions = 0;
            stats.hit_rate = 0.0;
            stats.expired_cleaned = 0;
            stats.last_updated = Utc::now();
        }
        
        info!("Cache cleared successfully");
        Ok(())
    }
    
    /// Clean expired entries
    pub async fn clean_expired(&self) -> Result<usize> {
        debug!("Cleaning expired cache entries");
        
        let mut total_cleaned = 0;
        let now = Utc::now();
        
        for shard in &self.shards {
            let mut shard_guard = shard.write().await;
            let mut expired_keys = Vec::new();
            
            for (key, entry) in shard_guard.entries.iter() {
                if let Some(expires_at) = entry.expires_at {
                    if now > expires_at {
                        expired_keys.push(key.clone());
                    }
                }
            }
            
            for key in expired_keys {
                if let Some(entry) = shard_guard.entries.remove(&key) {
                    shard_guard.lru_order.retain(|k| k != &key);
                    shard_guard.current_size_bytes -= entry.size_bytes;
                    shard_guard.stats.entries -= 1;
                    total_cleaned += 1;
                }
            }
        }
        
        // Update global statistics
        if total_cleaned > 0 {
            let mut stats = self.stats.write().await;
            stats.total_entries -= total_cleaned;
            stats.expired_cleaned += total_cleaned as u64;
            stats.last_updated = Utc::now();
        }
        
        debug!("Cleaned {} expired cache entries", total_cleaned);
        Ok(total_cleaned)
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        
        // Aggregate shard statistics
        let mut total_entries = 0;
        let mut total_size = 0;
        
        for shard in &self.shards {
            let shard_guard = shard.read().await;
            total_entries += shard_guard.stats.entries;
            total_size += shard_guard.stats.size_bytes;
        }
        
        stats.total_entries = total_entries;
        stats.current_size_bytes = total_size;
        
        stats
    }
    
    /// Shutdown cache
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down KV cache");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = CacheState::ShuttingDown;
        }
        
        // Clear all entries
        self.clear().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = CacheState::Shutdown;
        }
        
        info!("KV cache shutdown complete");
        Ok(())
    }
    
    /// Get shard index for key
    fn get_shard_index(&self, key: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.config.shard_count
    }
    
    /// Evict entry from shard based on policy
    async fn evict_from_shard(&self, shard: &mut CacheShard) -> Result<bool> {
        if shard.entries.is_empty() {
            return Ok(false);
        }
        
        let key_to_remove = match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                // Remove least recently used
                shard.lru_order.first().cloned()
            }
            EvictionPolicy::LFU => {
                // Remove least frequently used
                shard.entries.iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(key, _)| key.clone())
            }
            EvictionPolicy::FIFO => {
                // Remove oldest entry
                shard.entries.iter()
                    .min_by_key(|(_, entry)| entry.created_at)
                    .map(|(key, _)| key.clone())
            }
            EvictionPolicy::Random => {
                // Remove random entry
                let keys: Vec<String> = shard.entries.keys().cloned().collect();
                if !keys.is_empty() {
                    let index = (rand::random::<usize>() % keys.len()) as usize;
                    Some(keys[index].clone())
                } else {
                    None
                }
            }
            EvictionPolicy::None => {
                return Err(InferenceError::CacheError("Cache full and eviction disabled".to_string()).into());
            }
        };
        
        if let Some(key) = key_to_remove {
            if let Some(entry) = shard.entries.remove(&key) {
                shard.lru_order.retain(|k| k != &key);
                shard.current_size_bytes -= entry.size_bytes;
                shard.stats.entries -= 1;
                shard.stats.evictions += 1;
                
                // Update global statistics
                {
                    let mut stats = self.stats.write().await;
                    stats.total_entries -= 1;
                    stats.current_size_bytes -= entry.size_bytes;
                    stats.evictions += 1;
                }
                
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Update hit rate
    fn update_hit_rate(&self, stats: &mut CacheStats) {
        let total_requests = stats.hits + stats.misses;
        if total_requests > 0 {
            stats.hit_rate = stats.hits as f64 / total_requests as f64;
        }
    }
}

impl CacheEntry {
    /// Create new cache entry
    pub fn new(
        key_tensor: Vec<f32>,
        value_tensor: Vec<f32>,
        sequence_length: usize,
        num_layers: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Self {
        let now = Utc::now();
        Self {
            key: String::new(),
            key_tensor,
            value_tensor,
            sequence_length,
            num_layers,
            head_dim,
            num_heads,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            size_bytes: 0,
            expires_at: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}
