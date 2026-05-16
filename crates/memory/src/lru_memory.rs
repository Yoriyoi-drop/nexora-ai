//! Memory Management dengan LRU Cache
//! 
//! Implementasi memory management dengan LRU (Least Recently Used) cache
//! untuk optimalisasi memory usage dan performance

use nexora_core::{MemoryLayer, CoreError, CoreResult};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use parking_lot::RwLock as ParkingRwLock;
use std::time::{Duration, Instant};
use tracing::{debug, info};
use serde::{Deserialize, Serialize};

/// Memory entry dengan metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub layer: MemoryLayer,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u64,
    pub size_bytes: usize,
    pub ttl_ms: Option<u64>, // Time to live
    pub tags: Vec<String>,
}

impl MemoryEntry {
    pub fn new(key: String, value: String, layer: MemoryLayer) -> Self {
        let size_bytes = key.len() + value.len();
        let now = Self::current_timestamp();
        
        Self {
            key,
            value,
            layer,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            ttl_ms: None,
            tags: Vec::new(),
        }
    }
    
    pub fn with_ttl(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = Some(ttl_ms);
        self
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn touch(&mut self) {
        self.last_accessed = Self::current_timestamp();
        self.access_count += 1;
    }
    
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl_ms {
            let elapsed = Self::current_timestamp() - self.created_at;
            elapsed > ttl
        } else {
            false
        }
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// LRU Cache node
#[derive(Debug, Clone)]
struct LruNode {
    key: String,
    prev: Option<String>,
    next: Option<String>,
}

/// LRU Cache implementation
pub struct LruCache {
    capacity: usize,
    max_size_bytes: usize,
    entries: ParkingRwLock<HashMap<String, MemoryEntry>>,
    access_order: ParkingRwLock<VecDeque<String>>,
    current_size_bytes: ParkingRwLock<usize>,
    stats: ParkingRwLock<CacheStats>,
}

impl LruCache {
    pub fn new(capacity: usize, max_size_bytes: usize) -> Self {
        Self {
            capacity,
            max_size_bytes,
            entries: ParkingRwLock::new(HashMap::new()),
            access_order: ParkingRwLock::new(VecDeque::new()),
            current_size_bytes: ParkingRwLock::new(0),
            stats: ParkingRwLock::new(CacheStats::default()),
        }
    }
    
    /// Get value dari cache
    pub fn get(&self, key: &str) -> Option<MemoryEntry> {
        let mut entries = self.entries.write();
        
        if let Some(entry) = entries.get_mut(key) {
            if entry.is_expired() {
                // Remove expired entry
                let size_bytes = entry.size_bytes;
                let _ = entry;
                entries.remove(key);
                self.remove_from_access_order(key);
                self.update_size_bytes(-(size_bytes as isize));
                
                let mut stats = self.stats.write();
                stats.misses += 1;
                stats.expired += 1;
                
                return None;
            }
            
            // Update access
            entry.touch();
            let entry_clone = entry.clone();
            let _ = entry;
            self.move_to_back(key);
            
            let mut stats = self.stats.write();
            stats.hits += 1;
            
            Some(entry_clone)
        } else {
            let mut stats = self.stats.write();
            stats.misses += 1;
            None
        }
    }
    
    /// Put value ke cache
    pub fn put(&self, entry: MemoryEntry) -> CoreResult<()> {
        let key = entry.key.clone();
        
        // Check if entry already exists
        let mut entries = self.entries.write();
        let existing_size = if let Some(existing) = entries.get(&key) {
            existing.size_bytes
        } else {
            0
        };
        
        // Check capacity constraints
        let size_diff = entry.size_bytes as isize - existing_size as isize;
        let new_total_size = (*self.current_size_bytes.read() as isize + size_diff) as usize;
        
        if new_total_size > self.max_size_bytes as usize {
            // Need to evict some entries
            self.evict_until_fit(entry.size_bytes)?;
        }
        
        if entries.len() >= self.capacity && !entries.contains_key(&key) {
            // Evict LRU entry
            self.evict_lru()?;
        }
        
        // Insert or update entry
        entries.insert(key.clone(), entry);
        
        if existing_size == 0 {
            // New entry, add to access order
            self.add_to_back(&key);
        } else {
            // Existing entry, move to back
            self.move_to_back(&key);
        }
        
        self.update_size_bytes(size_diff);
        
        let mut stats = self.stats.write();
        stats.inserts += 1;
        if existing_size > 0 {
            stats.updates += 1;
        }
        
        Ok(())
    }
    
    /// Remove entry dari cache
    pub fn remove(&self, key: &str) -> Option<MemoryEntry> {
        let mut entries = self.entries.write();
        
        if let Some(entry) = entries.remove(key) {
            self.remove_from_access_order(key);
            self.update_size_bytes(-(entry.size_bytes as isize));
            
            let mut stats = self.stats.write();
            stats.removals += 1;
            
            Some(entry)
        } else {
            None
        }
    }
    
    /// Clear cache
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        let size = entries.len();
        entries.clear();
        
        let mut access_order = self.access_order.write();
        access_order.clear();
        
        *self.current_size_bytes.write() = 0;
        
        let mut stats = self.stats.write();
        stats.clears += 1;
        
        info!("Cache cleared: {} entries removed", size);
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read();
        let entries = self.entries.read();
        
        CacheStats {
            entries: entries.len(),
            size_bytes: *self.current_size_bytes.read(),
            hits: stats.hits,
            misses: stats.misses,
            inserts: stats.inserts,
            updates: stats.updates,
            removals: stats.removals,
            clears: stats.clears,
            expired: stats.expired,
            evictions: stats.evictions,
            hit_rate: if stats.hits + stats.misses > 0 {
                stats.hits as f64 / (stats.hits + stats.misses) as f64
            } else {
                0.0
            },
        }
    }
    
    /// Cleanup expired entries
    pub fn cleanup_expired(&self) -> usize {
        let mut entries = self.entries.write();
        let mut expired_keys = Vec::new();
        
        for (key, entry) in entries.iter() {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }
        
        for key in &expired_keys {
            if let Some(entry) = entries.remove(key) {
                self.remove_from_access_order(key);
                self.update_size_bytes(-(entry.size_bytes as isize));
            }
        }
        
        if !expired_keys.is_empty() {
            let mut stats = self.stats.write();
            stats.expired += expired_keys.len() as u64;
        }
        
        expired_keys.len()
    }
    
    // Private methods
    
    fn evict_until_fit(&self, required_size: usize) -> CoreResult<()> {
        let current_size = *self.current_size_bytes.read();
        let target_size = self.max_size_bytes - required_size;
        
        while current_size > target_size {
            self.evict_lru()?;
            let new_size = *self.current_size_bytes.read();
            if new_size == current_size {
                break; // Can't evict more
            }
            break;
        }
        
        Ok(())
    }
    
    fn evict_lru(&self) -> CoreResult<()> {
        let mut access_order = self.access_order.write();
        let mut entries = self.entries.write();
        
        if let Some(key) = access_order.pop_front() {
            if let Some(entry) = entries.remove(&key) {
                self.update_size_bytes(-(entry.size_bytes as isize));
                
                let mut stats = self.stats.write();
                stats.evictions += 1;
                
                debug!("Evicted LRU entry: {} ({} bytes)", key, entry.size_bytes);
                Ok(())
            } else {
                Err(CoreError::MemoryAccess("Failed to evict LRU entry".to_string()))
            }
        } else {
            Err(CoreError::MemoryAccess("No entries to evict".to_string()))
        }
    }
    
    fn add_to_back(&self, key: &str) {
        let mut access_order = self.access_order.write();
        access_order.push_back(key.to_string());
    }
    
    fn move_to_back(&self, key: &str) {
        let mut access_order = self.access_order.write();
        
        // Remove from current position
        access_order.retain(|k| k != key);
        
        // Add to back
        access_order.push_back(key.to_string());
    }
    
    fn remove_from_access_order(&self, key: &str) {
        let mut access_order = self.access_order.write();
        access_order.retain(|k| k != key);
    }
    
    fn update_size_bytes(&self, delta: isize) {
        let mut size = self.current_size_bytes.write();
        if delta >= 0 {
            *size = size.saturating_add(delta as usize);
        } else {
            *size = size.saturating_sub((-delta) as usize);
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entries: usize,
    pub size_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub updates: u64,
    pub removals: u64,
    pub clears: u64,
    pub expired: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

/// Enhanced memory manager dengan LRU cache
pub struct LruMemoryManager {
    short_term_cache: Arc<LruCache>,
    session_cache: Arc<LruCache>,
    long_term_cache: Arc<LruCache>,
    knowledge_cache: Arc<LruCache>,
    global_stats: Arc<RwLock<MemoryStats>>,
    cleanup_interval: Duration,
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
}

/// Memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub total_hits: u64,
    pub total_misses: u64,
    pub layer_stats: HashMap<MemoryLayer, CacheStats>,
    pub last_cleanup: Option<Instant>,
}

impl LruMemoryManager {
    pub fn new() -> Self {
        Self::with_config(MemoryConfig::default())
    }
    
    pub fn with_config(config: MemoryConfig) -> Self {
        let short_term_cache = Arc::new(LruCache::new(
            config.short_term_capacity,
            config.short_term_max_size
        ));
        
        let session_cache = Arc::new(LruCache::new(
            config.session_capacity,
            config.session_max_size
        ));
        
        let long_term_cache = Arc::new(LruCache::new(
            config.long_term_capacity,
            config.long_term_max_size
        ));
        
        let knowledge_cache = Arc::new(LruCache::new(
            config.knowledge_capacity,
            config.knowledge_max_size
        ));
        
        Self {
            short_term_cache,
            session_cache,
            long_term_cache,
            knowledge_cache,
            global_stats: Arc::new(RwLock::new(MemoryStats::default())),
            cleanup_interval: config.cleanup_interval,
            cleanup_task: None,
        }
    }
    
    /// Start background cleanup task
    pub async fn start_cleanup_task(&mut self) {
        if self.cleanup_task.is_some() {
            return;
        }
        
        let short_term = Arc::clone(&self.short_term_cache);
        let session = Arc::clone(&self.session_cache);
        let long_term = Arc::clone(&self.long_term_cache);
        let knowledge = Arc::clone(&self.knowledge_cache);
        let global_stats = Arc::clone(&self.global_stats);
        let interval = self.cleanup_interval;
        
        let task = tokio::spawn(async move {
            let mut cleanup_interval = tokio::time::interval(interval);
            
            loop {
                cleanup_interval.tick().await;
                
                debug!("Starting memory cleanup");
                
                // Cleanup expired entries in all caches
                let short_expired = short_term.cleanup_expired();
                let session_expired = session.cleanup_expired();
                let long_expired = long_term.cleanup_expired();
                let knowledge_expired = knowledge.cleanup_expired();
                
                let total_expired = short_expired + session_expired + long_expired + knowledge_expired;
                
                if total_expired > 0 {
                    info!("Cleaned up {} expired memory entries", total_expired);
                }
                
                // Update global stats
                let mut stats = global_stats.write().await;
                stats.last_cleanup = Some(Instant::now());
                
                debug!("Memory cleanup completed");
            }
        });
        
        self.cleanup_task = Some(task);
        info!("Memory cleanup task started");
    }
    
    /// Store data ke specific memory layer
    pub async fn store(&self, layer: MemoryLayer, key: String, value: String) -> CoreResult<()> {
        let entry = MemoryEntry::new(key.clone(), value, layer);
        self.store_entry(entry).await
    }
    
    /// Store dengan TTL
    pub async fn store_with_ttl(&self, layer: MemoryLayer, key: String, value: String, ttl_ms: u64) -> CoreResult<()> {
        let entry = MemoryEntry::new(key.clone(), value, layer).with_ttl(ttl_ms);
        self.store_entry(entry).await
    }
    
    /// Store dengan tags
    pub async fn store_with_tags(&self, layer: MemoryLayer, key: String, value: String, tags: Vec<String>) -> CoreResult<()> {
        let entry = MemoryEntry::new(key.clone(), value, layer).with_tags(tags);
        self.store_entry(entry).await
    }
    
    async fn store_entry(&self, entry: MemoryEntry) -> CoreResult<()> {
        let cache = self.get_cache_for_layer(entry.layer);
        cache.put(entry)?;
        
        self.update_global_stats().await;
        Ok(())
    }
    
    /// Retrieve data dari memory layer
    pub async fn retrieve(&self, layer: MemoryLayer, key: &str) -> Option<String> {
        let cache = self.get_cache_for_layer(layer);
        cache.get(key).map(|entry| entry.value)
    }
    
    /// Retrieve dengan metadata
    pub async fn retrieve_with_metadata(&self, layer: MemoryLayer, key: &str) -> Option<MemoryEntry> {
        let cache = self.get_cache_for_layer(layer);
        cache.get(key)
    }
    
    /// Search entries by tags
    pub async fn search_by_tags(&self, layer: MemoryLayer, tags: &[String]) -> Vec<MemoryEntry> {
        let cache = self.get_cache_for_layer(layer);
        let entries = cache.entries.read();
        
        entries.values()
            .filter(|entry| {
                tags.iter().any(|tag| entry.tags.contains(tag))
            })
            .cloned()
            .collect()
    }
    
    /// Remove entry
    pub async fn remove(&self, layer: MemoryLayer, key: &str) -> Option<String> {
        let cache = self.get_cache_for_layer(layer);
        cache.remove(key).map(|entry| entry.value)
    }
    
    /// Clear specific layer
    pub async fn clear_layer(&self, layer: MemoryLayer) {
        let cache = self.get_cache_for_layer(layer);
        cache.clear();
        self.update_global_stats().await;
    }
    
    /// Clear all layers
    pub async fn clear_all(&self) {
        self.short_term_cache.clear();
        self.session_cache.clear();
        self.long_term_cache.clear();
        self.knowledge_cache.clear();
        self.update_global_stats().await;
    }
    
    /// Get memory statistics
    pub async fn get_stats(&self) -> MemoryStats {
        self.update_global_stats().await;
        self.global_stats.read().await.clone()
    }
    
    /// Get cache statistics for specific layer
    pub fn get_layer_stats(&self, layer: MemoryLayer) -> CacheStats {
        let cache = self.get_cache_for_layer(layer);
        cache.get_stats()
    }
    
    // Private methods
    
    fn get_cache_for_layer(&self, layer: MemoryLayer) -> &LruCache {
        match layer {
            MemoryLayer::Short => &self.short_term_cache,
            MemoryLayer::Session => &self.session_cache,
            MemoryLayer::Long => &self.long_term_cache,
            MemoryLayer::Knowledge => &self.knowledge_cache,
        }
    }
    
    async fn update_global_stats(&self) {
        let mut stats = self.global_stats.write().await;
        
        let short_stats = self.short_term_cache.get_stats();
        let session_stats = self.session_cache.get_stats();
        let long_stats = self.long_term_cache.get_stats();
        let knowledge_stats = self.knowledge_cache.get_stats();
        
        stats.total_entries = short_stats.entries + session_stats.entries + 
                            long_stats.entries + knowledge_stats.entries;
        stats.total_size_bytes = short_stats.size_bytes + session_stats.size_bytes + 
                               long_stats.size_bytes + knowledge_stats.size_bytes;
        stats.total_hits = short_stats.hits + session_stats.hits + 
                          long_stats.hits + knowledge_stats.hits;
        stats.total_misses = short_stats.misses + session_stats.misses + 
                            long_stats.misses + knowledge_stats.misses;
        
        stats.layer_stats.insert(MemoryLayer::Short, short_stats);
        stats.layer_stats.insert(MemoryLayer::Session, session_stats);
        stats.layer_stats.insert(MemoryLayer::Long, long_stats);
        stats.layer_stats.insert(MemoryLayer::Knowledge, knowledge_stats);
    }
}

impl Drop for LruMemoryManager {
    fn drop(&mut self) {
        if let Some(task) = self.cleanup_task.take() {
            task.abort();
        }
    }
}

/// Memory configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub short_term_capacity: usize,
    pub short_term_max_size: usize,
    pub session_capacity: usize,
    pub session_max_size: usize,
    pub long_term_capacity: usize,
    pub long_term_max_size: usize,
    pub knowledge_capacity: usize,
    pub knowledge_max_size: usize,
    pub cleanup_interval: Duration,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_term_capacity: 1000,
            short_term_max_size: 100 * 1024 * 1024, // 100MB
            session_capacity: 5000,
            session_max_size: 500 * 1024 * 1024, // 500MB
            long_term_capacity: 10000,
            long_term_max_size: 1024 * 1024 * 1024, // 1GB
            knowledge_capacity: 50000,
            knowledge_max_size: 2048 * 1024 * 1024, // 2GB
            cleanup_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

impl Default for LruMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_lru_cache_basic() {
        let cache = LruCache::new(3, 1024);
        
        // Insert entries
        let entry1 = MemoryEntry::new("key1".to_string(), "value1".to_string(), MemoryLayer::Short);
        let entry2 = MemoryEntry::new("key2".to_string(), "value2".to_string(), MemoryLayer::Short);
        let entry3 = MemoryEntry::new("key3".to_string(), "value3".to_string(), MemoryLayer::Short);
        
        cache.put(entry1).unwrap();
        cache.put(entry2).unwrap();
        cache.put(entry3).unwrap();
        
        // Get entries
        assert_eq!(cache.get("key1").unwrap().value, "value1");
        assert_eq!(cache.get("key2").unwrap().value, "value2");
        assert_eq!(cache.get("key3").unwrap().value, "value3");
        
        // Insert fourth entry (should evict LRU)
        let entry4 = MemoryEntry::new("key4".to_string(), "value4".to_string(), MemoryLayer::Short);
        cache.put(entry4).unwrap();
        
        // key1 should be evicted (least recently used)
        assert!(cache.get("key1").is_none());
        assert!(cache.get("key4").is_some());
    }
    
    #[tokio::test]
    async fn test_memory_manager() {
        let mut manager = LruMemoryManager::new();
        manager.start_cleanup_task().await;
        
        // Store some data
        manager.store(MemoryLayer::Short, "test1".to_string(), "value1".to_string()).await.unwrap();
        manager.store(MemoryLayer::Session, "test2".to_string(), "value2".to_string()).await.unwrap();
        
        // Retrieve data
        assert_eq!(manager.retrieve(MemoryLayer::Short, "test1").await, Some("value1".to_string()));
        assert_eq!(manager.retrieve(MemoryLayer::Session, "test2").await, Some("value2".to_string()));
        
        // Check stats
        let stats = manager.get_stats().await;
        assert!(stats.total_entries >= 2);
    }
    
    #[tokio::test]
    async fn test_ttl_expiration() {
        let cache = LruCache::new(10, 1024);
        
        // Insert entry with short TTL
        let entry = MemoryEntry::new("ttl_key".to_string(), "ttl_value".to_string(), MemoryLayer::Short)
            .with_ttl(10); // 10ms TTL
        
        cache.put(entry).unwrap();
        
        // Should be available immediately
        assert!(cache.get("ttl_key").is_some());
        
        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        // Should be expired now
        assert!(cache.get("ttl_key").is_none());
    }
}
