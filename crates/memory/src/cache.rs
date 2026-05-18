//! Cache System - LRU Cache dan memory optimization
//! 
//! Implementasi LRU cache untuk fast access dan memory optimization

use std::collections::HashMap;
use std::hash::Hash;
use serde::{Serialize, Deserialize};
use tracing::{debug, trace};

/// LRU (Least Recently Used) Cache implementation
#[derive(Debug)]
pub struct LRUCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, usize)>, // key -> (value, access_order)
    access_counter: usize,
}

impl<K: Clone + Eq + Hash + std::fmt::Debug, V: Clone> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            access_counter: 0,
        }
    }
    
    /// Get value dari cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some((value, _)) = self.map.get(key) {
            let _ = value;
            if let Some((_, counter)) = self.map.get_mut(key) {
                *counter = self.access_counter;
                self.access_counter += 1;
            }
            if let Some((value, _)) = self.map.get(key) {
                trace!("Cache hit for key: {:?}", key);
                Some(value.clone())
            } else {
                None
            }
        } else {
            trace!("Cache miss for key: {:?}", key);
            None
        }
    }
    
    /// Put value ke cache
    pub fn put(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            self.map.insert(key.clone(), (value, self.access_counter));
        } else {
            if self.map.len() >= self.capacity {
                let lru_key = self.map.iter()
                    .min_by_key(|(_, (_, c))| *c)
                    .map(|(k, _)| k.clone());
                if let Some(lru_key) = lru_key {
                    self.map.remove(&lru_key);
                    debug!("Evicted LRU key: {:?}", lru_key);
                }
            }
            self.map.insert(key.clone(), (value, self.access_counter));
        }
        
        self.access_counter += 1;
        trace!("Cache updated, current size: {}", self.map.len());
    }
    
    /// Remove key dari cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some((value, _)) = self.map.remove(key) {
            debug!("Removed key from cache: {:?}", key);
            Some(value)
        } else {
            None
        }
    }
    
    /// Check jika key exists
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }
    
    /// Clear cache
    pub fn clear(&mut self) {
        self.map.clear();
        self.access_counter = 0;
        debug!("Cache cleared");
    }
    
    /// Get cache size
    pub fn len(&self) -> usize {
        self.map.len()
    }
    
    /// Check jika cache kosong
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get keys dalam access order (most recent first)
    pub fn keys(&self) -> Vec<&K> {
        let mut keys: Vec<&K> = self.map.keys().collect();
        keys.sort_by_key(|k| {
            let (_, counter) = &self.map[k];
            std::cmp::Reverse(*counter)
        });
        keys
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            size: self.map.len(),
            capacity: self.capacity,
            utilization: self.map.len() as f32 / self.capacity as f32,
        }
    }
    
    /// Resize cache
    pub fn resize(&mut self, new_capacity: usize) {
        if new_capacity < self.capacity {
            while self.map.len() > new_capacity {
                let lru_key = self.map.iter()
                    .min_by_key(|(_, (_, c))| *c)
                    .map(|(k, _)| k.clone());
                if let Some(lru_key) = lru_key {
                    self.map.remove(&lru_key);
                }
            }
        }
        
        self.capacity = new_capacity;
        debug!("Cache resized to capacity: {}", new_capacity);
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub utilization: f32,
}

/// Memory cache dengan multiple levels
#[derive(Debug)]
pub struct MemoryCache {
    l1_cache: LRUCache<String, String>,  // Fast cache (hot data)
    l2_cache: LRUCache<String, String>,  // Slower cache (warm data)
    l1_capacity: usize,
    l2_capacity: usize,
    total_accesses: usize,
    l1_hits: usize,
    l2_hits: usize,
    misses: usize,
}

impl MemoryCache {
    pub fn new(l1_capacity: usize, l2_capacity: usize) -> Self {
        Self {
            l1_cache: LRUCache::new(l1_capacity),
            l2_cache: LRUCache::new(l2_capacity),
            l1_capacity,
            l2_capacity,
            total_accesses: 0,
            l1_hits: 0,
            l2_hits: 0,
            misses: 0,
        }
    }
    
    /// Get value dari multi-level cache
    pub fn get(&mut self, key: &str) -> Option<String> {
        self.total_accesses += 1;
        
        // Check L1 cache first
        if let Some(value) = self.l1_cache.get(&key.to_string()) {
            self.l1_hits += 1;
            trace!("L1 cache hit: {}", key);
            return Some(value);
        }
        
        // Check L2 cache
        if let Some(value) = self.l2_cache.get(&key.to_string()) {
            self.l2_hits += 1;
            trace!("L2 cache hit: {}", key);
            
            // Promote to L1 cache
            self.l1_cache.put(key.to_string(), value.clone());
            
            return Some(value);
        }
        
        // Cache miss
        self.misses += 1;
        trace!("Cache miss: {}", key);
        None
    }
    
    /// Put value ke cache
    pub fn put(&mut self, key: String, value: String) {
        let key_clone = key.clone();
        
        // Always put in L1 cache first
        self.l1_cache.put(key.clone(), value.clone());
        
        // Also put in L2 if it's not already there
        if !self.l2_cache.contains(&key) {
            self.l2_cache.put(key, value);
        }
        
        trace!("Cache put: {}", key_clone);
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> MultiCacheStats {
        let l1_hit_rate = if self.total_accesses > 0 {
            self.l1_hits as f32 / self.total_accesses as f32
        } else {
            0.0
        };
        
        let l2_hit_rate = if self.total_accesses > 0 {
            self.l2_hits as f32 / self.total_accesses as f32
        } else {
            0.0
        };
        
        let total_hit_rate = if self.total_accesses > 0 {
            (self.l1_hits + self.l2_hits) as f32 / self.total_accesses as f32
        } else {
            0.0
        };
        
        MultiCacheStats {
            l1_size: self.l1_cache.len(),
            l1_capacity: self.l1_capacity,
            l1_hit_rate,
            
            l2_size: self.l2_cache.len(),
            l2_capacity: self.l2_capacity,
            l2_hit_rate,
            
            total_hit_rate,
            total_accesses: self.total_accesses,
            misses: self.misses,
        }
    }
    
    /// Clear all caches
    pub fn clear(&mut self) {
        self.l1_cache.clear();
        self.l2_cache.clear();
        self.total_accesses = 0;
        self.l1_hits = 0;
        self.l2_hits = 0;
        self.misses = 0;
        debug!("All caches cleared");
    }
    
    /// Resize caches
    pub fn resize(&mut self, l1_capacity: usize, l2_capacity: usize) {
        self.l1_cache.resize(l1_capacity);
        self.l2_cache.resize(l2_capacity);
        self.l1_capacity = l1_capacity;
        self.l2_capacity = l2_capacity;
        debug!("Caches resized: L1={}, L2={}", l1_capacity, l2_capacity);
    }
}

/// Multi-level cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCacheStats {
    pub l1_size: usize,
    pub l1_capacity: usize,
    pub l1_hit_rate: f32,
    
    pub l2_size: usize,
    pub l2_capacity: usize,
    pub l2_hit_rate: f32,
    
    pub total_hit_rate: f32,
    pub total_accesses: usize,
    pub misses: usize,
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new(100, 1000)
    }
}

impl<K: Clone + Eq + Hash + std::fmt::Debug, V: Clone> Default for LRUCache<K, V> {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lru_cache_basic() {
        let mut cache = LRUCache::new(3);
        
        // Test put and get
        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");
        
        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));
        assert_eq!(cache.get(&"key3"), Some("value3"));
        
        // Test eviction
        cache.put("key4", "value4"); // Should evict key1 (least recently used)
        assert_eq!(cache.get(&"key1"), None);
        assert_eq!(cache.get(&"key4"), Some("value4"));
        assert_eq!(cache.len(), 3);
    }
    
    #[test]
    fn test_lru_cache_access_order() {
        let mut cache = LRUCache::new(3);
        
        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");
        
        // Access key1 to make it most recently used
        cache.get(&"key1");
        
        // Add key4, should evict key2 (now least recently used)
        cache.put("key4", "value4");
        
        assert_eq!(cache.get(&"key1"), Some("value1")); // Should still be there
        assert_eq!(cache.get(&"key2"), None); // Should be evicted
        assert_eq!(cache.get(&"key4"), Some("value4"));
    }
    
    #[test]
    fn test_memory_cache() {
        let mut cache = MemoryCache::new(2, 4);
        
        // Put some values
        cache.put("key1".to_string(), "value1".to_string());
        cache.put("key2".to_string(), "value2".to_string());
        cache.put("key3".to_string(), "value3".to_string());
        
        // Test get
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("key2"), Some("value2".to_string()));
        assert_eq!(cache.get("key3"), Some("value3".to_string()));
        
        // Test statistics
        let stats = cache.get_stats();
        assert_eq!(stats.total_accesses, 3);
        assert!(stats.total_hit_rate > 0.0);
    }
    
    #[test]
    fn test_cache_stats() {
        let mut cache = LRUCache::new(5);
        
        cache.put("key1", "value1");
        cache.put("key2", "value2");
        
        let stats = cache.get_stats();
        assert_eq!(stats.size, 2);
        assert_eq!(stats.capacity, 5);
        assert_eq!(stats.utilization, 0.4);
    }
}
