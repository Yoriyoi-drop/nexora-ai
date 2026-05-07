//! Cache System - LRU Cache dan memory optimization
//! 
//! Implementasi LRU cache untuk fast access dan memory optimization

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use serde::{Serialize, Deserialize};
use tracing::{debug, trace};

/// LRU (Least Recently Used) Cache implementation
#[derive(Debug)]
pub struct LRUCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, usize)>, // key -> (value, access_order)
    access_order: VecDeque<K>,   // Track access order (front = most recent)
    access_counter: usize,
}

impl<K: Clone + Eq + Hash + std::fmt::Debug, V: Clone> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            access_order: VecDeque::new(),
            access_counter: 0,
        }
    }
    
    /// Get value dari cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some((value, _)) = self.map.get(key) {
            let key_clone = key.clone();
            // Update access order after getting value
            drop(value); // Release the borrow
            self.update_access_order(key_clone);
            // Get value again after updating order
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
        // Check if key already exists
        if self.map.contains_key(&key) {
            // Update existing entry
            self.map.insert(key.clone(), (value, self.access_counter));
            self.update_access_order(key);
        } else {
            // Check capacity
            if self.map.len() >= self.capacity {
                // Evict least recently used
                if let Some(lru_key) = self.access_order.pop_back() {
                    self.map.remove(&lru_key);
                    debug!("Evicted LRU key: {:?}", lru_key);
                }
            }
            
            // Insert new entry
            self.map.insert(key.clone(), (value, self.access_counter));
            self.access_order.push_front(key);
        }
        
        self.access_counter += 1;
        trace!("Cache updated, current size: {}", self.map.len());
    }
    
    /// Remove key dari cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some((value, _)) = self.map.remove(key) {
            // Remove from access order
            self.access_order.retain(|k| k != key);
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
        self.access_order.clear();
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
    
    /// Update access order untuk key
    fn update_access_order(&mut self, key: K) {
        // Remove from current position
        self.access_order.retain(|k| k != &key);
        // Add to front (most recent)
        self.access_order.push_front(key);
    }
    
    /// Get keys dalam access order (most recent first)
    pub fn keys(&self) -> Vec<&K> {
        self.access_order.iter().collect()
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
            // Evict excess entries
            while self.map.len() > new_capacity {
                if let Some(lru_key) = self.access_order.pop_back() {
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

/// Weighted cache dengan priority scoring
#[derive(Debug)]
pub struct WeightedCache<K, V> {
    items: HashMap<K, CacheItem<V>>,
    total_weight: f32,
    max_weight: f32,
    access_counter: usize,
}

#[derive(Debug, Clone)]
struct CacheItem<V> {
    value: V,
    weight: f32,
    last_access: usize,
    access_count: usize,
}

impl<K: Clone + Eq + Hash + std::fmt::Debug, V: Clone> WeightedCache<K, V> {
    pub fn new(max_weight: f32) -> Self {
        Self {
            items: HashMap::new(),
            total_weight: 0.0,
            max_weight,
            access_counter: 0,
        }
    }
    
    /// Get value dari weighted cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(item) = self.items.get_mut(key) {
            item.last_access = self.access_counter;
            item.access_count += 1;
            self.access_counter += 1;
            
            trace!("Weighted cache hit: {:?}", key);
            Some(item.value.clone())
        } else {
            trace!("Weighted cache miss: {:?}", key);
            None
        }
    }
    
    /// Put value dengan weight
    pub fn put(&mut self, key: K, value: V, weight: f32) {
        // Check if we need to evict items
        if self.items.contains_key(&key) {
            // Update existing item
            if let Some(existing_item) = self.items.get_mut(&key) {
                self.total_weight -= existing_item.weight;
                existing_item.value = value;
                existing_item.weight = weight;
                existing_item.last_access = self.access_counter;
                existing_item.access_count += 1;
            }
        } else {
            // Add new item, evict if necessary
            while self.total_weight + weight > self.max_weight && !self.items.is_empty() {
                self.evict_lowest_priority();
            }
            
            let item = CacheItem {
                value,
                weight,
                last_access: self.access_counter,
                access_count: 1,
            };
            
            self.items.insert(key.clone(), item);
            self.total_weight += weight;
        }
        
        self.access_counter += 1;
        trace!("Weighted cache updated, total weight: {:.2}", self.total_weight);
    }
    
    /// Evict item dengan lowest priority
    fn evict_lowest_priority(&mut self) {
        if self.items.is_empty() {
            return;
        }
        
        // Find item dengan lowest priority (weight * recency_factor)
        let mut lowest_key = None;
        let mut lowest_priority = f32::INFINITY;
        
        for (key, item) in &self.items {
            let recency_factor = 1.0 / (1.0 + (self.access_counter - item.last_access) as f32);
            let priority = item.weight * recency_factor;
            
            if priority < lowest_priority {
                lowest_priority = priority;
                lowest_key = Some(key.clone());
            }
        }
        
        if let Some(key) = lowest_key {
            if let Some(item) = self.items.remove(&key) {
                self.total_weight -= item.weight;
                debug!("Evicted low priority item: {:?} (priority: {:.2})", key, lowest_priority);
            }
        }
    }
    
    /// Get cache size
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    /// Get total weight
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }
    
    /// Clear cache
    pub fn clear(&mut self) {
        self.items.clear();
        self.total_weight = 0.0;
        self.access_counter = 0;
        debug!("Weighted cache cleared");
    }
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
    fn test_weighted_cache() {
        let mut cache = WeightedCache::new(10.0);
        
        // Add items with different weights
        cache.put("key1", "value1", 3.0);
        cache.put("key2", "value2", 4.0);
        cache.put("key3", "value3", 5.0);
        
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.total_weight(), 12.0);
        
        // Add item that would exceed capacity
        cache.put("key4", "value4", 2.0); // Total would be 14.0, max is 10.0
        
        // Should have evicted some items
        assert!(cache.len() <= 3);
        assert!(cache.total_weight() <= 10.0);
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
