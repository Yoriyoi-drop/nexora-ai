//! Controller Cache - Context caching dengan LRU policy

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::ContextInfo;
use super::controller_types::{LruContextCache, ContextEntry};

impl LruContextCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            max_size,
            cleanup_threshold: max_size / 2,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<ContextInfo> {
        if let Some(entry) = self.cache.get_mut(key) {
            entry.last_accessed = Self::current_timestamp_ms();
            entry.access_count += 1;
            
            let context = entry.context.clone();
            
            // Move to back of access order (LRU)
            self.access_order.retain(|k| k != key);
            self.access_order.push_back(key.to_string());
            
            Some(context)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: String, context: &ContextInfo, ttl_ms: u64) {
        let current_time = Self::current_timestamp_ms();
        
        let entry = ContextEntry {
            context: context.clone(),
            created_at: current_time,
            last_accessed: current_time,
            access_count: 1,
            ttl_ms,
        };
        
        // If cache is full, remove oldest entry
        if self.cache.len() >= self.max_size {
            if let Some(oldest_key) = self.access_order.pop_front() {
                self.cache.remove(&oldest_key);
            }
        }
        
        self.cache.insert(key.clone(), entry);
        self.access_order.push_back(key);
    }

    pub fn cleanup_expired(&mut self) {
        let current_time = Self::current_timestamp_ms();
        
        // Remove expired entries
        self.cache.retain(|_, entry| {
            current_time.saturating_sub(entry.created_at) < entry.ttl_ms
        });
        
        // Update access order
        let keys_to_retain: Vec<String> = self.access_order.iter()
            .filter(|key| self.cache.contains_key(*key))
            .cloned()
            .collect();
        self.access_order = keys_to_retain.into();
        
        // Force cleanup if cache is too large
        if self.cache.len() > self.cleanup_threshold {
            let to_remove = self.cache.len() - self.cleanup_threshold;
            for _ in 0..to_remove {
                if let Some(oldest_key) = self.access_order.pop_front() {
                    self.cache.remove(&oldest_key);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}
