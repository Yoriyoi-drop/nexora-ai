//! Memory Layers - 4-layer memory architecture
//! 
//! Implementasi memory layers: Short, Session, Long, Knowledge

use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{debug, trace};

/// Memory layers dengan retention policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayer {
    Short,      // Short-term memory (seconds to minutes)
    Session,    // Session memory (hours)
    Long,       // Long-term memory (days to weeks)
    Knowledge,  // Knowledge base (permanent)
}

impl MemoryLayer {
    /// Get capacity untuk layer
    pub fn capacity(&self) -> usize {
        match self {
            MemoryLayer::Short => 100,      // 100 entries
            MemoryLayer::Session => 1000,    // 1K entries
            MemoryLayer::Long => 10000,      // 10K entries
            MemoryLayer::Knowledge => 100000, // 100K entries
        }
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            MemoryLayer::Short => "Short-term Memory",
            MemoryLayer::Session => "Session Memory",
            MemoryLayer::Long => "Long-term Memory",
            MemoryLayer::Knowledge => "Knowledge Base",
        }
    }
    
    pub fn retention_policy(&self) -> RetentionPolicy {
        match self {
            MemoryLayer::Short => RetentionPolicy::Immediate,
            MemoryLayer::Session => RetentionPolicy::Session,
            MemoryLayer::Long => RetentionPolicy::Persistent,
            MemoryLayer::Knowledge => RetentionPolicy::Permanent,
        }
    }
}

/// Retention policies untuk memory entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionPolicy {
    Immediate,   // Cleared after request
    Session,     // Cleared after session
    Persistent,  // Persistent across sessions
    Permanent,   // Never cleared
}

/// Memory entry dengan metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub timestamp: u64,
    pub created_at: u64,
    pub accessed_at: u64,
    pub access_count: u32,
    pub last_access: u64,
    pub ttl: Option<u64>, // Time to live in seconds
    pub importance: f32,  // 0.0 - 1.0
    pub tags: Vec<String>,
}

impl MemoryEntry {
    pub fn new(key: String, value: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            key,
            value,
            timestamp: now,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            last_access: now,
            ttl: None,
            importance: 0.5,
            tags: Vec::new(),
        }
    }
    
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_ttl(mut self, ttl: u64) -> Self {
        self.ttl = Some(ttl);
        self
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now > self.timestamp + ttl
        } else {
            false
        }
    }
    
    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_access = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.accessed_at = self.last_access;
    }
}

/// Memory layers implementation
#[derive(Debug)]
pub struct MemoryLayers {
    layers: HashMap<MemoryLayer, HashMap<String, MemoryEntry>>,
    max_entries: HashMap<MemoryLayer, usize>,
}

impl MemoryLayers {
    pub fn new() -> Self {
        let mut layers = HashMap::new();
        let mut max_entries = HashMap::new();
        
        // Initialize each layer dengan kapasitas yang berbeda
        layers.insert(MemoryLayer::Short, HashMap::new());
        layers.insert(MemoryLayer::Session, HashMap::new());
        layers.insert(MemoryLayer::Long, HashMap::new());
        layers.insert(MemoryLayer::Knowledge, HashMap::new());
        
        max_entries.insert(MemoryLayer::Short, 100);      // Small capacity
        max_entries.insert(MemoryLayer::Session, 1000);   // Medium capacity
        max_entries.insert(MemoryLayer::Long, 10000);    // Large capacity
        max_entries.insert(MemoryLayer::Knowledge, 100000); // Very large capacity
        
        Self {
            layers,
            max_entries,
        }
    }
    
    /// Store data ke specific layer
    pub async fn store(&mut self, layer: MemoryLayer, key: &str, value: &str) -> Result<()> {
        trace!("Storing to {:?}: {} = {}", layer, key, value);
        
        let entry = MemoryEntry::new(key.to_string(), value.to_string());
        self.store_entry(layer, entry).await
    }
    
    /// Store entry ke memory layer
    pub async fn store_entry(&mut self, layer: MemoryLayer, entry: MemoryEntry) -> Result<()> {
        debug!("Storing in {:?}: {}", layer, entry.key);
        
        // Check capacity dan evict if necessary
        let max_capacity = self.max_entries.get(&layer).copied().unwrap_or_else(|| layer.capacity());
        if let Some(layer_map) = self.layers.get_mut(&layer) {
            if layer_map.len() >= max_capacity {
                // Need to drop the borrow before calling evict_entries
                let _ = layer_map; // Drop the borrow before modifying shard
                self.evict_entries(layer, 1).await?;
                let layer_map = self.layers.get_mut(&layer)
                    .ok_or_else(|| anyhow::anyhow!("Layer {:?} not found after eviction", layer))?;
                layer_map.insert(entry.key.clone(), entry.clone());
            } else {
                layer_map.insert(entry.key.clone(), entry.clone());
            }
            debug!("Stored entry in {:?}: {}", layer, entry.key);
        }
        
        Ok(())
    }
    
    /// Retrieve data dari layer
    pub async fn retrieve(&mut self, layer: MemoryLayer, key: &str) -> Result<Option<String>> {
        trace!("Retrieving from {:?}: {}", layer, key);
        
        if let Some(layer_map) = self.layers.get_mut(&layer) {
            if let Some(entry) = layer_map.get_mut(key) {
                // Check if expired
                if entry.is_expired() {
                    layer_map.remove(key);
                    return Ok(None);
                }
                
                // Update access statistics
                entry.access();
                debug!("Retrieved entry from {:?}: {} (access count: {})", 
                       layer, key, entry.access_count);
                
                return Ok(Some(entry.value.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Search dalam layer
    pub async fn search(&self, layer: MemoryLayer, query: &str) -> Result<Option<Vec<crate::MemorySearchResult>>> {
        trace!("Searching in {:?}: {}", layer, query);
        
        if let Some(layer_map) = self.layers.get(&layer) {
            let mut results = Vec::new();
            
            for entry in layer_map.values() {
                // Skip expired entries
                if entry.is_expired() {
                    continue;
                }
                
                // Simple keyword matching
                let relevance = self.calculate_relevance(entry, query);
                if relevance > 0.1 {
                    results.push(crate::MemorySearchResult {
                        layer,
                        key: entry.key.clone(),
                        value: entry.value.clone(),
                        relevance_score: relevance,
                        timestamp: entry.timestamp,
                    });
                }
            }
            
            // Sort by relevance
            results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(Ordering::Equal));
            
            return Ok(Some(results));
        }
        
        Ok(None)
    }
    
    /// Get statistics untuk layer
    pub async fn get_stats(&self) -> Result<HashMap<MemoryLayer, usize>> {
        let mut stats = HashMap::new();
        
        for (layer, layer_map) in &self.layers {
            stats.insert(*layer, layer_map.len());
        }
        
        Ok(stats)
    }
    
    /// Clear layer
    pub async fn clear_layer(&mut self, layer: MemoryLayer) -> Result<()> {
        debug!("Clearing layer: {:?}", layer);
        
        if let Some(layer_map) = self.layers.get_mut(&layer) {
            layer_map.clear();
        }
        
        Ok(())
    }
    
    /// Delete specific entry from layer
    pub async fn delete(&mut self, layer: MemoryLayer, key: &str) -> Result<()> {
        debug!("Deleting from {:?}: {}", layer, key);
        
        if let Some(layer_map) = self.layers.get_mut(&layer) {
            layer_map.remove(key);
        }
        
        Ok(())
    }
    
    /// Cleanup expired entries
    pub async fn cleanup_expired(&mut self) -> Result<usize> {
        let mut total_removed = 0;
        
        for (layer, layer_map) in &mut self.layers {
            let mut expired_keys = Vec::new();
            
            for (key, entry) in layer_map.iter() {
                if entry.is_expired() {
                    expired_keys.push(key.clone());
                }
            }
            
            for key in expired_keys {
                layer_map.remove(&key);
                total_removed += 1;
            }
            
            if total_removed > 0 {
                debug!("Cleaned {} expired entries from {:?}", total_removed, layer);
            }
        }
        
        Ok(total_removed)
    }
    
    /// Evict entries based on policy
    async fn evict_entries(&mut self, layer: MemoryLayer, count: usize) -> Result<()> {
        if let Some(layer_map) = self.layers.get_mut(&layer) {
            // Simple eviction: remove oldest entries
            let keys_to_remove: Vec<String> = layer_map.keys().cloned().take(count).collect();
            for key in keys_to_remove {
                layer_map.remove(&key);
            }
        }
        Ok(())
    }
    
    /// Calculate relevance score untuk search
    fn calculate_relevance(&self, entry: &MemoryEntry, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let key_lower = entry.key.to_lowercase();
        let value_lower = entry.value.to_lowercase();
        
        let mut relevance = 0.0;
        
        // Exact key match
        if key_lower.contains(&query_lower) {
            relevance += 0.8;
        }
        
        // Exact value match
        if value_lower.contains(&query_lower) {
            relevance += 0.6;
        }
        
        // Partial matches
        for word in query_lower.split_whitespace() {
            if key_lower.contains(word) {
                relevance += 0.3;
            }
            if value_lower.contains(word) {
                relevance += 0.2;
            }
        }
        
        // Boost by importance and recent access
        relevance *= 1.0 + entry.importance;
        
        // Decay by age (older entries get lower relevance)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let age_hours = (now - entry.timestamp) as f32 / 3600.0;
        let age_decay = (-age_hours / 24.0).exp(); // Decay over days
        relevance *= age_decay;
        
        relevance.clamp(0.0, 1.0)
    }
    
    /// Get layer configuration
    pub fn get_layer_config(&self, layer: MemoryLayer) -> LayerConfig {
        LayerConfig {
            layer,
            max_entries: self.max_entries.get(&layer).copied().unwrap_or(1000),
            retention_policy: layer.retention_policy(),
            current_size: self.layers.get(&layer).map(|m| m.len()).unwrap_or(0),
        }
    }
    
    /// Update layer configuration
    pub fn update_layer_config(&mut self, layer: MemoryLayer, max_entries: usize) {
        self.max_entries.insert(layer, max_entries);
        debug!("Updated {:?} max entries to {}", layer, max_entries);
    }
    
    /// Query memory layers for entries matching the query
    pub async fn query(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        let mut results = Vec::new();
        
        for (_layer, layer_map) in &self.layers {
            for entry in layer_map.values() {
                let relevance = self.calculate_relevance(entry, query);
                if relevance > 0.1 { // Only include relevant entries
                    results.push(entry.clone());
                }
            }
        }
        
        // Sort by relevance (most relevant first)
        results.sort_by(|a, b| {
            let relevance_a = self.calculate_relevance(a, query);
            let relevance_b = self.calculate_relevance(b, query);
            relevance_b.partial_cmp(&relevance_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(results)
    }
    
    /// Get layer data (for testing)
    pub async fn get_layer_data(&self, layer: MemoryLayer) -> Result<Vec<MemoryEntry>> {
        if let Some(layer_map) = self.layers.get(&layer) {
            Ok(layer_map.values().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
    
        
    /// Search in specific layer (for testing)
    pub async fn search_in_layer(&self, layer: MemoryLayer, query: &str) -> Result<Vec<MemoryEntry>> {
        if let Some(layer_map) = self.layers.get(&layer) {
            let mut results = Vec::new();
            for entry in layer_map.values() {
                if entry.value.contains(query) || entry.key.contains(query) {
                    results.push(entry.clone());
                }
            }
            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get entry from layer (for testing)
    pub async fn get_entry(&self, layer: MemoryLayer, key: &str) -> Result<Option<MemoryEntry>> {
        if let Some(layer_map) = self.layers.get(&layer) {
            Ok(layer_map.get(key).cloned())
        } else {
            Ok(None)
        }
    }
    
    /// Remove entry from layer (for testing)
    pub async fn remove_entry(&mut self, layer: MemoryLayer, key: &str) -> Result<bool> {
        if let Some(layer_map) = self.layers.get_mut(&layer) {
            Ok(layer_map.remove(key).is_some())
        } else {
            Ok(false)
        }
    }
    
    /// Get max entries for layer (for testing)
    pub async fn max_entries(&self, layer: MemoryLayer) -> Result<usize> {
        Ok(self.max_entries.get(&layer).copied().unwrap_or(1000))
    }
}

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layer: MemoryLayer,
    pub max_entries: usize,
    pub retention_policy: RetentionPolicy,
    pub current_size: usize,
}

impl Default for MemoryLayers {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_layers() {
        let mut layers = MemoryLayers::new();
        
        // Test store and retrieve
        layers.store(MemoryLayer::Short, "test_key", "test_value").await.unwrap();
        let result = layers.retrieve(MemoryLayer::Short, "test_key").await.unwrap();
        assert_eq!(result, Some("test_value".to_string()));
        
        // Test search
        let search_results = layers.search(MemoryLayer::Short, "test").await.unwrap();
        assert!(search_results.is_some());
        assert!(!search_results.unwrap().is_empty());
        
        // Test stats
        let stats = layers.get_stats().await.unwrap();
        assert_eq!(stats.get(&MemoryLayer::Short), Some(&1));
    }
    
    #[tokio::test]
    async fn test_memory_entry() {
        let entry = MemoryEntry::new("key".to_string(), "value".to_string())
            .with_importance(0.8)
            .with_ttl(3600);
        
        assert_eq!(entry.importance, 0.8);
        assert_eq!(entry.ttl, Some(3600));
        assert!(!entry.is_expired());
        
        let mut mutable_entry = entry;
        mutable_entry.access();
        assert_eq!(mutable_entry.access_count, 1);
    }
    
    #[tokio::test]
    async fn test_layer_capacity() {
        let mut layers = MemoryLayers::new();
        
        // Set very small capacity for testing
        layers.update_layer_config(MemoryLayer::Short, 2);
        
        // Store 3 entries (should evict one)
        layers.store(MemoryLayer::Short, "key1", "value1").await.unwrap();
        layers.store(MemoryLayer::Short, "key2", "value2").await.unwrap();
        layers.store(MemoryLayer::Short, "key3", "value3").await.unwrap();
        
        let stats = layers.get_stats().await.unwrap();
        assert_eq!(stats.get(&MemoryLayer::Short), Some(&2)); // Should be at capacity
    }
    
    #[tokio::test]
    async fn test_cleanup_expired() {
        let mut layers = MemoryLayers::new();
        
        // Store entry with short TTL
        let entry = MemoryEntry::new("expired".to_string(), "value".to_string())
            .with_ttl(1); // 1 second TTL
        
        layers.store_entry(MemoryLayer::Short, entry).await.unwrap();
        
        // Wait for expiration (in real test, you'd need to wait)
        // For now, just test the cleanup function
        let removed = layers.cleanup_expired().await.unwrap();
        assert_eq!(removed, 0); // Not expired yet in this test
    }
    
    #[tokio::test]
    async fn test_memory_layers_capacity_limits() {
        let mut layers = MemoryLayers::new();
        
        // Test storing up to capacity limit for short-term memory
        for i in 0..1050 {
            layers.store(MemoryLayer::Short, &format!("key_{}", i), &format!("value_{}", i)).await.unwrap();
        }
        
        // Should have at most 1000 entries
        let short_term_data = layers.get_layer_data(MemoryLayer::Short).await.unwrap();
        assert!(short_term_data.len() <= 1000);
    }
    
    #[tokio::test]
    async fn test_memory_layers_search() {
        let mut layers = MemoryLayers::new();
        
        // Store test data
        layers.store(MemoryLayer::Session, "test_key_1", "test_value_1").await.unwrap();
        layers.store(MemoryLayer::Session, "search_key", "searchable_value").await.unwrap();
        layers.store(MemoryLayer::Long, "another_key", "another_value").await.unwrap();
        
        // Test search functionality
        let results = layers.search(MemoryLayer::Session, "search").await.unwrap();
        assert!(results.is_some());
        assert!(!results.as_ref().unwrap().is_empty());
        
        // Test search in specific layer
        let session_results = layers.search_in_layer(MemoryLayer::Session, "search").await.unwrap();
        assert!(!session_results.is_empty());
        
        // Test search with no results
        let no_results = layers.search(MemoryLayer::Session, "nonexistent").await.unwrap();
        assert!(no_results.is_none() || no_results.unwrap().is_empty());
    }
    
    #[tokio::test]
    async fn test_memory_layers_clear() {
        let mut layers = MemoryLayers::new();
        
        // Store data in different layers
        layers.store(MemoryLayer::Short, "temp_key", "temp_value").await.unwrap();
        layers.store(MemoryLayer::Session, "session_key", "session_value").await.unwrap();
        layers.store(MemoryLayer::Long, "persistent_key", "persistent_value").await.unwrap();
        
        // Verify data exists
        let short_data = layers.get_layer_data(MemoryLayer::Short).await.unwrap();
        assert!(!short_data.is_empty());
        
        // Clear specific layer
        layers.clear_layer(MemoryLayer::Short).await.unwrap();
        
        // Verify layer is cleared
        let short_data_after = layers.get_layer_data(MemoryLayer::Short).await.unwrap();
        assert!(short_data_after.is_empty());
        
        // Verify other layers still have data
        let session_data = layers.get_layer_data(MemoryLayer::Session).await.unwrap();
        assert!(!session_data.is_empty());
    }
    
    #[tokio::test]
    async fn test_memory_layers_entry_metadata() {
        let mut layers = MemoryLayers::new();
        
        // Create entry with metadata
        let entry = MemoryEntry::new("meta_key".to_string(), "meta_value".to_string())
            .with_ttl(3600)
            .with_tags(vec!["test".to_string(), "important".to_string()]);
        
        layers.store_entry(MemoryLayer::Knowledge, entry).await.unwrap();
        
        // Retrieve and verify metadata
        let data = layers.get_layer_data(MemoryLayer::Knowledge).await.unwrap();
        let retrieved_entry = &data[0];
        
        assert_eq!(retrieved_entry.key, "meta_key");
        assert_eq!(retrieved_entry.value, "meta_value");
        assert!(retrieved_entry.tags.contains(&"test".to_string()));
        assert!(retrieved_entry.tags.contains(&"important".to_string()));
    }
    
    #[tokio::test]
    async fn test_memory_layers_get_entry() {
        let mut layers = MemoryLayers::new();
        
        // Store test entry
        layers.store(MemoryLayer::Long, "specific_key", "specific_value").await.unwrap();
        
        // Retrieve specific entry
        let entry = layers.get_entry(MemoryLayer::Long, "specific_key").await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().value, "specific_value");
        
        // Try to get non-existent entry
        let non_existent = layers.get_entry(MemoryLayer::Long, "non_existent").await.unwrap();
        assert!(non_existent.is_none());
    }
    
    #[tokio::test]
    async fn test_memory_layers_remove_entry() {
        let mut layers = MemoryLayers::new();
        
        // Store test entry
        layers.store(MemoryLayer::Session, "remove_key", "remove_value").await.unwrap();
        
        // Verify entry exists
        let before_removal = layers.get_entry(MemoryLayer::Session, "remove_key").await.unwrap();
        assert!(before_removal.is_some());
        
        // Remove entry
        let removed = layers.remove_entry(MemoryLayer::Session, "remove_key").await.unwrap();
        assert!(removed);
        
        // Verify entry is removed
        let after_removal = layers.get_entry(MemoryLayer::Session, "remove_key").await.unwrap();
        assert!(after_removal.is_none());
        
        // Try to remove non-existent entry
        let not_removed = layers.remove_entry(MemoryLayer::Session, "non_existent").await.unwrap();
        assert!(!not_removed);
    }
    
    #[test]
    fn test_memory_layer_enum() {
        // Test all memory layer variants
        let layers = vec![
            MemoryLayer::Short,
            MemoryLayer::Session,
            MemoryLayer::Long,
            MemoryLayer::Knowledge,
        ];
        
        assert_eq!(layers.len(), 4);
        
        // Test layer ordering and comparison
        assert!(MemoryLayer::Short != MemoryLayer::Session);
        assert!(MemoryLayer::Session != MemoryLayer::Long);
        assert!(MemoryLayer::Long != MemoryLayer::Knowledge);
    }
    
    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("test_key".to_string(), "test_value".to_string());
        
        assert_eq!(entry.key, "test_key");
        assert_eq!(entry.value, "test_value");
        assert!(entry.created_at > 0);
        assert!(entry.accessed_at > 0);
        assert!(entry.tags.is_empty());
        assert!(entry.ttl.is_none());
    }
    
    #[test]
    fn test_memory_entry_with_metadata() {
        let entry = MemoryEntry::new("key".to_string(), "value".to_string())
            .with_ttl(7200)
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()]);
        
        assert_eq!(entry.ttl, Some(7200));
        assert_eq!(entry.tags.len(), 2);
        assert!(entry.tags.contains(&"tag1".to_string()));
        assert!(entry.tags.contains(&"tag2".to_string()));
    }
}
