//! Memory management untuk Nexora Core

use nexora_core::{MemoryLayer, CoreResult};
use std::collections::HashMap;
use tracing::{debug, info};

/// Memory manager untuk mengelola akses memory layer
pub struct MemoryManager {
    memory_layers: HashMap<MemoryLayer, HashMap<String, String>>,
    enabled: bool,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            memory_layers: HashMap::new(),
            enabled: true,
        }
    }
    
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
    
    /// Store data ke memory layer
    pub async fn store(&mut self, layer: MemoryLayer, key: String, data: String) -> CoreResult<()> {
        if !self.enabled {
            return Ok(());
        }
        
        debug!("Storing to memory: layer={:?}, key={}", layer, key);
        
        let layer_memory = self.memory_layers.entry(layer).or_insert_with(HashMap::new);
        layer_memory.insert(key, data);
        
        info!("Data stored successfully");
        Ok(())
    }
    
    /// Retrieve data dari memory layer
    pub async fn retrieve(&self, layer: MemoryLayer, key: &str) -> CoreResult<Option<String>> {
        if !self.enabled {
            return Ok(None);
        }
        
        debug!("Retrieving from memory: layer={:?}, key={}", layer, key);
        
        if let Some(layer_memory) = self.memory_layers.get(&layer) {
            let result = layer_memory.get(key).cloned();
            info!("Data retrieved: found={}", result.is_some());
            Ok(result)
        } else {
            Ok(None)
        }
    }
    
    /// Check apakah memory layer memiliki data
    pub fn has_data(&self, layer: MemoryLayer) -> bool {
        self.memory_layers.get(&layer).map_or(false, |layer| !layer.is_empty())
    }
    
    /// Clear memory layer
    pub fn clear_layer(&mut self, layer: MemoryLayer) {
        if let Some(layer_memory) = self.memory_layers.get_mut(&layer) {
            layer_memory.clear();
            info!("Memory layer cleared: {:?}", layer);
        }
    }
    
    /// Clear all memory
    pub fn clear_all(&mut self) {
        self.memory_layers.clear();
        info!("All memory cleared");
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}
