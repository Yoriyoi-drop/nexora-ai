//! Memory Model Core Logic
//! 
//! Extracted dari memory_model.rs untuk modular structure

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
use rand::Rng;
use crate::model::{MemoryEntry, MemoryType};
use crate::types::{MemoryConfig, MemoryStats};

/// Hebbian Memory Model
pub struct HebbianMemoryModel {
    memories: HashMap<u32, MemoryEntry>,
    next_memory_id: u32,
    config: MemoryConfig,
    stats: MemoryStats,
}

impl HebbianMemoryModel {
    /// Create new memory model with default config
    pub fn new() -> Self {
        Self::with_config(MemoryConfig::default())
    }
    
    /// Create new memory model with custom config
    pub fn with_config(config: MemoryConfig) -> Self {
        Self {
            memories: HashMap::new(),
            next_memory_id: 1,
            config,
            stats: MemoryStats::default(),
        }
    }
    
    /// Add new memory entry
    pub fn add_memory(
        &mut self,
        memory_type: MemoryType,
        content: Option<String>,
        relevance: Option<f64>,
        emotional_salience: Option<f64>,
    ) -> Result<u32> {
        let memory_id = self.next_memory_id;
        self.next_memory_id += 1;
        
        let memory = MemoryEntry::new(
            memory_id,
            memory_type,
            content,
            relevance.unwrap_or_else(|| memory_type.default_relevance()),
            emotional_salience.unwrap_or_else(|| memory_type.default_emotional_salience()),
        );
        
        // Validate memory before adding
        memory.validate()
            .map_err(|e| anyhow::anyhow!("Invalid memory: {}", e))?;
        
        self.memories.insert(memory_id, memory);
        self.update_stats();
        
        Ok(memory_id)
    }
    
    /// Get memory by ID
    pub fn get_memory(&self, memory_id: u32) -> Option<&MemoryEntry> {
        self.memories.get(&memory_id)
    }
    
    /// Get mutable memory by ID
    pub fn get_memory_mut(&mut self, memory_id: u32) -> Option<&mut MemoryEntry> {
        self.memories.get_mut(&memory_id)
    }
    
    /// Activate memory (increase activation and update timestamp)
    pub fn activate_memory(&mut self, memory_id: u32, activation_boost: f64) -> Result<()> {
        if let Some(memory) = self.memories.get_mut(&memory_id) {
            memory.activate(activation_boost);
            self.update_stats();
            Ok(())
        } else {
            Err(anyhow::anyhow!("Memory {} not found", memory_id))
        }
    }
    
    /// Update all memory strengths based on decay
    pub fn update_all_strengths(&mut self) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        for memory in self.memories.values_mut() {
            if self.config.decay_enabled {
                memory.update_strength(memory.memory_type.decay_rate(), current_time);
            }
            
            if self.config.emotional_modulation_enabled {
                memory.apply_emotional_decay(memory.memory_type.decay_rate() * 0.5, current_time);
                memory.apply_relevance_decay(memory.memory_type.decay_rate() * 0.3, current_time);
            }
        }
        
        self.update_stats();
    }
    
    /// Remove expired memories
    pub fn remove_expired_memories(&mut self) -> Vec<u32> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        let expired_ids: Vec<u32> = self.memories
            .values()
            .filter(|memory| memory.is_expired(self.config.forgetting_threshold, current_time))
            .map(|memory| memory.memory_id)
            .collect();
        
        for id in &expired_ids {
            self.memories.remove(id);
        }
        
        self.update_stats();
        expired_ids
    }
    
    /// Get memories by type
    pub fn get_memories_by_type(&self, memory_type: MemoryType) -> Vec<&MemoryEntry> {
        self.memories
            .values()
            .filter(|memory| memory.memory_type == memory_type)
            .collect()
    }
    
    /// Get strongest memories (top N)
    pub fn get_strongest_memories(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut memories: Vec<&MemoryEntry> = self.memories.values().collect();
        memories.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
        memories.into_iter().take(limit).collect()
    }
    
    /// Get most relevant memories for query (simple content matching)
    pub fn get_relevant_memories(&self, query: &str, limit: usize) -> Vec<&MemoryEntry> {
        let mut relevant_memories: Vec<(&MemoryEntry, f64)> = self.memories
            .values()
            .filter_map(|memory| {
                if let Some(content) = &memory.content {
                    let score = self.calculate_relevance_score(query, content);
                    Some((memory, score))
                } else {
                    None
                }
            })
            .collect();
        
        relevant_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        relevant_memories
            .into_iter()
            .take(limit)
            .map(|(memory, _)| memory)
            .collect()
    }
    
    /// Calculate simple relevance score between query and content
    fn calculate_relevance_score(&self, query: &str, content: &str) -> f64 {
        let query_words: Vec<String> = query.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
        let content_words: Vec<String> = content.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
        
        if query_words.is_empty() || content_words.is_empty() {
            return 0.0;
        }
        
        let matches = query_words
            .iter()
            .filter(|&q| content_words.contains(q))
            .count();
        
        matches as f64 / query_words.len() as f64
    }
    
    /// Consolidate weak memories (remove those below threshold)
    pub fn consolidate_memories(&mut self) -> Vec<u32> {
        if !self.config.consolidation_enabled {
            return Vec::new();
        }
        
        let consolidated_ids: Vec<u32> = self.memories
            .values()
            .filter(|memory| !memory.should_consolidate(self.config.consolidation_threshold))
            .map(|memory| memory.memory_id)
            .collect();
        
        for id in &consolidated_ids {
            self.memories.remove(id);
        }
        
        self.update_stats();
        consolidated_ids
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }
    
    /// Get configuration
    pub fn get_config(&self) -> &MemoryConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: MemoryConfig) {
        self.config = config;
    }
    
    /// Get total memory count
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }
    
    /// Check if memory limit is reached
    pub fn is_at_capacity(&self) -> bool {
        self.memories.len() >= self.config.max_memories
    }
    
    /// Get working memories (temporary memories)
    pub fn get_working_memories(&self) -> Vec<&MemoryEntry> {
        self.get_memories_by_type(MemoryType::Working)
            .into_iter()
            .take(self.config.working_memory_capacity)
            .collect()
    }
    
    /// Clear working memories
    pub fn clear_working_memories(&mut self) {
        let working_ids: Vec<u32> = self.memories
            .values()
            .filter(|memory| memory.memory_type == MemoryType::Working)
            .map(|memory| memory.memory_id)
            .collect();
        
        for id in working_ids {
            self.memories.remove(&id);
        }
        
        self.update_stats();
    }
    
    /// Update memory statistics
    fn update_stats(&mut self) {
        self.stats.total_memories = self.memories.len();
        
        self.stats.memories_by_type.clear();
        for memory in self.memories.values() {
            *self.stats.memories_by_type.entry(memory.memory_type).or_insert(0) += 1;
        }
        
        if !self.memories.is_empty() {
            self.stats.average_strength = self.memories
                .values()
                .map(|m| m.strength)
                .sum::<f64>() / self.memories.len() as f64;
            
            self.stats.total_activations = self.memories
                .values()
                .map(|m| m.activation)
                .sum::<f64>();
            
            // Find oldest and newest timestamps manually
            let timestamps: Vec<f64> = self.memories.values().map(|m| m.timestamp).collect();
            if let Some(oldest) = timestamps.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
                self.stats.oldest_memory_timestamp = Some(*oldest);
            }
            if let Some(newest) = timestamps.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
                self.stats.newest_memory_timestamp = Some(*newest);
            }
        }
    }
    
    /// Validate all memories
    pub fn validate_all_memories(&self) -> Result<Vec<String>> {
        let mut errors = Vec::new();
        
        for (id, memory) in &self.memories {
            if let Err(e) = memory.validate() {
                errors.push(format!("Memory {}: {}", id, e));
            }
        }
        
        if errors.is_empty() {
            Ok(Vec::new())
        } else {
            Err(anyhow::anyhow!("Validation errors: {}", errors.join("; ")))
        }
    }
}

impl Default for HebbianMemoryModel {
    fn default() -> Self {
        Self::new()
    }
}
