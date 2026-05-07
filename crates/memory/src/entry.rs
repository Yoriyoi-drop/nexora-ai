//! Memory Entry Operations
//! 
//! Extracted dari memory_model.rs untuk modular structure

use std::time::{SystemTime, UNIX_EPOCH};
use crate::model::{MemoryEntry, MemoryType};

impl MemoryEntry {
    /// Create new memory entry
    pub fn new(
        memory_id: u32,
        memory_type: MemoryType,
        content: Option<String>,
        initial_relevance: f64,
        initial_emotional_salience: f64,
    ) -> Self {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        Self {
            memory_id,
            memory_type,
            activation: 1.0, // Initial activation
            relevance: initial_relevance,
            emotional_salience: initial_emotional_salience,
            timestamp: current_time,
            strength: initial_relevance * initial_emotional_salience,
            content,
            embedding: None,
            embedding_dim: 0,
        }
    }
    
    /// Create memory entry with defaults
    pub fn new_with_defaults(
        memory_id: u32,
        memory_type: MemoryType,
        content: Option<String>,
    ) -> Self {
        Self::new(
            memory_id,
            memory_type,
            content,
            memory_type.default_relevance(),
            memory_type.default_emotional_salience(),
        )
    }
    
    /// Calculate memory strength using Hebbian formula
    /// M_i(t) = A_ik × R_ik × E_ik × e^(-λ(t - t_k))
    pub fn calculate_strength(&self, decay_rate: f64, current_time: f64) -> f64 {
        let time_diff = current_time - self.timestamp;
        let decay_factor = (-decay_rate * time_diff).exp();
        
        self.activation * self.relevance * self.emotional_salience * decay_factor
    }
    
    /// Update memory strength based on decay
    pub fn update_strength(&mut self, decay_rate: f64, current_time: f64) {
        self.strength = self.calculate_strength(decay_rate, current_time);
    }
    
    /// Activate memory (increase activation and update timestamp)
    pub fn activate(&mut self, activation_boost: f64) {
        self.activation += activation_boost;
        self.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
    }
    
    /// Set embedding vector
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding_dim = embedding.len();
        self.embedding = Some(embedding);
    }
    
    /// Get embedding vector
    pub fn get_embedding(&self) -> Option<&Vec<f32>> {
        self.embedding.as_ref()
    }
    
    /// Check if memory is expired based on threshold
    pub fn is_expired(&self, threshold: f64, current_time: f64) -> bool {
        let decay_rate = self.memory_type.decay_rate();
        let current_strength = self.calculate_strength(decay_rate, current_time);
        current_strength < threshold
    }
    
    /// Check if memory should be consolidated
    pub fn should_consolidate(&self, threshold: f64) -> bool {
        self.strength >= threshold
    }
    
    /// Get age of memory in seconds
    pub fn age(&self) -> f64 {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        current_time - self.timestamp
    }
    
    /// Get memory importance score (combination of relevance and emotional salience)
    pub fn importance_score(&self) -> f64 {
        self.relevance * self.emotional_salience * self.activation
    }
    
    /// Update emotional salience
    pub fn update_emotional_salience(&mut self, new_salience: f64) {
        self.emotional_salience = new_salience.clamp(0.0, 1.0);
    }
    
    /// Update relevance
    pub fn update_relevance(&mut self, new_relevance: f64) {
        self.relevance = new_relevance.clamp(0.0, 1.0);
    }
    
    /// Apply emotional decay (reduce emotional salience over time)
    pub fn apply_emotional_decay(&mut self, decay_rate: f64, current_time: f64) {
        let time_diff = current_time - self.timestamp;
        let decay_factor = (-decay_rate * time_diff).exp();
        self.emotional_salience *= decay_factor;
        self.emotional_salience = self.emotional_salience.max(0.1); // Minimum emotional salience
    }
    
    /// Apply relevance decay (reduce relevance over time for certain memory types)
    pub fn apply_relevance_decay(&mut self, decay_rate: f64, current_time: f64) {
        // Only apply relevance decay to certain memory types
        match self.memory_type {
            MemoryType::Working | MemoryType::Episodic => {
                let time_diff = current_time - self.timestamp;
                let decay_factor = (-decay_rate * time_diff).exp();
                self.relevance *= decay_factor;
                self.relevance = self.relevance.max(0.1); // Minimum relevance
            }
            _ => {} // No relevance decay for semantic, procedural, user memories
        }
    }
    
    /// Get memory summary
    pub fn summary(&self) -> String {
        format!(
            "Memory[{}]: type={}, strength={:.3}, relevance={:.3}, emotional={:.3}, age={:.1}s",
            self.memory_id,
            self.memory_type,
            self.strength,
            self.relevance,
            self.emotional_salience,
            self.age()
        )
    }
    
    /// Validate memory entry
    pub fn validate(&self) -> Result<(), String> {
        if self.memory_id == 0 {
            return Err("Memory ID cannot be 0".to_string());
        }
        
        if self.activation < 0.0 {
            return Err("Activation cannot be negative".to_string());
        }
        
        if self.relevance < 0.0 || self.relevance > 1.0 {
            return Err("Relevance must be between 0.0 and 1.0".to_string());
        }
        
        if self.emotional_salience < 0.0 || self.emotional_salience > 1.0 {
            return Err("Emotional salience must be between 0.0 and 1.0".to_string());
        }
        
        if self.strength < 0.0 {
            return Err("Strength cannot be negative".to_string());
        }
        
        if let Some(ref embedding) = self.embedding {
            if embedding.len() != self.embedding_dim {
                return Err("Embedding dimension mismatch".to_string());
            }
        }
        
        Ok(())
    }
}
