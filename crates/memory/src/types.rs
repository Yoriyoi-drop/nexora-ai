//! Memory Model Types
//! 
//! Extracted dari memory_model.rs untuk modular structure

use serde::{Serialize, Deserialize};
pub use crate::memory_model::MemoryType;



/// Memory entry with all properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub memory_id: u32,           // Unique identifier for this memory
    pub memory_type: MemoryType,  // Memory type (episodic/semantic/procedural)
    pub activation: f64,         // A_ik: How often accessed/activated
    pub relevance: f64,          // R_ik: Context/goal/reward importance
    pub emotional_salience: f64, // E_ik: Emotional weight (limbic modulation)
    pub timestamp: f64,          // t_k: When this memory was created/last updated
    pub strength: f64,           // M_i(t): Current memory strength
    pub content: Option<String>,  // Memory content (optional)
    pub embedding: Option<Vec<f32>>, // Semantic embedding vector (optional)
    pub embedding_dim: usize,    // Dimension of embedding
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub memories_by_type: HashMap<MemoryType, usize>,
    pub average_strength: f64,
    pub total_activations: f64,
    pub oldest_memory_timestamp: Option<f64>,
    pub newest_memory_timestamp: Option<f64>,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_memories: 0,
            memories_by_type: HashMap::new(),
            average_strength: 0.0,
            total_activations: 0.0,
            oldest_memory_timestamp: None,
            newest_memory_timestamp: None,
        }
    }
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_memories: usize,
    pub decay_enabled: bool,
    pub consolidation_enabled: bool,
    pub emotional_modulation_enabled: bool,
    pub embedding_enabled: bool,
    pub working_memory_capacity: usize,
    pub consolidation_threshold: f64,
    pub forgetting_threshold: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memories: 10000,
            decay_enabled: true,
            consolidation_enabled: true,
            emotional_modulation_enabled: true,
            embedding_enabled: true,
            working_memory_capacity: 100,
            consolidation_threshold: 0.7,
            forgetting_threshold: 0.01,
        }
    }
}

use std::collections::HashMap;
