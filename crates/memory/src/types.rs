//! Memory Model Types
//! 
//! Extracted dari memory_model.rs untuk modular structure

use serde::{Serialize, Deserialize};

/// Memory types (Level 3: Multi-type memory system)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    Episodic = 0,    // Specific events/experiences (fast decay, high emotional)
    Semantic = 1,    // General knowledge/facts (slow decay, low emotional)
    Procedural = 2,  // Skills/habits (very slow decay, reinforcement-based)
    Working = 3,     // Working memory (temporary, active processing)
    User = 4,        // User-specific memories
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::Episodic => write!(f, "Episodic"),
            MemoryType::Semantic => write!(f, "Semantic"),
            MemoryType::Procedural => write!(f, "Procedural"),
            MemoryType::Working => write!(f, "Working"),
            MemoryType::User => write!(f, "User"),
        }
    }
}

impl MemoryType {
    /// Get decay rate for this memory type
    pub fn decay_rate(&self) -> f64 {
        match self {
            MemoryType::Episodic => 0.1,    // Fast decay
            MemoryType::Semantic => 0.01,   // Slow decay
            MemoryType::Procedural => 0.001, // Very slow decay
            MemoryType::Working => 0.5,     // Very fast decay
            MemoryType::User => 0.05,       // Medium decay
        }
    }
    
    /// Get default emotional salience for this memory type
    pub fn default_emotional_salience(&self) -> f64 {
        match self {
            MemoryType::Episodic => 0.8,    // High emotional
            MemoryType::Semantic => 0.2,    // Low emotional
            MemoryType::Procedural => 0.4,  // Medium emotional
            MemoryType::Working => 0.1,     // Low emotional (temporary)
            MemoryType::User => 0.9,       // Very high emotional
        }
    }
    
    /// Get default relevance for this memory type
    pub fn default_relevance(&self) -> f64 {
        match self {
            MemoryType::Episodic => 0.6,    // Medium relevance
            MemoryType::Semantic => 0.8,    // High relevance
            MemoryType::Procedural => 0.7,  // High relevance
            MemoryType::Working => 0.3,     // Low relevance (temporary)
            MemoryType::User => 0.9,       // Very high relevance
        }
    }
}

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
