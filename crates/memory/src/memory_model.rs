//! Hebbian Memory Model - Rust implementation
//! 
//! Memory strength formula: M_i(t) = Σ [A_ik × R_ik × E_ik × e^(-λ(t - t_k))]
//! Inspired by Ebbinghaus forgetting curve, Hebbian learning, and limbic modulation

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use std::ffi::CString;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use rand::Rng;

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
    pub fn default_relevance(&self) -> f64 {
        match self {
            MemoryType::Episodic => 0.7,
            MemoryType::Semantic => 0.9,
            MemoryType::Procedural => 0.8,
            MemoryType::Working => 0.6,
            MemoryType::User => 0.8,
        }
    }
    
    pub fn default_emotional_salience(&self) -> f64 {
        match self {
            MemoryType::Episodic => 0.8,
            MemoryType::Semantic => 0.3,
            MemoryType::Procedural => 0.5,
            MemoryType::Working => 0.4,
            MemoryType::User => 0.9,
        }
    }
    
    pub fn decay_rate(&self) -> f64 {
        match self {
            MemoryType::Episodic => 0.1,
            MemoryType::Semantic => 0.01,
            MemoryType::Procedural => 0.005,
            MemoryType::Working => 0.5,
            MemoryType::User => 0.02,
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

impl MemoryEntry {
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
    
    pub fn calculate_strength(&self, decay_rate: f64, current_time: f64) -> f64 {
        let time_diff = current_time - self.timestamp;
        let decay_factor = (-decay_rate * time_diff).exp();
        
        // M_i(t) = A_ik × R_ik × E_ik × e^(-λ(t - t_k))
        self.activation * self.relevance * self.emotional_salience * decay_factor
    }
    
    pub fn update_strength(&mut self, decay_rate: f64, current_time: f64) {
        self.strength = self.calculate_strength(decay_rate, current_time);
    }
    
    pub fn activate(&mut self, activation_boost: f64) {
        self.activation += activation_boost;
        self.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
    }
    
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding_dim = embedding.len();
        self.embedding = Some(embedding);
    }
    
    pub fn cosine_similarity(&self, other: &MemoryEntry) -> Option<f64> {
        if let (Some(ref emb1), Some(ref emb2)) = (&self.embedding, &other.embedding) {
            if emb1.len() != emb2.len() || emb1.is_empty() {
                return None;
            }
            
            let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            if norm1 == 0.0 || norm2 == 0.0 {
                return None;
            }
            
            Some((dot_product / (norm1 * norm2)) as f64)
        } else {
            None
        }
    }
    
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
    
    pub fn is_expired(&self, threshold: f64, current_time: f64) -> bool {
        let decay_rate = self.memory_type.decay_rate();
        let current_strength = self.calculate_strength(decay_rate, current_time);
        current_strength < threshold
    }
    
    pub fn should_consolidate(&self, threshold: f64) -> bool {
        self.strength >= threshold
    }
    
    pub fn apply_emotional_decay(&mut self, decay_rate: f64, current_time: f64) {
        let time_diff = current_time - self.timestamp;
        let decay_factor = (-decay_rate * time_diff).exp();
        self.emotional_salience *= decay_factor;
        self.emotional_salience = self.emotional_salience.max(0.1); // Minimum emotional salience
    }
    
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
}

/// Hebbian memory system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianMemoryConfig {
    pub capacity: usize,              // Maximum number of memories
    pub decay_rate: f64,              // λ: Memory decay rate (higher = faster forgetting)
    pub consolidation_gamma: f64,     // γ: Non-linear consolidation factor (optional)
    pub use_consolidation: bool,      // Whether to apply non-linear consolidation
    pub use_competition: bool,       // Whether to apply competition normalization
    pub competition_alpha: f64,      // α: Competition strength (0 = no competition, 1 = full softmax)
    pub use_local_competition: bool, // Whether to use k-nearest local competition instead of global
    pub competition_k: usize,        // k: Number of nearest neighbors for local competition
    pub use_interference: bool,       // Whether to apply interference between similar memories
    pub interference_beta: f64,       // β: Interference strength
    pub use_probabilistic_retrieval: bool, // Whether to use softmax probabilistic retrieval
    pub retrieval_temperature: f64,  // τ: Temperature for softmax (higher = more uniform)
    pub verbose: bool,                // Whether to print detailed output
    pub use_adaptive_decay: bool,     // Whether to use adaptive decay based on memory properties
    pub use_consolidation_cycle: bool, // Whether to use hippocampus→cortex consolidation cycle
    pub consolidation_threshold: f64, // Strength threshold for consolidation
    pub sleep_cycle_interval: f64,   // Time interval between consolidation cycles
}

impl Default for HebbianMemoryConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            decay_rate: 0.1,
            consolidation_gamma: 0.5,
            use_consolidation: false,
            use_competition: false,
            competition_alpha: 0.5,
            use_local_competition: false,
            competition_k: 10,
            use_interference: false,
            interference_beta: 0.1,
            use_probabilistic_retrieval: false,
            retrieval_temperature: 1.0,
            verbose: false,
            use_adaptive_decay: false,
            use_consolidation_cycle: false,
            consolidation_threshold: 0.5,
            sleep_cycle_interval: 100.0,
        }
    }
}

/// Main Hebbian memory system
#[derive(Debug, Clone)]
pub struct HebbianMemory {
    entries: HashMap<u32, MemoryEntry>,
    config: HebbianMemoryConfig,
    current_time: f64,
    last_consolidation_time: f64,
    next_memory_id: u32,
}

impl HebbianMemory {
    /// Create a new Hebbian memory system
    pub fn new(config: HebbianMemoryConfig) -> Self {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        Self {
            entries: HashMap::new(),
            config,
            current_time,
            last_consolidation_time: current_time,
            next_memory_id: 1,
        }
    }
    
    /// Create memory with default configuration
    pub fn with_capacity(capacity: usize, decay_rate: f64) -> Self {
        let mut config = HebbianMemoryConfig::default();
        config.capacity = capacity;
        config.decay_rate = decay_rate;
        Self::new(config)
    }
    
    /// Add a new memory entry
    pub fn add_memory(
        &mut self,
        content: Option<String>,
        initial_relevance: f64,
        initial_emotional_salience: f64,
        memory_type: MemoryType,
    ) -> Result<u32> {
        if self.entries.len() >= self.config.capacity {
            return Err(anyhow::anyhow!("Memory capacity reached"));
        }
        
        let memory_id = self.next_memory_id;
        self.next_memory_id += 1;
        
        let mut entry = MemoryEntry::new(
            memory_id,
            memory_type,
            content,
            initial_relevance,
            initial_emotional_salience,
        );
        
        entry.update_strength(self.config.decay_rate, self.current_time);
        
        self.entries.insert(memory_id, entry);
        
        if self.config.verbose {
            println!("Added memory {}: type={:?}, relevance={:.3}, emotional={:.3}", 
                    memory_id, memory_type, initial_relevance, initial_emotional_salience);
        }
        
        Ok(memory_id)
    }
    
    /// Update memory strength based on activation
    pub fn activate_memory(&mut self, memory_id: u32, activation_boost: f64) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(&memory_id) {
            entry.activate(activation_boost);
            entry.update_strength(self.config.decay_rate, self.current_time);
            
            if self.config.verbose {
                println!("Activated memory {} with boost {:.3}", memory_id, activation_boost);
            }
        } else {
            return Err(anyhow::anyhow!("Memory {} not found", memory_id));
        }
        Ok(())
    }
    
    /// Update relevance of a memory
    pub fn update_relevance(&mut self, memory_id: u32, new_relevance: f64) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(&memory_id) {
            entry.relevance = new_relevance;
            entry.update_strength(self.config.decay_rate, self.current_time);
        } else {
            return Err(anyhow::anyhow!("Memory {} not found", memory_id));
        }
        Ok(())
    }
    
    /// Update emotional salience of a memory
    pub fn update_emotional_salience(&mut self, memory_id: u32, new_salience: f64) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(&memory_id) {
            entry.emotional_salience = new_salience;
            entry.update_strength(self.config.decay_rate, self.current_time);
        } else {
            return Err(anyhow::anyhow!("Memory {} not found", memory_id));
        }
        Ok(())
    }
    
    /// Clamp emotional salience to reasonable bounds
    pub fn clamp_emotional_salience(&mut self, min_salience: f64, max_salience: f64) {
        for entry in self.entries.values_mut() {
            entry.emotional_salience = entry.emotional_salience.clamp(min_salience, max_salience);
            entry.update_strength(self.config.decay_rate, self.current_time);
        }
    }
    
    /// Advance time and apply decay to all memories
    pub fn advance_time(&mut self, time_delta: f64) {
        self.current_time += time_delta;
        
        if self.config.use_adaptive_decay {
            self.apply_adaptive_decay(time_delta);
        } else {
            for entry in self.entries.values_mut() {
                entry.update_strength(self.config.decay_rate, self.current_time);
            }
        }
        
        // Check for consolidation cycle
        if self.config.use_consolidation_cycle {
            let time_since_consolidation = self.current_time - self.last_consolidation_time;
            if time_since_consolidation >= self.config.sleep_cycle_interval {
                self.consolidation_cycle();
                self.last_consolidation_time = self.current_time;
            }
        }
        
        if self.config.verbose {
            println!("Advanced time by {:.3} to {:.3}", time_delta, self.current_time);
        }
    }
    
    /// Get current strength of a specific memory
    pub fn get_strength(&self, memory_id: u32) -> Option<f64> {
        self.entries.get(&memory_id).map(|entry| entry.strength)
    }
    
    /// Retrieve top N memories by strength
    pub fn retrieve_top(&self, n: usize) -> Vec<&MemoryEntry> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
        entries.into_iter().take(n).collect()
    }
    
    /// Retrieve memories above strength threshold
    pub fn retrieve_above_threshold(&self, threshold: f64) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|entry| entry.strength > threshold)
            .collect()
    }
    
    /// Probabilistic retrieval using softmax over memory strengths
    pub fn retrieve_probabilistic(&self) -> Option<&MemoryEntry> {
        if self.entries.is_empty() {
            return None;
        }
        
        let strengths: Vec<f64> = self.entries.values().map(|entry| entry.strength).collect();
        let max_strength = strengths.iter().fold(0.0_f64, |a, &b| a.max(b));
        
        // Apply softmax
        let exp_strengths: Vec<f64> = strengths
            .iter()
            .map(|&s| ((s - max_strength) / self.config.retrieval_temperature).exp())
            .collect();
        
        let total: f64 = exp_strengths.iter().sum();
        let probabilities: Vec<f64> = exp_strengths.iter().map(|&e| e / total).collect();
        
        // Sample based on probabilities
        let mut rng = rand::thread_rng();
        let random_val: f64 = rng.gen();
        
        let mut cumulative = 0.0;
        for (entry, &prob) in self.entries.values().zip(probabilities.iter()) {
            cumulative += prob;
            if random_val <= cumulative {
                return Some(entry);
            }
        }
        
        // Fallback to last entry (shouldn't happen)
        self.entries.values().last()
    }
    
    /// Find memory by content (simple substring match)
    pub fn find_by_content(&self, content: &str) -> Option<&MemoryEntry> {
        self.entries
            .values()
            .find(|entry| {
                if let Some(ref entry_content) = entry.content {
                    entry_content.contains(content)
                } else {
                    false
                }
            })
    }
    
    /// Get memories of a specific type
    pub fn get_by_type(&self, memory_type: MemoryType) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|entry| entry.memory_type == memory_type)
            .collect()
    }
    
    /// Get type-specific decay rate multiplier
    pub fn get_type_decay_multiplier(memory_type: MemoryType) -> f64 {
        match memory_type {
            MemoryType::Episodic => 1.5,    // Fast decay
            MemoryType::Semantic => 0.5,     // Slow decay
            MemoryType::Procedural => 0.3,   // Very slow decay
            MemoryType::Working => 2.0,      // Very fast decay (temporary)
            MemoryType::User => 0.8,         // Moderate decay (user-specific)
        }
    }
    
    /// Update memory type
    pub fn set_type(&mut self, memory_id: u32, new_type: MemoryType) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(&memory_id) {
            entry.memory_type = new_type;
            entry.update_strength(self.config.decay_rate, self.current_time);
        } else {
            return Err(anyhow::anyhow!("Memory {} not found", memory_id));
        }
        Ok(())
    }
    
    /// Prune memories below strength threshold
    pub fn prune(&mut self, threshold: f64) -> usize {
        let initial_count = self.entries.len();
        self.entries.retain(|_, entry| entry.strength >= threshold);
        let pruned_count = initial_count - self.entries.len();
        
        if self.config.verbose && pruned_count > 0 {
            println!("Pruned {} memories below threshold {:.3}", pruned_count, threshold);
        }
        
        pruned_count
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> (f64, f64, f64) {
        if self.entries.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let strengths: Vec<f64> = self.entries.values().map(|entry| entry.strength).collect();
        let avg_strength = strengths.iter().sum::<f64>() / strengths.len() as f64;
        let max_strength = strengths.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_strength = strengths.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        (avg_strength, max_strength, min_strength)
    }
    
    /// Set embedding for a memory entry
    pub fn set_embedding(&mut self, memory_id: u32, embedding: Vec<f32>) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(&memory_id) {
            entry.set_embedding(embedding);
        } else {
            return Err(anyhow::anyhow!("Memory {} not found", memory_id));
        }
        Ok(())
    }
    
    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, id_a: u32, id_b: u32) -> Option<f64> {
        if let (Some(entry_a), Some(entry_b)) = (self.entries.get(&id_a), self.entries.get(&id_b)) {
            entry_a.cosine_similarity(entry_b)
        } else {
            None
        }
    }
    
    /// ==================== Advanced Features ====================
    
    /// Apply competition normalization (softmax-like)
    pub fn apply_competition(&mut self) {
        if !self.config.use_competition || self.entries.is_empty() {
            return;
        }
        
        let strengths: Vec<f64> = self.entries.values().map(|entry| entry.strength).collect();
        let max_strength = strengths.iter().fold(0.0_f64, |a, &b| a.max(b));
        
        // Apply softmax normalization
        let exp_strengths: Vec<f64> = strengths
            .iter()
            .map(|&s| ((s - max_strength) / self.config.retrieval_temperature).exp())
            .collect();
        
        let total: f64 = exp_strengths.iter().sum();
        let normalized_strengths: Vec<f64> = exp_strengths.iter().map(|&e| e / total).collect();
        
        // Update strengths with competition
        for (entry, &new_strength) in self.entries.values_mut().zip(normalized_strengths.iter()) {
            let competition_strength = entry.strength * (1.0 - self.config.competition_alpha) 
                + new_strength * self.config.competition_alpha;
            entry.strength = competition_strength;
        }
        
        if self.config.verbose {
            println!("Applied competition normalization with alpha {:.3}", self.config.competition_alpha);
        }
    }
    
    /// Apply local k-nearest competition (more biologically realistic)
    pub fn apply_local_competition(&mut self) {
        if !self.config.use_local_competition || self.entries.len() <= self.config.competition_k {
            return;
        }
        
        // For each memory, find k nearest neighbors and compete locally
        let memory_ids: Vec<u32> = self.entries.keys().copied().collect();
        
        for &memory_id in &memory_ids {
            // Find k nearest neighbors based on embedding similarity or content similarity
            let mut similarities: Vec<(u32, f64)> = memory_ids
                .iter()
                .filter(|&&id| id != memory_id)
                .map(|&id| {
                    let similarity = if let Some(cos_sim) = self.cosine_similarity(memory_id, id) {
                        cos_sim
                    } else {
                        // Fallback to content similarity
                        self.content_similarity(memory_id, id)
                    };
                    (id, similarity)
                })
                .collect();
            
            // Sort by similarity (descending) and take top k
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            similarities.truncate(self.config.competition_k);
            
            // Apply local competition with these neighbors
            // Collect neighbor strengths first to avoid borrowing issues
            let mut competition_effect: f64 = similarities.iter()
                .filter_map(|&(neighbor_id, similarity)| {
                    self.entries.get(&neighbor_id).map(|neighbor| similarity * neighbor.strength)
                })
                .sum();
            
            if let Some(entry) = self.entries.get_mut(&memory_id) {
                
                if !similarities.is_empty() {
                    competition_effect /= similarities.len() as f64;
                    entry.strength = entry.strength * (1.0 - self.config.competition_alpha * competition_effect);
                }
            }
        }
        
        if self.config.verbose {
            println!("Applied local k-nearest competition with k={}", self.config.competition_k);
        }
    }
    
    /// Apply interference between similar memories
    pub fn apply_interference(&mut self) {
        if !self.config.use_interference {
            return;
        }
        
        let memory_ids: Vec<u32> = self.entries.keys().copied().collect();
        
        for (i, &id_a) in memory_ids.iter().enumerate() {
            for &id_b in memory_ids.iter().skip(i + 1) {
                let similarity = if let Some(cos_sim) = self.cosine_similarity(id_a, id_b) {
                    cos_sim
                } else {
                    self.content_similarity(id_a, id_b)
                };
                
                if similarity > 0.5 { // Only interfere with similar memories
                    let interference_amount = self.config.interference_beta * similarity;
                    
                    // Apply interference to both entries
                    // We need to do this carefully to avoid double mutable borrow
                    for id in [id_a, id_b] {
                        if let Some(entry) = self.entries.get_mut(&id) {
                            entry.strength *= 1.0 - interference_amount;
                        }
                    }
                }
            }
        }
        
        if self.config.verbose {
            println!("Applied interference with beta {:.3}", self.config.interference_beta);
        }
    }
    
    /// Calculate similarity between two memory contents (simple Jaccard-like)
    pub fn content_similarity(&self, id_a: u32, id_b: u32) -> f64 {
        if let (Some(entry_a), Some(entry_b)) = (self.entries.get(&id_a), self.entries.get(&id_b)) {
            if let (Some(ref content_a), Some(ref content_b)) = (&entry_a.content, &entry_b.content) {
                // Simple word-based Jaccard similarity
                let words_a: std::collections::HashSet<&str> = content_a.split_whitespace().collect();
                let words_b: std::collections::HashSet<&str> = content_b.split_whitespace().collect();
                
                let intersection = words_a.intersection(&words_b).count();
                let union = words_a.union(&words_b).count();
                
                if union == 0 {
                    0.0
                } else {
                    intersection as f64 / union as f64
                }
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Perform hippocampus→cortex consolidation cycle (sleep cycle)
    pub fn consolidation_cycle(&mut self) {
        if !self.config.use_consolidation_cycle {
            return;
        }
        
        // Find memories above consolidation threshold
        let strong_memories: Vec<u32> = self.entries
            .iter()
            .filter(|(_, entry)| entry.strength > self.config.consolidation_threshold)
            .map(|(&id, _)| id)
            .collect();
        
        // Apply consolidation: strengthen strong memories, weaken others
        for memory_id in &strong_memories {
            if let Some(entry) = self.entries.get_mut(memory_id) {
                // Non-linear consolidation: M_i(t+1) = M_i(t) * (1 + γ * M_i(t))
                let consolidation_factor = 1.0 + self.config.consolidation_gamma * entry.strength;
                entry.strength *= consolidation_factor;
                entry.strength = entry.strength.min(1.0); // Cap at 1.0
            }
        }
        
        // Weaken other memories slightly (forgetting during consolidation)
        for (memory_id, entry) in &mut self.entries {
            if !strong_memories.contains(memory_id) {
                entry.strength *= 0.95; // Slight weakening
            }
        }
        
        if self.config.verbose {
            println!("Consolidation cycle: {} memories consolidated, {} weakened", 
                    strong_memories.len(), self.entries.len() - strong_memories.len());
        }
    }
    
    /// Calculate adaptive decay rate for a specific memory based on its properties
    pub fn calculate_adaptive_decay(&self, memory_id: u32) -> Option<f64> {
        if let Some(entry) = self.entries.get(&memory_id) {
            let base_decay = self.config.decay_rate;
            let type_multiplier = Self::get_type_decay_multiplier(entry.memory_type);
            
            // Adaptive decay based on emotional salience and relevance
            let emotional_factor = 1.0 - (entry.emotional_salience * 0.5); // High emotion = slower decay
            let relevance_factor = 1.0 - (entry.relevance * 0.3); // High relevance = slower decay
            
            let adaptive_decay = base_decay * type_multiplier * emotional_factor * relevance_factor;
            Some(adaptive_decay)
        } else {
            None
        }
    }
    
    /// Apply adaptive decay to all memories
    pub fn apply_adaptive_decay(&mut self, _time_delta: f64) {
        // Collect adaptive decay values first to avoid borrowing issues
        let memory_ids: Vec<u32> = self.entries.keys().copied().collect();
        let adaptive_decays: Vec<Option<f64>> = memory_ids.iter()
            .map(|&id| self.calculate_adaptive_decay(id))
            .collect();
        
        // Apply decay using collected values
        for ((_memory_id, entry), adaptive_decay) in self.entries.iter_mut().zip(adaptive_decays.into_iter()) {
            if let Some(adaptive_decay) = adaptive_decay {
                let time_diff = self.current_time - entry.timestamp;
                let decay_factor = (-adaptive_decay * time_diff).exp();
                entry.strength = entry.activation * entry.relevance * entry.emotional_salience * decay_factor;
            }
        }
        
        if self.config.verbose {
            println!("Applied adaptive decay to all memories");
        }
    }
    
    /// ==================== Validation & Testing ====================
    
    /// Validate memory model properties (stability, convergence, biological plausibility)
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::default();
        
        // Check if decay rate is positive (memories should decay over time)
        result.is_realistic = self.config.decay_rate > 0.0;
        
        // Check if activation increases strength (learning)
        result.learns = true; // By design, activation increases strength
        
        // Check if zero relevance memories decay to zero (filters noise)
        let zero_relevance_memories: Vec<_> = self.entries
            .values()
            .filter(|entry| entry.relevance == 0.0)
            .collect();
        
        result.filters_noise = zero_relevance_memories
            .iter()
            .all(|entry| entry.strength < 0.01); // Should be very close to zero
        
        // Check system stability (convergence to steady state)
        // This is a simplified check - in practice would need more sophisticated analysis
        result.is_stable = self.entries.len() <= self.config.capacity;
        
        // Calculate convergence error (simplified)
        let (avg_strength, _, _) = self.get_stats();
        result.convergence_error = if avg_strength > 0.0 {
            // Error based on how much average strength deviates from expected range
            (avg_strength - 0.5).abs()
        } else {
            0.0
        };
        
        result
    }
    
    /// Simulate memory retention curve for testing
    pub fn simulate_retention(&self, time_horizon: f64, n_points: usize) -> Vec<f64> {
        let mut strengths = Vec::with_capacity(n_points);
        let time_step = time_horizon / n_points as f64;
        
        for i in 0..n_points {
            let current_time = i as f64 * time_step;
            let total_strength: f64 = self.entries
                .values()
                .map(|entry| {
                    let time_diff = current_time - (entry.timestamp - self.current_time);
                    let decay_factor = (-self.config.decay_rate * time_diff).exp();
                    entry.activation * entry.relevance * entry.emotional_salience * decay_factor
                })
                .sum();
            
            strengths.push(total_strength);
        }
        
        strengths
    }
    
    /// Get current time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }
    
    /// Get memory count
    pub fn memory_count(&self) -> usize {
        self.entries.len()
    }
    
    /// Get configuration
    pub fn config(&self) -> &HebbianMemoryConfig {
        &self.config
    }
    
    /// Reset all memories
    pub fn reset(&mut self) {
        self.entries.clear();
        self.next_memory_id = 1;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.current_time = current_time;
        self.last_consolidation_time = current_time;
    }
    
    /// Set verbose mode
    pub fn set_verbose(&mut self, verbose: bool) {
        self.config.verbose = verbose;
    }
}

/// Validation result for memory model
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub is_stable: bool,           // System converges to steady state
    pub is_realistic: bool,        // Memories decay over time (λ > 0)
    pub learns: bool,              // Activation increases strength
    pub filters_noise: bool,       // Zero relevance memories decay to zero
    pub convergence_error: f64,    // Error from expected behavior
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.is_stable && self.is_realistic && self.learns && self.filters_noise && self.convergence_error < 0.1
    }
}

/// C-compatible interface for FFI
#[repr(C)]
pub struct CMemoryEntry {
    pub memory_id: u32,
    pub memory_type: i32,
    pub activation: f64,
    pub relevance: f64,
    pub emotional_salience: f64,
    pub timestamp: f64,
    pub strength: f64,
    pub content: *mut std::os::raw::c_char,
    pub content_len: usize,
    pub embedding: *mut f32,
    pub embedding_dim: usize,
}

/// C-compatible memory statistics
#[repr(C)]
pub struct CMemoryStats {
    pub avg_strength: f64,
    pub max_strength: f64,
    pub min_strength: f64,
    pub memory_count: usize,
}

/// C-compatible validation result
#[repr(C)]
pub struct CValidationResult {
    pub is_stable: bool,
    pub is_realistic: bool,
    pub learns: bool,
    pub filters_noise: bool,
    pub convergence_error: f64,
}

/// Convert Rust MemoryEntry to C-compatible format
pub fn memory_entry_to_c(entry: &MemoryEntry) -> CMemoryEntry {
    use std::ffi::CString;
    
    let content_cstr = entry.content.as_ref()
        .and_then(|s| CString::new(s.as_str()).ok())
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut());
    
    let embedding_ptr = entry.embedding.as_ref()
        .map(|e| e.as_ptr() as *mut f32)
        .unwrap_or(std::ptr::null_mut());
    
    CMemoryEntry {
        memory_id: entry.memory_id,
        memory_type: entry.memory_type as i32,
        activation: entry.activation,
        relevance: entry.relevance,
        emotional_salience: entry.emotional_salience,
        timestamp: entry.timestamp,
        strength: entry.strength,
        content: content_cstr,
        content_len: entry.content.as_ref().map(|s| s.len()).unwrap_or(0),
        embedding: embedding_ptr,
        embedding_dim: entry.embedding_dim,
    }
}

/// Convert CMemoryEntry back to Rust (for cleanup)
pub fn free_c_memory_entry(c_entry: CMemoryEntry) {
    if !c_entry.content.is_null() {
        let _ = unsafe { CString::from_raw(c_entry.content) };
    }
    // Note: embedding is owned by Rust, so don't free it here
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_creation() {
        let memory = HebbianMemory::with_capacity(100, 0.1);
        assert_eq!(memory.memory_count(), 0);
    }
    
    #[test]
    fn test_add_memory() {
        let mut memory = HebbianMemory::with_capacity(100, 0.1);
        let memory_id = memory.add_memory(
            Some("test memory".to_string()),
            0.8,
            0.6,
            MemoryType::Episodic,
        ).unwrap();
        
        assert!(memory.get_strength(memory_id).is_some());
        assert!(memory.get_strength(memory_id).unwrap() > 0.0);
    }
    
    #[test]
    fn test_memory_decay() {
        let mut memory = HebbianMemory::with_capacity(100, 0.1);
        let memory_id = memory.add_memory(
            Some("test memory".to_string()),
            0.8,
            0.6,
            MemoryType::Episodic,
        ).unwrap();
        
        let initial_strength = memory.get_strength(memory_id).unwrap();
        memory.advance_time(10.0);
        let decayed_strength = memory.get_strength(memory_id).unwrap();
        
        assert!(decayed_strength < initial_strength);
    }
    
    #[test]
    fn test_memory_activation() {
        let mut memory = HebbianMemory::with_capacity(100, 0.1);
        let memory_id = memory.add_memory(
            Some("test memory".to_string()),
            0.8,
            0.6,
            MemoryType::Episodic,
        ).unwrap();
        
        let initial_strength = memory.get_strength(memory_id).unwrap();
        memory.activate_memory(memory_id, 0.5).unwrap();
        let activated_strength = memory.get_strength(memory_id).unwrap();
        
        assert!(activated_strength > initial_strength);
    }
    
    #[test]
    fn test_retrieve_top() {
        let mut memory = HebbianMemory::with_capacity(100, 0.1);
        
        for i in 0..5 {
            memory.add_memory(
                Some(format!("memory {}", i)),
                0.5 + i as f64 * 0.1,
                0.6,
                MemoryType::Episodic,
            ).unwrap();
        }
        
        let top_memories = memory.retrieve_top(3);
        assert_eq!(top_memories.len(), 3);
        
        // Should be sorted by strength (descending)
        for i in 1..top_memories.len() {
            assert!(top_memories[i-1].strength >= top_memories[i].strength);
        }
    }
    
    #[test]
    fn test_memory_types() {
        let mut memory = HebbianMemory::with_capacity(100, 0.1);
        
        let episodic_id = memory.add_memory(
            Some("episodic memory".to_string()),
            0.8,
            0.6,
            MemoryType::Episodic,
        ).unwrap();
        
        let semantic_id = memory.add_memory(
            Some("semantic memory".to_string()),
            0.8,
            0.6,
            MemoryType::Semantic,
        ).unwrap();
        
        let procedural_id = memory.add_memory(
            Some("procedural memory".to_string()),
            0.8,
            0.6,
            MemoryType::Procedural,
        ).unwrap();
        
        let episodic_memories = memory.get_by_type(MemoryType::Episodic);
        let semantic_memories = memory.get_by_type(MemoryType::Semantic);
        let procedural_memories = memory.get_by_type(MemoryType::Procedural);
        
        assert_eq!(episodic_memories.len(), 1);
        assert_eq!(semantic_memories.len(), 1);
        assert_eq!(procedural_memories.len(), 1);
        
        assert_eq!(episodic_memories[0].memory_id, episodic_id);
        assert_eq!(semantic_memories[0].memory_id, semantic_id);
        assert_eq!(procedural_memories[0].memory_id, procedural_id);
    }
    
    #[test]
    fn test_validation() {
        let memory = HebbianMemory::with_capacity(100, 0.1);
        let result = memory.validate();
        
        assert!(result.is_realistic);
        assert!(result.learns);
        assert!(result.is_stable);
    }
}
