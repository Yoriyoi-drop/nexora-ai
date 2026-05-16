//! Episodic Memory Retention (EMR)
//!
//! Memory eksternal dengan selective-write untuk:
//! - Mengurangi memory overwrite
//! - Mempertahankan informasi penting
//! - Scalable episodic context
//! - Long-term memory retention

use crate::{DLResult, DeepLearningError};
use crate::star_x::core::EpisodicMemory;
use ndarray::{ArrayD, Array1, Array2};
use std::collections::HashMap;

/// Episodic memory entry
#[derive(Debug, Clone)]
pub struct EmrMemoryEntry {
    pub id: usize,
    pub content: ArrayD<f32>,
    pub timestamp: usize,
    pub priority: f32,
    pub access_count: usize,
    pub last_access: usize,
    pub relevance_score: f32,
}

impl EmrMemoryEntry {
    pub fn new(id: usize, content: ArrayD<f32>, timestamp: usize) -> Self {
        Self {
            id,
            content,
            timestamp,
            priority: 0.0,
            access_count: 0,
            last_access: timestamp,
            relevance_score: 0.0,
        }
    }
    
    pub fn update_access(&mut self, current_time: usize) {
        self.access_count += 1;
        self.last_access = current_time;
    }
    
    pub fn compute_decay_factor(&self, current_time: usize, decay_rate: f32) -> f32 {
        let time_diff = current_time - self.last_access;
        decay_rate.powf(time_diff as f32)
    }
}

/// Episodic Memory Retention implementation
#[derive(Debug, Clone)]
pub struct EpisodicMemoryRetention {
    // Memory storage
    memory_entries: HashMap<usize, EmrMemoryEntry>,
    memory_matrix: Array2<f32>, // [max_size x hidden_size]
    
    // Memory management
    max_size: usize,
    current_size: usize,
    next_id: usize,
    current_time: usize,
    
    // Priority computation parameters
    relevance_weight: f32,
    gradient_weight: f32,
    recency_weight: f32,
    frequency_weight: f32,
    
    // Memory decay and cleanup
    decay_rate: f32,
    cleanup_threshold: f32,
    cleanup_interval: usize,
    last_cleanup: usize,
    
    // Retrieval parameters
    retrieval_top_k: usize,
    similarity_threshold: f32,
    
    // Statistics
    total_writes: usize,
    total_reads: usize,
    cleanup_count: usize,
    avg_priority: f32,
}

impl EpisodicMemoryRetention {
    pub fn new(
        max_size: usize,
        hidden_size: usize,
        _write_threshold: f32,
    ) -> DLResult<Self> {
        if hidden_size == 0 {
            return Err(DeepLearningError::Configuration {
                reason: "Hidden size must be greater than 0".to_string(),
            });
        }
        
        Ok(Self {
            memory_entries: HashMap::new(),
            memory_matrix: Array2::zeros((max_size, hidden_size)),
            max_size,
            current_size: 0,
            next_id: 0,
            current_time: 0,
            relevance_weight: 0.4,
            gradient_weight: 0.3,
            recency_weight: 0.2,
            frequency_weight: 0.1,
            decay_rate: 0.99,
            cleanup_threshold: 0.1,
            cleanup_interval: 100,
            last_cleanup: 0,
            retrieval_top_k: 10,
            similarity_threshold: 0.5,
            total_writes: 0,
            total_reads: 0,
            cleanup_count: 0,
            avg_priority: 0.0,
        })
    }
    
    /// Compute priority score untuk memory write decision
    pub fn compute_priority_score(&self, 
        state: &ArrayD<f32>,
        gradient: &ArrayD<f32>,
        relevance: f32
    ) -> DLResult<f32> {
        
        // H(x) - entropy/complexity of state
        let state_entropy = self.compute_state_entropy(state);
        
        // ||∇h|| - gradient magnitude
        let gradient_norm = self.compute_gradient_norm(gradient);
        
        // r_t - relevance score
        let relevance_score = relevance;
        
        // Combined priority score
        let priority = self.relevance_weight * relevance_score +
                      self.gradient_weight * gradient_norm.log10().max(0.0) +
                      self.recency_weight * state_entropy +
                      self.frequency_weight * 0.1; // Base frequency contribution
        
        Ok(priority)
    }
    
    /// Compute entropy/complexity of state
    fn compute_state_entropy(&self, state: &ArrayD<f32>) -> f32 {
        let state_flat = state.as_slice().expect("tensor should be contiguous");
        
        // Normalize to probability distribution
        let mut sum = 0.0;
        for &val in state_flat {
            sum += val.abs();
        }
        
        if sum == 0.0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &val in state_flat {
            let prob = val.abs() / sum;
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        
        entropy
    }
    
    /// Compute gradient norm
    fn compute_gradient_norm(&self, gradient: &ArrayD<f32>) -> f32 {
        let mut sum_sq = 0.0;
        for &val in gradient.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }
    
    /// Find available memory slot
    fn find_available_slot(&self) -> Option<usize> {
        if self.current_size < self.max_size {
            return Some(self.current_size);
        }
        
        // Find lowest priority entry for replacement
        let mut min_priority = f32::INFINITY;
        let mut min_id = None;
        
        for entry in self.memory_entries.values() {
            let decayed_priority = entry.priority * entry.compute_decay_factor(self.current_time, self.decay_rate);
            if decayed_priority < min_priority {
                min_priority = decayed_priority;
                min_id = Some(entry.id);
            }
        }
        
        min_id
    }
    
    /// Write state ke episodic memory
    fn write_to_memory(&mut self, state: &ArrayD<f32>, priority: f32) -> DLResult<usize> {
        let slot = self.find_available_slot()
            .ok_or_else(|| DeepLearningError::MemoryAllocation { 
                reason: "No available memory slots".to_string() 
            })?;
        
        let entry_id = self.next_id;
        self.next_id += 1;
        
        // Create new memory entry
        let mut entry = EmrMemoryEntry::new(entry_id, state.clone(), self.current_time);
        entry.priority = priority;
        entry.relevance_score = priority;
        
        // Store in matrix
        let state_flat = state.as_slice().expect("tensor should be contiguous");
        if slot < self.max_size && state_flat.len() <= self.memory_matrix.shape()[1] {
            for (i, &val) in state_flat.iter().enumerate() {
                self.memory_matrix[[slot, i]] = val;
            }
        }
        
        // Store entry
        self.memory_entries.insert(entry_id, entry);
        
        // Update size if new slot
        if slot == self.current_size {
            self.current_size += 1;
        }
        
        // Update statistics
        self.total_writes += 1;
        self.update_average_priority();
        
        Ok(entry_id)
    }
    
    /// Update average priority
    fn update_average_priority(&mut self) {
        if self.memory_entries.is_empty() {
            self.avg_priority = 0.0;
            return;
        }
        
        let total_priority: f32 = self.memory_entries.values()
            .map(|e| e.priority * e.compute_decay_factor(self.current_time, self.decay_rate))
            .sum();
        
        self.avg_priority = total_priority / self.memory_entries.len() as f32;
    }
    
    /// Compute cosine similarity
    fn compute_similarity(&self, query: &ArrayD<f32>, memory: &ArrayD<f32>) -> DLResult<f32> {
        let query_flat = query.as_slice().expect("tensor should be contiguous");
        let memory_flat = memory.as_slice().expect("tensor should be contiguous");
        
        if query_flat.len() != memory_flat.len() {
            return Ok(0.0);
        }
        
        let mut dot_product = 0.0;
        let mut query_norm = 0.0;
        let mut memory_norm = 0.0;
        
        for (q, m) in query_flat.iter().zip(memory_flat.iter()) {
            dot_product += q * m;
            query_norm += q * q;
            memory_norm += m * m;
        }
        
        if query_norm > 0.0 && memory_norm > 0.0 {
            Ok(dot_product / (query_norm.sqrt() * memory_norm.sqrt()))
        } else {
            Ok(0.0)
        }
    }
    
    /// Retrieve top-k most similar memories
    fn retrieve_similar_memories(&self, query: &ArrayD<f32>) -> DLResult<Vec<(usize, f32, ArrayD<f32>)>> {
        let mut similarities = Vec::new();
        
        for (id, entry) in &self.memory_entries {
            let similarity = self.compute_similarity(query, &entry.content)?;
            
            if similarity >= self.similarity_threshold {
                // Apply temporal decay to similarity
                let decayed_similarity = similarity * entry.compute_decay_factor(self.current_time, self.decay_rate);
                similarities.push((id, decayed_similarity, entry.content.clone()));
            }
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top-k
        Ok(similarities.into_iter().take(self.retrieval_top_k).map(|(&id, sim, content)| (id, sim, content)).collect())
    }
    
    /// Aggregate retrieved memories
    fn aggregate_memories(&self, retrieved: &[(usize, f32, ArrayD<f32>)]) -> DLResult<ArrayD<f32>> {
        if retrieved.is_empty() {
            return Ok(ArrayD::zeros(vec![self.memory_matrix.shape()[1]]));
        }
        
        let first_content = &retrieved[0].2;
        let mut aggregated = Array1::zeros(first_content.len());
        let agg_flat = aggregated.as_slice_mut().expect("tensor should be contiguous");
        
        let mut total_weight = 0.0;
        
        for (_, similarity, content) in retrieved {
            let content_flat = content.as_slice().expect("tensor should be contiguous");
            let weight = similarity; // Use similarity as weight
            
            for (i, &val) in content_flat.iter().enumerate().take(agg_flat.len()) {
                agg_flat[i] += weight * val;
            }
            total_weight += weight;
        }
        
        // Normalize by total weight
        if total_weight > 0.0 {
            for val in agg_flat.iter_mut() {
                *val /= total_weight;
            }
        }
        
        Ok(aggregated.into_dyn())
    }
    
    /// Cleanup old/low-priority memories
    fn cleanup_memory(&mut self) -> DLResult<usize> {
        let mut to_remove = Vec::new();
        let mut removed_count = 0;
        
        for (id, entry) in &self.memory_entries {
            let decayed_priority = entry.priority * entry.compute_decay_factor(self.current_time, self.decay_rate);
            
            if decayed_priority < self.cleanup_threshold {
                to_remove.push(*id);
            }
        }
        
        for id in to_remove {
            if let Some(_entry) = self.memory_entries.remove(&id) {
                // Clear matrix slot (simplified - in practice would need slot mapping)
                removed_count += 1;
            }
        }
        
        self.current_size = self.memory_entries.len();
        self.cleanup_count += 1;
        
        Ok(removed_count)
    }
    
    /// Get memory statistics
    pub fn get_memory_stats(&self) -> (usize, usize, f32, f32, usize, usize) {
        (
            self.current_size,
            self.max_size,
            self.avg_priority,
            self.get_utilization(),
            self.total_writes,
            self.total_reads,
        )
    }
    
    /// Get memory utilization
    pub fn get_utilization(&self) -> f32 {
        self.current_size as f32 / self.max_size as f32
    }
    
    /// Set priority weights
    pub fn set_priority_weights(&mut self, 
        relevance: f32,
        gradient: f32,
        recency: f32,
        frequency: f32
    ) {
        let total = relevance + gradient + recency + frequency;
        if total > 0.0 {
            self.relevance_weight = relevance / total;
            self.gradient_weight = gradient / total;
            self.recency_weight = recency / total;
            self.frequency_weight = frequency / total;
        }
    }
    
    /// Advance time
    pub fn advance_time(&mut self) {
        self.current_time += 1;
        
        // Periodic cleanup
        if self.current_time - self.last_cleanup >= self.cleanup_interval {
            let _ = self.cleanup_memory();
            self.last_cleanup = self.current_time;
        }
    }
}

impl EpisodicMemory for EpisodicMemoryRetention {
    fn write_memory(&mut self, 
        state: &ArrayD<f32>,
        gradient: &ArrayD<f32>,
        relevance: f32,
        threshold: f32
    ) -> DLResult<bool> {
        
        // Compute priority score
        let priority = self.compute_priority_score(state, gradient, relevance)?;
        
        // Write if priority exceeds threshold
        if priority >= threshold {
            self.write_to_memory(state, priority)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    fn read_memory(&self, query: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let retrieved = self.retrieve_similar_memories(query)?;
        self.aggregate_memories(&retrieved)
    }
    
    fn get_memory_utilization(&self) -> f32 {
        self.get_utilization()
    }
    
    fn cleanup_memory(&mut self) -> DLResult<usize> {
        self.cleanup_memory()
    }
}

/// Advanced memory operations
impl EpisodicMemoryRetention {
    /// Temporal memory retrieval dengan time window
    pub fn temporal_retrieval(&self, query: &ArrayD<f32>, time_window: usize) -> DLResult<ArrayD<f32>> {
        let mut filtered_memories = Vec::new();
        
        for (id, entry) in &self.memory_entries {
            let time_diff = self.current_time - entry.timestamp;
            
            if time_diff <= time_window {
                let similarity = self.compute_similarity(query, &entry.content)?;
                if similarity >= self.similarity_threshold {
                    let decayed_similarity = similarity * entry.compute_decay_factor(self.current_time, self.decay_rate);
                    filtered_memories.push((id, decayed_similarity, entry.content.clone()));
                }
            }
        }
        
        // Sort and aggregate
        filtered_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.aggregate_memories(&filtered_memories.into_iter().take(self.retrieval_top_k).map(|(&id, sim, content)| (id, sim, content)).collect::<Vec<_>>())
    }
    
    /// Associative memory retrieval dengan multiple queries
    pub fn associative_retrieval(&self, queries: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        let mut all_retrieved = Vec::new();
        
        for query in queries {
            let retrieved = self.retrieve_similar_memories(query)?;
            all_retrieved.extend(retrieved);
        }
        
        // Remove duplicates and re-sort
        all_retrieved.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_retrieved.dedup_by(|a, b| a.0 == b.0);
        
        self.aggregate_memories(&all_retrieved.into_iter().take(self.retrieval_top_k).collect::<Vec<_>>())
    }
    
    /// Memory consolidation untuk long-term retention
    pub fn consolidate_memory(&mut self) -> DLResult<usize> {
        let mut consolidated = Vec::new();
        
        // Group similar memories
        let mut groups: HashMap<Vec<usize>, Vec<(usize, f32, ArrayD<f32>)>> = HashMap::new();
        
        for (id, entry) in &self.memory_entries {
            // Simple grouping by priority ranges
            let priority_group = (entry.priority * 10.0) as usize;
            let group_key = vec![priority_group];
            
            groups.entry(group_key).or_insert_with(Vec::new).push((
                *id,
                entry.priority,
                entry.content.clone()
            ));
        }
        
        // Consolidate each group
        for (_group_key, memories) in groups {
            if memories.len() > 1 {
                // Average the memories in the group
                let aggregated = self.aggregate_memories(&memories)?;
                let avg_priority = memories.iter().map(|(_, p, _)| *p).sum::<f32>() / memories.len() as f32;
                
                consolidated.push((aggregated, avg_priority));
            } else {
                // Keep single memories as-is
                for (_id, priority, content) in memories {
                    consolidated.push((content, priority));
                }
            }
        }
        
        // Rebuild memory with consolidated entries
        self.memory_entries.clear();
        self.current_size = 0;
        
        for (content, priority) in consolidated {
            let _ = self.write_to_memory(&content, priority);
        }
        
        Ok(self.current_size)
    }
}
