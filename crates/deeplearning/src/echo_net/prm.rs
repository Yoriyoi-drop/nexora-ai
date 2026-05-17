//! Persistent Resonance Memory (PRM)
//!
//! Block 5 dari ECHO-Net Ω
//!
//! Memory yang tidak lagi overwrite linear tapi dengan adaptive update:
//! H_t = γ_t * H_{t-1} + η_t * (Ψ_t ⊗ K_t)
//!
//! Novelty score:
//! N_t = 1 - cos(H_{t-1}, Ψ_t)
//!
//! Decay:
//! γ_t = e^{-α * N_t}
//!
//! Write strength:
//! η_t = σ(W_n * N_t)
//!
//! Efek:
//! - Informasi lama stabil
//! - Informasi baru adaptif
//! - Memory saturation turun drastis

use crate::{DLResult, DeepLearningError};
use crate::echo_net::HolographicWave;
use ndarray::{ArrayD, Array2, Array1};
use std::collections::HashMap;

/// Memory entry with metadata
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub data: ArrayD<f32>,
    pub timestamp: usize,
    pub novelty_score: f32,
    pub importance: f32,
    pub access_count: u64,
    pub last_access: usize,
    pub resonance_strength: f32,
}

impl MemoryEntry {
    pub fn new(data: ArrayD<f32>, timestamp: usize) -> Self {
        Self {
            data,
            timestamp,
            novelty_score: 0.0,
            importance: 0.0,
            access_count: 0,
            last_access: timestamp,
            resonance_strength: 0.0,
        }
    }
}

/// Persistent Resonance Memory implementation
#[derive(Debug, Clone)]
pub struct PersistentResonanceMemory {
    // Memory storage
    memory_size: usize,
    memory_entries: Vec<Option<MemoryEntry>>,
    free_slots: Vec<usize>,
    
    // Resonance parameters
    decay_alpha: f32,
    write_threshold: f32,
    novelty_threshold: f32,
    
    // Adaptive parameters
    novelty_weights: Array1<f32>,
    importance_decay: f32,
    access_boost: f32,
    
    // Memory management
    current_timestamp: usize,
    total_writes: u64,
    total_reads: u64,
    
    // Resonance cache
    resonance_cache: HashMap<(usize, usize), f32>,
    cache_size: usize,
    
    // Memory statistics
    memory_utilization: f32,
    average_novelty: f32,
    saturation_level: f32,
    
    // Kernel for resonance operations
    resonance_kernel: Array2<f32>,
    
    // Decay tracking
    decay_history: Vec<f32>,
    novelty_history: Vec<f32>,
}

impl PersistentResonanceMemory {
    /// Create new Persistent Resonance Memory
    pub fn new(
        memory_size: usize,
        decay_alpha: f32,
        write_threshold: f32,
        novelty_threshold: f32,
    ) -> DLResult<Self> {
        // Initialize memory entries
        let memory_entries = vec![None; memory_size];
        let free_slots: Vec<usize> = (0..memory_size).collect();
        
        // Initialize novelty weights
        let novelty_weights = Array1::from_vec(vec![0.5, 0.3, 0.2]); // Energy, entropy, coherence weights
        
        // Initialize resonance kernel
        let resonance_kernel = Array2::from_elem((8, 8), 0.125); // Simple averaging kernel
        
        Ok(Self {
            memory_size,
            memory_entries,
            free_slots,
            decay_alpha,
            write_threshold,
            novelty_threshold,
            novelty_weights,
            importance_decay: 0.99,
            access_boost: 1.1,
            current_timestamp: 0,
            total_writes: 0,
            total_reads: 0,
            resonance_cache: HashMap::new(),
            cache_size: 10000,
            memory_utilization: 0.0,
            average_novelty: 0.0,
            saturation_level: 0.0,
            resonance_kernel,
            decay_history: Vec::new(),
            novelty_history: Vec::new(),
        })
    }
    
    /// Forward pass - write holographic wave to persistent memory
    pub fn forward(&mut self, wave: &HolographicWave, timestamp: usize) -> DLResult<usize> {
        self.current_timestamp = timestamp;
        
        // Convert wave to tensor representation
        let tensor = self.wave_to_tensor(wave)?;
        
        // Calculate novelty score
        let novelty_score = self.calculate_novelty_score(&tensor)?;
        
        // Check if should write to memory
        if novelty_score < self.write_threshold {
            return Ok(usize::MAX); // Not novel enough to write
        }
        
        // Find memory slot
        let memory_slot = self.find_memory_slot()?;
        
        // Calculate decay factor
        let decay_factor = self.calculate_decay_factor(novelty_score);
        
        // Calculate write strength
        let write_strength = self.calculate_write_strength(novelty_score);
        
        // Apply resonance operation
        let resonated_data = self.apply_resonance_operation(&tensor, write_strength)?;
        
        // Create memory entry
        let mut entry = MemoryEntry::new(resonated_data, timestamp);
        entry.novelty_score = novelty_score;
        entry.importance = self.calculate_importance(&entry, novelty_score);
        entry.resonance_strength = write_strength;
        
        // Write to memory
        self.write_to_memory(memory_slot, entry, decay_factor)?;
        
        // Update statistics
        self.update_statistics()?;
        
        Ok(memory_slot)
    }
    
    /// Retrieve from memory
    pub fn retrieve(&mut self, query: &HolographicWave, k: usize) -> DLResult<Vec<(usize, f32, ArrayD<f32>)>> {
        self.total_reads += 1;
        
        // Convert query to tensor
        let query_tensor = self.wave_to_tensor(query)?;
        
        // Calculate resonance with all memory entries
        let mut resonances = Vec::with_capacity(self.memory_entries.iter().filter(|e| e.is_some()).count());
        
        let entries_to_process: Vec<(usize, ArrayD<f32>)> = self.memory_entries.iter()
            .enumerate()
            .filter_map(|(slot_idx, entry)| {
                entry.as_ref().map(|memory_entry| (slot_idx, memory_entry.data.clone()))
            })
            .collect();
        
        for (slot_idx, data) in entries_to_process {
            let resonance = self.calculate_resonance(&query_tensor, &data, slot_idx)?;
            resonances.push((slot_idx, resonance, data.clone()));
            
            // Update access statistics
            self.update_access_stats(slot_idx);
        }
        
        // Sort by resonance strength
        resonances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top-k results
        Ok(resonances.into_iter().take(k).collect())
    }
    
    /// Convert holographic wave to tensor
    fn wave_to_tensor(&self, wave: &HolographicWave) -> DLResult<ArrayD<f32>> {
        let total_size = wave.amplitude.len() + wave.phase.len() + wave.frequency.len();
        let mut tensor = ArrayD::zeros(vec![total_size]);
        
        let mut offset = 0;
        
        // Copy amplitude
        for (i, &amp) in wave.amplitude.iter().enumerate() {
            tensor[offset + i] = amp;
        }
        offset += wave.amplitude.len();
        
        // Copy phase
        for (i, &phase) in wave.phase.iter().enumerate() {
            tensor[offset + i] = phase;
        }
        offset += wave.phase.len();
        
        // Copy frequency
        for (i, &freq) in wave.frequency.iter().enumerate() {
            tensor[offset + i] = freq;
        }
        
        Ok(tensor)
    }
    
    /// Calculate novelty score
    fn calculate_novelty_score(&mut self, tensor: &ArrayD<f32>) -> DLResult<f32> {
        if self.memory_entries.is_empty() || self.memory_entries.iter().all(|e| e.is_none()) {
            return Ok(1.0); // Empty memory is maximally novel
        }
        
        // Find most similar memory entry
        let mut max_similarity = 0.0;
        let mut similarities = Vec::new();
        
        for entry in self.memory_entries.iter().flatten() {
            let similarity = self.calculate_similarity(tensor, &entry.data)?;
            similarities.push(similarity);
            max_similarity = (max_similarity as f32).max(similarity);
        }
        
        // Novelty = 1 - max_similarity
        let novelty = 1.0 - max_similarity;
        
        // Update novelty history
        self.novelty_history.push(novelty);
        if self.novelty_history.len() > 1000 {
            self.novelty_history.remove(0);
        }
        
        Ok(novelty)
    }
    
    /// Calculate similarity between tensors
    fn calculate_similarity(&self, tensor1: &ArrayD<f32>, tensor2: &ArrayD<f32>) -> DLResult<f32> {
        if tensor1.shape() != tensor2.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: tensor1.shape().to_vec(),
                actual: tensor2.shape().to_vec(),
            });
        }
        
        // Cosine similarity
        let dot_product: f32 = tensor1.iter().zip(tensor2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = tensor1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = tensor2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm1 * norm2))
    }
    
    /// Calculate decay factor
    fn calculate_decay_factor(&self, novelty_score: f32) -> f32 {
        (-self.decay_alpha * novelty_score).exp()
    }
    
    /// Calculate write strength
    fn calculate_write_strength(&self, novelty_score: f32) -> f32 {
        // Sigmoid activation
        let weighted_novelty = novelty_score * self.novelty_weights[0] + 
                              novelty_score * self.novelty_weights[1] + 
                              novelty_score * self.novelty_weights[2];
        1.0 / (1.0 + (-weighted_novelty).exp())
    }
    
    /// Apply resonance operation
    fn apply_resonance_operation(&self, tensor: &ArrayD<f32>, write_strength: f32) -> DLResult<ArrayD<f32>> {
        // Apply convolution with resonance kernel
        let shape = tensor.shape();
        if shape.len() != 1 {
            return Err(DeepLearningError::Configuration {
                reason: "Only 1D tensors supported for resonance operation".to_string(),
            });
        }
        
        let size = shape[0];
        let kernel_size = self.resonance_kernel.nrows();
        let mut resonated = ArrayD::zeros(shape.to_vec());
        
        for i in 0..size {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            for ki in 0..kernel_size {
                let tensor_idx = if i + ki >= kernel_size / 2 {
                    i + ki - kernel_size / 2
                } else {
                    0
                };
                
                if tensor_idx < size {
                    sum += tensor[tensor_idx] * self.resonance_kernel[[ki, 0]];
                    weight_sum += self.resonance_kernel[[ki, 0]];
                }
            }
            
            if weight_sum > 0.0 {
                resonated[i] = (sum / weight_sum) * write_strength;
            }
        }
        
        Ok(resonated)
    }
    
    /// Find available memory slot
    fn find_memory_slot(&mut self) -> DLResult<usize> {
        // Try to use free slot first
        if let Some(slot) = self.free_slots.pop() {
            return Ok(slot);
        }
        
        // Find least important entry to replace
        let mut min_importance = f32::INFINITY;
        let mut min_slot = 0;
        
        for (slot_idx, entry) in self.memory_entries.iter().enumerate() {
            if let Some(memory_entry) = entry {
                let decayed_importance = memory_entry.importance * 
                    self.importance_decay.powf((self.current_timestamp - memory_entry.timestamp) as f32);
                
                if decayed_importance < min_importance {
                    min_importance = decayed_importance;
                    min_slot = slot_idx;
                }
            }
        }
        
        Ok(min_slot)
    }
    
    /// Write to memory with decay
    fn write_to_memory(&mut self, slot: usize, entry: MemoryEntry, decay_factor: f32) -> DLResult<()> {
        // Apply decay to existing memory if present
        if let Some(existing_entry) = &self.memory_entries[slot] {
            let decayed_data = existing_entry.data.mapv(|x| x * decay_factor);
            
            // Combine with new data
            let mut combined_data = entry.data.clone();
            for (i, &decayed_val) in decayed_data.iter().enumerate() {
                if i < combined_data.len() {
                    combined_data[i] += decayed_val;
                }
            }
            
            let mut combined_entry = entry;
            combined_entry.data = combined_data;
            self.memory_entries[slot] = Some(combined_entry);
        } else {
            self.memory_entries[slot] = Some(entry);
        }
        
        self.total_writes += 1;
        
        // Update decay history
        self.decay_history.push(decay_factor);
        if self.decay_history.len() > 1000 {
            self.decay_history.remove(0);
        }
        
        Ok(())
    }
    
    /// Calculate importance of memory entry
    fn calculate_importance(&self, entry: &MemoryEntry, novelty_score: f32) -> f32 {
        let time_factor = 1.0 / (1.0 + (self.current_timestamp - entry.timestamp) as f32 * 0.001);
        let access_factor = 1.0 + (entry.access_count as f32).ln() * 0.1;
        
        novelty_score * time_factor * access_factor * entry.resonance_strength
    }
    
    /// Calculate resonance between query and memory
    fn calculate_resonance(&mut self, query: &ArrayD<f32>, memory: &ArrayD<f32>, slot_idx: usize) -> DLResult<f32> {
        // Check cache first
        let cache_key = (slot_idx, self.current_timestamp);
        if let Some(&cached_resonance) = self.resonance_cache.get(&cache_key) {
            return Ok(cached_resonance);
        }
        
        // Calculate resonance
        let similarity = self.calculate_similarity(query, memory)?;
        let energy_resonance = self.calculate_energy_resonance(query, memory)?;
        let phase_resonance = self.calculate_phase_resonance(query, memory)?;
        
        // Combined resonance score
        let resonance = 0.5 * similarity + 0.3 * energy_resonance + 0.2 * phase_resonance;
        
        // Update cache
        if self.resonance_cache.len() < self.cache_size {
            self.resonance_cache.insert(cache_key, resonance);
        }
        
        Ok(resonance)
    }
    
    /// Calculate energy resonance
    fn calculate_energy_resonance(&self, query: &ArrayD<f32>, memory: &ArrayD<f32>) -> DLResult<f32> {
        let query_energy: f32 = query.iter().map(|&x| x * x).sum();
        let memory_energy: f32 = memory.iter().map(|&x| x * x).sum();
        
        if query_energy == 0.0 || memory_energy == 0.0 {
            return Ok(0.0);
        }
        
        let energy_ratio = (query_energy / memory_energy).min(memory_energy / query_energy);
        Ok(energy_ratio.sqrt())
    }
    
    /// Calculate phase resonance
    fn calculate_phase_resonance(&self, query: &ArrayD<f32>, memory: &ArrayD<f32>) -> DLResult<f32> {
        // Extract phase information (simplified)
        let query_phase = self.extract_phase_info(query)?;
        let memory_phase = self.extract_phase_info(memory)?;
        
        let phase_diff: f32 = query_phase.iter().zip(memory_phase.iter())
            .map(|(&qp, &mp)| (qp - mp).abs())
            .sum();
        
        let avg_phase_diff = phase_diff / query_phase.len() as f32;
        Ok((-avg_phase_diff).exp()) // Higher resonance for lower phase difference
    }
    
    /// Extract phase information from tensor
    fn extract_phase_info(&self, tensor: &ArrayD<f32>) -> DLResult<Array1<f32>> {
        // Simplified phase extraction using arctan2
        let mut phase_info = Vec::with_capacity(tensor.len() / 2);
        
        for i in 0..tensor.len() / 2 {
            if i + 1 < tensor.len() {
                let real_part = tensor[i];
                let imag_part = tensor[i + 1];
                let phase = imag_part.atan2(real_part);
                phase_info.push(phase);
            }
        }
                    
        Ok(Array1::from(phase_info))
    }
    
    /// Update access statistics
    fn update_access_stats(&mut self, slot: usize) {
        if let Some(entry) = &mut self.memory_entries[slot] {
            entry.access_count += 1;
            entry.last_access = self.current_timestamp;
            entry.importance *= self.access_boost;
        }
    }
    
    /// Update memory statistics
    fn update_statistics(&mut self) -> DLResult<()> {
        // Memory utilization
        let used_slots = self.memory_entries.iter().filter(|e| e.is_some()).count();
        self.memory_utilization = used_slots as f32 / self.memory_size as f32;
        
        // Average novelty
        if !self.novelty_history.is_empty() {
            self.average_novelty = self.novelty_history.iter().sum::<f32>() / self.novelty_history.len() as f32;
        }
        
        // Saturation level
        let high_importance_count = self.memory_entries.iter()
            .filter_map(|e| e.as_ref())
            .filter(|e| e.importance > 0.8)
            .count();
        self.saturation_level = high_importance_count as f32 / self.memory_size as f32;
        
        Ok(())
    }
    
    /// Get memory statistics
    pub fn get_statistics(&self) -> MemoryStatistics {
        MemoryStatistics {
            memory_size: self.memory_size,
            used_slots: self.memory_entries.iter().filter(|e| e.is_some()).count(),
            memory_utilization: self.memory_utilization,
            average_novelty: self.average_novelty,
            saturation_level: self.saturation_level,
            total_writes: self.total_writes,
            total_reads: self.total_reads,
            cache_hit_rate: if self.total_reads > 0 {
                self.resonance_cache.len() as f32 / self.total_reads as f32
            } else {
                0.0
            },
        }
    }
    
    /// Get memory entry
    pub fn get_entry(&self, slot: usize) -> DLResult<Option<&MemoryEntry>> {
        if slot >= self.memory_size {
            return Err(DeepLearningError::InvalidDimension { dim: slot });
        }
        
        Ok(self.memory_entries[slot].as_ref())
    }
    
    /// Reset memory
    pub fn reset(&mut self) -> DLResult<()> {
        self.memory_entries.fill(None);
        self.free_slots = (0..self.memory_size).collect();
        self.current_timestamp = 0;
        self.total_writes = 0;
        self.total_reads = 0;
        self.resonance_cache.clear();
        self.memory_utilization = 0.0;
        self.average_novelty = 0.0;
        self.saturation_level = 0.0;
        self.decay_history.clear();
        self.novelty_history.clear();
        
        Ok(())
    }
    
    /// Set decay alpha
    pub fn set_decay_alpha(&mut self, alpha: f32) {
        self.decay_alpha = alpha;
    }
    
    /// Set write threshold
    pub fn set_write_threshold(&mut self, threshold: f32) {
        self.write_threshold = threshold;
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub memory_size: usize,
    pub used_slots: usize,
    pub memory_utilization: f32,
    pub average_novelty: f32,
    pub saturation_level: f32,
    pub total_writes: u64,
    pub total_reads: u64,
    pub cache_hit_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use ndarray::ArrayD;
    
    #[test]
    fn test_prm_creation() {
        let prm = PersistentResonanceMemory::new(1000, 0.01, 0.1, 0.3).unwrap();
        assert_eq!(prm.memory_size, 1000);
        assert_eq!(prm.decay_alpha, 0.01);
        assert_eq!(prm.write_threshold, 0.1);
    }
    
    #[test]
    fn test_memory_entry_creation() {
        let data = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let entry = MemoryEntry::new(data, 100);
        
        assert_eq!(entry.timestamp, 100);
        assert_eq!(entry.access_count, 0);
        assert_eq!(entry.novelty_score, 0.0);
    }
    
    #[test]
    fn test_similarity_calculation() {
        let prm = PersistentResonanceMemory::new(100, 0.01, 0.1, 0.3).unwrap();
        
        let tensor1 = ArrayD::from_shape_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let tensor2 = ArrayD::from_shape_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let tensor3 = ArrayD::from_shape_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap();
        
        let sim1 = prm.calculate_similarity(&tensor1, &tensor2).unwrap();
        let sim2 = prm.calculate_similarity(&tensor1, &tensor3).unwrap();
        
        assert!((sim1 - 1.0).abs() < 1e-6);
        assert!((sim2 - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_decay_factor() {
        let prm = PersistentResonanceMemory::new(100, 0.1, 0.1, 0.3).unwrap();
        
        let decay1 = prm.calculate_decay_factor(0.0); // No novelty
        let decay2 = prm.calculate_decay_factor(1.0); // Maximum novelty
        
        assert!((decay1 - 1.0).abs() < 1e-6);
        assert!(decay2 < decay1);
    }
}
