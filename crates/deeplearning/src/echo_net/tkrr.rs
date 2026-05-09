//! Top-K Resonance Routing (TKRR)
//!
//! Block 8 dari ECHO-Net Ω
//!
//! Hanya resonansi paling penting yang diproses:
//! R' = TopK(R, k)
//!
//! Sparse routing:
//! ctx = Σ_{i∈K} R_i * H_i
//!
//! Inference menjadi sangat murah dengan O(k) complexity
//! bukan O(T) atau O(T²).

use crate::{DLResult, DeepLearningError};
use crate::echo_net::{HolographicWave, ComplexTensor};
use crate::echo_net::utils::{ResonanceCalculator, Complex};
use ndarray::{ArrayD, Array2, Array1, s};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::hash::{Hasher, DefaultHasher};

/// Resonance routing candidate
#[derive(Debug, Clone)]
pub struct ResonanceCandidate {
    pub index: usize,
    pub resonance_score: f32,
    pub resonance_data: ArrayD<f32>,
    pub holographic_component: ArrayD<f32>,
    pub confidence: f32,
    pub relevance: f32,
}

impl ResonanceCandidate {
    pub fn new(index: usize, resonance_score: f32, resonance_data: ArrayD<f32>, holographic_component: ArrayD<f32>) -> Self {
        Self {
            index,
            resonance_score,
            resonance_data,
            holographic_component,
            confidence: 0.0,
            relevance: 0.0,
        }
    }
}

/// Ordering for max-heap (reverse order for min-heap behavior)
impl Ord for ResonanceCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.resonance_score.partial_cmp(&self.resonance_score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ResonanceCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ResonanceCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.resonance_score == other.resonance_score
    }
}

impl Eq for ResonanceCandidate {}

/// Top-K Resonance Routing implementation
#[derive(Debug, Clone)]
pub struct TopKResonanceRouting {
    // Routing parameters
    top_k: usize,
    routing_threshold: f32,
    diversity_weight: f32,
    confidence_threshold: f32,
    
    // Resonance computation
    resonance_calculator: ResonanceCalculator,
    
    // Routing cache
    routing_cache: std::collections::HashMap<usize, Vec<usize>>,
    cache_size: usize,
    cache_hit_count: u64,
    cache_miss_count: u64,
    
    // Diversity management
    diversity_window: usize,
    similarity_threshold: f32,
    
    // Performance metrics
    average_routing_time: f32,
    cache_hit_rate: f32,
    routing_efficiency: f32,
    
    // Adaptive routing
    adaptive_k: bool,
    min_k: usize,
    max_k: usize,
    k_adjustment_rate: f32,
    
    // Routing history
    routing_history: Vec<RoutingEvent>,
    selected_indices_history: Vec<Vec<usize>>,
    
    // Relevance scoring
    relevance_weights: Array1<f32>,
    temporal_decay: f32,
}

impl TopKResonanceRouting {
    /// Create new Top-K Resonance Routing
    pub fn new(
        top_k: usize,
        routing_threshold: f32,
        diversity_weight: f32,
        confidence_threshold: f32,
    ) -> DLResult<Self> {
        Ok(Self {
            top_k,
            routing_threshold,
            diversity_weight,
            confidence_threshold,
            resonance_calculator: ResonanceCalculator,
            routing_cache: std::collections::HashMap::new(),
            cache_size: 10000,
            cache_hit_count: 0,
            cache_miss_count: 0,
            diversity_window: 16,
            similarity_threshold: 0.7,
            average_routing_time: 0.0,
            cache_hit_rate: 0.0,
            routing_efficiency: 0.0,
            adaptive_k: true,
            min_k: 4,
            max_k: 128,
            k_adjustment_rate: 0.1,
            routing_history: Vec::new(),
            selected_indices_history: Vec::new(),
            relevance_weights: Array1::from_vec(vec![0.4, 0.3, 0.2, 0.1]), // Energy, entropy, coherence, novelty
            temporal_decay: 0.95,
        })
    }
    
    /// Forward pass - perform top-k resonance routing
    pub fn forward(&mut self, query: &HolographicWave, resonance_data: &[ArrayD<f32>], holographic_memory: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        let cache_key = self.calculate_cache_key(query)?;
        if let Some(cached_indices) = self.routing_cache.get(&cache_key) {
            self.cache_hit_count += 1;
            return self.route_from_cached_indices(&cached_indices, resonance_data, holographic_memory);
        }
        
        self.cache_miss_count += 1;
        
        // Calculate resonance scores for all candidates
        let mut candidates = Vec::new();
        
        for (idx, (resonance, holographic)) in resonance_data.iter().zip(holographic_memory.iter()).enumerate() {
            let resonance_score = self.calculate_resonance_score(query, resonance, holographic)?;
            
            if resonance_score >= self.routing_threshold {
                let mut candidate = ResonanceCandidate::new(idx, resonance_score, resonance.clone(), holographic.clone());
                
                // Calculate confidence and relevance
                candidate.confidence = self.calculate_confidence(&candidate)?;
                candidate.relevance = self.calculate_relevance(&candidate)?;
                
                candidates.push(candidate);
            }
        }
        
        // Apply top-k selection
        let selected_candidates = self.select_top_k_candidates(&mut candidates)?;
        
        // Apply diversity filtering
        let diverse_candidates = self.apply_diversity_filtering(&selected_candidates)?;
        
        // Generate sparse routing context
        let context = self.generate_sparse_routing_context(&diverse_candidates, resonance_data, holographic_memory)?;
        
        // Update cache
        let selected_indices: Vec<usize> = diverse_candidates.iter().map(|c| c.index).collect();
        self.update_cache(cache_key, selected_indices.clone())?;
        
        // Record routing event
        let routing_time = start_time.elapsed().as_secs_f32();
        self.record_routing_event(&candidates, &diverse_candidates, routing_time)?;
        
        // Update performance metrics
        self.update_performance_metrics(routing_time)?;
        
        Ok(context)
    }
    
    /// Calculate cache key for query
    fn calculate_cache_key(&self, query: &HolographicWave) -> DLResult<usize> {
        // Simple hash-based cache key
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash amplitude
        for &amp in query.amplitude.iter() {
            hasher.write_u64(amp.to_bits() as u64);
        }
        
        // Hash phase
        for &phase in query.phase.iter() {
            hasher.write_u64(phase.to_bits() as u64);
        }
        
        // Hash frequency
        for &freq in query.frequency.iter() {
            hasher.write_u64(freq.to_bits() as u64);
        }
        
        Ok(hasher.finish() as usize)
    }
    
    /// Route from cached indices
    fn route_from_cached_indices(&self, indices: &[usize], resonance_data: &[ArrayD<f32>], holographic_memory: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        let mut cached_candidates = Vec::new();
        
        for &idx in indices {
            if idx < resonance_data.len() && idx < holographic_memory.len() {
                let resonance_score = 1.0; // Cached candidates are pre-filtered
                let candidate = ResonanceCandidate::new(
                    idx,
                    resonance_score,
                    resonance_data[idx].clone(),
                    holographic_memory[idx].clone(),
                );
                cached_candidates.push(candidate);
            }
        }
        
        self.generate_sparse_routing_context(&cached_candidates, resonance_data, holographic_memory)
    }
    
    /// Calculate resonance score between query and candidate
    fn calculate_resonance_score(&self, query: &HolographicWave, resonance_data: &ArrayD<f32>, holographic_component: &ArrayD<f32>) -> DLResult<f32> {
        // Convert query to tensor
        let query_tensor = self.wave_to_tensor(query)?;
        
        // Calculate basic resonance
        let basic_resonance = ResonanceCalculator::resonance_coefficient(
            &self.to_1d_array(&query_tensor)?,
            &self.to_1d_array(resonance_data)?
        );
        
        // Calculate holographic alignment
        let holographic_alignment = self.calculate_holographic_alignment(&query_tensor, holographic_component)?;
        
        // Combine scores
        let combined_score = 0.7 * basic_resonance + 0.3 * holographic_alignment;
        
        Ok(combined_score.clamp(0.0, 1.0))
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
    
    /// Convert to 1D array
    fn to_1d_array(&self, tensor: &ArrayD<f32>) -> DLResult<Array1<f32>> {
        let vec: Vec<f32> = tensor.iter().cloned().collect();
        Ok(Array1::from(vec))
    }
    
    /// Calculate holographic alignment
    fn calculate_holographic_alignment(&self, query: &ArrayD<f32>, holographic: &ArrayD<f32>) -> DLResult<f32> {
        if query.shape() != holographic.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: query.shape().to_vec(),
                actual: holographic.shape().to_vec(),
            });
        }
        
        // Phase alignment calculation
        let mut phase_alignment = 0.0;
        let chunk_size = 2; // Assume complex-like data
        
        let query_1d = self.to_1d_array(query)?;
        let holographic_1d = self.to_1d_array(holographic)?;
        
        let query_chunks = query_1d.exact_chunks(chunk_size).into_iter();
        let holographic_chunks = holographic_1d.exact_chunks(chunk_size).into_iter();
        
        for (query_chunk, holographic_chunk) in query_chunks.zip(holographic_chunks) {
            if query_chunk.len() >= 2 && holographic_chunk.len() >= 2 {
                let query_phase = query_chunk[1].atan2(query_chunk[0]);
                let holographic_phase = holographic_chunk[1].atan2(holographic_chunk[0]);
                let phase_diff = (query_phase - holographic_phase).abs();
                phase_alignment += (-phase_diff).exp();
            }
        }
        
        let alignment = phase_alignment / (query.len() / chunk_size) as f32;
        Ok(alignment.clamp(0.0, 1.0))
    }
    
    /// Select top-k candidates using heap for efficiency
    fn select_top_k_candidates(&self, candidates: &mut Vec<ResonanceCandidate>) -> DLResult<Vec<ResonanceCandidate>> {
        let effective_k = if self.adaptive_k {
            self.calculate_adaptive_k(candidates.len())?
        } else {
            self.top_k
        };
        
        // Use heap for efficient top-k selection
        let mut heap = BinaryHeap::new();
        
        for candidate in candidates.drain(..) {
            if heap.len() < effective_k {
                heap.push(candidate);
            } else if candidate.resonance_score > heap.peek().unwrap().resonance_score {
                heap.pop();
                heap.push(candidate);
            }
        }
        
        // Convert heap to sorted vector
        let mut selected: Vec<ResonanceCandidate> = heap.into_iter().collect();
        selected.sort_by(|a, b| b.resonance_score.partial_cmp(&a.resonance_score).unwrap());
        
        Ok(selected)
    }
    
    /// Calculate adaptive k based on candidate quality
    fn calculate_adaptive_k(&self, total_candidates: usize) -> DLResult<usize> {
        if !self.adaptive_k {
            return Ok(self.top_k);
        }
        
        // Adaptive k based on total candidates
        let adaptive_k = if total_candidates < self.min_k {
            total_candidates
        } else if total_candidates > self.max_k {
            self.max_k
        } else {
            // Scale between min and max based on candidate count
            let ratio = (total_candidates - self.min_k) as f32 / (self.max_k - self.min_k) as f32;
            (self.min_k as f32 + ratio * (self.top_k - self.min_k) as f32) as usize
        };
        
        Ok(adaptive_k.clamp(self.min_k, self.max_k))
    }
    
    /// Apply diversity filtering to selected candidates
    fn apply_diversity_filtering(&self, candidates: &[ResonanceCandidate]) -> DLResult<Vec<ResonanceCandidate>> {
        if self.diversity_weight == 0.0 {
            return Ok(candidates.to_vec());
        }
        
        let mut diverse_candidates = Vec::new();
        let mut selected_data = Vec::new();
        
        for candidate in candidates {
            let is_diverse = if selected_data.is_empty() {
                true
            } else {
                // Check diversity against already selected candidates
                let min_similarity = selected_data.iter()
                    .map(|selected| self.calculate_similarity(&candidate.resonance_data, selected))
                    .fold(1.0_f32, |acc, sim| acc.min(sim.unwrap_or(1.0_f32)));
                
                min_similarity < self.similarity_threshold
            };
            
            if is_diverse || diverse_candidates.len() < self.min_k {
                diverse_candidates.push(candidate.clone());
                selected_data.push(candidate.resonance_data.clone());
                
                if diverse_candidates.len() >= self.top_k {
                    break;
                }
            }
        }
        
        Ok(diverse_candidates)
    }
    
    /// Calculate similarity between two tensors
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
    
    /// Generate sparse routing context
    fn generate_sparse_routing_context(&self, candidates: &[ResonanceCandidate], resonance_data: &[ArrayD<f32>], holographic_memory: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        if candidates.is_empty() {
            return Err(DeepLearningError::Configuration {
                reason: "No candidates selected for routing".to_string(),
            });
        }
        
        // Initialize context with zeros
        let context_shape = candidates[0].resonance_data.shape().to_vec();
        let mut context = ArrayD::zeros(context_shape.clone());
        
        // Weighted sum of selected candidates
        let total_weight: f32 = candidates.iter().map(|c| c.resonance_score).sum();
        
        if total_weight == 0.0 {
            // Equal weighting if no resonance scores
            for candidate in candidates {
                for (i, &val) in candidate.resonance_data.iter().enumerate() {
                    if i < context.len() {
                        context[i] += val / candidates.len() as f32;
                    }
                }
            }
        } else {
            // Weighted by resonance scores
            for candidate in candidates {
                let weight = candidate.resonance_score / total_weight;
                
                for (i, &val) in candidate.resonance_data.iter().enumerate() {
                    if i < context.len() {
                        context[i] += weight * val;
                    }
                }
                
                // Add holographic contribution
                for (i, &val) in candidate.holographic_component.iter().enumerate() {
                    if i < context.len() {
                        context[i] += weight * val * self.diversity_weight;
                    }
                }
            }
        }
        
        Ok(context)
    }
    
    /// Calculate confidence for candidate
    fn calculate_confidence(&self, candidate: &ResonanceCandidate) -> DLResult<f32> {
        // Confidence based on resonance score and data quality
        let data_energy: f32 = candidate.resonance_data.iter().map(|&x| x * x).sum();
        let normalized_energy = (data_energy.sqrt() / 100.0).tanh(); // Normalize
        
        let confidence = 0.7 * candidate.resonance_score + 0.3 * normalized_energy;
        Ok(confidence.clamp(0.0, 1.0))
    }
    
    /// Calculate relevance for candidate
    fn calculate_relevance(&self, candidate: &ResonanceCandidate) -> DLResult<f32> {
        // Multi-factor relevance scoring
        let energy_factor = self.calculate_energy_factor(&candidate.resonance_data)?;
        let entropy_factor = self.calculate_entropy_factor(&candidate.resonance_data)?;
        let coherence_factor = self.calculate_coherence_factor(&candidate.resonance_data)?;
        let novelty_factor = 1.0 - candidate.resonance_score; // Higher resonance = lower novelty
        
        // Weighted combination
        let relevance = self.relevance_weights[0] * energy_factor
                      + self.relevance_weights[1] * entropy_factor
                      + self.relevance_weights[2] * coherence_factor
                      + self.relevance_weights[3] * novelty_factor;
        
        Ok(relevance.clamp(0.0, 1.0))
    }
    
    /// Calculate energy factor
    fn calculate_energy_factor(&self, data: &ArrayD<f32>) -> DLResult<f32> {
        let energy: f32 = data.iter().map(|&x| x * x).sum();
        Ok((energy.sqrt() / 50.0).tanh().clamp(0.0, 1.0))
    }
    
    /// Calculate entropy factor
    fn calculate_entropy_factor(&self, data: &ArrayD<f32>) -> DLResult<f32> {
        let values: Vec<f32> = data.iter().cloned().collect();
        if values.is_empty() {
            return Ok(0.0);
        }
        
        // Simple entropy calculation
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < 1e-10 {
            return Ok(0.0);
        }
        
        let mut histogram = vec![0.0; 10];
        let bin_size = (max_val - min_val) / 10.0;
        
        for &value in &values {
            let bin_index = ((value - min_val) / bin_size) as usize;
            if bin_index < 10 {
                histogram[bin_index] += 1.0;
            }
        }
        
        let total_samples = values.len() as f32;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0.0 {
                let probability = count / total_samples;
                entropy -= probability * probability.ln();
            }
        }
        
        Ok((entropy / 2.3).clamp(0.0, 1.0)) // Normalize by ln(10)
    }
    
    /// Calculate coherence factor
    fn calculate_coherence_factor(&self, data: &ArrayD<f32>) -> DLResult<f32> {
        let values: Vec<f32> = data.iter().cloned().collect();
        if values.len() < 2 {
            return Ok(1.0);
        }
        
        // Calculate local coherence
        let mut coherence_sum = 0.0;
        for i in 1..values.len() {
            let diff = (values[i] - values[i-1]).abs();
            coherence_sum += (-diff).exp();
        }
        
        let coherence = coherence_sum / (values.len() - 1) as f32;
        Ok(coherence.clamp(0.0, 1.0))
    }
    
    /// Update routing cache
    fn update_cache(&mut self, key: usize, indices: Vec<usize>) -> DLResult<()> {
        if self.routing_cache.len() >= self.cache_size {
            // Remove oldest entry (simple FIFO)
            if let Some(oldest_key) = self.routing_cache.keys().next().cloned() {
                self.routing_cache.remove(&oldest_key);
            }
        }
        
        self.routing_cache.insert(key, indices);
        Ok(())
    }
    
    /// Record routing event
    fn record_routing_event(&mut self, all_candidates: &[ResonanceCandidate], selected_candidates: &[ResonanceCandidate], routing_time: f32) -> DLResult<()> {
        let event = RoutingEvent {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            total_candidates: all_candidates.len(),
            selected_candidates: selected_candidates.len(),
            routing_time,
            average_resonance: selected_candidates.iter().map(|c| c.resonance_score).sum::<f32>() / selected_candidates.len() as f32,
        };
        
        self.routing_history.push(event);
        
        // Keep only recent history
        if self.routing_history.len() > 1000 {
            self.routing_history.remove(0);
        }
        
        // Store selected indices
        let selected_indices: Vec<usize> = selected_candidates.iter().map(|c| c.index).collect();
        self.selected_indices_history.push(selected_indices);
        
        if self.selected_indices_history.len() > 1000 {
            self.selected_indices_history.remove(0);
        }
        
        Ok(())
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self, routing_time: f32) -> DLResult<()> {
        // Update average routing time
        if self.routing_history.is_empty() {
            self.average_routing_time = routing_time;
        } else {
            self.average_routing_time = (self.average_routing_time * 0.9 + routing_time * 0.1);
        }
        
        // Update cache hit rate
        let total_requests = self.cache_hit_count + self.cache_miss_count;
        if total_requests > 0 {
            self.cache_hit_rate = self.cache_hit_count as f32 / total_requests as f32;
        }
        
        // Update routing efficiency
        if !self.routing_history.is_empty() {
            let recent_events: Vec<&RoutingEvent> = self.routing_history
                .iter()
                .rev()
                .take(10)
                .collect();
            
            let avg_selected: f32 = recent_events.iter().map(|e| e.selected_candidates as f32).sum::<f32>() / recent_events.len() as f32;
            let avg_total: f32 = recent_events.iter().map(|e| e.total_candidates as f32).sum::<f32>() / recent_events.len() as f32;
            
            self.routing_efficiency = if avg_total > 0.0 {
                avg_selected / avg_total
            } else {
                0.0
            };
        }
        
        Ok(())
    }
    
    /// Get routing statistics
    pub fn get_statistics(&self) -> RoutingStatistics {
        RoutingStatistics {
            top_k: self.top_k,
            routing_threshold: self.routing_threshold,
            cache_hit_rate: self.cache_hit_rate,
            average_routing_time: self.average_routing_time,
            routing_efficiency: self.routing_efficiency,
            adaptive_k: self.adaptive_k,
            diversity_weight: self.diversity_weight,
        }
    }
    
    /// Get recent routing events
    pub fn get_recent_events(&self, count: usize) -> &[RoutingEvent] {
        let start = if self.routing_history.len() > count {
            self.routing_history.len() - count
        } else {
            0
        };
        
        &self.routing_history[start..]
    }
    
    /// Set top-k parameter
    pub fn set_top_k(&mut self, k: usize) {
        self.top_k = k.clamp(self.min_k, self.max_k);
    }
    
    /// Set routing threshold
    pub fn set_routing_threshold(&mut self, threshold: f32) {
        self.routing_threshold = threshold.clamp(0.0, 1.0);
    }
    
    /// Set diversity weight
    pub fn set_diversity_weight(&mut self, weight: f32) {
        self.diversity_weight = weight.clamp(0.0, 1.0);
    }
    
    /// Enable/disable adaptive k
    pub fn set_adaptive_k(&mut self, adaptive: bool) {
        self.adaptive_k = adaptive;
    }
    
    /// Reset routing state
    pub fn reset(&mut self) -> DLResult<()> {
        self.routing_cache.clear();
        self.cache_hit_count = 0;
        self.cache_miss_count = 0;
        self.routing_history.clear();
        self.selected_indices_history.clear();
        self.average_routing_time = 0.0;
        self.cache_hit_rate = 0.0;
        self.routing_efficiency = 0.0;
        
        Ok(())
    }
}

/// Routing event record
#[derive(Debug, Clone)]
pub struct RoutingEvent {
    pub timestamp: u64,
    pub total_candidates: usize,
    pub selected_candidates: usize,
    pub routing_time: f32,
    pub average_resonance: f32,
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStatistics {
    pub top_k: usize,
    pub routing_threshold: f32,
    pub cache_hit_rate: f32,
    pub average_routing_time: f32,
    pub routing_efficiency: f32,
    pub adaptive_k: bool,
    pub diversity_weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    
    #[test]
    fn test_tkrr_creation() {
        let tkrr = TopKResonanceRouting::new(64, 0.1, 0.3, 0.5).unwrap();
        assert_eq!(tkrr.top_k, 64);
        assert_eq!(tkrr.routing_threshold, 0.1);
        assert_eq!(tkrr.diversity_weight, 0.3);
        assert_eq!(tkrr.confidence_threshold, 0.5);
    }
    
    #[test]
    fn test_resonance_candidate_creation() {
        let data = ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap();
        let holographic = ArrayD::from_shape_vec(vec![3], vec![0.5, 1.0, 1.5]).unwrap();
        let candidate = ResonanceCandidate::new(0, 0.8, data.clone(), holographic.clone());
        
        assert_eq!(candidate.index, 0);
        assert_eq!(candidate.resonance_score, 0.8);
        assert_eq!(candidate.resonance_data, data);
        assert_eq!(candidate.holographic_component, holographic);
    }
    
    #[test]
    fn test_candidate_ordering() {
        let candidate1 = ResonanceCandidate::new(0, 0.5, ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap(), ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap());
        let candidate2 = ResonanceCandidate::new(1, 0.8, ArrayD::from_shape_vec(vec![1], vec![2.0]).unwrap(), ArrayD::from_shape_vec(vec![1], vec![2.0]).unwrap());
        let candidate3 = ResonanceCandidate::new(2, 0.3, ArrayD::from_shape_vec(vec![1], vec![3.0]).unwrap(), ArrayD::from_shape_vec(vec![1], vec![3.0]).unwrap());
        
        assert!(candidate2 > candidate1);
        assert!(candidate1 > candidate3);
        assert!(candidate2 > candidate3);
    }
    
    #[test]
    fn test_similarity_calculation() {
        let tkrr = TopKResonanceRouting::new(10, 0.1, 0.3, 0.5).unwrap();
        
        let vec1 = ArrayD::from_shape_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let vec2 = ArrayD::from_shape_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let vec3 = ArrayD::from_shape_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap();
        
        let sim1 = tkrr.calculate_similarity(&vec1, &vec2).unwrap();
        let sim2 = tkrr.calculate_similarity(&vec1, &vec3).unwrap();
        
        assert!((sim1 - 1.0).abs() < 1e-6);
        assert!((sim2 - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_adaptive_k_calculation() {
        let mut tkrr = TopKResonanceRouting::new(32, 0.1, 0.3, 0.5).unwrap();
        tkrr.min_k = 4;
        tkrr.max_k = 64;
        
        let k1 = tkrr.calculate_adaptive_k(10).unwrap();
        let k2 = tkrr.calculate_adaptive_k(100).unwrap();
        let k3 = tkrr.calculate_adaptive_k(2).unwrap();
        
        assert!(k1 >= tkrr.min_k && k1 <= tkrr.max_k);
        assert_eq!(k2, tkrr.max_k);
        assert_eq!(k3, 2); // Less than min_k
    }
}
