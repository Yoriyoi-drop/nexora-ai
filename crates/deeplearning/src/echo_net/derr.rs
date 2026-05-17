//! Dual Entropic Resonance Retrieval (DERR)
//!
//! Block 7 dari ECHO-Net Ω
//!
//! Retrieval sekarang mempertimbangkan tiga faktor:
//! - Energy: E(R) = Σ R_i²
//! - Entropy: H(R) = -Σ p_i * log(p_i)
//! - Coherence: C(R) = |Σ|R_i|| / |ΣR_i|
//!
//! Gate:
//! g = σ(αE - βH + δC)
//!
//! Context retrieval:
//! ctx = g ⋅ R
//!
//! Ini jauh lebih stabil dibanding entropy-only gating.

use crate::{DLResult, DeepLearningError};
use crate::echo_net::HolographicWave;
use ndarray::ArrayD;

/// Retrieval candidate with dual metrics
#[derive(Debug, Clone)]
pub struct RetrievalCandidate {
    pub data: ArrayD<f32>,
    pub energy: f32,
    pub entropy: f32,
    pub coherence: f32,
    pub gate_score: f32,
    pub similarity: f32,
    pub resonance_strength: f32,
}

impl RetrievalCandidate {
    pub fn new(data: ArrayD<f32>) -> Self {
        Self {
            data,
            energy: 0.0,
            entropy: 0.0,
            coherence: 0.0,
            gate_score: 0.0,
            similarity: 0.0,
            resonance_strength: 0.0,
        }
    }
}

/// Dual Entropic Resonance Retrieval implementation
#[derive(Debug, Clone)]
pub struct DualEntropicResonanceRetrieval {
    // Retrieval parameters
    energy_weight: f32,
    entropy_weight: f32,
    coherence_weight: f32,
    gate_threshold: f32,
    
    // Retrieval memory
    retrieval_memory: Vec<RetrievalCandidate>,
    max_memory_size: usize,
    
    // Gate parameters
    gate_alpha: f32,
    gate_beta: f32,
    gate_delta: f32,
    
    // Entropy calculation
    entropy_bins: usize,
    min_entropy: f32,
    max_entropy: f32,
    
    // Coherence calculation
    coherence_window: usize,
    phase_alignment_weight: f32,
    
    // Retrieval statistics
    retrieval_history: Vec<RetrievalEvent>,
    average_gate_score: f32,
    retrieval_efficiency: f32,
    
    // Adaptive parameters
    adaptive_weights: bool,
    weight_update_rate: f32,
    
    // Performance metrics
    retrieval_accuracy: f32,
    gate_stability: f32,
    energy_distribution: Vec<f32>,
}

impl DualEntropicResonanceRetrieval {
    /// Create new Dual Entropic Resonance Retrieval
    pub fn new(
        energy_weight: f32,
        entropy_weight: f32,
        coherence_weight: f32,
        gate_threshold: f32,
        max_memory_size: usize,
    ) -> DLResult<Self> {
        Ok(Self {
            energy_weight,
            entropy_weight,
            coherence_weight,
            gate_threshold,
            retrieval_memory: Vec::new(),
            max_memory_size,
            gate_alpha: 1.0,
            gate_beta: 0.5,
            gate_delta: 0.3,
            entropy_bins: 100,
            min_entropy: 0.0,
            max_entropy: 10.0,
            coherence_window: 8,
            phase_alignment_weight: 0.7,
            retrieval_history: Vec::new(),
            average_gate_score: 0.0,
            retrieval_efficiency: 0.0,
            adaptive_weights: true,
            weight_update_rate: 0.01,
            retrieval_accuracy: 0.0,
            gate_stability: 0.0,
            energy_distribution: Vec::new(),
        })
    }
    
    /// Forward pass - retrieve context using dual entropic resonance
    pub fn forward(&mut self, query: &HolographicWave, candidates: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        // Convert query to tensor
        let query_tensor = self.wave_to_tensor(query)?;
        
        // Process candidates and calculate metrics
        let mut processed_candidates = Vec::with_capacity(candidates.len());
        
        for candidate_data in candidates {
            let mut candidate = RetrievalCandidate::new(candidate_data.clone());
            
            // Calculate similarity with query
            candidate.similarity = self.calculate_similarity(&query_tensor, candidate_data)?;
            
            // Calculate energy
            candidate.energy = self.calculate_energy(candidate_data)?;
            
            // Calculate entropy
            candidate.entropy = self.calculate_entropy(candidate_data)?;
            
            // Calculate coherence
            candidate.coherence = self.calculate_coherence(candidate_data)?;
            
            // Calculate gate score
            candidate.gate_score = self.calculate_gate_score(
                candidate.energy,
                candidate.entropy,
                candidate.coherence
            )?;
            
            // Calculate resonance strength
            candidate.resonance_strength = self.calculate_resonance_strength(&candidate);
            
            processed_candidates.push(candidate);
        }
        
        // Filter candidates by gate threshold
        let filtered_candidates: Vec<&RetrievalCandidate> = processed_candidates
            .iter()
            .filter(|c| c.gate_score >= self.gate_threshold)
            .collect();
        
        // If no candidates pass threshold, use top candidates
        let final_candidates = if filtered_candidates.is_empty() {
            let mut sorted_candidates: Vec<&RetrievalCandidate> = processed_candidates
                .iter()
                .collect();
            sorted_candidates.sort_by(|a, b| b.gate_score.partial_cmp(&a.gate_score).unwrap_or(std::cmp::Ordering::Equal));
            sorted_candidates.into_iter().take(5).collect()
        } else {
            filtered_candidates
        };
        
        // Generate context retrieval
        let context = self.generate_context_retrieval(&final_candidates)?;
        
        // Update retrieval statistics
        self.update_statistics(&processed_candidates, &final_candidates)?;
        
        // Store candidates in memory
        self.update_memory(processed_candidates)?;
        
        Ok(context)
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
    
    /// Calculate similarity between query and candidate
    fn calculate_similarity(&self, query: &ArrayD<f32>, candidate: &ArrayD<f32>) -> DLResult<f32> {
        if query.shape() != candidate.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: query.shape().to_vec(),
                actual: candidate.shape().to_vec(),
            });
        }
        
        // Cosine similarity
        let dot_product: f32 = query.iter().zip(candidate.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = candidate.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm1 * norm2))
    }
    
    /// Calculate energy of candidate
    fn calculate_energy(&self, candidate: &ArrayD<f32>) -> DLResult<f32> {
        let energy: f32 = candidate.iter().map(|&x| x * x).sum();
        Ok(energy.sqrt())
    }
    
    /// Calculate entropy of candidate
    fn calculate_entropy(&self, candidate: &ArrayD<f32>) -> DLResult<f32> {
        // Create histogram for entropy calculation
        let values: Vec<f32> = candidate.iter().cloned().collect();
        
        if values.is_empty() {
            return Ok(0.0);
        }
        
        // Find min and max values
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < 1e-10 {
            return Ok(0.0);
        }
        
        // Create histogram
        let mut histogram = vec![0.0; self.entropy_bins];
        let bin_size = (max_val - min_val) / self.entropy_bins as f32;
        
        for &value in &values {
            let bin_index = ((value - min_val) / bin_size) as usize;
            if bin_index < self.entropy_bins {
                histogram[bin_index] += 1.0;
            }
        }
        
        // Calculate entropy
        let total_samples = values.len() as f32;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0.0 {
                let probability = count / total_samples;
                entropy -= probability * probability.ln();
            }
        }
        
        // Normalize entropy
        let max_possible_entropy = (self.entropy_bins as f32).ln();
        let normalized_entropy = entropy / max_possible_entropy;
        
        Ok(normalized_entropy * self.max_entropy)
    }
    
    /// Calculate coherence of candidate
    fn calculate_coherence(&self, candidate: &ArrayD<f32>) -> DLResult<f32> {
        let values: Vec<f32> = candidate.iter().cloned().collect();
        
        if values.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate absolute sum and regular sum
        let absolute_sum: f32 = values.iter().map(|&x| x.abs()).sum();
        let regular_sum: f32 = values.iter().sum();
        
        if absolute_sum == 0.0 {
            return Ok(0.0);
        }
        
        // Basic coherence: ratio of absolute sum to regular sum
        let basic_coherence = absolute_sum / regular_sum.abs();
        
        // Calculate phase alignment (for complex-like data)
        let phase_alignment = self.calculate_phase_alignment(&values)?;
        
        // Combine basic coherence with phase alignment
        let coherence = 0.6 * basic_coherence + 0.4 * phase_alignment;
        
        Ok(coherence.clamp(0.0, 1.0))
    }
    
    /// Calculate phase alignment for coherence
    fn calculate_phase_alignment(&self, values: &[f32]) -> DLResult<f32> {
        if values.len() < 2 {
            return Ok(1.0);
        }
        
        let window_size = self.coherence_window.min(values.len());
        let mut alignments = Vec::new();
        
        for i in 0..=(values.len() - window_size) {
            let window = &values[i..i + window_size];
            
            // Calculate local phase alignment
            let mut alignment = 0.0;
            for j in 1..window_size {
                let phase_diff = (window[j] - window[j-1]).abs();
                alignment += (-phase_diff).exp(); // Higher alignment for smaller differences
            }
            
            alignments.push(alignment / (window_size - 1) as f32);
        }
        
        // Average alignment across windows
        let average_alignment = alignments.iter().sum::<f32>() / alignments.len() as f32;
        
        Ok(average_alignment)
    }
    
    /// Calculate gate score using dual entropic metrics
    fn calculate_gate_score(&self, energy: f32, entropy: f32, coherence: f32) -> DLResult<f32> {
        // Normalize metrics
        let normalized_energy = self.normalize_energy(energy)?;
        let normalized_entropy = self.normalize_entropy(entropy)?;
        let normalized_coherence = coherence; // Already normalized
        
        // Gate function: g = σ(αE - βH + δC)
        let gate_input = self.gate_alpha * normalized_energy 
                       - self.gate_beta * normalized_entropy 
                       + self.gate_delta * normalized_coherence;
        
        // Apply sigmoid activation
        let gate_score = 1.0 / (1.0 + (-gate_input).exp());
        
        Ok(gate_score)
    }
    
    /// Normalize energy metric
    fn normalize_energy(&self, energy: f32) -> DLResult<f32> {
        // Simple normalization based on typical energy ranges
        let normalized = (energy / 100.0).tanh(); // Assuming typical energy < 100
        Ok(normalized.clamp(0.0, 1.0))
    }
    
    /// Normalize entropy metric
    fn normalize_entropy(&self, entropy: f32) -> DLResult<f32> {
        let normalized = (entropy / self.max_entropy).clamp(0.0, 1.0);
        Ok(normalized)
    }
    
    /// Calculate resonance strength
    fn calculate_resonance_strength(&self, candidate: &RetrievalCandidate) -> f32 {
        // Combine all metrics into resonance strength
        let resonance = 0.3 * candidate.similarity 
                      + 0.3 * candidate.gate_score 
                      + 0.2 * candidate.energy.sqrt() 
                      + 0.1 * candidate.coherence 
                      + 0.1 * (1.0 - candidate.entropy);
        
        resonance.clamp(0.0, 1.0)
    }
    
    /// Generate context retrieval from filtered candidates
    fn generate_context_retrieval(&self, candidates: &[&RetrievalCandidate]) -> DLResult<ArrayD<f32>> {
        if candidates.is_empty() {
            return Err(DeepLearningError::Configuration {
                reason: "No candidates available for context retrieval".to_string(),
            });
        }
        
        // Weighted combination of candidates
        let total_resonance: f32 = candidates.iter().map(|c| c.resonance_strength).sum();
        
        if total_resonance == 0.0 {
            // Equal weighting if no resonance
            let mut context = ArrayD::zeros(candidates[0].data.shape().to_vec());
            
            for candidate in candidates {
                for (i, &val) in candidate.data.iter().enumerate() {
                    if i < context.len() {
                        context[i] += val / candidates.len() as f32;
                    }
                }
            }
            
            return Ok(context);
        }
        
        // Weighted by resonance strength
        let mut context = ArrayD::zeros(candidates[0].data.shape().to_vec());
        
        for candidate in candidates {
            let weight = candidate.resonance_strength / total_resonance;
            
            for (i, &val) in candidate.data.iter().enumerate() {
                if i < context.len() {
                    context[i] += weight * val;
                }
            }
        }
        
        Ok(context)
    }
    
    /// Update retrieval statistics
    fn update_statistics(&mut self, all_candidates: &[RetrievalCandidate], filtered_candidates: &[&RetrievalCandidate]) -> DLResult<()> {
        // Update average gate score
        if !all_candidates.is_empty() {
            let total_gate: f32 = all_candidates.iter().map(|c| c.gate_score).sum();
            self.average_gate_score = total_gate / all_candidates.len() as f32;
        }
        
        // Update retrieval efficiency
        self.retrieval_efficiency = filtered_candidates.len() as f32 / all_candidates.len() as f32;
        
        // Update energy distribution
        self.energy_distribution.clear();
        for candidate in all_candidates {
            self.energy_distribution.push(candidate.energy);
        }
        
        // Record retrieval event
        let event = RetrievalEvent {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_secs(),
            total_candidates: all_candidates.len(),
            retrieved_candidates: filtered_candidates.len(),
            average_gate_score: self.average_gate_score,
            retrieval_efficiency: self.retrieval_efficiency,
        };
        
        self.retrieval_history.push(event);
        
        // Keep only recent history
        if self.retrieval_history.len() > 1000 {
            self.retrieval_history.remove(0);
        }
        
        // Update adaptive weights if enabled
        if self.adaptive_weights {
            self.update_adaptive_weights()?;
        }
        
        Ok(())
    }
    
    /// Update adaptive weights based on performance
    fn update_adaptive_weights(&mut self) -> DLResult<()> {
        if self.retrieval_history.len() < 10 {
            return Ok(());
        }
        
        // Calculate recent performance
        let recent_events: Vec<&RetrievalEvent> = self.retrieval_history
            .iter()
            .rev()
            .take(10)
            .collect();
        
        let avg_efficiency: f32 = recent_events.iter().map(|e| e.retrieval_efficiency).sum::<f32>() / 10.0;
        let avg_gate_score: f32 = recent_events.iter().map(|e| e.average_gate_score).sum::<f32>() / 10.0;
        
        // Adjust weights based on performance
        if avg_efficiency < 0.5 {
            // Increase energy weight to improve filtering
            self.energy_weight += self.weight_update_rate;
        }
        
        if avg_gate_score < 0.3 {
            // Decrease entropy weight to reduce strictness
            self.entropy_weight = (self.entropy_weight - self.weight_update_rate).max(0.1);
        }
        
        // Normalize weights
        let total_weight = self.energy_weight + self.entropy_weight + self.coherence_weight;
        self.energy_weight /= total_weight;
        self.entropy_weight /= total_weight;
        self.coherence_weight /= total_weight;
        
        Ok(())
    }
    
    /// Update memory with new candidates
    fn update_memory(&mut self, candidates: Vec<RetrievalCandidate>) -> DLResult<()> {
        // Add new candidates to memory
        for candidate in candidates {
            self.retrieval_memory.push(candidate);
        }
        
        // Maintain memory size limit
        if self.retrieval_memory.len() > self.max_memory_size {
            // Remove least important candidates
            self.retrieval_memory.sort_by(|a, b| a.resonance_strength.partial_cmp(&b.resonance_strength).unwrap_or(std::cmp::Ordering::Equal));
            self.retrieval_memory.drain(0..self.retrieval_memory.len() - self.max_memory_size);
        }
        
        Ok(())
    }
    
    /// Get retrieval statistics
    pub fn get_statistics(&self) -> RetrievalStatistics {
        RetrievalStatistics {
            memory_size: self.retrieval_memory.len(),
            max_memory_size: self.max_memory_size,
            average_gate_score: self.average_gate_score,
            retrieval_efficiency: self.retrieval_efficiency,
            energy_weight: self.energy_weight,
            entropy_weight: self.entropy_weight,
            coherence_weight: self.coherence_weight,
            adaptive_weights: self.adaptive_weights,
        }
    }
    
    /// Get recent retrieval events
    pub fn get_recent_events(&self, count: usize) -> &[RetrievalEvent] {
        let start = if self.retrieval_history.len() > count {
            self.retrieval_history.len() - count
        } else {
            0
        };
        
        &self.retrieval_history[start..]
    }
    
    /// Set gate parameters
    pub fn set_gate_parameters(&mut self, alpha: f32, beta: f32, delta: f32) {
        self.gate_alpha = alpha;
        self.gate_beta = beta;
        self.gate_delta = delta;
    }
    
    /// Set gate threshold
    pub fn set_gate_threshold(&mut self, threshold: f32) {
        self.gate_threshold = threshold;
    }
    
    /// Enable/disable adaptive weights
    pub fn set_adaptive_weights(&mut self, adaptive: bool) {
        self.adaptive_weights = adaptive;
    }
    
    /// Reset retrieval state
    pub fn reset(&mut self) -> DLResult<()> {
        self.retrieval_memory.clear();
        self.retrieval_history.clear();
        self.average_gate_score = 0.0;
        self.retrieval_efficiency = 0.0;
        self.energy_distribution.clear();
        
        Ok(())
    }
}

/// Retrieval event record
#[derive(Debug, Clone)]
pub struct RetrievalEvent {
    pub timestamp: u64,
    pub total_candidates: usize,
    pub retrieved_candidates: usize,
    pub average_gate_score: f32,
    pub retrieval_efficiency: f32,
}

/// Retrieval statistics
#[derive(Debug, Clone)]
pub struct RetrievalStatistics {
    pub memory_size: usize,
    pub max_memory_size: usize,
    pub average_gate_score: f32,
    pub retrieval_efficiency: f32,
    pub energy_weight: f32,
    pub entropy_weight: f32,
    pub coherence_weight: f32,
    pub adaptive_weights: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    
    #[test]
    fn test_derr_creation() {
        let derr = DualEntropicResonanceRetrieval::new(1.0, 0.5, 0.3, 0.5, 1000).unwrap();
        assert_eq!(derr.energy_weight, 1.0);
        assert_eq!(derr.entropy_weight, 0.5);
        assert_eq!(derr.coherence_weight, 0.3);
        assert_eq!(derr.max_memory_size, 1000);
    }
    
    #[test]
    fn test_retrieval_candidate_creation() {
        let data = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let candidate = RetrievalCandidate::new(data.clone());
        
        assert_eq!(candidate.data, data);
        assert_eq!(candidate.energy, 0.0);
        assert_eq!(candidate.entropy, 0.0);
        assert_eq!(candidate.coherence, 0.0);
    }
    
    #[test]
    fn test_energy_calculation() {
        let derr = DualEntropicResonanceRetrieval::new(1.0, 0.5, 0.3, 0.5, 100).unwrap();
        
        let data = ArrayD::from_shape_vec(vec![2], vec![3.0, 4.0]).unwrap(); // 3-4-5 triangle
        let energy = derr.calculate_energy(&data).unwrap();
        
        assert!((energy - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
    }
    
    #[test]
    fn test_similarity_calculation() {
        let derr = DualEntropicResonanceRetrieval::new(1.0, 0.5, 0.3, 0.5, 100).unwrap();
        
        let vec1 = ArrayD::from_shape_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let vec2 = ArrayD::from_shape_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let vec3 = ArrayD::from_shape_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap();
        
        let sim1 = derr.calculate_similarity(&vec1, &vec2).unwrap();
        let sim2 = derr.calculate_similarity(&vec1, &vec3).unwrap();
        
        assert!((sim1 - 1.0).abs() < 1e-6);
        assert!((sim2 - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_gate_score_calculation() {
        let derr = DualEntropicResonanceRetrieval::new(1.0, 0.5, 0.3, 0.5, 100).unwrap();
        
        let gate_score = derr.calculate_gate_score(1.0, 0.5, 0.8).unwrap();
        
        assert!(gate_score >= 0.0);
        assert!(gate_score <= 1.0);
    }
}
