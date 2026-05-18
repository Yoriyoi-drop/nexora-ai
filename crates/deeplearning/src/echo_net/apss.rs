//! Adaptive Phase Separation Stabilizer (APSS)
//!
//! Block 2 dari ECHO-Net Ω
//!
//! Memperbaiki masalah interferensi destruktif dengan menyesuaikan fase antar token
//! berdasarkan similarity semantik.
//!
//! Formula:
//! Δ_ij = 1 - cos(E_i, E_j)
//! φ'_i = φ_i + λ * Σ_j Δ_ij
//!
//! Efek:
//! - Konsep mirip → resonansi menguat
//! - Konsep konflik → dipisah fase
//! - Mencegah semantic cancellation
//! - Mencegah resonance collapse
//! - Mencegah memory corruption

use crate::{DLResult, DeepLearningError};
use crate::echo_net::HolographicWave;
use ndarray::{ArrayD, Array1, Array2, s};
use std::f32::consts::PI;

/// Adaptive Phase Separation Stabilizer implementation
#[derive(Debug, Clone)]
pub struct AdaptivePhaseSeparationStabilizer {
    // Phase adjustment parameters
    phase_separation_strength: f32,
    similarity_threshold: f32,
    max_phase_adjustment: f32,
    
    // Semantic similarity computation
    embedding_dim: usize,
    similarity_matrix: Option<Array2<f32>>,
    
    // Phase stabilization
    phase_history: Vec<ArrayD<f32>>,
    history_length: usize,
    momentum_factor: f32,
    
    // Conflict detection
    conflict_threshold: f32,
    conflict_penalty: f32,
    
    // Phase normalization
    target_phase_range: (f32, f32),
    normalization_strength: f32,
}

impl AdaptivePhaseSeparationStabilizer {
    /// Create new Adaptive Phase Separation Stabilizer
    pub fn new(
        embedding_dim: usize,
        phase_separation_strength: f32,
        similarity_threshold: f32,
        max_phase_adjustment: f32,
    ) -> DLResult<Self> {
        Ok(Self {
            phase_separation_strength,
            similarity_threshold,
            max_phase_adjustment,
            embedding_dim,
            similarity_matrix: None,
            phase_history: Vec::new(),
            history_length: 5,
            momentum_factor: 0.9,
            conflict_threshold: 0.3,
            conflict_penalty: 2.0,
            target_phase_range: (-PI, PI),
            normalization_strength: 0.1,
        })
    }
    
    /// Forward pass - stabilize phases in holographic wave
    pub fn forward(&mut self, wave: &mut HolographicWave, embeddings: &ArrayD<f32>) -> DLResult<()> {
        if wave.amplitude.shape() != embeddings.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: wave.amplitude.shape().to_vec(),
                actual: embeddings.shape().to_vec(),
            });
        }
        
        // Compute semantic similarity matrix
        let similarity_matrix = self.compute_similarity_matrix(embeddings)?;
        
        // Apply phase separation based on semantic similarity
        self.apply_phase_separation(wave, &similarity_matrix)?;
        
        // Detect and handle phase conflicts
        self.handle_phase_conflicts(wave, &similarity_matrix)?;
        
        // Normalize phases to target range
        self.normalize_phases(&mut wave.phase)?;
        
        // Update phase history for momentum
        self.update_phase_history(&wave.phase)?;
        
        Ok(())
    }
    
    /// Compute semantic similarity matrix between all token pairs
    fn compute_similarity_matrix(&self, embeddings: &ArrayD<f32>) -> DLResult<Array2<f32>> {
        let num_tokens = embeddings.shape()[0];
        let mut similarity_matrix = Array2::zeros((num_tokens, num_tokens));
        
        for i in 0..num_tokens {
            for j in 0..num_tokens {
                if i == j {
                    similarity_matrix[[i, j]] = 1.0;
                } else {
                    let emb_i = embeddings.slice(s![i, ..]).to_owned().into_dimensionality().expect("dimensions match");
                    let emb_j = embeddings.slice(s![j, ..]).to_owned().into_dimensionality().expect("dimensions match");
                    
                    let similarity = self.cosine_similarity(&emb_i, &emb_j);
                    similarity_matrix[[i, j]] = similarity;
                }
            }
        }
        
        Ok(similarity_matrix)
    }
    
    /// Apply phase separation based on semantic similarity
    fn apply_phase_separation(&mut self, wave: &mut HolographicWave, similarity_matrix: &Array2<f32>) -> DLResult<()> {
        let num_tokens = wave.amplitude.shape()[0];
        let phase_dim = wave.phase.shape()[1];
        
        for i in 0..num_tokens {
            let mut phase_adjustment: Array1<f32> = Array1::zeros(phase_dim);
            
            // Compute phase adjustment based on all other tokens
            for j in 0..num_tokens {
                if i != j {
                    let similarity = similarity_matrix[[i, j]];
                    let dissimilarity = 1.0 - similarity;
                    
                    // Only adjust for dissimilar concepts
                    if similarity < self.similarity_threshold {
                        let phase_j = wave.phase.slice(s![j, ..]).to_owned();
                        let phase_i = wave.phase.slice(s![i, ..]).to_owned();
                        
                        // Compute phase difference
                        let mut phase_diff = Array1::zeros(phase_dim);
                        for k in 0..phase_dim {
                            phase_diff[k] = phase_j[k] - phase_i[k];
                            
                            // Wrap to [-π, π]
                            while phase_diff[k] > PI {
                                phase_diff[k] -= 2.0 * PI;
                            }
                            while phase_diff[k] < -PI {
                                phase_diff[k] += 2.0 * PI;
                            }
                        }
                        
                        // Weight by dissimilarity
                        for k in 0..phase_dim {
                            phase_adjustment[k] += self.phase_separation_strength * dissimilarity * phase_diff[k];
                        }
                    }
                }
            }
            
            // Apply phase adjustment with momentum
            let current_phase = wave.phase.slice(s![i, ..]).to_owned();
            let momentum_correction = if self.phase_history.len() > 0 {
                let last_phase = &self.phase_history[self.phase_history.len() - 1];
                let last_phase_i = last_phase.slice(s![i, ..]).to_owned();
                
                let mut momentum = Array1::zeros(phase_dim);
                for k in 0..phase_dim {
                    momentum[k] = self.momentum_factor * (current_phase[k] - last_phase_i[k]);
                }
                momentum
            } else {
                Array1::zeros(phase_dim)
            };
            
            // Apply adjustments with limits
            for k in 0..phase_dim {
                let total_adjustment: f32 = phase_adjustment[k] + momentum_correction[k];
                
                // Limit maximum adjustment
                let clamped_adjustment = total_adjustment
                    .max(-self.max_phase_adjustment)
                    .min(self.max_phase_adjustment);
                
                wave.phase[[i, k]] += clamped_adjustment;
            }
        }
        
        Ok(())
    }
    
    /// Detect and handle phase conflicts
    fn handle_phase_conflicts(&mut self, wave: &mut HolographicWave, similarity_matrix: &Array2<f32>) -> DLResult<()> {
        let num_tokens = wave.amplitude.shape()[0];
        let phase_dim = wave.phase.shape()[1];
        
        for i in 0..num_tokens {
            for j in (i + 1)..num_tokens {
                let similarity = similarity_matrix[[i, j]];
                
                // Check for conflict (low similarity but similar phases)
                if similarity < self.conflict_threshold {
                    let phase_i = wave.phase.slice(s![i, ..]).to_owned();
                    let phase_j = wave.phase.slice(s![j, ..]).to_owned();
                    
                    let phase_similarity = self.phase_similarity(&phase_i, &phase_j);
                    
                    if phase_similarity > 0.8 {
                        // Conflict detected - separate phases
                        let separation_amount = self.conflict_penalty * (1.0 - similarity);
                        
                        for k in 0..phase_dim {
                            let diff = phase_j[k] - phase_i[k];
                            
                            // Apply separation
                            wave.phase[[i, k]] -= separation_amount * diff * 0.5;
                            wave.phase[[j, k]] += separation_amount * diff * 0.5;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Normalize phases to target range
    fn normalize_phases(&self, phases: &mut ArrayD<f32>) -> DLResult<()> {
        let (min_range, max_range) = self.target_phase_range;
        
        phases.mapv_inplace(|phase| {
            let mut normalized_phase = phase;
            
            // Wrap to [-π, π] first
            while normalized_phase > PI {
                normalized_phase -= 2.0 * PI;
            }
            while normalized_phase < -PI {
                normalized_phase += 2.0 * PI;
            }
            
            // Apply gentle normalization towards target range
            if normalized_phase < min_range {
                normalized_phase = min_range + (normalized_phase - min_range) * self.normalization_strength;
            } else if normalized_phase > max_range {
                normalized_phase = max_range + (normalized_phase - max_range) * self.normalization_strength;
            }
            
            normalized_phase
        });
        
        Ok(())
    }
    
    /// Update phase history for momentum calculations
    fn update_phase_history(&mut self, current_phase: &ArrayD<f32>) -> DLResult<()> {
        self.phase_history.push(current_phase.clone());
        
        // Keep only recent history
        if self.phase_history.len() > self.history_length {
            self.phase_history.remove(0);
        }
        
        Ok(())
    }
    
    /// Compute cosine similarity between two embeddings
    fn cosine_similarity(&self, emb1: &Array1<f32>, emb2: &Array1<f32>) -> f32 {
        let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = emb1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
    
    /// Compute phase similarity (circular correlation)
    fn phase_similarity(&self, phase1: &Array1<f32>, phase2: &Array1<f32>) -> f32 {
        let mut similarity = 0.0;
        let count = phase1.len();
        
        for (&p1, &p2) in phase1.iter().zip(phase2.iter()) {
            similarity += (p1 - p2).cos();
        }
        
        similarity / count as f32
    }
    
    /// Get phase separation statistics
    pub fn get_phase_statistics(&self) -> PhaseStatistics {
        let avg_separation = if self.phase_history.len() >= 2 {
            let current = &self.phase_history[self.phase_history.len() - 1];
            let previous = &self.phase_history[self.phase_history.len() - 2];
            
            let mut total_diff = 0.0;
            let count = current.len();
            
            for i in 0..count {
                total_diff += (current[i] - previous[i]).abs();
            }
            
            total_diff / count as f32
        } else {
            0.0
        };
        
        PhaseStatistics {
            average_phase_separation: avg_separation,
            conflict_count: 0, // Would need to track during forward pass
            stability_score: 1.0 / (1.0 + avg_separation),
        }
    }
    
    /// Reset internal state
    pub fn reset(&mut self) -> DLResult<()> {
        self.similarity_matrix = None;
        self.phase_history.clear();
        Ok(())
    }
    
    /// Set phase separation strength
    pub fn set_phase_separation_strength(&mut self, strength: f32) {
        self.phase_separation_strength = strength;
    }
    
    /// Set similarity threshold
    pub fn set_similarity_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold;
    }
    
    /// Set maximum phase adjustment
    pub fn set_max_phase_adjustment(&mut self, max_adjustment: f32) {
        self.max_phase_adjustment = max_adjustment;
    }
}

/// Statistics for phase separation monitoring
#[derive(Debug, Clone)]
pub struct PhaseStatistics {
    pub average_phase_separation: f32,
    pub conflict_count: usize,
    pub stability_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_apss_creation() {
        let apss = AdaptivePhaseSeparationStabilizer::new(512, 0.1, 0.5, 0.5).unwrap();
        assert_eq!(apss.embedding_dim, 512);
        assert_eq!(apss.phase_separation_strength, 0.1);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let apss = AdaptivePhaseSeparationStabilizer::new(4, 0.1, 0.5, 0.5).unwrap();
        
        let vec1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let vec2 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let vec3 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        
        assert!((apss.cosine_similarity(&vec1, &vec2) - 1.0).abs() < 1e-6);
        assert!((apss.cosine_similarity(&vec1, &vec3) - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_phase_similarity() {
        let apss = AdaptivePhaseSeparationStabilizer::new(4, 0.1, 0.5, 0.5).unwrap();
        
        let phase1 = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let phase2 = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let phase3 = Array1::from_vec(vec![std::f32::consts::PI, 0.0, 0.0, 0.0]);
        
        assert!((apss.phase_similarity(&phase1, &phase2) - 1.0).abs() < 1e-6);
        assert!((apss.phase_similarity(&phase1, &phase3) - 0.5).abs() < 1e-6);
    }
}
