//! Semantic Spectral Embedding (SSE)
//!
//! Block 1 dari ECHO-Net Ω
//! 
//! Mengubah token biasa menjadi representasi spektral holografik:
//! - Amplitude spectrum: kekuatan fitur
//! - Semantic phase: posisi konseptual  
//! - Resonance energy: domain semantik
//!
//! Formula:
//! E_t = Embed(x_t)
//! A_t = W_A * E_t
//! φ_t = W_φ * E_t + P(t)
//! ω_t = W_ω * E_t
//!
//! Representasi gelombang:
//! Ψ_t(ω) = A_t(ω) * e^(i(ω_t + φ_t))

use crate::{DLResult, DeepLearningError};
use crate::echo_net::{HolographicWave, ComplexTensor};
use ndarray::{ArrayD, Array2, Array1, ArrayView1, s};
use rand::Rng;

/// Semantic Spectral Embedding implementation
#[derive(Debug, Clone)]
pub struct SemanticSpectralEmbedding {
    // Embedding parameters
    vocab_size: usize,
    embedding_dim: usize,
    embedding_matrix: Array2<f32>,
    
    // Spectral transformation weights
    amplitude_weights: Array2<f32>,
    phase_weights: Array2<f32>,
    frequency_weights: Array2<f32>,
    
    // Spectral dimensions
    amplitude_dim: usize,
    phase_dim: usize,
    resonance_dim: usize,
    
    // Positional encoding
    max_position: usize,
    positional_encoding: Array2<f32>,
    
    // Normalization parameters
    amplitude_scale: f32,
    phase_scale: f32,
    frequency_scale: f32,
}

impl SemanticSpectralEmbedding {
    /// Create new Semantic Spectral Embedding
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        amplitude_dim: usize,
        phase_dim: usize,
        resonance_dim: usize,
        max_position: usize,
    ) -> DLResult<Self> {
        // Initialize embedding matrix
        let embedding_matrix = Self::xavier_init(vocab_size, embedding_dim);
        
        // Initialize spectral transformation weights
        let amplitude_weights = Self::xavier_init(embedding_dim, amplitude_dim);
        let phase_weights = Self::xavier_init(embedding_dim, phase_dim);
        let frequency_weights = Self::xavier_init(embedding_dim, resonance_dim);
        
        // Initialize positional encoding
        let positional_encoding = Self::create_positional_encoding(max_position, embedding_dim);
        
        Ok(Self {
            vocab_size,
            embedding_dim,
            embedding_matrix,
            amplitude_weights,
            phase_weights,
            frequency_weights,
            amplitude_dim,
            phase_dim,
            resonance_dim,
            max_position,
            positional_encoding,
            amplitude_scale: 1.0,
            phase_scale: 1.0,
            frequency_scale: 1.0,
        })
    }
    
    /// Forward pass - convert tokens to spectral representation
    pub fn forward(&self, token_ids: &[usize], positions: &[usize]) -> DLResult<HolographicWave> {
        if token_ids.len() != positions.len() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![token_ids.len()],
                actual: vec![positions.len()],
            });
        }
        
        let batch_size = token_ids.len();
        let mut amplitude = ArrayD::zeros(vec![batch_size, self.amplitude_dim]);
        let mut phase = ArrayD::zeros(vec![batch_size, self.phase_dim]);
        let mut frequency = ArrayD::zeros(vec![batch_size, self.resonance_dim]);
        
        for (i, (&token_id, &pos)) in token_ids.iter().zip(positions.iter()).enumerate() {
            if token_id >= self.vocab_size {
                return Err(DeepLearningError::InvalidDimension { dim: token_id });
            }
            
            // Get embedding
            let mut embedding = self.embedding_matrix.row(token_id).to_owned();
            
            // Add positional encoding
            if pos < self.max_position {
                let pos_encoding = self.positional_encoding.row(pos);
                embedding = embedding + pos_encoding;
            }
            
            // Transform to spectral components
            let embedding_view: ArrayView1<f32> = embedding.view().into_dimensionality().unwrap();
            
            // Amplitude spectrum
            let amp_result: Array1<f32> = {
                let weights_view = self.amplitude_weights.view();
                embedding_view.dot(&weights_view.t())
            };
            for j in 0..self.amplitude_dim {
                amplitude[[i, j]] = amp_result[j] * self.amplitude_scale;
            }
            
            // Semantic phase
            let phase_result: Array1<f32> = {
                let weights_view = self.phase_weights.view();
                embedding_view.dot(&weights_view.t())
            };
            for j in 0..self.phase_dim {
                phase[[i, j]] = phase_result[j] * self.phase_scale;
            }
            
            // Resonance frequency
            let freq_result: Array1<f32> = {
                let weights_view = self.frequency_weights.view();
                embedding_view.dot(&weights_view.t())
            };
            for j in 0..self.resonance_dim {
                frequency[[i, j]] = freq_result[j] * self.frequency_scale;
            }
        }
        
        Ok(HolographicWave {
            amplitude,
            phase,
            frequency,
        })
    }
    
    /// Convert single token to spectral representation
    pub fn embed_token(&self, token_id: usize, position: usize) -> DLResult<HolographicWave> {
        self.forward(&[token_id], &[position])
    }
    
    /// Batch embedding for multiple tokens
    pub fn embed_batch(&self, token_ids: &[usize]) -> DLResult<HolographicWave> {
        let positions: Vec<usize> = (0..token_ids.len()).collect();
        self.forward(token_ids, &positions)
    }
    
    /// Get raw embedding for a token
    pub fn get_embedding(&self, token_id: usize) -> DLResult<Array1<f32>> {
        if token_id >= self.vocab_size {
            return Err(DeepLearningError::InvalidDimension { dim: token_id });
        }
        
        Ok(self.embedding_matrix.row(token_id).to_owned())
    }
    
    /// Create positional encoding using sinusoidal functions
    fn create_positional_encoding(max_position: usize, embedding_dim: usize) -> Array2<f32> {
        let mut encoding = Array2::zeros((max_position, embedding_dim));
        
        for pos in 0..max_position {
            for i in (0..embedding_dim).step_by(2) {
                let div_term = (pos as f32) / (10000.0f32.powf(i as f32 / embedding_dim as f32));
                encoding[[pos, i]] = div_term.sin();
                
                if i + 1 < embedding_dim {
                    encoding[[pos, i + 1]] = div_term.cos();
                }
            }
        }
        
        encoding
    }
    
    /// Xavier initialization for weights
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        let scale = (6.0 / (rows + cols) as f32).sqrt();
        
        Array2::from_shape_fn((rows, cols), |_| {
            rng.gen_range(-scale..scale)
        })
    }
    
    /// Normalize spectral components
    pub fn normalize_spectral(&self, wave: &mut HolographicWave) -> DLResult<()> {
        // Normalize amplitude to [0, 1]
        let amp_max = wave.amplitude.iter().fold(0.0f32, |acc, &x| acc.max(x));
        if amp_max > 0.0 {
            wave.amplitude.mapv_inplace(|x| x / amp_max);
        }
        
        // Normalize phase to [-π, π]
        wave.phase.mapv_inplace(|x| {
            let mut p = x;
            while p > std::f32::consts::PI {
                p -= 2.0 * std::f32::consts::PI;
            }
            while p < -std::f32::consts::PI {
                p += 2.0 * std::f32::consts::PI;
            }
            p
        });
        
        // Normalize frequency to positive range
        let freq_min = wave.frequency.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        if freq_min < 0.0 {
            wave.frequency.mapv_inplace(|x| x - freq_min);
        }
        
        Ok(())
    }
    
    /// Apply frequency band filtering
    pub fn apply_frequency_filter(&self, wave: &mut HolographicWave, min_freq: f32, max_freq: f32) -> DLResult<()> {
        wave.frequency.mapv_inplace(|f| {
            if f < min_freq || f > max_freq {
                0.0
            } else {
                f
            }
        });
        
        // Zero out amplitude for filtered frequencies
        for ((idx, &freq), amp) in wave.frequency.indexed_iter().zip(wave.amplitude.iter_mut()) {
            if freq < min_freq || freq > max_freq {
                *amp = 0.0;
            }
        }
        
        Ok(())
    }
    
    /// Compute spectral similarity between two waves
    pub fn spectral_similarity(&self, wave1: &HolographicWave, wave2: &HolographicWave) -> DLResult<f32> {
        if wave1.amplitude.shape() != wave2.amplitude.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: wave1.amplitude.shape().to_vec(),
                actual: wave2.amplitude.shape().to_vec(),
            });
        }
        
        // Amplitude similarity
        let amp_sim = self.cosine_similarity(&wave1.amplitude, &wave2.amplitude);
        
        // Phase similarity
        let phase_sim = self.phase_similarity(&wave1.phase, &wave2.phase);
        
        // Frequency similarity
        let freq_sim = self.cosine_similarity(&wave1.frequency, &wave2.frequency);
        
        // Weighted combination
        Ok(0.4 * amp_sim + 0.3 * phase_sim + 0.3 * freq_sim)
    }
    
    /// Cosine similarity between two arrays
    fn cosine_similarity(&self, arr1: &ArrayD<f32>, arr2: &ArrayD<f32>) -> f32 {
        let dot_product: f32 = arr1.iter().zip(arr2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = arr1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = arr2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
    
    /// Phase similarity (circular correlation)
    fn phase_similarity(&self, phase1: &ArrayD<f32>, phase2: &ArrayD<f32>) -> f32 {
        let mut similarity = 0.0;
        let count = phase1.len();
        
        for (&p1, &p2) in phase1.iter().zip(phase2.iter()) {
            similarity += (p1 - p2).cos();
        }
        
        similarity / count as f32
    }
    
    /// Convert holographic wave to complex tensor
    pub fn to_complex(&self, wave: &HolographicWave) -> DLResult<ComplexTensor> {
        ComplexTensor::from_polar(&wave.amplitude, &wave.phase)
    }
    
    /// Update embedding weights (for training)
    pub fn update_weights(&mut self, amplitude_grad: &Array2<f32>, phase_grad: &Array2<f32>, frequency_grad: &Array2<f32>, learning_rate: f32) -> DLResult<()> {
        // Update spectral transformation weights
        self.amplitude_weights = &self.amplitude_weights - &(amplitude_grad * learning_rate);
        self.phase_weights = &self.phase_weights - &(phase_grad * learning_rate);
        self.frequency_weights = &self.frequency_weights - &(frequency_grad * learning_rate);
        
        Ok(())
    }
    
    /// Get model parameters for training
    pub fn get_parameters(&self) -> Vec<&Array2<f32>> {
        vec![
            &self.embedding_matrix,
            &self.amplitude_weights,
            &self.phase_weights,
            &self.frequency_weights,
        ]
    }
    
    /// Get mutable parameters for training
    pub fn get_parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![
            &mut self.embedding_matrix,
            &mut self.amplitude_weights,
            &mut self.phase_weights,
            &mut self.frequency_weights,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sse_creation() {
        let sse = SemanticSpectralEmbedding::new(1000, 512, 256, 128, 128, 1024).unwrap();
        assert_eq!(sse.vocab_size, 1000);
        assert_eq!(sse.embedding_dim, 512);
        assert_eq!(sse.amplitude_dim, 256);
    }
    
    #[test]
    fn test_token_embedding() {
        let sse = SemanticSpectralEmbedding::new(1000, 512, 256, 128, 128, 1024).unwrap();
        let wave = sse.embed_token(42, 0).unwrap();
        
        assert_eq!(wave.amplitude.shape(), &[1, 256]);
        assert_eq!(wave.phase.shape(), &[1, 128]);
        assert_eq!(wave.frequency.shape(), &[1, 128]);
    }
    
    #[test]
    fn test_batch_embedding() {
        let sse = SemanticSpectralEmbedding::new(1000, 512, 256, 128, 128, 1024).unwrap();
        let tokens = vec![1, 2, 3, 4, 5];
        let wave = sse.embed_batch(&tokens).unwrap();
        
        assert_eq!(wave.amplitude.shape(), &[5, 256]);
        assert_eq!(wave.phase.shape(), &[5, 128]);
        assert_eq!(wave.frequency.shape(), &[5, 128]);
    }
}
