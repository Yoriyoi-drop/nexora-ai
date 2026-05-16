//! Harmonic Temporal Encoding (HTE)
//!
//! Encoding temporal yang menangkap:
//! - Periodicity patterns
//! - Rhythm dan cyclic dependencies
//! - Complex temporal relationships
//! - Multimodal temporal coherence

use crate::{DLResult, DeepLearningError};
use crate::star_x::core::TemporalProcessor;
use crate::star_x::tensor_pool::PooledTensor1D;
use crate::star_x::fused_ops::{FusedLinearActivation, FusedElementWise};
use ndarray::{ArrayD, Array1};
use std::f32::consts::PI;
use rand;
use once_cell;

/// Harmonic Temporal Encoding implementation
pub struct HarmonicTemporalEncoding {
    // Encoding parameters
    frequencies: Vec<f32>,
    amplitudes: Vec<f32>,
    phases: Vec<f32>,
    embedding_dim: usize,
    num_harmonics: usize,
    
    // Adaptive parameters
    frequency_decay: f32,
    amplitude_modulation: f32,
    phase_shift: f32,
    
    // Fused operations for optimization
    #[allow(dead_code)]
    fused_linear: Option<FusedLinearActivation>,
    #[allow(dead_code)]
    fused_element_wise: Option<FusedElementWise>,
    
    // Temporal context
    max_position: usize,
    current_position: usize,
    temporal_cache: Vec<ArrayD<f32>>,
}

impl std::fmt::Debug for HarmonicTemporalEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarmonicTemporalEncoding")
            .field("embedding_dim", &self.embedding_dim)
            .field("num_harmonics", &self.num_harmonics)
            .field("frequency_decay", &self.frequency_decay)
            .field("max_position", &self.max_position)
            .field("current_position", &self.current_position)
            .field("has_fused_linear", &self.fused_linear.is_some())
            .field("has_fused_element_wise", &self.fused_element_wise.is_some())
            .finish()
    }
}

impl Clone for HarmonicTemporalEncoding {
    fn clone(&self) -> Self {
        Self {
            frequencies: self.frequencies.clone(),
            amplitudes: self.amplitudes.clone(),
            phases: self.phases.clone(),
            embedding_dim: self.embedding_dim,
            num_harmonics: self.num_harmonics,
            frequency_decay: self.frequency_decay,
            amplitude_modulation: self.amplitude_modulation,
            phase_shift: self.phase_shift,
            fused_linear: None, // Cannot clone fused operations, recreate as needed
            fused_element_wise: None,
            max_position: self.max_position,
            current_position: self.current_position,
            temporal_cache: self.temporal_cache.clone(),
        }
    }
}

impl HarmonicTemporalEncoding {
    pub fn new(
        embedding_dim: usize,
        num_harmonics: usize,
        max_position: usize,
    ) -> DLResult<Self> {
        if embedding_dim < num_harmonics * 2 {
            return Err(DeepLearningError::Configuration {
                reason: format!("embedding_dim {} must be at least 2x num_harmonics {}", embedding_dim, num_harmonics),
            });
        }
        
        // Initialize frequencies dengan geometric progression
        let mut frequencies = Vec::with_capacity(num_harmonics);
        for i in 0..num_harmonics {
            let base_freq = 2.0 * PI * (i as f32 + 1.0);
            frequencies.push(base_freq);
        }
        
        // Initialize amplitudes dengan decay
        let mut amplitudes = Vec::with_capacity(num_harmonics);
        for i in 0..num_harmonics {
            let amplitude = 1.0 / (1.0 + i as f32 * 0.1); // Decay factor
            amplitudes.push(amplitude);
        }
        
        // Initialize phases dengan random values
        let mut phases = Vec::with_capacity(num_harmonics);
        for _i in 0..num_harmonics {
            let phase = (rand::random::<f32>() * 2.0 - 1.0) * PI;
            phases.push(phase);
        }
        
        Ok(Self {
            frequencies,
            amplitudes,
            phases,
            embedding_dim,
            num_harmonics,
            frequency_decay: 0.99,
            amplitude_modulation: 1.0,
            phase_shift: 0.0,
            fused_linear: None,
            fused_element_wise: None,
            max_position,
            current_position: 0,
            temporal_cache: Vec::new(),
        })
    }
    
    /// Compute harmonic encoding untuk position tertentu
    pub fn compute_harmonic_encoding(&self, position: usize) -> DLResult<ArrayD<f32>> {
        let pos_f32 = position as f32;
        
        // Use pooled tensor untuk mengurangi alokasi
        let mut pooled_tensor = PooledTensor1D::new(self.embedding_dim)?;
        let encoding = pooled_tensor.get_mut();
        let encoding_flat = encoding.as_slice_mut().expect("tensor should be contiguous");
        
        for i in 0..self.num_harmonics {
            let freq = self.frequencies[i];
            let phase = self.phases[i];
            let harmonic_val = (freq * pos_f32 + phase).sin();
            
            // Store both sin and cos components
            if i * 2 < self.embedding_dim {
                encoding_flat[i * 2] = harmonic_val;
                if i * 2 + 1 < self.embedding_dim {
                    encoding_flat[i * 2 + 1] = (freq * pos_f32 + phase).cos();
                }
            }
        }
        
        // Add positional encoding untuk remaining dimensions
        for i in self.num_harmonics * 2..self.embedding_dim {
            let pos_enc = if i % 2 == 0 {
                (pos_f32 / 10000.0_f32.powf(i as f32 / self.embedding_dim as f32)).sin()
            } else {
                (pos_f32 / 10000.0_f32.powf(i as f32 / self.embedding_dim as f32)).cos()
            };
            encoding_flat[i] = pos_enc;
        }
        
        Ok(encoding.clone().into_dyn())
    }
    
    /// Compute encoding dengan temporal decay untuk long sequences
    pub fn compute_decay_encoding(&self, position: usize, decay_factor: f32) -> DLResult<ArrayD<f32>> {
        let base_encoding = self.compute_harmonic_encoding(position)?;
        let mut decayed_encoding = base_encoding.clone();
        
        // Apply temporal decay
        let decay = decay_factor.powf(position as f32);
        for val in decayed_encoding.iter_mut() {
            *val *= decay;
        }
        
        Ok(decayed_encoding)
    }
    
    /// Compute relative temporal encoding
    pub fn compute_relative_encoding(&self, query_pos: usize, key_pos: usize) -> DLResult<ArrayD<f32>> {
        let relative_pos = query_pos as f32 - key_pos as f32;
        let mut encoding = Array1::zeros(self.embedding_dim);
        let encoding_flat = encoding.as_slice_mut().expect("tensor should be contiguous");
        
        for i in 0..self.num_harmonics {
            let freq = self.frequencies[i];
            let amp = self.amplitudes[i];
            let phase = self.phases[i];
            
            // Relative sin component
            let sin_val = amp * (freq * relative_pos + phase).sin();
            if 2 * i < encoding_flat.len() {
                encoding_flat[2 * i] = sin_val;
            }
            
            // Relative cos component  
            let cos_val = amp * (freq * relative_pos + phase).cos();
            if 2 * i + 1 < encoding_flat.len() {
                encoding_flat[2 * i + 1] = cos_val;
            }
        }
        
        Ok(encoding.into_dyn())
    }
    
    /// Compute periodicity-aware encoding
    pub fn compute_periodic_encoding(&self, position: usize, period: usize) -> DLResult<ArrayD<f32>> {
        let normalized_pos = (position % period) as f32 / period as f32;
        let mut encoding = Array1::zeros(self.embedding_dim);
        let encoding_flat = encoding.as_slice_mut().expect("tensor should be contiguous");
        
        for i in 0..self.num_harmonics {
            let freq = 2.0 * PI * (i as f32 + 1.0) * normalized_pos;
            let amp = self.amplitudes[i];
            let phase = self.phases[i];
            
            // Periodic sin component
            let sin_val = amp * (freq + phase).sin();
            if 2 * i < encoding_flat.len() {
                encoding_flat[2 * i] = sin_val;
            }
            
            // Periodic cos component
            let cos_val = amp * (freq + phase).cos();
            if 2 * i + 1 < encoding_flat.len() {
                encoding_flat[2 * i + 1] = cos_val;
            }
        }
        
        Ok(encoding.into_dyn())
    }
    
    /// Compute multimodal temporal encoding (untuk audio, video, etc.)
    pub fn compute_multimodal_encoding(&self, 
        position: usize, 
        modality_frequencies: &[f32]
    ) -> DLResult<ArrayD<f32>> {
        let mut encoding = Array1::zeros(self.embedding_dim);
        let encoding_flat = encoding.as_slice_mut().expect("tensor should be contiguous");
        
        // Base harmonic encoding
        let base_encoding = self.compute_harmonic_encoding(position)?;
        let base_flat = base_encoding.as_slice().expect("tensor should be contiguous");
        
        // Blend with modality-specific frequencies
        for i in 0..self.embedding_dim.min(base_flat.len()) {
            encoding_flat[i] = base_flat[i];
        }
        
        // Add modality-specific harmonics
        for (mod_idx, &mod_freq) in modality_frequencies.iter().enumerate() {
            if mod_idx < self.num_harmonics {
                let pos_f32 = position as f32;
                let freq = mod_freq * 2.0 * PI;
                let amp = self.amplitudes[mod_idx] * 0.5; // Reduced amplitude for modality
                
                // Modality sin component
                let sin_val = amp * (freq * pos_f32).sin();
                if 2 * mod_idx < encoding_flat.len() {
                    encoding_flat[2 * mod_idx] += sin_val;
                }
                
                // Modality cos component
                let cos_val = amp * (freq * pos_f32).cos();
                if 2 * mod_idx + 1 < encoding_flat.len() {
                    encoding_flat[2 * mod_idx + 1] += cos_val;
                }
            }
        }
        
        Ok(encoding.into_dyn())
    }
    
    /// Adaptive frequency adjustment based on sequence statistics
    pub fn adapt_frequencies(&mut self, sequence_variance: f32) {
        // Higher variance -> higher frequencies
        // Lower variance -> lower frequencies  
        let adaptation_factor = 1.0 + sequence_variance;
        
        for freq in &mut self.frequencies {
            *freq *= adaptation_factor;
        }
    }
    
    /// Learnable amplitude modulation
    pub fn update_amplitude_modulation(&mut self, gradient: f32, learning_rate: f32) {
        self.amplitude_modulation -= learning_rate * gradient;
        self.amplitude_modulation = self.amplitude_modulation.max(0.1).min(2.0);
    }
    
    /// Phase shift adjustment for temporal alignment
    pub fn adjust_phase_shift(&mut self, target_alignment: f32) {
        // Adjust phase to align with target temporal pattern
        self.phase_shift = target_alignment;
    }
    
    /// Cache temporal encoding untuk efficiency
    pub fn cache_encoding(&mut self, position: usize) -> DLResult<()> {
        if position >= self.temporal_cache.len() {
            self.temporal_cache.resize(position + 1, ArrayD::zeros(vec![self.embedding_dim]));
        }
        
        let encoding = self.compute_harmonic_encoding(position)?;
        self.temporal_cache[position] = encoding;
        
        Ok(())
    }
    
    /// Get cached encoding
    pub fn get_cached_encoding(&self, position: usize) -> Option<&ArrayD<f32>> {
        self.temporal_cache.get(position)
    }
    
    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.temporal_cache.clear();
    }
    
    /// Get temporal statistics
    pub fn get_temporal_stats(&self) -> (f32, f32, f32) {
        let avg_frequency = self.frequencies.iter().sum::<f32>() / self.frequencies.len() as f32;
        let avg_amplitude = self.amplitudes.iter().sum::<f32>() / self.amplitudes.len() as f32;
        let avg_phase = self.phases.iter().sum::<f32>() / self.phases.len() as f32;
        
        (avg_frequency, avg_amplitude, avg_phase)
    }
}

impl TemporalProcessor for HarmonicTemporalEncoding {
    fn process_temporal(&self, input: &ArrayD<f32>, temporal_pos: usize) -> DLResult<ArrayD<f32>> {
        // Get temporal encoding
        let temporal_encoding = self.compute_harmonic_encoding(temporal_pos)?;
        
        // Combine input dengan temporal encoding
        let input_flat = input.as_slice().expect("tensor should be contiguous");
        let temp_flat = temporal_encoding.as_slice().expect("tensor should be contiguous");
        
        let mut combined = Vec::with_capacity(input_flat.len());
        for (i, &input_val) in input_flat.iter().enumerate() {
            let temp_val = if i < temp_flat.len() { temp_flat[i] } else { 0.0 };
            combined.push(input_val + temp_val * 0.1); // Small temporal influence
        }
        
        Ok(Array1::from_vec(combined).into_dyn())
    }
    
    fn update_temporal_state(&mut self, _new_state: ArrayD<f32>) -> DLResult<()> {
        // Update current position based on state
        self.current_position = (self.current_position + 1) % self.max_position;
        
        // Cache encoding untuk current position
        self.cache_encoding(self.current_position)?;
        
        Ok(())
    }
    
    fn get_temporal_state(&self) -> &ArrayD<f32> {
        if let Some(cached) = self.get_cached_encoding(self.current_position) {
            cached
        } else {
            // Return zero encoding if not cached
            static ZERO_ENCODING: once_cell::sync::Lazy<ArrayD<f32>> = 
                once_cell::sync::Lazy::new(|| Array1::zeros(64).into_dyn());
            &ZERO_ENCODING
        }
    }
}

/// Advanced temporal patterns
impl HarmonicTemporalEncoding {
    /// Detect periodicity dalam sequence
    pub fn detect_periodicity(&self, sequence: &[ArrayD<f32>]) -> DLResult<Vec<f32>> {
        if sequence.len() < 4 {
            return Ok(vec![0.0; self.num_harmonics]);
        }
        
        let mut periodicities = Vec::with_capacity(self.num_harmonics);
        
        for i in 0..self.num_harmonics {
            let period = (i + 1) * 2; // Test different periods
            
            if sequence.len() > period {
                // Compute autocorrelation untuk detect periodicity
                let mut correlation = 0.0;
                let count = sequence.len() - period;
                
                for j in 0..count {
                    let seq_j_flat = sequence[j].as_slice().expect("tensor should be contiguous");
                    let seq_jp_flat = sequence[j + period].as_slice().expect("tensor should be contiguous");
                    
                    for (a, b) in seq_j_flat.iter().zip(seq_jp_flat.iter()) {
                        correlation += a * b;
                    }
                }
                
                periodicities.push(correlation / count as f32);
            } else {
                periodicities.push(0.0);
            }
        }
        
        Ok(periodicities)
    }
    
    /// Compute temporal coherence score
    pub fn compute_coherence_score(&self, sequence: &[ArrayD<f32>]) -> DLResult<f32> {
        if sequence.len() < 2 {
            return Ok(1.0);
        }
        
        let mut total_coherence = 0.0;
        let mut comparisons = 0;
        
        // Compute pairwise coherence
        for i in 0..sequence.len() - 1 {
            let seq_i_flat = sequence[i].as_slice().expect("tensor should be contiguous");
            let seq_ip1_flat = sequence[i + 1].as_slice().expect("tensor should be contiguous");
            
            // Compute cosine similarity
            let mut dot_product = 0.0;
            let mut norm_i = 0.0;
            let mut norm_ip1 = 0.0;
            
            for (a, b) in seq_i_flat.iter().zip(seq_ip1_flat.iter()) {
                dot_product += a * b;
                norm_i += a * a;
                norm_ip1 += b * b;
            }
            
            if norm_i > 0.0 && norm_ip1 > 0.0 {
                let coherence = dot_product / (norm_i.sqrt() * norm_ip1.sqrt());
                total_coherence += coherence;
                comparisons += 1;
            }
        }
        
        Ok(if comparisons > 0 { total_coherence / comparisons as f32 } else { 0.0 })
    }
}

impl crate::traits::Forward for HarmonicTemporalEncoding {
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;
    
    fn forward(&self, _input: &Self::Input) -> DLResult<Self::Output> {
        // Use current position for encoding
        self.compute_harmonic_encoding(self.current_position)
    }
}
