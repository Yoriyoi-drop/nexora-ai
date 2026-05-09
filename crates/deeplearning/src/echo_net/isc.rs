//! Inverse Spectral Collapse (ISC)
//!
//! Block 9 dari ECHO-Net Ω
//!
//! Final block yang mengubah distribusi resonansi kembali ke representasi token deterministik.
//! 
//! Wave collapse:
//! ĥ_t = IFFT(ctx)
//!
//! Lalu:
//! y = softmax(W_o * ĥ_t)
//!
//! Ini mengubah distribusi resonansi kembali ke representasi token yang dapat digunakan
//! untuk output atau layer berikutnya.

use crate::{DLResult, DeepLearningError};
use crate::echo_net::{HolographicWave, ComplexTensor};
use crate::echo_net::utils::{HolographicFFT, SpectralAnalyzer, Complex};
use ndarray::{ArrayD, Array2, Array1, s};
use std::f32::consts::PI;

/// Spectral collapse configuration
#[derive(Debug, Clone)]
pub struct SpectralCollapseConfig {
    pub output_size: usize,
    pub temperature: f32,
    pub collapse_strength: f32,
    pub phase_preservation: f32,
    pub amplitude_normalization: bool,
    pub frequency_filtering: bool,
    pub min_frequency: f32,
    pub max_frequency: f32,
}

impl Default for SpectralCollapseConfig {
    fn default() -> Self {
        Self {
            output_size: 512,
            temperature: 1.0,
            collapse_strength: 1.0,
            phase_preservation: 0.8,
            amplitude_normalization: true,
            frequency_filtering: true,
            min_frequency: 0.1,
            max_frequency: 10.0,
        }
    }
}

/// Inverse Spectral Collapse implementation
#[derive(Debug, Clone)]
pub struct InverseSpectralCollapse {
    // Configuration
    config: SpectralCollapseConfig,
    
    // Output projection weights
    output_weights: Array2<f32>,
    output_bias: Array1<f32>,
    
    // Collapse parameters
    collapse_kernel: Array2<Complex>,
    phase_reconstruction: Array1<f32>,
    
    // Frequency filtering
    frequency_filters: Vec<Array1<f32>>,
    filter_banks: usize,
    
    // Normalization parameters
    amplitude_scale: f32,
    phase_scale: f32,
    
    // Collapse statistics
    collapse_history: Vec<CollapseEvent>,
    average_energy: f32,
    average_phase_coherence: f32,
    collapse_efficiency: f32,
    
    // Adaptive parameters
    adaptive_temperature: bool,
    adaptive_strength: bool,
    temperature_decay: f32,
    strength_growth: f32,
    
    // Performance metrics
    collapse_latency: f32,
    output_quality: f32,
    spectral_fidelity: f32,
}

impl InverseSpectralCollapse {
        /// Calculate Shannon entropy of the output
        fn calculate_entropy(&self, output: &Array1<f32>) -> f32 {
            let mut entropy = 0.0_f32;
            
            for &value in output {
                if value <= 0.0 {
                    continue;
                }
                
                let prob = value / output.sum();
                if prob > 0.0 {
                    entropy -= prob * prob.ln();
                }
            }
            
            entropy
        }
        
        /// Calculate phase coherence
        fn calculate_phase_coherence(&self, input: &ArrayD<f32>) -> DLResult<f32> {
            let values: Vec<f32> = input.iter().cloned().collect();
            if values.len() < 2 {
                return Ok(1.0);
            }
            
            // Calculate phase coherence based on value consistency
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            
            // Higher coherence for lower variance
            Ok((-variance).exp())
        }
        
        /// Record collapse event
        fn record_collapse_event(&mut self, input: &ArrayD<f32>, output: &Array1<f32>, collapse_time: f32, timestamp: usize) -> DLResult<()> {
            // Calculate metrics
            let input_energy: f32 = input.iter().map(|&x| x * x).sum();
            let output_entropy = self.calculate_entropy(output);
            let phase_coherence = self.calculate_phase_coherence(input)?;
            
            let event = CollapseEvent {
                timestamp: timestamp as u64,
                collapse_time,
                input_energy,
                output_entropy,
                phase_coherence,
                temperature: self.config.temperature,
                collapse_strength: self.config.collapse_strength,
            };
            
            self.collapse_history.push(event);
            
            // Keep only recent history
            if self.collapse_history.len() > 1000 {
                self.collapse_history.remove(0);
            }
            
            // Update running averages
            self.average_energy = self.average_energy * 0.9 + input_energy * 0.1;
            
            Ok(())
        }
        
        /// Get performance statistics
        pub fn get_statistics(&self) -> CollapseStatistics {
            CollapseStatistics {
                average_energy: self.average_energy,
                average_phase_coherence: self.average_phase_coherence,
                collapse_efficiency: self.collapse_efficiency,
                collapse_latency: self.collapse_latency,
                output_quality: 0.0,
                spectral_fidelity: 0.0,
                current_temperature: 0.0,
                current_collapse_strength: 0.0,
            }
        }
    
    /// Create new Inverse Spectral Collapse
    pub fn new(input_dim: usize, config: SpectralCollapseConfig) -> DLResult<Self> {
        // Initialize output projection weights
        let output_weights = Self::xavier_init(input_dim, config.output_size);
        let output_bias = Array1::zeros(config.output_size);
        
        // Initialize collapse kernel
        let collapse_kernel = Self::create_collapse_kernel(16)?; // 16x16 kernel
        
        // Initialize phase reconstruction
        let phase_reconstruction = Array1::from_vec(vec![0.5, 0.3, 0.2]); // Harmonic reconstruction weights
        
        // Initialize frequency filters
        let mut frequency_filters = Vec::new();
        let filter_banks = 8;
        
        for i in 0..filter_banks {
            let filter = Self::create_frequency_filter(i, filter_banks, &config)?;
            frequency_filters.push(filter);
        }
        
        Ok(Self {
            config,
            output_weights,
            output_bias,
            collapse_kernel,
            phase_reconstruction,
            frequency_filters,
            filter_banks,
            amplitude_scale: 1.0,
            phase_scale: 1.0,
            collapse_history: Vec::new(),
            average_energy: 0.0,
            average_phase_coherence: 0.0,
            collapse_efficiency: 0.0,
            adaptive_temperature: true,
            adaptive_strength: true,
            temperature_decay: 0.99,
            strength_growth: 1.01,
            collapse_latency: 0.0,
            output_quality: 0.0,
            spectral_fidelity: 0.0,
        })
    }
    
    /// Forward pass - perform inverse spectral collapse
    pub fn forward(&mut self, context: &ArrayD<f32>, timestamp: usize) -> DLResult<Array1<f32>> {
        let start_time = std::time::Instant::now();
        
        // Convert context to complex representation
        let complex_context = self.context_to_complex(context)?;
        
        // Apply frequency filtering if enabled
        let filtered_context = if self.config.frequency_filtering {
            self.apply_frequency_filtering(&complex_context)?
        } else {
            complex_context
        };
        
        // Perform inverse FFT (wave collapse)
        let collapsed_wave = self.perform_wave_collapse(&filtered_context)?;
        
        // Apply phase preservation
        let phase_preserved = self.apply_phase_preservation(&collapsed_wave)?;
        
        // Normalize amplitude if enabled
        let normalized = if self.config.amplitude_normalization {
            self.normalize_amplitude(&phase_preserved)?
        } else {
            phase_preserved
        };
        
        // Convert back to real representation
        let real_representation = self.complex_to_real(&normalized)?;
        
        // Apply output projection
        let projected = self.apply_output_projection(&real_representation)?;
        
        // Apply softmax with temperature
        let output = self.apply_softmax(&projected)?;
        
        // Record collapse event
        let collapse_time = start_time.elapsed().as_secs_f32();
        self.record_collapse_event(context, &output, collapse_time, timestamp)?;
        
        // Update adaptive parameters
        self.update_adaptive_parameters(&output)?;
        
        // Update performance metrics
        self.update_performance_metrics(collapse_time, &output)?;
        
        Ok(output)
    }
    
    /// Convert context to complex representation
    fn context_to_complex(&self, context: &ArrayD<f32>) -> DLResult<Array2<Complex>> {
        let total_elements = context.len();
        let size = (total_elements as f32).sqrt() as usize;
        
        if size * size != total_elements {
            return Err(DeepLearningError::Configuration {
                reason: "Context size must be a perfect square for 2D FFT".to_string(),
            });
        }
        
        let mut complex_array = Array2::zeros((size, size));
        
        for (i, &val) in context.iter().enumerate() {
            let row = i / size;
            let col = i % size;
            
            // Create complex representation with phase information
            let phase = self.estimate_phase(val, i, total_elements);
            complex_array[[row, col]] = Complex::from_polar(val.abs(), phase);
        }
        
        Ok(complex_array)
    }
    
    /// Estimate phase for real value
    fn estimate_phase(&self, value: f32, position: usize, total_size: usize) -> f32 {
        // Simple phase estimation based on position and value
        let normalized_pos = position as f32 / total_size as f32;
        let base_phase = 2.0 * PI * normalized_pos;
        
        // Modulate phase based on value magnitude
        let magnitude_factor = value.abs().tanh();
        base_phase * magnitude_factor
    }
    
    /// Apply frequency filtering
    fn apply_frequency_filtering(&self, context: &Array2<Complex>) -> DLResult<Array2<Complex>> {
        let mut filtered = context.clone();
        
        // Apply each filter bank
        for filter in &self.frequency_filters {
            let filtered_bank = self.apply_single_filter(context, filter)?;
            
            // Combine with existing filtered result
            for ((i, j), &filtered_val) in filtered_bank.indexed_iter() {
                filtered[[i, j]] = filtered[[i, j]] + filtered_val * 0.125; // Equal weight for 8 filters
            }
        }
        
        Ok(filtered)
    }
    
    /// Apply single frequency filter
    fn apply_single_filter(&self, context: &Array2<Complex>, filter: &Array1<f32>) -> DLResult<Array2<Complex>> {
        let (rows, cols) = context.dim();
        let mut filtered = Array2::zeros((rows, cols));
        
        for i in 0..rows {
            for j in 0..cols {
                let freq_idx = (i * cols + j) % filter.len();
                let filter_value = filter[freq_idx];
                
                // Apply frequency-domain filtering
                filtered[[i, j]] = context[[i, j]] * filter_value;
            }
        }
        
        Ok(filtered)
    }
    
    /// Perform wave collapse using inverse FFT
    fn perform_wave_collapse(&self, context: &Array2<Complex>) -> DLResult<Array2<Complex>> {
        // Apply 2D inverse FFT
        let mut collapsed = Array2::zeros(context.dim());
        
        // Simple inverse FFT implementation (for demonstration)
        // In practice, would use optimized FFT library
        for i in 0..context.nrows() {
            for j in 0..context.ncols() {
                let mut sum = Complex::new(0.0, 0.0);
                
                for ki in 0..context.nrows() {
                    for kj in 0..context.ncols() {
                        let phase = 2.0 * PI * (i as f32 * ki as f32 / context.nrows() as f32 
                                            + j as f32 * kj as f32 / context.ncols() as f32);
                        let kernel_val = self.collapse_kernel[[ki % self.collapse_kernel.nrows(), 
                                                              kj % self.collapse_kernel.ncols()]];
                        
                        sum = sum + context[[ki, kj]] * kernel_val * Complex::from_polar(1.0, phase);
                    }
                }
                
                collapsed[[i, j]] = sum * self.config.collapse_strength;
            }
        }
        
        Ok(collapsed)
    }
    
    /// Apply phase preservation
    fn apply_phase_preservation(&self, collapsed: &Array2<Complex>) -> DLResult<Array2<Complex>> {
        let mut preserved = collapsed.clone();
        
        for ((i, j), &complex_val) in collapsed.indexed_iter() {
            let current_phase = complex_val.phase();
            let current_amplitude = complex_val.magnitude();
            
            // Harmonic phase reconstruction
            let reconstructed_phase = self.reconstruct_phase(current_phase, i, j);
            
            // Blend original and reconstructed phase
            let final_phase = self.config.phase_preservation * current_phase 
                            + (1.0 - self.config.phase_preservation) * reconstructed_phase;
            
            preserved[[i, j]] = Complex::from_polar(current_amplitude, final_phase);
        }
        
        Ok(preserved)
    }
    
    /// Reconstruct phase using harmonic analysis
    fn reconstruct_phase(&self, phase: f32, row: usize, col: usize) -> f32 {
        // Use harmonic reconstruction weights
        let fundamental = phase;
        let harmonic2 = (2.0 * phase).sin() * self.phase_reconstruction[1];
        let harmonic3 = (3.0 * phase).sin() * self.phase_reconstruction[2];
        
        // Add spatial modulation
        let spatial_mod = ((row + col) as f32 * 0.1).sin();
        
        fundamental + harmonic2 + harmonic3 + spatial_mod * self.phase_reconstruction[0]
    }
    
    /// Normalize amplitude
    fn normalize_amplitude(&self, collapsed: &Array2<Complex>) -> DLResult<Array2<Complex>> {
        let mut normalized = collapsed.clone();
        
        // Find maximum amplitude
        let max_amplitude = collapsed.iter()
            .map(|c| c.magnitude())
            .fold(0.0f32, |acc, amp| acc.max(amp));
        
        if max_amplitude > 0.0 {
            let scale = self.amplitude_scale / max_amplitude;
            
            for ((i, j), complex_val) in collapsed.indexed_iter() {
                normalized[[i, j]] = Complex::from_polar(
                    complex_val.magnitude() * scale,
                    complex_val.phase()
                );
            }
        }
        
        Ok(normalized)
    }
    
    /// Convert complex to real representation
    fn complex_to_real(&self, complex: &Array2<Complex>) -> DLResult<Array1<f32>> {
        let (rows, cols) = complex.dim();
        let mut real_vec = Vec::with_capacity(rows * cols);
        
        for complex_val in complex.iter() {
            // Use real part as primary representation
            real_vec.push(complex_val.real);
            
            // Optionally include imaginary part
            if real_vec.len() < rows * cols {
                real_vec.push(complex_val.imag);
            }
        }
        
        // Truncate or pad to match expected size
        while real_vec.len() > self.output_weights.nrows() {
            real_vec.pop();
        }
        
        while real_vec.len() < self.output_weights.nrows() {
            real_vec.push(0.0);
        }
        
        Ok(Array1::from(real_vec))
    }
    
    /// Apply output projection
    fn apply_output_projection(&self, real_representation: &Array1<f32>) -> DLResult<Array1<f32>> {
        let projected = real_representation.dot(&self.output_weights);
        let biased = projected + &self.output_bias;
        Ok(biased)
    }
    
    /// Apply softmax with temperature
    fn apply_softmax(&self, input: &Array1<f32>) -> DLResult<Array1<f32>> {
        // Apply temperature scaling
        let scaled = input.mapv(|x| x / self.config.temperature);
        
        // Compute softmax
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let exp_shifted: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_shifted.iter().sum();
        
        if sum_exp == 0.0 {
            return Ok(Array1::zeros(input.len()));
        }
        
        let softmax: Vec<f32> = exp_shifted.iter().map(|&x| x / sum_exp).collect();
        Ok(Array1::from(softmax))
    }
    
        
        
    /// Update adaptive parameters
    fn update_adaptive_parameters(&mut self, output: &Array1<f32>) -> DLResult<()> {
        if !self.adaptive_temperature && !self.adaptive_strength {
            return Ok(());
        }
        
        let output_entropy = self.calculate_entropy(output);
        
        // Adaptive temperature: decrease for high entropy (more confident outputs)
        if self.adaptive_temperature {
            if output_entropy > 2.0 {
                self.config.temperature *= self.temperature_decay;
            } else if output_entropy < 1.0 {
                self.config.temperature = (self.config.temperature / self.temperature_decay).min(2.0);
            }
        }
        
        // Adaptive strength: increase for low coherence
        if self.adaptive_strength {
            let coherence = self.calculate_phase_coherence(&output.view().into_dimensionality().unwrap().to_owned())?;
            if coherence < 0.5 {
                self.config.collapse_strength *= self.strength_growth;
            } else if coherence > 0.8 {
                self.config.collapse_strength /= self.strength_growth;
            }
        }
        
        Ok(())
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self, collapse_time: f32, output: &Array1<f32>) -> DLResult<()> {
        // Update collapse latency
        self.collapse_latency = self.collapse_latency * 0.9 + collapse_time * 0.1;
        
        // Update output quality (based on entropy)
        let output_entropy = self.calculate_entropy(output);
        
        self.output_quality = self.output_quality * 0.9 + (1.0 / (1.0 + output_entropy)) * 0.1;
        
        // Update spectral fidelity (based on phase coherence)
        let coherence = self.calculate_phase_coherence(&output.to_owned().into_dimensionality().unwrap())?;
        self.spectral_fidelity = self.spectral_fidelity * 0.9 + coherence * 0.1;
        
        // Update collapse efficiency
        self.collapse_efficiency = self.output_quality * self.spectral_fidelity;
        
        Ok(())
    }
    
    /// Create collapse kernel
    fn create_collapse_kernel(size: usize) -> DLResult<Array2<Complex>> {
        let mut kernel = Array2::zeros((size, size));
        
        for i in 0..size {
            for j in 0..size {
                // Gaussian kernel with phase
                let center = size as f32 / 2.0;
                let dx = i as f32 - center;
                let dy = j as f32 - center;
                let distance = (dx * dx + dy * dy).sqrt();
                
                let sigma = size as f32 / 4.0;
                let amplitude = (-(distance * distance) / (2.0 * sigma * sigma)).exp();
                let phase = 2.0 * PI * distance / size as f32;
                
                kernel[[i, j]] = Complex::from_polar(amplitude, phase);
            }
        }
        
        Ok(kernel)
    }
    
    /// Create frequency filter
    fn create_frequency_filter(bank_idx: usize, total_banks: usize, config: &SpectralCollapseConfig) -> DLResult<Array1<f32>> {
        let filter_size = 64;
        let mut filter = Array1::zeros(filter_size);
        
        let min_freq = config.min_frequency;
        let max_freq = config.max_frequency;
        let bandwidth = (max_freq - min_freq) / total_banks as f32;
        let center_freq = min_freq + bank_idx as f32 * bandwidth;
        
        for i in 0..filter_size {
            let freq = min_freq + (max_freq - min_freq) * i as f32 / filter_size as f32;
            
            // Gaussian bandpass filter
            let sigma = bandwidth / 4.0;
            let filter_value = (-(freq - center_freq).powi(2) / (2.0 * sigma * sigma)).exp();
            filter[i] = filter_value;
        }
        
        Ok(filter)
    }
    
    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let scale = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            rand::random::<f32>() * 2.0 * scale - scale
        })
    }
    
        
    /// Get recent collapse events
    pub fn get_recent_events(&self, count: usize) -> &[CollapseEvent] {
        let start = if self.collapse_history.len() > count {
            self.collapse_history.len() - count
        } else {
            0
        };
        
        &self.collapse_history[start..]
    }
    
    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature.max(0.1);
    }
    
    /// Set collapse strength
    pub fn set_collapse_strength(&mut self, strength: f32) {
        self.config.collapse_strength = strength.max(0.1);
    }
    
    /// Set phase preservation
    pub fn set_phase_preservation(&mut self, preservation: f32) {
        self.config.phase_preservation = preservation.clamp(0.0, 1.0);
    }
    
    /// Enable/disable adaptive parameters
    pub fn set_adaptive_temperature(&mut self, adaptive: bool) {
        self.adaptive_temperature = adaptive;
    }
    
    pub fn set_adaptive_strength(&mut self, adaptive: bool) {
        self.adaptive_strength = adaptive;
    }
    
    /// Reset collapse state
    pub fn reset(&mut self) -> DLResult<()> {
        self.collapse_history.clear();
        self.average_energy = 0.0;
        self.average_phase_coherence = 0.0;
        self.collapse_efficiency = 0.0;
        self.collapse_latency = 0.0;
        self.output_quality = 0.0;
        self.spectral_fidelity = 0.0;
        
        // Reset adaptive parameters
        self.config.temperature = 1.0;
        self.config.collapse_strength = 1.0;
        
        Ok(())
    }
}

/// Collapse event record
#[derive(Debug, Clone)]
pub struct CollapseEvent {
    pub timestamp: u64,
    pub collapse_time: f32,
    pub input_energy: f32,
    pub output_entropy: f32,
    pub phase_coherence: f32,
    pub temperature: f32,
    pub collapse_strength: f32,
}

/// Collapse statistics
#[derive(Debug, Clone)]
pub struct CollapseStatistics {
    pub average_energy: f32,
    pub average_phase_coherence: f32,
    pub collapse_efficiency: f32,
    pub collapse_latency: f32,
    pub output_quality: f32,
    pub spectral_fidelity: f32,
    pub current_temperature: f32,
    pub current_collapse_strength: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    
    #[test]
    fn test_isc_creation() {
        let config = SpectralCollapseConfig::default();
        let isc = InverseSpectralCollapse::new(1024, config).unwrap();
        
        assert_eq!(isc.config.output_size, 512);
        assert_eq!(isc.config.temperature, 1.0);
        assert_eq!(isc.config.collapse_strength, 1.0);
    }
    
    #[test]
    fn test_spectral_collapse_config_default() {
        let config = SpectralCollapseConfig::default();
        
        assert_eq!(config.output_size, 512);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.collapse_strength, 1.0);
        assert_eq!(config.phase_preservation, 0.8);
        assert!(config.amplitude_normalization);
        assert!(config.frequency_filtering);
    }
    
    #[test]
    fn test_entropy_calculation() {
        let isc = InverseSpectralCollapse::new(10, SpectralCollapseConfig::default()).unwrap();
        
        let uniform = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let concentrated = Array1::from_vec(vec![0.9, 0.05, 0.03, 0.02]);
        
        let entropy_uniform = isc.calculate_entropy(&uniform);
        let entropy_concentrated = isc.calculate_entropy(&concentrated);
        
        assert!(entropy_uniform > entropy_concentrated);
    }
    
    #[test]
    fn test_phase_coherence_calculation() {
        let isc = InverseSpectralCollapse::new(10, SpectralCollapseConfig::default()).unwrap();
        
        let coherent = ArrayD::from_shape_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let incoherent = ArrayD::from_shape_vec(vec![4], vec![1.0, -1.0, 1.0, -1.0]).unwrap();
        
        let coherence_coherent = isc.calculate_phase_coherence(&coherent).unwrap();
        let coherence_incoherent = isc.calculate_phase_coherence(&incoherent).unwrap();
        
        assert!(coherence_coherent > coherence_incoherent);
    }
    
    #[test]
    fn test_softmax_application() {
        let isc = InverseSpectralCollapse::new(4, SpectralCollapseConfig::default()).unwrap();
        
        let input = Array1::from_vec(vec![2.0, 1.0, 0.1, 0.0]);
        let softmax = isc.apply_softmax(&input).unwrap();
        
        // Check softmax properties
        let sum: f32 = softmax.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check ordering preservation
        assert!(softmax[0] > softmax[1]);
        assert!(softmax[1] > softmax[2]);
        assert!(softmax[2] > softmax[3]);
    }
}
