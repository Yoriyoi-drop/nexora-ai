//! Multi-Band Holographic Writer (MBHW)
//!
//! Block 3 dari ECHO-Net Ω
//!
//! Menulis hologram ke multiple frequency bands untuk menangani skala berbeda:
//! - Low frequency: tema global
//! - Mid frequency: struktur reasoning  
//! - High frequency: detail lokal
//!
//! Write operation:
//! H_b ← H_b + Ψ_b ⊗ K_b
//!
//! Kompleksitas: O(BKd) dengan B ≪ T

use crate::{DLResult, DeepLearningError};
use crate::echo_net::{HolographicWave, ComplexTensor};
use crate::echo_net::utils::Complex;
use ndarray::{Array2, Array1};
use std::f32::consts::PI;

/// Frequency band configuration
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    pub id: usize,
    pub frequency_range: (f32, f32),
    pub kernel_size: usize,
    pub description: String,
}

impl FrequencyBand {
    pub fn new(id: usize, min_freq: f32, max_freq: f32, kernel_size: usize, description: &str) -> Self {
        Self {
            id,
            frequency_range: (min_freq, max_freq),
            kernel_size,
            description: description.to_string(),
        }
    }
}

/// Multi-Band Holographic Writer implementation
#[derive(Debug, Clone)]
pub struct MultiBandHolographicWriter {
    // Frequency bands
    bands: Vec<FrequencyBand>,
    num_bands: usize,
    
    // Holographic memory for each band
    holographic_memory: Vec<Array2<Complex>>,
    memory_size: usize,
    
    // Kernels for each band
    kernels: Vec<Array2<Complex>>,
    
    // Write parameters
    write_strength: f32,
    interference_threshold: f32,
    decay_factor: f32,
    
    // Frequency filtering
    frequency_filters: Vec<Array1<f32>>,
    
    // Memory management
    write_positions: Vec<usize>,
    memory_utilization: Vec<f32>,
    
    // Interference management
    interference_matrix: Option<Array2<f32>>,
    last_write_timestamp: Vec<usize>,
}

impl MultiBandHolographicWriter {
    /// Create new Multi-Band Holographic Writer
    pub fn new(
        bands: Vec<FrequencyBand>,
        memory_size: usize,
        write_strength: f32,
        interference_threshold: f32,
    ) -> DLResult<Self> {
        let num_bands = bands.len();
        
        // Initialize holographic memory for each band
        let mut holographic_memory = Vec::with_capacity(bands.len());
        let mut kernels = Vec::with_capacity(bands.len());
        let mut frequency_filters = Vec::with_capacity(bands.len());
        
        for band in &bands {
            // Initialize memory for this band
            let memory = Array2::zeros((memory_size, band.kernel_size));
            holographic_memory.push(memory);
            
            // Initialize kernel for this band
            let kernel = Self::create_holographic_kernel(band.kernel_size, band.frequency_range)?;
            kernels.push(kernel);
            
            // Create frequency filter
            let filter = Self::create_frequency_filter(band.frequency_range, band.kernel_size)?;
            frequency_filters.push(filter);
        }
        
        Ok(Self {
            bands,
            num_bands,
            holographic_memory,
            memory_size,
            kernels,
            write_strength,
            interference_threshold,
            decay_factor: 0.99,
            frequency_filters,
            write_positions: vec![0; num_bands],
            memory_utilization: vec![0.0; num_bands],
            interference_matrix: None,
            last_write_timestamp: vec![0; num_bands],
        })
    }
    
    /// Forward pass - write holographic wave to multiple bands
    pub fn forward(&mut self, wave: &HolographicWave, timestamp: usize) -> DLResult<Vec<Array2<Complex>>> {
        // Convert wave to complex representation
        let complex_wave = wave.to_complex()?;
        
        // Filter wave into frequency bands
        let band_waves = self.filter_into_bands(&complex_wave)?;
        
        // Write to each band
        let mut written_memories = Vec::with_capacity(band_waves.len());
        for (band_idx, band_wave) in band_waves.into_iter().enumerate() {
            let written_memory = self.write_to_band(band_idx, &band_wave, timestamp)?;
            written_memories.push(written_memory);
        }
        
        // Update interference matrix
        self.update_interference_matrix(&written_memories)?;
        
        // Update memory utilization
        self.update_memory_utilization()?;
        
        Ok(written_memories)
    }
    
    /// Filter complex wave into frequency bands
    fn filter_into_bands(&self, wave: &ComplexTensor) -> DLResult<Vec<Array2<Complex>>> {
        let mut band_waves = Vec::with_capacity(self.bands.len());
        
        for (band_idx, band) in self.bands.iter().enumerate() {
            let filtered_wave = self.apply_frequency_filter(wave, band_idx)?;
            band_waves.push(filtered_wave);
        }
        
        Ok(band_waves)
    }
    
    /// Apply frequency filter to complex wave
    fn apply_frequency_filter(&self, wave: &ComplexTensor, band_idx: usize) -> DLResult<Array2<Complex>> {
        let band = &self.bands[band_idx];
        let filter = &self.frequency_filters[band_idx];
        
        // Convert to 2D representation for convolution
        let wave_2d = self.reshape_to_2d(wave, band.kernel_size)?;
        
        // Apply frequency filtering in frequency domain
        let mut filtered = Array2::zeros(wave_2d.dim());
        
        for ((i, j), &complex_val) in wave_2d.indexed_iter() {
            let freq_idx = (i + j) % filter.len();
            let filter_value = filter[freq_idx];
            
            if filter_value > 0.0 {
                filtered[[i, j]] = complex_val * filter_value;
            }
        }
        
        Ok(filtered)
    }
    
    /// Write to specific band
    fn write_to_band(&mut self, band_idx: usize, band_wave: &Array2<Complex>, timestamp: usize) -> DLResult<Array2<Complex>> {
        let write_pos = self.write_positions[band_idx];
        let kernel = &self.kernels[band_idx];
        
        // Apply convolution with kernel
        let convolved = self.convolve_with_kernel(band_wave, kernel)?;
        
        // Apply write strength
        let scaled_convolution = convolved.mapv(|c| c * self.write_strength);
        
        // Get current memory state
        let current_memory = self.holographic_memory[band_idx].clone();
        
        // Apply decay to existing memory
        let decayed_memory = current_memory.mapv(|c| c * self.decay_factor);
        
        // Write new data with interference handling
        let new_memory = self.apply_interference_handling(&decayed_memory, &scaled_convolution, write_pos)?;
        
        // Update memory
        self.holographic_memory[band_idx] = new_memory.clone();
        
        // Update write position
        self.write_positions[band_idx] = (write_pos + 1) % self.memory_size;
        self.last_write_timestamp[band_idx] = timestamp;
        
        Ok(new_memory)
    }
    
    /// Convolve with holographic kernel
    fn convolve_with_kernel(&self, input: &Array2<Complex>, kernel: &Array2<Complex>) -> DLResult<Array2<Complex>> {
        let (input_rows, input_cols) = input.dim();
        let (kernel_rows, kernel_cols) = kernel.dim();
        
        let output_rows = input_rows + kernel_rows - 1;
        let output_cols = input_cols + kernel_cols - 1;
        let mut output = Array2::zeros((output_rows, output_cols));
        
        // Simple convolution (can be optimized with FFT)
        for i in 0..input_rows {
            for j in 0..input_cols {
                for ki in 0..kernel_rows {
                    for kj in 0..kernel_cols {
                        let output_i = i + ki;
                        let output_j = j + kj;
                        output[[output_i, output_j]] = output[[output_i, output_j]] + input[[i, j]] * kernel[[ki, kj]];
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Apply interference handling during write
    fn apply_interference_handling(&self, current_memory: &Array2<Complex>, new_data: &Array2<Complex>, write_pos: usize) -> DLResult<Array2<Complex>> {
        let mut result = current_memory.clone();
        let (rows, cols) = new_data.dim();
        
        for i in 0..rows {
            for j in 0..cols {
                let memory_pos = (write_pos + i * cols + j) % self.memory_size;
                
                if memory_pos < rows {
                    let current_val = result[[memory_pos, j % cols]];
                    let new_val = new_data[[i, j]];
                    
                    // Check for destructive interference
                    let interference = self.calculate_interference(&current_val, &new_val);
                    
                    if interference > self.interference_threshold {
                        // Apply constructive interference enhancement
                        let enhanced_val = Complex::new(
                            (current_val.real + new_val.real) * 1.2,
                            (current_val.imag + new_val.imag) * 1.2,
                        );
                        result[[memory_pos, j % cols]] = enhanced_val;
                    } else {
                        // Normal addition
                        result[[memory_pos, j % cols]] = current_val + new_val;
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Calculate interference between two complex values
    fn calculate_interference(&self, val1: &Complex, val2: &Complex) -> f32 {
        // Phase difference
        let phase1 = val1.phase();
        let phase2 = val2.phase();
        let phase_diff = (phase1 - phase2).abs();
        
        // Amplitude difference
        let amp_diff = (val1.magnitude() - val2.magnitude()).abs();
        
        // Interference strength (0 = constructive, π = destructive)
        let interference_strength = phase_diff.cos();
        
        // Weight by amplitude difference
        interference_strength * (1.0 - amp_diff / (val1.magnitude() + val2.magnitude() + 1e-8))
    }
    
    /// Update interference matrix
    fn update_interference_matrix(&mut self, memories: &[Array2<Complex>]) -> DLResult<()> {
        let mut interference_matrix = Array2::zeros((self.num_bands, self.num_bands));
        
        for i in 0..self.num_bands {
            for j in 0..self.num_bands {
                if i != j {
                    let interference = self.calculate_band_interference(&memories[i], &memories[j])?;
                    interference_matrix[[i, j]] = interference;
                } else {
                    interference_matrix[[i, j]] = 1.0;
                }
            }
        }
        
        self.interference_matrix = Some(interference_matrix);
        Ok(())
    }
    
    /// Calculate interference between two bands
    fn calculate_band_interference(&self, band1: &Array2<Complex>, band2: &Array2<Complex>) -> DLResult<f32> {
        let (rows1, cols1) = band1.dim();
        let (rows2, cols2) = band2.dim();
        
        let min_rows = rows1.min(rows2);
        let min_cols = cols1.min(cols2);
        
        let mut total_interference = 0.0;
        let mut count = 0;
        
        for i in 0..min_rows {
            for j in 0..min_cols {
                let interference = self.calculate_interference(&band1[[i, j]], &band2[[i, j]]);
                total_interference += interference;
                count += 1;
            }
        }
        
        Ok(total_interference / count as f32)
    }
    
    /// Update memory utilization statistics
    fn update_memory_utilization(&mut self) -> DLResult<()> {
        for band_idx in 0..self.num_bands {
            let memory = &self.holographic_memory[band_idx];
            let mut active_cells = 0;
            let total_cells = memory.len();
            
            for complex_val in memory.iter() {
                if complex_val.magnitude() > 1e-6 {
                    active_cells += 1;
                }
            }
            
            self.memory_utilization[band_idx] = active_cells as f32 / total_cells as f32;
        }
        
        Ok(())
    }
    
    /// Create holographic kernel for frequency band
    fn create_holographic_kernel(kernel_size: usize, frequency_range: (f32, f32)) -> DLResult<Array2<Complex>> {
        let mut kernel = Array2::zeros((kernel_size, kernel_size));
        let (min_freq, max_freq) = frequency_range;
        
        for i in 0..kernel_size {
            for j in 0..kernel_size {
                // Create spatial frequency
                let spatial_freq_x = (i as f32 - kernel_size as f32 / 2.0) / kernel_size as f32;
                let spatial_freq_y = (j as f32 - kernel_size as f32 / 2.0) / kernel_size as f32;
                
                // Map to frequency range
                let freq_x = min_freq + (max_freq - min_freq) * (spatial_freq_x + 1.0) / 2.0;
                let freq_y = min_freq + (max_freq - min_freq) * (spatial_freq_y + 1.0) / 2.0;
                
                // Create complex kernel value
                let phase = 2.0 * PI * (freq_x * i as f32 + freq_y * j as f32) / kernel_size as f32;
                let amplitude = 1.0 / kernel_size as f32;
                
                kernel[[i, j]] = Complex::from_polar(amplitude, phase);
            }
        }
        
        Ok(kernel)
    }
    
    /// Create frequency filter
    fn create_frequency_filter(frequency_range: (f32, f32), filter_size: usize) -> DLResult<Array1<f32>> {
        let (min_freq, max_freq) = frequency_range;
        let mut filter = Array1::zeros(filter_size);
        
        for i in 0..filter_size {
            let freq = min_freq + (max_freq - min_freq) * i as f32 / filter_size as f32;
            
            // Gaussian filter centered in the band
            let center_freq = (min_freq + max_freq) / 2.0;
            let bandwidth = (max_freq - min_freq) / 2.0;
            let filter_value = (-(freq - center_freq).powi(2) / (2.0 * bandwidth * bandwidth)).exp();
            
            filter[i] = filter_value;
        }
        
        Ok(filter)
    }
    
    /// Reshape complex tensor to 2D array
    fn reshape_to_2d(&self, wave: &ComplexTensor, kernel_size: usize) -> DLResult<Array2<Complex>> {
        let total_elements = wave.real.len();
        let rows = (total_elements + kernel_size - 1) / kernel_size;
        let cols = kernel_size;
        
        let mut result = Array2::zeros((rows, cols));
        
        for (idx, (&real_val, &imag_val)) in wave.real.iter().zip(wave.imag.iter()).enumerate() {
            let row = idx / cols;
            let col = idx % cols;
            if row < rows {
                result[[row, col]] = Complex::new(real_val, imag_val);
            }
        }
        
        Ok(result)
    }
    
    /// Get holographic memory for a specific band
    pub fn get_band_memory(&self, band_idx: usize) -> DLResult<&Array2<Complex>> {
        if band_idx >= self.num_bands {
            return Err(DeepLearningError::InvalidDimension { dim: band_idx });
        }
        
        Ok(&self.holographic_memory[band_idx])
    }
    
    /// Get memory utilization statistics
    pub fn get_memory_utilization(&self) -> &[f32] {
        &self.memory_utilization
    }
    
    /// Get interference matrix
    pub fn get_interference_matrix(&self) -> Option<&Array2<f32>> {
        self.interference_matrix.as_ref()
    }
    
    /// Reset all holographic memory
    pub fn reset(&mut self) -> DLResult<()> {
        for memory in &mut self.holographic_memory {
            memory.fill(Complex::new(0.0, 0.0));
        }
        
        self.write_positions.fill(0);
        self.memory_utilization.fill(0.0);
        self.interference_matrix = None;
        self.last_write_timestamp.fill(0);
        
        Ok(())
    }
    
    /// Set write strength
    pub fn set_write_strength(&mut self, strength: f32) {
        self.write_strength = strength;
    }
    
    /// Set decay factor
    pub fn set_decay_factor(&mut self, decay: f32) {
        self.decay_factor = decay;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    
    #[test]
    fn test_mbhw_creation() {
        let bands = vec![
            FrequencyBand::new(0, 0.0, 1.0, 8, "Low frequency"),
            FrequencyBand::new(1, 1.0, 4.0, 16, "Mid frequency"),
            FrequencyBand::new(2, 4.0, 16.0, 32, "High frequency"),
        ];
        
        let mbhw = MultiBandHolographicWriter::new(bands, 1000, 0.1, 0.5).unwrap();
        assert_eq!(mbhw.num_bands, 3);
        assert_eq!(mbhw.memory_size, 1000);
    }
    
    #[test]
    fn test_frequency_band_creation() {
        let band = FrequencyBand::new(0, 1.0, 10.0, 16, "Test band");
        assert_eq!(band.id, 0);
        assert_eq!(band.frequency_range, (1.0, 10.0));
        assert_eq!(band.kernel_size, 16);
        assert_eq!(band.description, "Test band");
    }
    
    #[test]
    fn test_interference_calculation() {
        let mbhw = MultiBandHolographicWriter::new(vec![], 100, 0.1, 0.5).unwrap();
        
        let val1 = Complex::new(1.0, 0.0);
        let val2 = Complex::new(1.0, 0.0); // Same phase
        let val3 = Complex::new(-1.0, 0.0); // Opposite phase
        
        let interference1 = mbhw.calculate_interference(&val1, &val2);
        let interference2 = mbhw.calculate_interference(&val1, &val3);
        
        assert!(interference1 > interference2); // Constructive > Destructive
    }
}
