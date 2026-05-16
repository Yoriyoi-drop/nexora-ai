//! Utility functions untuk holographic operations
//!
//! Berisi fungsi-fungsi matematika untuk operasi holographic:
//! - FFT operations
//! - Complex number operations
//! - Spectral analysis
//! - Resonance calculations

use crate::{DLResult, DeepLearningError};
use ndarray::{Array1, Array2, s};
use std::f32::consts::PI;

/// FFT operations untuk holographic processing
pub struct HolographicFFT;

impl HolographicFFT {
    /// 1D FFT implementation (simplified)
    pub fn fft_1d(input: &Array1<f32>) -> DLResult<Array1<Complex>> {
        let n = input.len();
        if n & (n - 1) != 0 {
            return Err(DeepLearningError::Configuration {
                reason: "Input length must be power of 2".to_string(),
            });
        }
        
        let mut result = vec![Complex::new(0.0, 0.0); n];
        for (i, &val) in input.iter().enumerate() {
            result[i] = Complex::new(val, 0.0);
        }
        
        // Cooley-Tukey FFT implementation
        Self::fft_recursive(&mut result);
        
        Ok(Array1::from(result))
    }
    
    /// Inverse 1D FFT
    pub fn ifft_1d(input: &Array1<Complex>) -> DLResult<Array1<f32>> {
        let mut conj = input.to_vec();
        for complex in &mut conj {
            complex.imag = -complex.imag;
        }
        
        let mut result = conj;
        Self::fft_recursive(&mut result);
        
        let len = result.len() as f32;
        for complex in &mut result {
            complex.real /= len;
            complex.imag = -complex.imag / len;
        }
        
        Ok(Array1::from(result.iter().map(|c| c.real).collect::<Vec<_>>()))
    }
    
    /// Recursive FFT implementation
    fn fft_recursive(x: &mut [Complex]) {
        let n = x.len();
        if n <= 1 {
            return;
        }
        
        // Divide
        let (even, odd) = x.split_at_mut(n / 2);
        Self::fft_recursive(even);
        Self::fft_recursive(odd);
        
        // Combine
        for k in 0..n / 2 {
            let t = Complex::from_polar(
                1.0,
                -2.0 * PI * k as f32 / n as f32,
            ) * odd[k];
            odd[k] = even[k] - t;
            even[k] = even[k] + t;
        }
    }
    
    /// 2D FFT untuk holographic images
    pub fn fft_2d(input: &Array2<f32>) -> DLResult<Array2<Complex>> {
        let (rows, cols) = input.dim();
        let mut result = Array2::zeros((rows, cols));
        
        // FFT pada setiap row
        for i in 0..rows {
            let row = input.slice(s![i, ..]).to_owned();
            let fft_row = Self::fft_1d(&row)?;
            for j in 0..cols {
                result[[i, j]] = fft_row[j];
            }
        }
        
        // FFT pada setiap column
        for j in 0..cols {
            let col = result.column(j).to_owned();
            let mut col_vec = col.to_vec();
            Self::fft_recursive(&mut col_vec);
            for i in 0..rows {
                result[[i, j]] = col_vec[i];
            }
        }
        
        Ok(result)
    }
}

/// Complex number operations
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub real: f32,
    pub imag: f32,
}

impl Complex {
    pub fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }
    
    pub fn from_polar(magnitude: f32, phase: f32) -> Self {
        Self {
            real: magnitude * phase.cos(),
            imag: magnitude * phase.sin(),
        }
    }
    
    pub fn magnitude(&self) -> f32 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
    
    pub fn phase(&self) -> f32 {
        self.imag.atan2(self.real)
    }
    
    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl std::ops::Mul<f32> for Complex {
    type Output = Self;
    
    fn mul(self, scalar: f32) -> Self {
        Self {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl num_traits::identities::Zero for Complex {
    fn zero() -> Self {
        Self { real: 0.0, imag: 0.0 }
    }
    
    fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imag == 0.0
    }
}

/// Spectral analysis utilities
pub struct SpectralAnalyzer;

impl SpectralAnalyzer {
    /// Compute power spectrum
    pub fn power_spectrum(complex_signal: &Array1<Complex>) -> Array1<f32> {
        complex_signal.mapv(|c| c.magnitude().powi(2))
    }
    
    /// Compute phase spectrum
    pub fn phase_spectrum(complex_signal: &Array1<Complex>) -> Array1<f32> {
        complex_signal.mapv(|c| c.phase())
    }
    
    /// Compute spectral entropy
    pub fn spectral_entropy(power_spectrum: &Array1<f32>) -> f32 {
        let total: f32 = power_spectrum.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &power in power_spectrum.iter() {
            if power > 0.0 {
                let p = power / total;
                entropy -= p * p.ln();
            }
        }
        
        entropy
    }
    
    /// Find dominant frequencies
    pub fn dominant_frequencies(power_spectrum: &Array1<f32>, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = power_spectrum
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

/// Resonance calculations
#[derive(Debug, Clone)]
pub struct ResonanceCalculator;

impl ResonanceCalculator {
    /// Calculate resonance between two signals
    pub fn resonance_coefficient(signal1: &Array1<f32>, signal2: &Array1<f32>) -> f32 {
        if signal1.len() != signal2.len() {
            return 0.0;
        }
        
        let correlation = Self::correlation_coefficient(signal1, signal2);
        let phase_alignment = Self::phase_alignment(signal1, signal2);
        
        correlation * phase_alignment
    }
    
    /// Calculate correlation coefficient
    fn correlation_coefficient(signal1: &Array1<f32>, signal2: &Array1<f32>) -> f32 {
        let mean1: f32 = signal1.iter().sum::<f32>() / signal1.len() as f32;
        let mean2: f32 = signal2.iter().sum::<f32>() / signal2.len() as f32;
        
        let mut numerator = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        
        for (&s1, &s2) in signal1.iter().zip(signal2.iter()) {
            let diff1 = s1 - mean1;
            let diff2 = s2 - mean2;
            numerator += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }
        
        if var1 == 0.0 || var2 == 0.0 {
            return 0.0;
        }
        
        numerator / (var1 * var2).sqrt()
    }
    
    /// Calculate phase alignment
    fn phase_alignment(signal1: &Array1<f32>, signal2: &Array1<f32>) -> f32 {
        let fft1 = HolographicFFT::fft_1d(signal1).expect("FFT computation succeeded");
        let fft2 = HolographicFFT::fft_1d(signal2).expect("FFT computation succeeded");
        
        let phase1 = SpectralAnalyzer::phase_spectrum(&fft1);
        let phase2 = SpectralAnalyzer::phase_spectrum(&fft2);
        
        let mut alignment = 0.0;
        for (&p1, &p2) in phase1.iter().zip(phase2.iter()) {
            alignment += (p1 - p2).cos();
        }
        
        alignment / phase1.len() as f32
    }
    
    /// Calculate interference pattern
    pub fn interference_pattern(wave1: &HolographicWave, wave2: &HolographicWave) -> DLResult<HolographicWave> {
        if wave1.amplitude.shape() != wave2.amplitude.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: wave1.amplitude.shape().to_vec(),
                actual: wave2.amplitude.shape().to_vec(),
            });
        }
        
        let mut result = HolographicWave::new(wave1.amplitude.shape().to_vec());
        
        for ((idx, &amp1), &amp2) in wave1.amplitude.indexed_iter().zip(wave2.amplitude.iter()) {
            let phase1 = wave1.phase[idx.clone()];
            let phase2 = wave2.phase[idx.clone()];
            let freq1 = wave1.frequency[idx.clone()];
            let freq2 = wave2.frequency[idx.clone()];
            
            // Constructive and destructive interference
            result.amplitude[idx.clone()] = (amp1 * amp1 + amp2 * amp2 + 2.0 * amp1 * amp2 * (phase1 - phase2).cos()).sqrt();
            result.phase[idx.clone()] = (amp1 * phase1 + amp2 * phase2) / (amp1 + amp2 + 1e-8);
            result.frequency[idx.clone()] = (freq1 + freq2) / 2.0;
        }
        
        Ok(result)
    }
}

/// Memory compression utilities
pub struct MemoryCompressor;

impl MemoryCompressor {
    /// Compress holographic memory using FFT pooling
    pub fn fft_pool_compress(input: &Array2<f32>, pool_size: usize) -> DLResult<Array2<f32>> {
        let (rows, cols) = input.dim();
        let compressed_rows = (rows + pool_size - 1) / pool_size;
        let compressed_cols = (cols + pool_size - 1) / pool_size;
        
        let mut compressed = Array2::zeros((compressed_rows, compressed_cols));
        
        for i in 0..compressed_rows {
            for j in 0..compressed_cols {
                let start_row = i * pool_size;
                let end_row = (start_row + pool_size).min(rows);
                let start_col = j * pool_size;
                let end_col = (start_col + pool_size).min(cols);
                
                let patch = input.slice(s![start_row..end_row, start_col..end_col]).to_owned();
                let fft_patch = HolographicFFT::fft_2d(&patch)?;
                
                // Keep only low frequency components
                let mut sum = 0.0;
                let count = (pool_size / 2).min(fft_patch.nrows()).min(fft_patch.ncols());
                
                for di in 0..count {
                    for dj in 0..count {
                        sum += fft_patch[[di, dj]].magnitude();
                    }
                }
                
                compressed[[i, j]] = sum / (count * count) as f32;
            }
        }
        
        Ok(compressed)
    }
    
    /// Generate summary statistics
    pub fn generate_summary(input: &Array2<f32>) -> Array1<f32> {
        let mut summary = vec![0.0; 8]; // mean, std, min, max, median, entropy, sparsity, energy
        
        let mut values: Vec<f32> = input.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Basic statistics
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values[0];
        let max = values[values.len() - 1];
        let median = values[values.len() / 2];
        
        // Entropy
        let mut entropy = 0.0;
        for &val in &values {
            if val > 0.0 {
                let p = val / values.iter().sum::<f32>();
                entropy -= p * p.ln();
            }
        }
        
        // Sparsity
        let sparsity = values.iter().filter(|&&x| x.abs() < 1e-6).count() as f32 / values.len() as f32;
        
        // Energy
        let energy = values.iter().map(|x| x * x).sum::<f32>();
        
        summary[0] = mean;
        summary[1] = std;
        summary[2] = min;
        summary[3] = max;
        summary[4] = median;
        summary[5] = entropy;
        summary[6] = sparsity;
        summary[7] = energy;
        
        Array1::from(summary)
    }
}

/// Re-export HolographicWave from mod.rs
use super::HolographicWave;
