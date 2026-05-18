//! Noise generation and manipulation for DDPM

use crate::types::*;
use nexora_atqs::Tensor;
use rand::prelude::*;

/// Noise generator for DDPM
pub struct NoiseGenerator {
    rng: ThreadRng,
}

impl NoiseGenerator {
    /// Create new noise generator
    pub fn new() -> Self {
        Self {
            rng: thread_rng(),
        }
    }
    
    /// Generate Gaussian noise
    pub fn gaussian_noise(&mut self, shape: &[usize]) -> Tensor {
        let total_elements: usize = shape.iter().product();
        let mut noise_data = Vec::with_capacity(total_elements);
        
        for _ in 0..total_elements {
            noise_data.push(self.randn());
        }
        
        Tensor::new(noise_data, shape.to_vec())
    }
    
    /// Generate uniform noise
    pub fn uniform_noise(&mut self, shape: &[usize], min: f32, max: f32) -> Tensor {
        let total_elements: usize = shape.iter().product();
        let mut noise_data = Vec::with_capacity(total_elements);
        
        for _ in 0..total_elements {
            noise_data.push(self.rng.gen_range(min..max));
        }
        
        Tensor::new(noise_data, shape.to_vec())
    }
    
    /// Generate structured noise (for testing)
    pub fn structured_noise(&mut self, shape: &[usize], pattern: NoisePattern) -> Tensor {
        let total_elements: usize = shape.iter().product();
        let mut noise_data = Vec::with_capacity(total_elements);
        
        match pattern {
            NoisePattern::Checkerboard => {
                let (h, w) = if shape.len() >= 3 { (shape[1], shape[2]) } else { (shape[0], 1) };
                for y in 0..h {
                    for x in 0..w {
                        let value = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
                        noise_data.push(value);
                    }
                }
            }
            NoisePattern::Stripes => {
                let (h, w) = if shape.len() >= 3 { (shape[1], shape[2]) } else { (shape[0], 1) };
                for y in 0..h {
                    for x in 0..w {
                        let value = if x % 4 < 2 { 1.0 } else { -1.0 };
                        noise_data.push(value);
                    }
                }
            }
            NoisePattern::Radial => {
                let (h, w) = if shape.len() >= 3 { (shape[1], shape[2]) } else { (shape[0], 1) };
                let center_y = h as f32 / 2.0;
                let center_x = w as f32 / 2.0;
                
                for y in 0..h {
                    for x in 0..w {
                        let dy = y as f32 - center_y;
                        let dx = x as f32 - center_x;
                        let distance = (dy * dy + dx * dx).sqrt();
                        let value = (distance * 0.1).sin();
                        noise_data.push(value);
                    }
                }
            }
        }
        
        Tensor::new(noise_data, shape.to_vec())
    }
    
    /// Scale noise by factor
    pub fn scale_noise(&self, noise: &Tensor, scale: f32) -> Tensor {
        let noise_data = noise.data();
        let mut scaled_data = Vec::with_capacity(noise_data.len());
        
        for &val in noise_data {
            scaled_data.push(val * scale);
        }
        
        Tensor::new(scaled_data, noise.shape().to_vec())
    }
    
    /// Add noise to tensor
    pub fn add_noise(&self, tensor: &Tensor, noise: &Tensor) -> HLDVAResult<Tensor> {
        let tensor_data = tensor.data();
        let noise_data = noise.data();
        
        if tensor_data.len() != noise_data.len() {
            return Err(HLDVAError::Model("Tensor and noise must have same dimensions".to_string()));
        }
        
        let mut result_data = Vec::with_capacity(tensor_data.len());
        for (tensor_val, noise_val) in tensor_data.iter().zip(noise_data.iter()) {
            result_data.push(tensor_val + noise_val);
        }
        
        Ok(Tensor::new(result_data, tensor.shape().to_vec()))
    }
    
    /// Generate standard normal random number
    fn randn(&mut self) -> f32 {
        // Box-Muller transform
        let u1: f32 = self.rng.gen();
        let u2: f32 = self.rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Noise patterns for structured noise
#[derive(Debug, Clone)]
pub enum NoisePattern {
    Checkerboard,
    Stripes,
    Radial,
}

/// Noise utilities
pub struct NoiseUtils;

impl NoiseUtils {
    /// Compute signal-to-noise ratio
    pub fn compute_snr(signal: &Tensor, noise: &Tensor) -> HLDVAResult<f32> {
        let signal_data = signal.data();
        let noise_data = noise.data();
        
        if signal_data.len() != noise_data.len() {
            return Err(HLDVAError::Model("Signal and noise must have same dimensions".to_string()));
        }
        
        let mut signal_power = 0.0;
        let mut noise_power = 0.0;
        
        for (signal_val, noise_val) in signal_data.iter().zip(noise_data.iter()) {
            signal_power += signal_val * signal_val;
            noise_power += noise_val * noise_val;
        }
        
        if noise_power > 0.0 {
            Ok(signal_power / noise_power)
        } else {
            Ok(f32::INFINITY)
        }
    }
    
    /// Apply noise schedule
    pub fn apply_noise_schedule(noise: &Tensor, alpha_bar: f32) -> Tensor {
        let noise_data = noise.data();
        let mut scaled_noise = Vec::with_capacity(noise_data.len());
        
        for &val in noise_data {
            scaled_noise.push(val * (1.0 - alpha_bar).sqrt());
        }
        
        Tensor::new(scaled_noise, noise.shape().to_vec())
    }
    
    /// Generate colored noise (1/f noise)
    pub fn colored_noise(shape: &[usize], exponent: f32) -> Tensor {
        let total_elements: usize = shape.iter().product();
        let mut noise_data = Vec::with_capacity(total_elements);
        
        for i in 0..total_elements {
            let freq = (i as f32 + 1.0) / (total_elements as f32);
            let amplitude = 1.0 / (freq.powf(exponent));
            let value = amplitude * (2.0 * std::f32::consts::PI * freq).sin();
            noise_data.push(value);
        }
        
        Tensor::new(noise_data, shape.to_vec())
    }
}

impl Default for NoiseGenerator {
    fn default() -> Self {
        Self::new()
    }
}
