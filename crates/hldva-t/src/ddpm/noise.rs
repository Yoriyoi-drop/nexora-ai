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
        Self { rng: thread_rng() }
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
                let (h, w) = if shape.len() >= 3 {
                    (shape[1], shape[2])
                } else {
                    (shape[0], 1)
                };
                for y in 0..h {
                    for x in 0..w {
                        let value = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
                        noise_data.push(value);
                    }
                }
            }
            NoisePattern::Stripes => {
                let (h, w) = if shape.len() >= 3 {
                    (shape[1], shape[2])
                } else {
                    (shape[0], 1)
                };
                for _y in 0..h {
                    for x in 0..w {
                        let value = if x % 4 < 2 { 1.0 } else { -1.0 };
                        noise_data.push(value);
                    }
                }
            }
            NoisePattern::Radial => {
                let (h, w) = if shape.len() >= 3 {
                    (shape[1], shape[2])
                } else {
                    (shape[0], 1)
                };
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
            return Err(HLDVAError::Model(
                "Tensor and noise must have same dimensions".to_string(),
            ));
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
            return Err(HLDVAError::Model(
                "Signal and noise must have same dimensions".to_string(),
            ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_generator_new() {
        let mut gen = NoiseGenerator::new();
        let noise = gen.gaussian_noise(&[4, 4]);
        assert_eq!(noise.shape(), &[4, 4]);
    }

    #[test]
    fn test_noise_generator_uniform() {
        let mut gen = NoiseGenerator::new();
        let noise = gen.uniform_noise(&[3], -1.0, 1.0);
        assert_eq!(noise.shape(), &[3]);
        for &v in noise.data() {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_noise_generator_structured() {
        let mut gen = NoiseGenerator::new();
        let c = gen.structured_noise(&[4, 4], NoisePattern::Checkerboard);
        assert_eq!(c.shape(), &[4, 4]);
        let s = gen.structured_noise(&[4, 4], NoisePattern::Stripes);
        assert_eq!(s.shape(), &[4, 4]);
        let r = gen.structured_noise(&[4, 4], NoisePattern::Radial);
        assert_eq!(r.shape(), &[4, 4]);
    }

    #[test]
    fn test_noise_generator_scale() {
        let mut gen = NoiseGenerator::new();
        let noise = gen.gaussian_noise(&[4]);
        let scaled = gen.scale_noise(&noise, 2.0);
        assert_eq!(scaled.shape(), noise.shape());
    }

    #[test]
    fn test_noise_generator_add_noise() {
        let gen = NoiseGenerator::new();
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);
        let result = gen.add_noise(&a, &b).unwrap();
        assert_eq!(result.data(), &vec![4.0, 6.0]);
    }

    #[test]
    fn test_noise_generator_add_noise_mismatch() {
        let gen = NoiseGenerator::new();
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0], vec![1]);
        assert!(gen.add_noise(&a, &b).is_err());
    }

    #[test]
    fn test_noise_utils_compute_snr() {
        let signal = Tensor::new(vec![1.0, 2.0], vec![2]);
        let noise = Tensor::new(vec![0.1, 0.2], vec![2]);
        let snr = NoiseUtils::compute_snr(&signal, &noise).unwrap();
        assert!(snr > 0.0);
    }

    #[test]
    fn test_noise_utils_apply_schedule() {
        let noise = Tensor::new(vec![1.0, 2.0], vec![2]);
        let result = NoiseUtils::apply_noise_schedule(&noise, 0.5);
        assert_eq!(result.shape(), &[2]);
    }

    #[test]
    fn test_noise_utils_colored_noise() {
        let result = NoiseUtils::colored_noise(&[4], 1.0);
        assert_eq!(result.shape(), &[4]);
    }

    #[test]
    fn test_noise_pattern_variants() {
        assert!(matches!(
            NoisePattern::Checkerboard,
            NoisePattern::Checkerboard
        ));
        assert!(matches!(NoisePattern::Stripes, NoisePattern::Stripes));
        assert!(matches!(NoisePattern::Radial, NoisePattern::Radial));
    }

    #[test]
    fn test_noise_generator_default() {
        let gen = NoiseGenerator::default();
        let mut g = gen;
        let noise = g.gaussian_noise(&[2, 2]);
        assert_eq!(noise.shape(), &[2, 2]);
    }
}
