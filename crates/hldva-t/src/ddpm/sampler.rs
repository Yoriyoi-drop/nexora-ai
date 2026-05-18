//! Sampler module for DDPM

use crate::types::*;
use nexora_atqs::Tensor;
use rand::prelude::*;

/// DDPM Sampler
pub struct DDPMSampler {
    _config: SamplerConfig,
    rng: ThreadRng,
}

impl DDPMSampler {
    /// Create new DDPM sampler
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            _config: config,
            rng: thread_rng(),
        }
    }
    
    /// Sample from DDPM model
    pub fn sample(&mut self, model: &dyn NoisePredictor, shape: &[usize], num_steps: usize) -> HLDVAResult<Tensor> {
        // Start with pure noise
        let mut current = Tensor::new(
            vec![0.0; shape.iter().product()],
            shape.to_vec(),
        );
        
        // Add random noise
        let mut current_data = current.data_mut().to_vec();
        for i in 0..current_data.len() {
            current_data[i] = self.randn();
        }
        current = Tensor::new(current_data, shape.to_vec());
        
        // Reverse diffusion process
        for step in (0..num_steps).rev() {
            let timestep = step;
            let predicted_noise = model.predict_noise(&current, timestep)?;
            current = self.denoise_step(&current, &predicted_noise, timestep)?;
        }
        
        Ok(current)
    }
    
    /// Single denoising step
    fn denoise_step(&self, noisy: &Tensor, predicted_noise: &Tensor, timestep: usize) -> HLDVAResult<Tensor> {
        let noisy_data = noisy.data();
        let predicted_noise_data = predicted_noise.data();
        
        if noisy_data.len() != predicted_noise_data.len() {
            return Err(HLDVAError::Model("Noisy and predicted noise must have same dimensions".to_string()));
        }
        
        let mut denoised_data = Vec::with_capacity(noisy_data.len());
        for (noisy_val, pred_noise_val) in noisy_data.iter().zip(predicted_noise_data.iter()) {
            let denoised = noisy_val - pred_noise_val * 0.1; // Simplified denoising
            denoised_data.push(denoised);
        }
        
        Ok(Tensor::new(denoised_data, noisy.shape().to_vec()))
    }
    
    /// Generate standard normal random number
    fn randn(&mut self) -> f32 {
        // Box-Muller transform
        let u1: f32 = self.rng.gen();
        let u2: f32 = self.rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Noise Predictor trait
pub trait NoisePredictor {
    fn predict_noise(&self, noisy: &Tensor, timestep: usize) -> HLDVAResult<Tensor>;
}

/// Sampler configuration
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub num_inference_steps: usize,
    pub eta: f32, // DDIM eta parameter
    pub guidance_scale: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            eta: 0.0,
            guidance_scale: 1.0,
        }
    }
}
