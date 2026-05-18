//! Noise Schedule module for DDPM

use crate::types::*;
use nexora_atqs::Tensor;

/// Noise Schedule for diffusion process
pub struct NoiseSchedule {
    config: NoiseScheduleConfig,
    alphas: Vec<f32>,
    betas: Vec<f32>,
    alpha_bars: Vec<f32>,
    sqrt_alpha_bars: Vec<f32>,
    sqrt_one_minus_alpha_bars: Vec<f32>,
}

impl NoiseSchedule {
    /// Create new noise schedule
    pub fn new(config: NoiseScheduleConfig) -> HLDVAResult<Self> {
        let num_steps = config.num_steps;
        let mut betas = Vec::with_capacity(num_steps);
        let mut alphas = Vec::with_capacity(num_steps);
        let mut alpha_bars = Vec::with_capacity(num_steps);
        let mut sqrt_alpha_bars = Vec::with_capacity(num_steps);
        let mut sqrt_one_minus_alpha_bars = Vec::with_capacity(num_steps);
        
        // Generate beta values based on schedule type
        for t in 0..num_steps {
            let beta = match config.schedule_type {
                ScheduleType::Linear => {
                    config.beta_start + (config.beta_end - config.beta_start) * (t as f32) / (num_steps as f32 - 1.0)
                }
                ScheduleType::Cosine => {
                    let s = 0.008;
                    let f = ((t as f32) / (num_steps as f32) + s) / (1.0 + s) * std::f32::consts::PI / 2.0;
                    (f.cos()).powf(2.0)
                }
                ScheduleType::Quadratic => {
                    let progress = (t as f32) / (num_steps as f32);
                    config.beta_start + (config.beta_end - config.beta_start) * progress * progress
                }
            };
            
            betas.push(beta);
            alphas.push(1.0 - beta);
            
            if t == 0 {
                alpha_bars.push(alphas[0]);
                sqrt_alpha_bars.push(alphas[0].sqrt());
                sqrt_one_minus_alpha_bars.push((1.0 - alphas[0]).sqrt());
            } else {
                let alpha_bar = alpha_bars[t - 1] * alphas[t];
                alpha_bars.push(alpha_bar);
                sqrt_alpha_bars.push(alpha_bar.sqrt());
                sqrt_one_minus_alpha_bars.push((1.0 - alpha_bar).sqrt());
            }
        }
        
        Ok(Self {
            config,
            alphas,
            betas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
        })
    }
    
    /// Get beta at timestep
    pub fn get_beta(&self, timestep: usize) -> f32 {
        self.betas[timestep.min(self.betas.len() - 1)]
    }
    
    /// Get alpha at timestep
    pub fn get_alpha(&self, timestep: usize) -> f32 {
        self.alphas[timestep.min(self.alphas.len() - 1)]
    }
    
    /// Get alpha_bar at timestep
    pub fn get_alpha_bar(&self, timestep: usize) -> f32 {
        self.alpha_bars[timestep.min(self.alpha_bars.len() - 1)]
    }
    
    /// Get sqrt_alpha_bar at timestep
    pub fn get_sqrt_alpha_bar(&self, timestep: usize) -> f32 {
        self.sqrt_alpha_bars[timestep.min(self.sqrt_alpha_bars.len() - 1)]
    }
    
    /// Get sqrt_one_minus_alpha_bar at timestep
    pub fn get_sqrt_one_minus_alpha_bar(&self, timestep: usize) -> f32 {
        self.sqrt_one_minus_alpha_bars[timestep.min(self.sqrt_one_minus_alpha_bars.len() - 1)]
    }
    
    /// Add noise to image at timestep
    pub fn add_noise(&self, original: &Tensor, noise: &Tensor, timestep: usize) -> HLDVAResult<Tensor> {
        let original_data = original.data();
        let noise_data = noise.data();
        let sqrt_alpha_bar = self.get_sqrt_alpha_bar(timestep);
        let sqrt_one_minus_alpha_bar = self.get_sqrt_one_minus_alpha_bar(timestep);
        
        if original_data.len() != noise_data.len() {
            return Err(HLDVAError::Model("Original and noise must have same dimensions".to_string()));
        }
        
        let mut noisy_data = Vec::with_capacity(original_data.len());
        for (orig, noise_val) in original_data.iter().zip(noise_data.iter()) {
            noisy_data.push(orig * sqrt_alpha_bar + noise_val * sqrt_one_minus_alpha_bar);
        }
        
        Ok(Tensor::new(noisy_data, original.shape().to_vec()))
    }
    
    /// Remove noise from image at timestep (reverse process)
    pub fn remove_noise(&self, noisy: &Tensor, predicted_noise: &Tensor, timestep: usize) -> HLDVAResult<Tensor> {
        let noisy_data = noisy.data();
        let predicted_noise_data = predicted_noise.data();
        let alpha = self.get_alpha(timestep);
        let beta = self.get_beta(timestep);
        let alpha_bar = self.get_alpha_bar(timestep);
        let sqrt_one_minus_alpha_bar = self.get_sqrt_one_minus_alpha_bar(timestep);
        
        if noisy_data.len() != predicted_noise_data.len() {
            return Err(HLDVAError::Model("Noisy and predicted noise must have same dimensions".to_string()));
        }
        
        let mut denoised_data = Vec::with_capacity(noisy_data.len());
        for (noisy_val, pred_noise_val) in noisy_data.iter().zip(predicted_noise_data.iter()) {
            let denoised = (noisy_val - pred_noise_val * beta / sqrt_one_minus_alpha_bar) / alpha.sqrt();
            denoised_data.push(denoised);
        }
        
        Ok(Tensor::new(denoised_data, noisy.shape().to_vec()))
    }
    
    /// Get number of timesteps
    pub fn num_steps(&self) -> usize {
        self.config.num_steps
    }
    
    /// Get timesteps for sampling
    pub fn get_timesteps(&self, num_inference_steps: usize) -> Vec<usize> {
        let step_size = self.config.num_steps / num_inference_steps;
        (0..num_inference_steps).map(|i| i * step_size).collect()
    }
}

/// Noise Schedule configuration
#[derive(Debug, Clone)]
pub struct NoiseScheduleConfig {
    pub num_steps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
    pub schedule_type: ScheduleType,
}

#[derive(Debug, Clone)]
pub enum ScheduleType {
    Linear,
    Cosine,
    Quadratic,
}

impl Default for NoiseScheduleConfig {
    fn default() -> Self {
        Self {
            num_steps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            schedule_type: ScheduleType::Linear,
        }
    }
}
