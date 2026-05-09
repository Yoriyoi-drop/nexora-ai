//! DDPM (Denoising Diffusion Probabilistic Models) Implementation
//!
//! DDPM digunakan untuk noise schedule dan sampling:
//! - Forward process: Menambahkan noise secara bertahap
//! - Reverse process: Menghilangkan noise (denoising)
//! - Noise schedule: Linear, Cosine, Scaled Linear

pub mod schedule;
pub mod sampler;
pub mod noise;

use crate::hldva_t::{
    config::{DDPMConfig, NoiseScheduleType},
    types::*,
};
use crate::atqs::Tensor;

/// Main DDPM
pub struct DDPM {
    config: DDPMConfig,
    
    // Noise schedule
    noise_schedule: NoiseSchedule,
    
    // Sampler
    sampler: DDPM_Sampler,
}

impl DDPM {
    /// Create new DDPM
    pub fn new(config: &DDPMConfig) -> HLDVAResult<Self> {
        let noise_schedule = NoiseSchedule::new(config)?;
        let sampler = DDPM_Sampler::new(config)?;
        
        Ok(Self {
            config: config.clone(),
            noise_schedule,
            sampler,
        })
    }
    
    /// Get timesteps untuk inference
    pub fn get_timesteps(&self, num_steps: usize) -> Vec<Timestep> {
        self.sampler.get_timesteps(num_steps)
    }
    
    /// Add noise (forward process)
    pub fn add_noise(&self, original: &Tensor, noise: &Tensor, timestep: Timestep) -> HLDVAResult<Tensor> {
        let (alpha_bar, _) = self.noise_schedule.get_alpha_bar(timestep);
        
        let original_data = original.data();
        let noise_data = noise.data();
        
        let mut noisy = Vec::with_capacity(original_data.len());
        
        for i in 0..original_data.len() {
            let noise_val = if i < noise_data.len() { noise_data[i] } else { 0.0 };
            let noisy_val = (alpha_bar.sqrt() * original_data[i]) + ((1.0 - alpha_bar).sqrt() * noise_val);
            noisy.push(noisy_val);
        }
        
        Ok(Tensor::new(noisy, original.shape().to_vec()))
    }
    
    /// DDPM step (reverse process)
    pub fn step(&self, noisy_latent: Tensor, predicted_noise: Tensor, timestep: Timestep) -> HLDVAResult<Tensor> {
        self.sampler.step(noisy_latent, predicted_noise, timestep, &self.noise_schedule)
    }
    
    /// Sample from noise (full reverse process)
    pub fn sample(&self, model: &dyn NoisePredictor, shape: &[usize], num_steps: usize) -> HLDVAResult<Tensor> {
        let timesteps = self.get_timesteps(num_steps);
        
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
        
        // Reverse process
        for &timestep in timesteps.iter().rev() {
            let predicted_noise = model.predict_noise(&current, timestep)?;
            current = self.step(current, predicted_noise, timestep)?;
        }
        
        Ok(current)
    }
    
    /// Generate random normal number
    fn randn(&self) -> f32 {
        use std::f64::consts::PI;
        let u1: f64 = rand::random();
        let u2: f64 = rand::random();
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        z0 as f32
    }
    
    /// Get configuration
    pub fn config(&self) -> &DDPMConfig {
        &self.config
    }
}

/// Noise Schedule
pub struct NoiseSchedule {
    schedule_type: NoiseScheduleType,
    num_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    cosine_s: f32,
    
    // Precomputed values
    alphas: Vec<f32>,
    alpha_bars: Vec<f32>,
    sqrt_alpha_bars: Vec<f32>,
    sqrt_one_minus_alpha_bars: Vec<f32>,
}

impl NoiseSchedule {
    pub fn new(config: &DDPMConfig) -> HLDVAResult<Self> {
        let mut schedule = Self {
            schedule_type: config.schedule_type.clone(),
            num_timesteps: config.num_timesteps,
            beta_start: config.beta_start,
            beta_end: config.beta_end,
            cosine_s: config.cosine_s,
            alphas: Vec::new(),
            alpha_bars: Vec::new(),
            sqrt_alpha_bars: Vec::new(),
            sqrt_one_minus_alpha_bars: Vec::new(),
        };
        
        schedule.compute_schedule()?;
        Ok(schedule)
    }
    
    /// Compute noise schedule
    fn compute_schedule(&mut self) -> HLDVAResult<()> {
        self.alphas = Vec::with_capacity(self.num_timesteps);
        self.alpha_bars = Vec::with_capacity(self.num_timesteps);
        self.sqrt_alpha_bars = Vec::with_capacity(self.num_timesteps);
        self.sqrt_one_minus_alpha_bars = Vec::with_capacity(self.num_timesteps);
        
        let mut alpha_bar = 1.0;
        
        for t in 0..self.num_timesteps {
            let beta = self.compute_beta(t);
            let alpha = 1.0 - beta;
            alpha_bar *= alpha;
            
            self.alphas.push(alpha);
            self.alpha_bars.push(alpha_bar);
            self.sqrt_alpha_bars.push(alpha_bar.sqrt());
            self.sqrt_one_minus_alpha_bars.push((1.0 - alpha_bar).sqrt());
        }
        
        Ok(())
    }
    
    /// Compute beta untuk timestep tertentu
    fn compute_beta(&self, t: usize) -> f32 {
        match self.schedule_type {
            NoiseScheduleType::Linear => {
                let t_f = t as f32;
                let t_max = (self.num_timesteps - 1) as f32;
                self.beta_start + (self.beta_end - self.beta_start) * (t_f / t_max)
            }
            NoiseScheduleType::Cosine => {
                let t_f = t as f32;
                let t_max = (self.num_timesteps - 1) as f32;
                let s = self.cosine_s;
                
                let cosine_factor = ((t_f / t_max + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos();
                let cosine_factor_prev = (((t_f + 1.0) / t_max + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos();
                
                1.0 - (cosine_factor / cosine_factor_prev)
            }
            NoiseScheduleType::ScaledLinear => {
                let t_f = t as f32;
                let t_max = (self.num_timesteps - 1) as f32;
                let scale = self.beta_end / self.beta_start;
                self.beta_start * (scale.powf(t_f / t_max))
            }
        }
    }
    
    /// Get alpha_bar untuk timestep
    pub fn get_alpha_bar(&self, timestep: Timestep) -> (f32, f32) {
        let t = timestep.0.min(self.num_timesteps - 1);
        let alpha_bar = self.alpha_bars[t];
        let sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t];
        (alpha_bar, sqrt_one_minus_alpha_bar)
    }
    
    /// Get beta untuk timestep
    pub fn get_beta(&self, timestep: Timestep) -> f32 {
        let t = timestep.0.min(self.num_timesteps - 1);
        self.alphas[t]
    }
    
    /// Posterior variance
    pub fn posterior_variance(&self, timestep: Timestep) -> f32 {
        let t = timestep.0;
        if t == 0 {
            0.0
        } else {
            let alpha_bar_t = self.alpha_bars[t];
            let alpha_bar_t_prev = self.alpha_bars[t - 1];
            let beta_t = 1.0 - self.alphas[t];
            
            (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t) * beta_t
        }
    }
    
    /// Posterior mean coefficient
    pub fn posterior_mean_coef1(&self, timestep: Timestep) -> f32 {
        let t = timestep.0;
        if t == 0 {
            1.0
        } else {
            let alpha_bar_t = self.alpha_bars[t];
            let alpha_bar_t_prev = self.alpha_bars[t - 1];
            let beta_t = 1.0 - self.alphas[t];
            
            (alpha_bar_t_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t)
        }
    }
    
    /// Posterior mean coefficient 2
    pub fn posterior_mean_coef2(&self, timestep: Timestep) -> f32 {
        let t = timestep.0;
        if t == 0 {
            0.0
        } else {
            let alpha_bar_t = self.alpha_bars[t];
            let alpha_bar_t_prev = self.alpha_bars[t - 1];
            let beta_t = 1.0 - self.alphas[t];
            
            ((1.0 - alpha_bar_t_prev).sqrt() * (1.0 - alpha_bar_t).sqrt()) / (1.0 - alpha_bar_t)
        }
    }
}

/// DDPM Sampler
pub struct DDPM_Sampler {
    num_timesteps: usize,
}

impl DDPM_Sampler {
    pub fn new(config: &DDPMConfig) -> HLDVAResult<Self> {
        Ok(Self {
            num_timesteps: config.num_timesteps,
        })
    }
    
    /// Get timesteps untuk inference
    pub fn get_timesteps(&self, num_steps: usize) -> Vec<Timestep> {
        let step_size = self.num_timesteps / num_steps;
        (0..num_steps)
            .map(|i| Timestep(i * step_size))
            .collect()
    }
    
    /// DDPM step
    pub fn step(
        &self,
        noisy_latent: Tensor,
        predicted_noise: Tensor,
        timestep: Timestep,
        noise_schedule: &NoiseSchedule,
    ) -> HLDVAResult<Tensor> {
        let t = timestep.0;
        
        if t == 0 {
            // Final step: return denoised latent
            return Ok(noisy_latent);
        }
        
        let noisy_data = noisy_latent.data();
        let noise_data = predicted_noise.data();
        
        // Get schedule values
        let beta_t = noise_schedule.get_beta(timestep);
        let sqrt_one_minus_alpha_bar_t = noise_schedule.get_alpha_bar(timestep).1;
        let posterior_variance = noise_schedule.posterior_variance(timestep);
        let posterior_mean_coef1 = noise_schedule.posterior_mean_coef1(timestep);
        let posterior_mean_coef2 = noise_schedule.posterior_mean_coef2(timestep);
        
        let mut denoised = Vec::with_capacity(noisy_data.len());
        
        for i in 0..noisy_data.len() {
            let noise_val = if i < noise_data.len() { noise_data[i] } else { 0.0 };
            
            // Compute posterior mean
            let mean = posterior_mean_coef1 * noisy_data[i] + posterior_mean_coef2 * noise_val;
            
            // Add noise for stochastic sampling
            let noise = self.randn();
            let sample = mean + noise * posterior_variance.sqrt();
            
            denoised.push(sample);
        }
        
        Ok(Tensor::new(denoised, noisy_latent.shape().to_vec()))
    }
    
    /// Generate random normal number
    fn randn(&self) -> f32 {
        use std::f64::consts::PI;
        let u1: f64 = rand::random();
        let u2: f64 = rand::random();
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        z0 as f32
    }
}

/// Trait untuk noise predictor
pub trait NoisePredictor {
    fn predict_noise(&self, noisy_latent: &Tensor, timestep: Timestep) -> HLDVAResult<Tensor>;
}

/// DDPM Loss Calculator
pub struct DDPMLoss {
    config: DDPMConfig,
    noise_schedule: NoiseSchedule,
}

impl DDPMLoss {
    pub fn new(config: &DDPMConfig) -> HLDVAResult<Self> {
        let noise_schedule = NoiseSchedule::new(config)?;
        
        Ok(Self {
            config: config.clone(),
            noise_schedule,
        })
    }
    
    /// Add noise (forward process)
    pub fn add_noise(&self, original: &Tensor, noise: &Tensor, timestep: Timestep) -> HLDVAResult<Tensor> {
        let (alpha_bar, _) = self.noise_schedule.get_alpha_bar(timestep);
        
        let original_data = original.data();
        let noise_data = noise.data();
        
        let mut noisy_data = Vec::with_capacity(original_data.len());
        for i in 0..original_data.len().min(noise_data.len()) {
            noisy_data.push(
                original_data[i] * alpha_bar.sqrt() + 
                noise_data[i] * (1.0 - alpha_bar).sqrt()
            );
        }
        
        Ok(Tensor::new(noisy_data, original.shape().to_vec()))
    }
    
    /// Sample random noise
    pub fn sample_noise(&self, shape: &[usize]) -> HLDVAResult<Tensor> {
        let size = shape.iter().product();
        let noise_data: Vec<f32> = (0..size).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        Ok(Tensor::new(noise_data, shape.to_vec()))
    }
    
    /// Calculate DDPM loss
    pub fn calculate_loss(
        &self,
        model: &dyn NoisePredictor,
        clean_latent: &Tensor,
        timestep: Timestep,
    ) -> HLDVAResult<f32> {
        // Sample random noise
        let noise = self.sample_noise(clean_latent.shape())?;
        
        // Add noise to clean latent
        let noisy_latent = self.add_noise(clean_latent, &noise, timestep)?;
        
        // Predict noise
        let predicted_noise = model.predict_noise(&noisy_latent, timestep)?;
        
        // Calculate MSE loss
        let noise_data = noise.data();
        let predicted_data = predicted_noise.data();
        
        let mut mse = 0.0;
        let count = noise_data.len().min(predicted_data.len());
        
        for i in 0..count {
            let diff = noise_data[i] - predicted_data[i];
            mse += diff * diff;
        }
        
        Ok(mse / count as f32)
    }
    
        
    /// Generate random normal number
    fn randn(&self) -> f32 {
        use std::f64::consts::PI;
        let u1: f64 = rand::random();
        let u2: f64 = rand::random();
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        z0 as f32
    }
}

/// Classifier-Free Guidance
pub struct ClassifierFreeGuidance {
    guidance_scale: f32,
}

impl ClassifierFreeGuidance {
    pub fn new(guidance_scale: f32) -> Self {
        Self { guidance_scale }
    }
    
    /// Apply classifier-free guidance
    pub fn apply(
        &self,
        cond_noise: &Tensor,
        uncond_noise: &Tensor,
    ) -> HLDVAResult<Tensor> {
        let cond_data = cond_noise.data();
        let uncond_data = uncond_noise.data();
        
        let mut guided = Vec::with_capacity(cond_data.len());
        
        for i in 0..cond_data.len() {
            let uncond_val = if i < uncond_data.len() { uncond_data[i] } else { 0.0 };
            let guided_val = uncond_val + self.guidance_scale * (cond_data[i] - uncond_val);
            guided.push(guided_val);
        }
        
        Ok(Tensor::new(guided, cond_noise.shape().to_vec()))
    }
    
    /// Get guidance scale
    pub fn guidance_scale(&self) -> f32 {
        self.guidance_scale
    }
    
    /// Set guidance scale
    pub fn set_guidance_scale(&mut self, scale: f32) {
        self.guidance_scale = scale;
    }
}

/// Noise Level Conditioning
pub struct NoiseLevelConditioning {
    num_levels: usize,
    min_level: f32,
    max_level: f32,
}

impl NoiseLevelConditioning {
    pub fn new(num_levels: usize, min_level: f32, max_level: f32) -> Self {
        Self {
            num_levels,
            min_level,
            max_level,
        }
    }
    
    /// Get noise level untuk timestep
    pub fn get_noise_level(&self, timestep: Timestep, max_timesteps: usize) -> f32 {
        let t_f = timestep.0 as f32;
        let t_max = max_timesteps as f32;
        
        // Linear interpolation dari min ke max
        self.min_level + (self.max_level - self.min_level) * (t_f / t_max)
    }
    
    /// Quantize noise level
    pub fn quantize_level(&self, level: f32) -> usize {
        let step = (self.max_level - self.min_level) / self.num_levels as f32;
        let level_idx = ((level - self.min_level) / step) as usize;
        level_idx.min(self.num_levels - 1)
    }
    
    /// Get embedding untuk noise level
    pub fn get_level_embedding(&self, level: f32, embedding_dim: usize) -> Vec<f32> {
        let mut embedding = Vec::with_capacity(embedding_dim);
        
        // Sinusoidal encoding untuk noise level
        for i in 0..embedding_dim / 2 {
            let freq = 1.0 / (10000.0_f32.powf((2 * i) as f32 / embedding_dim as f32));
            embedding.push((level * freq).sin());
            embedding.push((level * freq).cos());
        }
        
        // Pad jika perlu
        while embedding.len() < embedding_dim {
            embedding.push(0.0);
        }
        
        embedding
    }
}
