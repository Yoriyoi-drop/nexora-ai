//! Latent Space module for HLDVA-T VAED

use crate::hldva_t::types::*;
use crate::atqs::Tensor;
use rand::{Rng, thread_rng};

/// Latent Space representation
#[derive(Debug, Clone)]
pub struct LatentSpace {
    pub data: Tensor,
    pub shape: Vec<usize>,
}

impl LatentSpace {
    /// Create new latent space
    pub fn new(data: Tensor, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
    
    /// Get latent dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        if self.shape.len() >= 4 {
            (self.shape[1], self.shape[2], self.shape[3])
        } else {
            (1, 1, self.data.data().len())
        }
    }
    
    /// Sample from latent distribution (reparameterization trick)
    pub fn sample(&self, rng: &mut rand::prelude::ThreadRng) -> HLDVAResult<Tensor> {
        let latent_data = self.data.data();
        let mut sampled = Vec::with_capacity(latent_data.len() / 2);
        
        for i in (0..latent_data.len()).step_by(2) {
            if i + 1 < latent_data.len() {
                let mu = latent_data[i];
                let log_var = latent_data[i + 1];
                let std = (log_var / 2.0).exp(); // log_var is actually log(σ²)
                let epsilon = rng.gen::<f32>(); // Standard normal
                let z = mu + std * epsilon;
                sampled.push(z);
            }
        }
        
        let len = sampled.len();
        Ok(Tensor::new(sampled, vec![len]))
    }
    
    /// Get mean and log variance from latent
    pub fn get_mu_logvar(&self) -> HLDVAResult<(Tensor, Tensor)> {
        let latent_data = self.data.data();
        let half_len = latent_data.len() / 2;
        
        let mu_data = latent_data[..half_len].to_vec();
        let log_var_data = latent_data[half_len..].to_vec();
        
        let mu = Tensor::new(mu_data, vec![half_len]);
        let log_var = Tensor::new(log_var_data, vec![half_len]);
        
        Ok((mu, log_var))
    }
    
    /// Compute KL divergence
    pub fn kl_divergence(&self) -> HLDVAResult<f32> {
        let (mu, log_var) = self.get_mu_logvar()?;
        let mu_data = mu.data();
        let log_var_data = log_var.data();
        
        let mut kl_sum = 0.0;
        for i in 0..mu_data.len() {
            let mu_val = mu_data[i];
            let log_var_val = log_var_data[i];
            let var_val = log_var_val.exp();
            
            // KL divergence: 0.5 * (μ² + σ - log(σ) - 1)
            kl_sum += 0.5 * (mu_val * mu_val + var_val - log_var_val - 1.0);
        }
        
        Ok(kl_sum / mu_data.len() as f32)
    }
    
    /// Apply latent space transformation
    pub fn transform(&self, transformation: &LatentTransform) -> HLDVAResult<LatentSpace> {
        let data = self.data.data();
        let mut transformed_data = Vec::with_capacity(data.len());
        
        match transformation {
            LatentTransform::Scale(scale) => {
                for &val in data {
                    transformed_data.push(val * scale);
                }
            }
            LatentTransform::Shift(shift) => {
                for &val in data {
                    transformed_data.push(val + shift);
                }
            }
            LatentTransform::Rotate(angle) => {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let (h, w, _) = self.dimensions();
                
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..3 {
                            let src_idx = (y * w + x) * 3 + c;
                            if src_idx < data.len() {
                                let new_x = (x as f32 * cos_a - y as f32 * sin_a) as usize;
                                let new_y = (x as f32 * sin_a + y as f32 * cos_a) as usize;
                                
                                if new_x < w && new_y < h {
                                    let dst_idx = (new_y * w + new_x) * 3 + c;
                                    if dst_idx < transformed_data.len() {
                                        transformed_data[dst_idx] = data[src_idx];
                                    } else {
                                        transformed_data.push(data[src_idx]);
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Fill remaining slots if needed
                while transformed_data.len() < data.len() {
                    transformed_data.push(0.0);
                }
            }
        }
        
        Ok(LatentSpace::new(
            Tensor::new(transformed_data, self.data.shape().to_vec()),
            self.shape.to_vec(),
        ))
    }
}

/// Latent space transformations
#[derive(Debug, Clone)]
pub enum LatentTransform {
    Scale(f32),
    Shift(f32),
    Rotate(f32),
}

/// Latent space operations
pub struct LatentOps;

impl LatentOps {
    /// Interpolate between two latent spaces
    pub fn interpolate(latent1: &LatentSpace, latent2: &LatentSpace, alpha: f32) -> HLDVAResult<LatentSpace> {
        let data1 = latent1.data.data();
        let data2 = latent2.data.data();
        
        if data1.len() != data2.len() {
            return Err(HLDVAError::Model("Latent dimensions must match for interpolation".to_string()));
        }
        
        let mut interpolated_data = Vec::with_capacity(data1.len());
        for (val1, val2) in data1.iter().zip(data2.iter()) {
            interpolated_data.push(val1 * (1.0 - alpha) + val2 * alpha);
        }
        
        Ok(LatentSpace::new(
            Tensor::new(interpolated_data, latent1.data.shape().to_vec()),
            latent1.shape.to_vec(),
        ))
    }
    
    /// Add noise to latent space
    pub fn add_noise(latent: &LatentSpace, noise_level: f32, rng: &mut rand::prelude::ThreadRng) -> HLDVAResult<LatentSpace> {
        let data = latent.data.data();
        let mut noisy_data = Vec::with_capacity(data.len());
        
        for &val in data {
            let noise: f32 = rng.gen_range(-1.0..1.0) * noise_level;
            noisy_data.push(val + noise);
        }
        
        Ok(LatentSpace::new(
            Tensor::new(noisy_data, latent.shape.clone()),
            latent.shape.clone(),
        ))
    }
    
    /// Compute latent space distance
    pub fn distance(latent1: &LatentSpace, latent2: &LatentSpace) -> HLDVAResult<f32> {
        let data1 = latent1.data.data();
        let data2 = latent2.data.data();
        
        if data1.len() != data2.len() {
            return Err(HLDVAError::Model("Latent dimensions must match for distance calculation".to_string()));
        }
        
        let mut distance_sq = 0.0;
        for (val1, val2) in data1.iter().zip(data2.iter()) {
            let diff = val1 - val2;
            distance_sq += diff * diff;
        }
        
        Ok(distance_sq.sqrt())
    }
}
