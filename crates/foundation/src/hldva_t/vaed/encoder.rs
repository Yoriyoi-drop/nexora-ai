//! VAED Encoder module for HLDVA-T

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Variational Autoencoder Encoder
pub struct VAEDncoder {
    config: VAEDncoderConfig,
}

impl VAEDncoder {
    /// Create new VAED encoder
    pub fn new(config: VAEDncoderConfig) -> HLDVAResult<Self> {
        Ok(Self { config })
    }
    
    /// Encode image to latent space
    pub fn encode(&self, image: &Tensor) -> HLDVAResult<LatentSpace> {
        let image_data = image.data();
        let latent_data = self.extract_latent_features(image_data)?;
        
        let shape = self.compute_latent_shape(image.shape())?;
        let resolution = Resolution::new(shape[2], shape[1]); // width, height
        let channels = shape[3];
        
        Ok(LatentSpace::new(
            latent_data,
            resolution,
            channels,
        ))
    }
    
    /// Extract latent features from image
    fn extract_latent_features(&self, image_data: &[f32]) -> HLDVAResult<Tensor> {
        let compression_factor = self.config.compression_factor;
        let latent_size = image_data.len() / compression_factor;
        
        let mut latent_data = Vec::with_capacity(latent_size);
        
        // Simple downsampling and feature extraction
        for i in 0..latent_size {
            let src_idx = i * compression_factor;
            if src_idx < image_data.len() {
                let mut feature = 0.0;
                for j in 0..compression_factor.min(image_data.len() - src_idx) {
                    feature += image_data[src_idx + j];
                }
                latent_data.push(feature / compression_factor as f32);
            } else {
                latent_data.push(0.0);
            }
        }
        
        Ok(Tensor::new(latent_data, vec![latent_size]))
    }
    
    /// Compute latent shape from input shape
    fn compute_latent_shape(&self, input_shape: &[usize]) -> HLDVAResult<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(HLDVAError::Model("Expected 4D input tensor".to_string()));
        }
        
        let (_, h, w, c) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        let latent_h = h / self.config.spatial_compression;
        let latent_w = w / self.config.spatial_compression;
        let latent_c = c * self.config.channel_expansion;
        
        Ok(vec![1, latent_h, latent_w, latent_c])
    }
    
    /// Apply convolution layers
    fn apply_convolution(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let mut output_data = Vec::with_capacity(input_data.len());
        
        // Simple convolution operation
        for i in 0..input_data.len() {
            let mut sum = 0.0;
            let kernel_size = self.config.kernel_size;
            
            for j in 0..kernel_size {
                let idx = if i >= j { i - j } else { 0 };
                let weight = ((j as f32 + 1.0) * (i as f32 + 1.0)).sin() / (kernel_size as f32);
                sum += input_data[idx] * weight;
            }
            
            output_data.push(sum);
        }
        
        Ok(Tensor::new(output_data, input.shape().to_vec()))
    }
}

/// VAED Encoder configuration
#[derive(Debug, Clone)]
pub struct VAEDncoderConfig {
    pub compression_factor: usize,
    pub spatial_compression: usize,
    pub channel_expansion: usize,
    pub kernel_size: usize,
    pub num_layers: usize,
}

impl Default for VAEDncoderConfig {
    fn default() -> Self {
        Self {
            compression_factor: 4,
            spatial_compression: 2,
            channel_expansion: 2,
            kernel_size: 3,
            num_layers: 4,
        }
    }
}
