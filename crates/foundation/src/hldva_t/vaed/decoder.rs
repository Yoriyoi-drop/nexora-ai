//! VAED Decoder module for HLDVA-T

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Variational Autoencoder Decoder
pub struct VAEDecoder {
    config: VAEDecoderConfig,
}

impl VAEDecoder {
    /// Create new VAED decoder
    pub fn new(config: VAEDecoderConfig) -> HLDVAResult<Self> {
        Ok(Self { config })
    }
    
    /// Decode latent space to image
    pub fn decode(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        let latent_data = latent.data.data();
        let shape = latent.shape(); // This returns (height, width, channels)
        let shape_vec = vec![shape.0, shape.1, shape.2];
        
        let image_data = self.reconstruct_from_latent(latent_data, &shape_vec)?;
        
        Ok(Tensor::new(image_data, self.compute_output_shape(&shape_vec)?))
    }
    
    /// Reconstruct image from latent features
    fn reconstruct_from_latent(&self, latent_data: &[f32], latent_shape: &[usize]) -> HLDVAResult<Vec<f32>> {
        let expansion_factor = self.config.expansion_factor;
        let output_size = latent_data.len() * expansion_factor;
        
        let mut image_data = Vec::with_capacity(output_size);
        
        // Simple upsampling and reconstruction
        for i in 0..output_size {
            let latent_idx = i / expansion_factor;
            if latent_idx < latent_data.len() {
                let base_value = latent_data[latent_idx];
                let offset = (i % expansion_factor) as f32;
                let reconstructed = base_value * (1.0 + 0.1 * offset.sin());
                image_data.push(reconstructed);
            } else {
                image_data.push(0.0);
            }
        }
        
        Ok(image_data)
    }
    
    /// Compute output shape from latent shape
    fn compute_output_shape(&self, latent_shape: &[usize]) -> HLDVAResult<Vec<usize>> {
        if latent_shape.len() != 4 {
            return Err(HLDVAError::Model("Expected 4D latent tensor".to_string()));
        }
        
        let (_, latent_h, latent_w, latent_c) = (
            latent_shape[0], 
            latent_shape[1], 
            latent_shape[2], 
            latent_shape[3]
        );
        
        let output_h = latent_h * self.config.spatial_expansion;
        let output_w = latent_w * self.config.spatial_expansion;
        let output_c = latent_c / self.config.channel_reduction;
        
        Ok(vec![1, output_h, output_w, output_c])
    }
    
    /// Apply transposed convolution
    fn apply_transposed_convolution(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let kernel_size = self.config.kernel_size;
        let stride = self.config.stride;
        
        let input_len = input_data.len();
        let output_len = (input_len - 1) * stride + kernel_size;
        let mut output_data = vec![0.0; output_len];
        
        for i in 0..input_len {
            for j in 0..kernel_size {
                let output_idx = i * stride + j;
                if output_idx < output_len {
                    let weight = ((j as f32 + 1.0) * (i as f32 + 1.0)).cos() / (kernel_size as f32);
                    output_data[output_idx] += input_data[i] * weight;
                }
            }
        }
        
        Ok(Tensor::new(output_data, vec![output_len]))
    }
    
    /// Apply activation function
    fn apply_activation(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let mut output_data = Vec::with_capacity(input_data.len());
        
        for &val in input_data {
            let activated = match self.config.activation {
                ActivationType::ReLU => val.max(0.0),
                ActivationType::Sigmoid => 1.0 / (1.0 + (-val).exp()),
                ActivationType::Tanh => val.tanh(),
                ActivationType::None => val,
            };
            output_data.push(activated);
        }
        
        Ok(Tensor::new(output_data, input.shape().to_vec()))
    }
}

/// VAED Decoder configuration
#[derive(Debug, Clone)]
pub struct VAEDecoderConfig {
    pub expansion_factor: usize,
    pub spatial_expansion: usize,
    pub channel_reduction: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub activation: ActivationType,
    pub num_layers: usize,
}

#[derive(Debug, Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    None,
}

impl Default for VAEDecoderConfig {
    fn default() -> Self {
        Self {
            expansion_factor: 4,
            spatial_expansion: 2,
            channel_reduction: 2,
            kernel_size: 3,
            stride: 2,
            activation: ActivationType::ReLU,
            num_layers: 4,
        }
    }
}
