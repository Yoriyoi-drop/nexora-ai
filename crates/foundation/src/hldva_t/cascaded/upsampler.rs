//! Upsampler module for HLDVA-T cascaded architecture

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Upsampler for increasing resolution
pub struct Upsampler {
    config: UpsamplerConfig,
}

impl Upsampler {
    /// Create new upsampler
    pub fn new(config: UpsamplerConfig) -> HLDVAResult<Self> {
        Ok(Self { config })
    }
    
    /// Upsample input tensor
    pub fn upsample(&self, input: Tensor, factor: usize) -> HLDVAResult<Tensor> {
        // Simplified upsampling - just duplicate pixels
        let input_data = input.data();
        let (_, h, w, c) = self.get_input_shape(&input)?;
        let new_h = h * factor;
        let new_w = w * factor;
        
        let mut output_data = Vec::with_capacity(new_h * new_w * c);
        
        for y in 0..new_h {
            for x in 0..new_w {
                let src_y = y / factor;
                let src_x = x / factor;
                
                for ch in 0..c {
                    let idx = (src_y * w + src_x) * c + ch;
                    output_data.push(input_data[idx]);
                }
            }
        }
        
        Ok(Tensor::new(output_data, vec![1, new_h, new_w, c]))
    }
    
    fn get_input_shape(&self, tensor: &Tensor) -> HLDVAResult<(usize, usize, usize, usize)> {
        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(HLDVAError::Model("Expected 4D tensor".to_string()));
        }
        Ok((shape[0], shape[1], shape[2], shape[3]))
    }
}

/// Upsampler configuration
#[derive(Debug, Clone)]
pub struct UpsamplerConfig {
    pub scale_factor: usize,
    pub method: UpsamplingMethod,
}

#[derive(Debug, Clone)]
pub enum UpsamplingMethod {
    Nearest,
    Bilinear,
    Bicubic,
}

impl Default for UpsamplerConfig {
    fn default() -> Self {
        Self {
            scale_factor: 2,
            method: UpsamplingMethod::Bilinear,
        }
    }
}
