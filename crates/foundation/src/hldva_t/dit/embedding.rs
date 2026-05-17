//! Embedding Layers for DiT
//!
//! Implementasi patch embedding, time embedding, dan position embedding

use crate::hldva_t::types::*;
use crate::atqs::Tensor;
pub use crate::hldva_t::types::HLDVAResult;

/// Patch Embedding untuk mengubah latent menjadi patches
pub struct PatchEmbedding {
    patch_size: usize,
    hidden_dim: usize,
    
    // Convolution-like projection
    projection: Linear,
    
    // VAE compression factor
    _compression_factor: usize,
}

impl PatchEmbedding {
    pub fn new(patch_size: usize, hidden_dim: usize) -> HLDVAResult<Self> {
        let input_dim = patch_size * patch_size * 4; // 4 channels untuk latent
        let projection = Linear::new(input_dim, hidden_dim)?;
        
        Ok(Self {
            patch_size,
            hidden_dim,
            projection,
            _compression_factor: 8, // Standard VAE compression
        })
    }
    
    /// Embed latent menjadi patches
    pub fn embed(&self, latent: &Tensor) -> HLDVAResult<Tensor> {
        let latent_shape = latent.shape();
        if latent_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid latent shape".to_string()));
        }
        
        let height = latent_shape[0];
        let width = latent_shape[1];
        let channels = latent_shape[2];
        
        // Calculate number of patches
        let patches_h = height / self.patch_size;
        let patches_w = width / self.patch_size;
        let num_patches = patches_h * patches_w;
        
        let latent_data = latent.data();
        let mut patches = Vec::with_capacity(num_patches * channels * self.patch_size * self.patch_size);
        
        // Extract patches
        for patch_y in 0..patches_h {
            for patch_x in 0..patches_w {
                for c in 0..channels {
                    for py in 0..self.patch_size {
                        for px in 0..self.patch_size {
                            let y = patch_y * self.patch_size + py;
                            let x = patch_x * self.patch_size + px;
                            
                            if y < height && x < width {
                                let idx = (y * width + x) * channels + c;
                                if idx < latent_data.len() {
                                    patches.push(latent_data[idx]);
                                } else {
                                    patches.push(0.0);
                                }
                            } else {
                                patches.push(0.0);
                            }
                        }
                    }
                }
            }
        }
        
        // Project ke hidden dimension
        let patch_tensor = Tensor::new(patches, vec![num_patches, channels * self.patch_size * self.patch_size]);
        let embedded = self.projection.forward(&patch_tensor)?;
        
        // Flatten ke sequence
        let embedded_data = embedded.data();
        let flattened = embedded_data.to_vec();
        
        Ok(Tensor::new(flattened, vec![num_patches, self.hidden_dim]))
    }
}

/// Time Embedding untuk timestep conditioning
pub struct TimeEmbedding {
    hidden_dim: usize,
    
    // MLP untuk time embedding
    linear1: Linear,
    linear2: Linear,
    
    // Positional encoding frequencies
    frequencies: Vec<f32>,
}

impl TimeEmbedding {
    pub fn new(hidden_dim: usize) -> HLDVAResult<Self> {
        let linear1 = Linear::new(hidden_dim, hidden_dim * 4)?;
        let linear2 = Linear::new(hidden_dim * 4, hidden_dim)?;
        
        // Calculate frequencies untuk sinusoidal embedding
        let mut frequencies = Vec::with_capacity(hidden_dim / 2);
        for i in 0..hidden_dim / 2 {
            let freq = 1.0 / (10000.0_f32.powf((2 * i) as f32 / hidden_dim as f32));
            frequencies.push(freq);
        }
        
        Ok(Self {
            hidden_dim,
            linear1,
            linear2,
            frequencies,
        })
    }
    
    /// Embed timestep
    pub fn embed(&self, timestep: Timestep) -> HLDVAResult<Tensor> {
        let t = timestep.0 as f32;
        
        // Sinusoidal positional encoding
        let mut encoding = Vec::with_capacity(self.hidden_dim);
        
        for (i, &freq) in self.frequencies.iter().enumerate() {
            encoding.push((t * freq).sin());
            encoding.push((t * freq).cos());
        }
        
        // Add extra dimensions jika hidden_dim ganjil
        while encoding.len() < self.hidden_dim {
            encoding.push(0.0);
        }
        
        let time_tensor = Tensor::new(encoding, vec![self.hidden_dim]);
        
        // MLP projection
        let hidden = self.linear1.forward(&time_tensor)?;
        let output = self.linear2.forward(&hidden)?;
        
        Ok(output)
    }
}

/// Position Embedding untuk spatial positions
pub struct PositionEmbedding {
    max_seq_len: usize,
    hidden_dim: usize,
    
    // Learnable position embeddings
    embeddings: Tensor,
    
    // Sinusoidal frequencies
    frequencies: Vec<f32>,
}

impl PositionEmbedding {
    pub fn new(max_seq_len: usize, hidden_dim: usize) -> HLDVAResult<Self> {
        // Initialize learnable embeddings
        let embedding_data = Self::init_position_embeddings(max_seq_len, hidden_dim);
        let embeddings = Tensor::new(embedding_data, vec![max_seq_len, hidden_dim]);
        
        // Calculate frequencies untuk 2D sinusoidal
        let mut frequencies = Vec::with_capacity(hidden_dim / 4);
        for i in 0..hidden_dim / 4 {
            let freq = 1.0 / (10000.0_f32.powf((2 * i) as f32 / (hidden_dim / 2) as f32));
            frequencies.push(freq);
        }
        
        Ok(Self {
            max_seq_len,
            hidden_dim,
            embeddings,
            frequencies,
        })
    }
    
    /// Add position embedding ke input
    pub fn add_to(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_data = input.data();
        let embedding_data = self.embeddings.data();
        
        let seq_len = input_data.len() / self.hidden_dim;
        let mut output = Vec::with_capacity(input_data.len());
        
        for pos in 0..seq_len.min(self.max_seq_len) {
            for dim in 0..self.hidden_dim {
                let input_idx = pos * self.hidden_dim + dim;
                let embedding_idx = pos * self.hidden_dim + dim;
                
                let input_val = if input_idx < input_data.len() {
                    input_data[input_idx]
                } else {
                    0.0
                };
                
                let embedding_val = if embedding_idx < embedding_data.len() {
                    embedding_data[embedding_idx]
                } else {
                    0.0
                };
                
                output.push(input_val + embedding_val);
            }
        }
        
        Ok(Tensor::new(output, input.shape().to_vec()))
    }
    
    /// Get 2D sinusoidal position embedding
    pub fn get_2d_embedding(&self, height: usize, width: usize) -> HLDVAResult<Tensor> {
        let mut embedding = Vec::with_capacity(height * width * self.hidden_dim);
        
        for y in 0..height {
            for x in 0..width {
                let pos_embedding = self.encode_2d_position(x, y)?;
                embedding.extend_from_slice(&pos_embedding);
            }
        }
        
        Ok(Tensor::new(embedding, vec![height * width, self.hidden_dim]))
    }
    
    /// Encode 2D position
    fn encode_2d_position(&self, x: usize, y: usize) -> HLDVAResult<Vec<f32>> {
        let mut encoding = Vec::with_capacity(self.hidden_dim);
        let x_f = x as f32;
        let y_f = y as f32;
        
        // Alternate between x and y encoding
        for (i, &freq) in self.frequencies.iter().enumerate() {
            // X position encoding
            encoding.push((x_f * freq).sin());
            encoding.push((x_f * freq).cos());
            
            // Y position encoding
            encoding.push((y_f * freq).sin());
            encoding.push((y_f * freq).cos());
        }
        
        // Pad jika perlu
        while encoding.len() < self.hidden_dim {
            encoding.push(0.0);
        }
        
        Ok(encoding)
    }
    
    /// Initialize position embeddings
    fn init_position_embeddings(max_seq_len: usize, hidden_dim: usize) -> Vec<f32> {
        let mut embeddings = Vec::with_capacity(max_seq_len * hidden_dim);
        
        for pos in 0..max_seq_len {
            let pos_embedding = Self::encode_position(pos, hidden_dim);
            embeddings.extend_from_slice(&pos_embedding);
        }
        
        embeddings
    }
    
    /// Encode position
    fn encode_position(pos: usize, hidden_dim: usize) -> Vec<f32> {
        let mut encoding = Vec::with_capacity(hidden_dim);
        let pos_f = pos as f32;
        
        for i in 0..hidden_dim / 2 {
            let freq = 1.0 / (10000.0_f32.powf((2 * i) as f32 / hidden_dim as f32));
            encoding.push((pos_f * freq).sin());
            encoding.push((pos_f * freq).cos());
        }
        
        // Pad jika hidden_dim ganjil
        while encoding.len() < hidden_dim {
            encoding.push(0.0);
        }
        
        encoding
    }
}

/// CLIP Conditioning Projection
pub struct ClipConditioningProjection {
    hidden_dim: usize,
    _clip_dim: usize,
    
    // Projections
    text_projection: Linear,
    image_projection: Linear,
    
    // Cross-attention projection
    cross_attention_proj: Linear,
    
    // Layer normalization
    layer_norm: LayerNorm,
}

impl ClipConditioningProjection {
    pub fn new(hidden_dim: usize, _clip_dim: usize) -> HLDVAResult<Self> {
        let text_projection = Linear::new(_clip_dim, hidden_dim)?;
        let image_projection = Linear::new(_clip_dim, hidden_dim)?;
        let cross_attention_proj = Linear::new(_clip_dim, hidden_dim)?;
        let layer_norm = LayerNorm::new(hidden_dim)?;
        
        Ok(Self {
            hidden_dim,
            _clip_dim,
            text_projection,
            image_projection,
            cross_attention_proj,
            layer_norm,
        })
    }
    
    /// Process CLIP embedding
    pub fn process(&self, clip_embedding: &ClipEmbedding) -> HLDVAResult<Tensor> {
        // Project text features
        let text_proj = self.text_projection.forward(&clip_embedding.text_features)?;
        
        // Project image features jika ada
        let image_proj = if let Some(ref image_features) = clip_embedding.image_features {
            self.image_projection.forward(image_features)?
        } else {
            Tensor::new(vec![0.0; self.hidden_dim], vec![self.hidden_dim])
        };
        
        // Combine text and image features
        let combined = self.combine_features(&text_proj, &image_proj)?;
        
        // Apply layer norm
        let normalized = self.layer_norm.forward(&combined)?;
        
        // Final projection untuk cross-attention
        let output = self.cross_attention_proj.forward(&normalized)?;
        
        Ok(output)
    }
    
    /// Combine text and image features
    fn combine_features(&self, text: &Tensor, image: &Tensor) -> HLDVAResult<Tensor> {
        let text_data = text.data();
        let image_data = image.data();
        
        let mut combined = Vec::with_capacity(self.hidden_dim);
        
        for i in 0..self.hidden_dim {
            let text_val = if i < text_data.len() { text_data[i] } else { 0.0 };
            let image_val = if i < image_data.len() { image_data[i] } else { 0.0 };
            
            // Simple averaging - bisa diganti dengan metode lain
            combined.push((text_val + image_val) / 2.0);
        }
        
        Ok(Tensor::new(combined, vec![self.hidden_dim]))
    }
}

// Re-export Linear dan LayerNorm dari attention module
use super::attention::{Linear, LayerNorm};
