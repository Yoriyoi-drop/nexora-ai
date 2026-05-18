//! VAE (Variational Autoencoder) Implementation
//!
//! VAE digunakan untuk kompresi gambar ke latent space dan decoding kembali:
//! - Encoder: Gambar → Latent (8x compression)
//! - Decoder: Latent → Gambar
//! - KL divergence untuk regularisasi

pub mod encoder;
pub mod decoder;
pub mod latent;

use crate::{
    config::VAEConfig,
    types::*,
};
use nexora_atqs::Tensor;

/// Main VAE
pub struct VAE {
    config: VAEConfig,
    
    encoder: VAEEncoder,
    decoder: VAEDecoder,
}

impl VAE {
    /// Create new VAE
    pub fn new(config: &VAEConfig) -> HLDVAResult<Self> {
        let encoder = VAEEncoder::new(config)?;
        let decoder = VAEDecoder::new(config)?;
        
        Ok(Self {
            config: config.clone(),
            encoder,
            decoder,
        })
    }
    
    /// Encode image ke latent
    pub fn encode(&self, image: &Tensor) -> HLDVAResult<LatentSpace> {
        self.encoder.encode(image)
    }
    
    /// Decode latent ke image
    pub fn decode(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        self.decoder.decode(latent)
    }
    
    /// Forward pass (encode then decode)
    pub fn forward(&self, image: &Tensor) -> HLDVAResult<(Tensor, Tensor, Tensor)> {
        let latent = self.encode(image)?;
        let reconstructed = self.decode(&latent)?;
        
        // Calculate KL divergence
        let kl_loss = self.calculate_kl_loss(&latent)?;
        
        Ok((reconstructed, latent.data, kl_loss))
    }
    
    /// Calculate KL divergence loss
    fn calculate_kl_loss(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        // Simplified KL loss calculation
        let latent_data = latent.data.data();
        let mut kl_loss = Vec::with_capacity(1);
        
        // Standard normal KL: KL(N(μ,σ²) || N(0,1))
        let mut kl_sum = 0.0;
        for i in (0..latent_data.len()).step_by(2) {
            if i + 1 < latent_data.len() {
                let mu = latent_data[i];
                let log_var = latent_data[i + 1];
                let kl = -0.5 * (1.0 + log_var - mu * mu - log_var.exp());
                kl_sum += kl;
            }
        }
        
        kl_loss.push(kl_sum / (latent_data.len() / 2) as f32);
        
        Ok(Tensor::new(kl_loss, vec![1]))
    }
    
    /// Get configuration
    pub fn config(&self) -> &VAEConfig {
        &self.config
    }
}

/// VAE Encoder
pub struct VAEEncoder {
    config: VAEConfig,
    
    // Convolutional layers
    conv_layers: Vec<ConvBlock>,
    
    // Residual blocks
    residual_blocks: Vec<ResidualBlock>,
    
    // Final layers
    final_conv: Conv2D,
    
    // Output layers for μ and log(σ²)
    mu_layer: Linear,
    log_var_layer: Linear,
}

impl VAEEncoder {
    pub fn new(config: &VAEConfig) -> HLDVAResult<Self> {
        let mut conv_layers = Vec::new();
        
        // Downsampling layers
        conv_layers.push(ConvBlock::new(3, 64, 4, 2, 1)?); // 256x256 -> 128x128
        conv_layers.push(ConvBlock::new(64, 128, 4, 2, 1)?); // 128x128 -> 64x64
        conv_layers.push(ConvBlock::new(128, 256, 4, 2, 1)?); // 64x64 -> 32x32
        conv_layers.push(ConvBlock::new(256, 512, 4, 2, 1)?); // 32x32 -> 16x16
        
        // Residual blocks
        let mut residual_blocks = Vec::new();
        for _ in 0..3 {
            residual_blocks.push(ResidualBlock::new(512)?);
        }
        
        let final_conv = Conv2D::new(512, 512, 3, 1, 1)?;
        
        // Calculate flattened size
        let flattened_size = 16 * 16 * 512; // After downsampling
        let mu_layer = Linear::new(flattened_size, config.latent_dim)?;
        let log_var_layer = Linear::new(flattened_size, config.latent_dim)?;
        
        Ok(Self {
            config: config.clone(),
            conv_layers,
            residual_blocks,
            final_conv,
            mu_layer,
            log_var_layer,
        })
    }
    
    /// Encode image ke latent
    pub fn encode(&self, image: &Tensor) -> HLDVAResult<LatentSpace> {
        let image_shape = image.shape();
        if image_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid image shape".to_string()));
        }
        
        let (height, width, channels) = (
            image_shape[0],
            image_shape[1],
            image_shape[2],
        );
        
        // Forward through conv layers
        let mut current = image.clone();
        let mut current_size = (height, width);
        
        for conv_layer in &self.conv_layers {
            current = conv_layer.forward(&current)?;
            current_size = (current_size.0 / 2, current_size.1 / 2); // Due to stride 2
        }
        
        // Forward through residual blocks
        for residual_block in &self.residual_blocks {
            current = residual_block.forward(&current)?;
        }
        
        // Final convolution
        current = self.final_conv.forward(&current)?;
        
        // Flatten
        let flattened = self.flatten(&current)?;
        
        // Get μ and log(σ²)
        let mu = self.mu_layer.forward(&flattened)?;
        let log_var = self.log_var_layer.forward(&flattened)?;
        
        // Combine μ and log(σ²) into single latent
        let latent_data = self.combine_mu_log_var(&mu, &log_var);
        
        // Calculate latent resolution
        let latent_resolution = Resolution::new(
            current_size.0 / 8, // VAE compression factor
            current_size.1 / 8,
        );
        
        Ok(LatentSpace::new(
            Tensor::new(latent_data, vec![latent_resolution.height, latent_resolution.width, self.config.latent_dim]),
            latent_resolution,
            self.config.latent_dim,
        ))
    }
    
    /// Combine μ and log(σ²)
    fn combine_mu_log_var(&self, mu: &Tensor, log_var: &Tensor) -> Vec<f32> {
        let mu_data = mu.data();
        let log_var_data = log_var.data();
        
        let mut combined = Vec::with_capacity(mu_data.len() + log_var_data.len());
        
        // Interleave μ and log(σ²)
        for i in 0..mu_data.len().min(log_var_data.len()) {
            combined.push(mu_data[i]);
            combined.push(log_var_data[i]);
        }
        
        combined
    }
    
    /// Flatten tensor
    fn flatten(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        let data = tensor.data();
        Ok(Tensor::new(data.to_vec(), vec![data.len()]))
    }
}

/// VAE Decoder
pub struct VAEDecoder {
    _config: VAEConfig,
    
    // Input layers
    input_projection: Linear,
    
    // Reshape layer
    reshape_size: (usize, usize, usize), // (height, width, channels)
    
    // Residual blocks
    residual_blocks: Vec<ResidualBlock>,
    
    // Upsampling layers
    upsample_layers: Vec<UpsampleBlock>,
    
    // Final convolution
    final_conv: Conv2D,
}

impl VAEDecoder {
    pub fn new(config: &VAEConfig) -> HLDVAResult<Self> {
        let flattened_size = 16 * 16 * 512; // Match encoder output
        let input_projection = Linear::new(config.latent_dim * 2, flattened_size)?; // *2 for μ and log(σ²)
        
        let reshape_size = (16, 16, 512);
        
        // Residual blocks
        let mut residual_blocks = Vec::new();
        for _ in 0..3 {
            residual_blocks.push(ResidualBlock::new(512)?);
        }
        
        // Upsampling layers
        let mut upsample_layers = Vec::new();
        upsample_layers.push(UpsampleBlock::new(512, 256)?); // 16x16 -> 32x32
        upsample_layers.push(UpsampleBlock::new(256, 128)?); // 32x32 -> 64x64
        upsample_layers.push(UpsampleBlock::new(128, 64)?);  // 64x64 -> 128x128
        upsample_layers.push(UpsampleBlock::new(64, 32)?);   // 128x128 -> 256x256
        
        let final_conv = Conv2D::new(32, 3, 3, 1, 1)?; // RGB output
        
        Ok(Self {
            _config: config.clone(),
            input_projection,
            reshape_size,
            residual_blocks,
            upsample_layers,
            final_conv,
        })
    }
    
    /// Decode latent ke image
    pub fn decode(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        // Sample from latent distribution (reparameterization trick)
        let sampled = self.reparameterize(latent)?;
        
        // Project and reshape
        let projected = self.input_projection.forward(&sampled)?;
        let reshaped = self.reshape(&projected)?;
        
        // Forward through residual blocks
        let mut current = reshaped;
        for residual_block in &self.residual_blocks {
            current = residual_block.forward(&current)?;
        }
        
        // Forward through upsampling layers
        for upsample_layer in &self.upsample_layers {
            current = upsample_layer.forward(&current)?;
        }
        
        // Final convolution
        let output = self.final_conv.forward(&current)?;
        
        // Apply sigmoid activation
        let activated = self.sigmoid_activation(&output)?;
        
        Ok(activated)
    }
    
    /// Reparameterization trick: z = μ + σ * ε
    fn reparameterize(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        let latent_data = latent.data.data();
        let mut sampled = Vec::with_capacity(latent_data.len() / 2);
        
        for i in (0..latent_data.len()).step_by(2) {
            if i + 1 < latent_data.len() {
                let mu = latent_data[i];
                let log_var = latent_data[i + 1];
                let std = (log_var / 2.0).exp(); // log_var is actually log(σ²)
                let epsilon = self.randn(); // Standard normal
                let z = mu + std * epsilon;
                sampled.push(z);
            }
        }
        
        Ok(Tensor::new(sampled.clone(), vec![sampled.len()]))
    }
    
    /// Reshape flat tensor to 3D
    fn reshape(&self, flat: &Tensor) -> HLDVAResult<Tensor> {
        let flat_data = flat.data();
        let (h, w, c) = self.reshape_size;
        let total_size = h * w * c;
        
        if flat_data.len() != total_size {
            return Err(HLDVAError::Model("Size mismatch in reshape".to_string()));
        }
        
        Ok(Tensor::new(flat_data.to_vec(), vec![h, w, c]))
    }
    
    /// Sigmoid activation
    fn sigmoid_activation(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        let data = tensor.data();
        let activated: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok(Tensor::new(activated, tensor.shape().to_vec()))
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

/// Convolutional Block
pub struct ConvBlock {
    conv: Conv2D,
    norm: LayerNorm2D,
    activation: ReLU,
}

impl ConvBlock {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> HLDVAResult<Self> {
        let conv = Conv2D::new(in_channels, out_channels, kernel_size, stride, padding)?;
        let norm = LayerNorm2D::new(out_channels)?;
        
        Ok(Self {
            conv,
            norm,
            activation: ReLU,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let conv_out = self.conv.forward(input)?;
        let norm_out = self.norm.forward(&conv_out)?;
        let activated = self.activation.forward(&norm_out)?;
        
        Ok(activated)
    }
}

/// Residual Block
pub struct ResidualBlock {
    conv1: Conv2D,
    conv2: Conv2D,
    norm1: LayerNorm2D,
    norm2: LayerNorm2D,
    activation: ReLU,
}

impl ResidualBlock {
    pub fn new(channels: usize) -> HLDVAResult<Self> {
        let conv1 = Conv2D::new(channels, channels, 3, 1, 1)?;
        let conv2 = Conv2D::new(channels, channels, 3, 1, 1)?;
        let norm1 = LayerNorm2D::new(channels)?;
        let norm2 = LayerNorm2D::new(channels)?;
        
        Ok(Self {
            conv1,
            conv2,
            norm1,
            norm2,
            activation: ReLU,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let conv1_out = self.conv1.forward(input)?;
        let norm1_out = self.norm1.forward(&conv1_out)?;
        let act1_out = self.activation.forward(&norm1_out)?;
        
        let conv2_out = self.conv2.forward(&act1_out)?;
        let norm2_out = self.norm2.forward(&conv2_out)?;
        
        // Residual connection
        let residual = self.add_tensors(input, &norm2_out)?;
        let activated = self.activation.forward(&residual)?;
        
        Ok(activated)
    }
    
    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
        let a_data = a.data();
        let b_data = b.data();
        
        let mut sum = Vec::with_capacity(a_data.len());
        for i in 0..a_data.len() {
            let b_val = if i < b_data.len() { b_data[i] } else { 0.0 };
            sum.push(a_data[i] + b_val);
        }
        
        Ok(Tensor::new(sum, a.shape().to_vec()))
    }
}

/// Upsampling Block
pub struct UpsampleBlock {
    conv: Conv2D,
    norm: LayerNorm2D,
    activation: ReLU,
}

impl UpsampleBlock {
    pub fn new(in_channels: usize, out_channels: usize) -> HLDVAResult<Self> {
        // Use transposed convolution for upsampling
        let conv = Conv2D::new(in_channels, out_channels, 3, 1, 1)?;
        let norm = LayerNorm2D::new(out_channels)?;
        
        Ok(Self {
            conv,
            norm,
            activation: ReLU,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        // Upsample by nearest neighbor
        let upsampled = self.nearest_neighbor_upsample(input, 2)?;
        
        // Apply convolution
        let conv_out = self.conv.forward(&upsampled)?;
        let norm_out = self.norm.forward(&conv_out)?;
        let activated = self.activation.forward(&norm_out)?;
        
        Ok(activated)
    }
    
    fn nearest_neighbor_upsample(&self, input: &Tensor, scale: usize) -> HLDVAResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid input shape".to_string()));
        }
        
        let (height, width, channels) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        );
        
        let new_height = height * scale;
        let new_width = width * scale;
        
        let input_data = input.data();
        let mut upsampled = Vec::with_capacity(new_height * new_width * channels);
        
        for c in 0..channels {
            for y in 0..new_height {
                for x in 0..new_width {
                    let src_y = y / scale;
                    let src_x = x / scale;
                    
                    if src_y < height && src_x < width {
                        let src_idx = (src_y * width + src_x) * channels + c;
                        if src_idx < input_data.len() {
                            upsampled.push(input_data[src_idx]);
                        } else {
                            upsampled.push(0.0);
                        }
                    } else {
                        upsampled.push(0.0);
                    }
                }
            }
        }
        
        Ok(Tensor::new(upsampled, vec![new_height, new_width, channels]))
    }
}

// Re-export dependencies
use super::dit::{Linear, LayerNorm};

/// Conv2D Layer
pub struct Conv2D {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    
    weight: Tensor,
    bias: Tensor,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> HLDVAResult<Self> {
        let weight_size = out_channels * in_channels * kernel_size * kernel_size;
        let weight_data = vec![0.0; weight_size];
        let weight = Tensor::new(weight_data, vec![out_channels, in_channels, kernel_size, kernel_size]);
        
        let bias_data = vec![0.0; out_channels];
        let bias = Tensor::new(bias_data, vec![out_channels]);
        
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid input shape".to_string()));
        }
        
        let (height, width, channels) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        );
        
        let output_height = (height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        let input_data = input.data();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();
        
        let mut output = Vec::with_capacity(output_height * output_width * self.out_channels);
        
        for out_c in 0..self.out_channels {
            for out_y in 0..output_height {
                for out_x in 0..output_width {
                    let mut sum = 0.0;
                    
                    for in_c in 0..self.in_channels {
                        for ky in 0..self.kernel_size {
                            for kx in 0..self.kernel_size {
                                let in_y = out_y * self.stride + ky - self.padding;
                                let in_x = out_x * self.stride + kx - self.padding;
                                
                                if true && in_y < height && true && in_x < width {
                                    let in_idx = (in_y * width + in_x) * channels + in_c;
                                    let weight_idx = (out_c * self.in_channels + in_c) * self.kernel_size * self.kernel_size + ky * self.kernel_size + kx;
                                    
                                    if in_idx < input_data.len() && weight_idx < weight_data.len() {
                                        sum += input_data[in_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    // Add bias
                    let bias_val = if out_c < bias_data.len() { bias_data[out_c] } else { 0.0 };
                    sum += bias_val;
                    
                    output.push(sum);
                }
            }
        }
        
        Ok(Tensor::new(output, vec![output_height, output_width, self.out_channels]))
    }
}

/// LayerNorm2D
pub struct LayerNorm2D {
    _num_channels: usize,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm2D {
    pub fn new(num_channels: usize) -> HLDVAResult<Self> {
        let weight = Tensor::new(vec![1.0; num_channels], vec![num_channels]);
        let bias = Tensor::new(vec![0.0; num_channels], vec![num_channels]);
        
        Ok(Self {
            _num_channels: num_channels,
            weight,
            bias,
            eps: 1e-6,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid input shape".to_string()));
        }
        
        let (height, width, channels) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        );
        
        let input_data = input.data();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();
        
        let mut output = Vec::with_capacity(input_data.len());
        
        for h in 0..height {
            for w in 0..width {
                // Calculate mean and variance for this spatial location
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                
                for c in 0..channels {
                    let idx = (h * width + w) * channels + c;
                    if idx < input_data.len() {
                        let val = input_data[idx];
                        sum += val;
                        sum_sq += val * val;
                    }
                }
                
                let mean = sum / channels as f32;
                let variance = (sum_sq / channels as f32) - mean * mean;
                let std = (variance + self.eps).sqrt();
                
                // Normalize and scale
                for c in 0..channels {
                    let idx = (h * width + w) * channels + c;
                    if idx < input_data.len() {
                        let normalized = (input_data[idx] - mean) / std;
                        let weight_val = if c < weight_data.len() { weight_data[c] } else { 1.0 };
                        let bias_val = if c < bias_data.len() { bias_data[c] } else { 0.0 };
                        let scaled = normalized * weight_val + bias_val;
                        output.push(scaled);
                    } else {
                        output.push(0.0);
                    }
                }
            }
        }
        
        Ok(Tensor::new(output, input_shape.to_vec()))
    }
}

/// ReLU Activation
pub struct ReLU;

impl ReLU {
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let data = input.data();
        let activated: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        Ok(Tensor::new(activated, input.shape().to_vec()))
    }
}
