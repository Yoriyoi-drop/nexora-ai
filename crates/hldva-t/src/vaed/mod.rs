//! VAE (Variational Autoencoder) Implementation
//!
//! GPU-accelerated VAE: encoder compresses image to latent space,
//! decoder reconstructs latent back to image.

pub mod decoder;
pub mod encoder;
pub mod latent;

use crate::{config::VAEConfig, gpu_ops, types::*};
use nexora_atqs::Tensor;

/// Main VAE
pub struct VAE {
    config: VAEConfig,
    encoder: VAEEncoder,
    decoder: VAEDecoder,
}

impl VAE {
    pub fn new(config: &VAEConfig) -> HLDVAResult<Self> {
        let encoder = VAEEncoder::new(config)?;
        let decoder = VAEDecoder::new(config)?;
        Ok(Self { config: config.clone(), encoder, decoder })
    }

    pub fn encode(&self, image: &Tensor) -> HLDVAResult<LatentSpace> {
        self.encoder.encode(image)
    }

    pub fn decode(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        self.decoder.decode(latent)
    }

    pub fn forward(&self, image: &Tensor) -> HLDVAResult<(Tensor, Tensor, Tensor)> {
        let latent = self.encode(image)?;
        let reconstructed = self.decode(&latent)?;
        let kl_loss = self.calculate_kl_loss(&latent)?;
        Ok((reconstructed, latent.data, kl_loss))
    }

    fn calculate_kl_loss(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        let latent_data = latent.data.data();
        let mut kl_sum = 0.0;
        for i in (0..latent_data.len()).step_by(2) {
            if i + 1 < latent_data.len() {
                let mu = latent_data[i];
                let log_var = latent_data[i + 1];
                kl_sum += -0.5 * (1.0 + log_var - mu * mu - log_var.exp());
            }
        }
        let n = (latent_data.len() / 2).max(1) as f32;
        Ok(Tensor::new(vec![kl_sum / n], vec![1]))
    }

    pub fn config(&self) -> &VAEConfig { &self.config }
}

/// VAE Encoder — GPU accelerated convolution
pub struct VAEEncoder {
    config: VAEConfig,
    conv_layers: Vec<ConvBlock>,
    residual_blocks: Vec<ResidualBlock>,
    final_conv: Conv2D,
    mu_layer: Linear,
    log_var_layer: Linear,
}

impl VAEEncoder {
    pub fn new(config: &VAEConfig) -> HLDVAResult<Self> {
        let mut conv_layers = Vec::new();
        conv_layers.push(ConvBlock::new(3, 64, 4, 2, 1)?);
        conv_layers.push(ConvBlock::new(64, 128, 4, 2, 1)?);
        conv_layers.push(ConvBlock::new(128, 256, 4, 2, 1)?);
        conv_layers.push(ConvBlock::new(256, 512, 4, 2, 1)?);

        let mut residual_blocks = Vec::new();
        for _ in 0..3 {
            residual_blocks.push(ResidualBlock::new(512)?);
        }

        let final_conv = Conv2D::new(512, 512, 3, 1, 1)?;
        let flattened_size = 16 * 16 * 512;
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

    pub fn encode(&self, image: &Tensor) -> HLDVAResult<LatentSpace> {
        let image_shape = image.shape();
        if image_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid image shape".to_string()));
        }
        let (height, width, _channels) = (image_shape[0], image_shape[1], image_shape[2]);

        let mut current = image.clone();
        let mut current_size = (height, width);

        for conv_layer in &self.conv_layers {
            current = conv_layer.forward(&current)?;
            current_size = (current_size.0 / 2, current_size.1 / 2);
        }

        for residual_block in &self.residual_blocks {
            current = residual_block.forward(&current)?;
        }

        current = self.final_conv.forward(&current)?;
        let flattened = self.flatten(&current)?;

        let mu = self.mu_layer.forward(&flattened)?;
        let log_var = self.log_var_layer.forward(&flattened)?;
        let latent_data = self.combine_mu_log_var(&mu, &log_var);

        let latent_resolution = Resolution::new(current_size.0 / 8, current_size.1 / 8);

        Ok(LatentSpace::new(
            Tensor::new(latent_data, vec![latent_resolution.height, latent_resolution.width, self.config.latent_dim]),
            latent_resolution,
            self.config.latent_dim,
        ))
    }

    fn combine_mu_log_var(&self, mu: &Tensor, log_var: &Tensor) -> Vec<f32> {
        let mu_data = mu.data();
        let log_var_data = log_var.data();
        let mut combined = Vec::with_capacity(mu_data.len() + log_var_data.len());
        for i in 0..mu_data.len().min(log_var_data.len()) {
            combined.push(mu_data[i]);
            combined.push(log_var_data[i]);
        }
        combined
    }

    fn flatten(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        let data = tensor.data();
        Ok(Tensor::new(data.to_vec(), vec![data.len()]))
    }
}

/// VAE Decoder — GPU accelerated
pub struct VAEDecoder {
    _config: VAEConfig,
    input_projection: Linear,
    reshape_size: (usize, usize, usize),
    residual_blocks: Vec<ResidualBlock>,
    upsample_layers: Vec<UpsampleBlock>,
    final_conv: Conv2D,
}

impl VAEDecoder {
    pub fn new(config: &VAEConfig) -> HLDVAResult<Self> {
        let flattened_size = 16 * 16 * 512;
        let input_projection = Linear::new(config.latent_dim * 2, flattened_size)?;
        let reshape_size = (16, 16, 512);

        let mut residual_blocks = Vec::new();
        for _ in 0..3 {
            residual_blocks.push(ResidualBlock::new(512)?);
        }

        let mut upsample_layers = Vec::new();
        upsample_layers.push(UpsampleBlock::new(512, 256)?);
        upsample_layers.push(UpsampleBlock::new(256, 128)?);
        upsample_layers.push(UpsampleBlock::new(128, 64)?);
        upsample_layers.push(UpsampleBlock::new(64, 32)?);

        let final_conv = Conv2D::new(32, 3, 3, 1, 1)?;

        Ok(Self {
            _config: config.clone(),
            input_projection,
            reshape_size,
            residual_blocks,
            upsample_layers,
            final_conv,
        })
    }

    pub fn decode(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        let sampled = self.reparameterize(latent)?;
        let projected = self.input_projection.forward(&sampled)?;
        let reshaped = self.reshape(&projected)?;

        let mut current = reshaped;
        for residual_block in &self.residual_blocks {
            current = residual_block.forward(&current)?;
        }

        for upsample_layer in &self.upsample_layers {
            current = upsample_layer.forward(&current)?;
        }

        let output = self.final_conv.forward(&current)?;
        self.sigmoid_activation(&output)
    }

    fn reparameterize(&self, latent: &LatentSpace) -> HLDVAResult<Tensor> {
        let latent_data = latent.data.data();
        let mut sampled = Vec::with_capacity(latent_data.len() / 2);
        for i in (0..latent_data.len()).step_by(2) {
            if i + 1 < latent_data.len() {
                let mu = latent_data[i];
                let log_var = latent_data[i + 1];
                let std = (log_var / 2.0).exp();
                let epsilon = self.randn();
                sampled.push(mu + std * epsilon);
            }
        }
        Ok(Tensor::new(sampled.clone(), vec![sampled.len()]))
    }

    fn reshape(&self, flat: &Tensor) -> HLDVAResult<Tensor> {
        let flat_data = flat.data();
        let (h, w, c) = self.reshape_size;
        let total_size = h * w * c;
        if flat_data.len() != total_size {
            return Err(HLDVAError::Model("Size mismatch in reshape".to_string()));
        }
        Ok(Tensor::new(flat_data.to_vec(), vec![h, w, c]))
    }

    fn sigmoid_activation(&self, tensor: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_sigmoid(tensor);
        }
        let data = tensor.data();
        let activated: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok(Tensor::new(activated, tensor.shape().to_vec()))
    }

    fn randn(&self) -> f32 {
        use std::f64::consts::PI;
        let u1: f64 = rand::random();
        let u2: f64 = rand::random();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()) as f32
    }
}

/// Convolutional Block — GPU accelerated
pub struct ConvBlock {
    conv: Conv2D,
    norm: LayerNorm2D,
    activation: ReLU,
}

impl ConvBlock {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> HLDVAResult<Self> {
        let conv = Conv2D::new(in_channels, out_channels, kernel_size, stride, padding)?;
        let norm = LayerNorm2D::new(out_channels)?;
        Ok(Self { conv, norm, activation: ReLU })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let conv_out = self.conv.forward(input)?;
        let norm_out = self.norm.forward(&conv_out)?;
        self.activation.forward(&norm_out)
    }
}

/// Residual Block — GPU accelerated
pub struct ResidualBlock {
    conv1: Conv2D,
    conv2: Conv2D,
    norm1: LayerNorm2D,
    norm2: LayerNorm2D,
    activation: ReLU,
}

impl ResidualBlock {
    pub fn new(channels: usize) -> HLDVAResult<Self> {
        Ok(Self {
            conv1: Conv2D::new(channels, channels, 3, 1, 1)?,
            conv2: Conv2D::new(channels, channels, 3, 1, 1)?,
            norm1: LayerNorm2D::new(channels)?,
            norm2: LayerNorm2D::new(channels)?,
            activation: ReLU,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let conv1_out = self.conv1.forward(input)?;
        let norm1_out = self.norm1.forward(&conv1_out)?;
        let act1_out = self.activation.forward(&norm1_out)?;

        let conv2_out = self.conv2.forward(&act1_out)?;
        let norm2_out = self.norm2.forward(&conv2_out)?;

        self.add_and_activate(input, &norm2_out)
    }

    fn add_and_activate(&self, a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
        let sum = if gpu_ops::gpu_available() {
            gpu_ops::gpu_add(a, b)?
        } else {
            let a_data = a.data();
            let b_data = b.data();
            let mut sum = Vec::with_capacity(a_data.len());
            for i in 0..a_data.len() {
                sum.push(a_data[i] + if i < b_data.len() { b_data[i] } else { 0.0 });
            }
            Tensor::new(sum, a.shape().to_vec())
        };
        self.activation.forward(&sum)
    }
}

/// Upsampling Block — GPU accelerated
pub struct UpsampleBlock {
    conv: Conv2D,
    norm: LayerNorm2D,
    activation: ReLU,
}

impl UpsampleBlock {
    pub fn new(in_channels: usize, out_channels: usize) -> HLDVAResult<Self> {
        Ok(Self {
            conv: Conv2D::new(in_channels, out_channels, 3, 1, 1)?,
            norm: LayerNorm2D::new(out_channels)?,
            activation: ReLU,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let upsampled = self.nearest_neighbor_upsample(input, 2)?;
        let conv_out = self.conv.forward(&upsampled)?;
        let norm_out = self.norm.forward(&conv_out)?;
        self.activation.forward(&norm_out)
    }

    fn nearest_neighbor_upsample(&self, input: &Tensor, scale: usize) -> HLDVAResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid input shape".to_string()));
        }
        let (height, width, channels) = (input_shape[0], input_shape[1], input_shape[2]);
        let (new_h, new_w) = (height * scale, width * scale);

        let input_data = input.data();
        let mut upsampled = Vec::with_capacity(new_h * new_w * channels);

        for c in 0..channels {
            for y in 0..new_h {
                for x in 0..new_w {
                    let sy = y / scale;
                    let sx = x / scale;
                    if sy < height && sx < width {
                        let idx = (sy * width + sx) * channels + c;
                        upsampled.push(if idx < input_data.len() { input_data[idx] } else { 0.0 });
                    } else {
                        upsampled.push(0.0);
                    }
                }
            }
        }

        Ok(Tensor::new(upsampled, vec![new_h, new_w, channels]))
    }
}

use super::dit::Linear;

/// GPU-accelerated Conv2D via im2col + GPU matmul (falls back to CPU)
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
        let weight = Tensor::new(vec![0.0; weight_size], vec![out_channels, in_channels, kernel_size, kernel_size]);
        let bias = Tensor::new(vec![0.0; out_channels], vec![out_channels]);
        Ok(Self { in_channels, out_channels, kernel_size, stride, padding, weight, bias })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_conv2d(input, &self.weight, &self.bias, self.stride, self.padding);
        }
        self.forward_cpu(input)
    }

    fn forward_cpu(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid input shape".to_string()));
        }
        let (height, width, channels) = (input_shape[0], input_shape[1], input_shape[2]);

        let out_h = (height + 2 * self.padding).saturating_sub(self.kernel_size) / self.stride + 1;
        let out_w = (width + 2 * self.padding).saturating_sub(self.kernel_size) / self.stride + 1;

        let input_data = input.data();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();

        let mut output = Vec::with_capacity(out_h * out_w * self.out_channels);

        for out_c in 0..self.out_channels {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut sum = 0.0;
                    for in_c in 0..self.in_channels {
                        for ky in 0..self.kernel_size {
                            for kx in 0..self.kernel_size {
                                let in_y = out_y * self.stride + ky - self.padding;
                                let in_x = out_x * self.stride + kx - self.padding;
                                if in_y < height && in_x < width {
                                    let in_idx = (in_y * width + in_x) * channels + in_c;
                                    let w_idx = ((out_c * self.in_channels + in_c) * self.kernel_size + ky) * self.kernel_size + kx;
                                    if in_idx < input_data.len() && w_idx < weight_data.len() {
                                        sum += input_data[in_idx] * weight_data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                    sum += if out_c < bias_data.len() { bias_data[out_c] } else { 0.0 };
                    output.push(sum);
                }
            }
        }

        Ok(Tensor::new(output, vec![out_h, out_w, self.out_channels]))
    }
}

/// GPU-accelerated LayerNorm2D
pub struct LayerNorm2D {
    _num_channels: usize,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm2D {
    pub fn new(num_channels: usize) -> HLDVAResult<Self> {
        Ok(Self {
            _num_channels: num_channels,
            weight: Tensor::new(vec![1.0; num_channels], vec![num_channels]),
            bias: Tensor::new(vec![0.0; num_channels], vec![num_channels]),
            eps: 1e-6,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_layer_norm(input, &self.weight, &self.bias, self.eps);
        }
        self.forward_cpu(input)
    }

    fn forward_cpu(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() < 3 {
            return Err(HLDVAError::Model("Invalid input shape".to_string()));
        }
        let (height, width, channels) = (input_shape[0], input_shape[1], input_shape[2]);
        let input_data = input.data();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();

        let mut output = Vec::with_capacity(input_data.len());
        for h in 0..height {
            for w in 0..width {
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

                for c in 0..channels {
                    let idx = (h * width + w) * channels + c;
                    if idx < input_data.len() {
                        let normalized = (input_data[idx] - mean) / std;
                        let wv = if c < weight_data.len() { weight_data[c] } else { 1.0 };
                        let bv = if c < bias_data.len() { bias_data[c] } else { 0.0 };
                        output.push(normalized * wv + bv);
                    }
                }
            }
        }
        Ok(Tensor::new(output, input_shape.to_vec()))
    }
}

/// ReLU Activation — GPU accelerated
pub struct ReLU;

impl ReLU {
    pub fn forward(&self, input: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_relu(input);
        }
        let data = input.data();
        let activated: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        Ok(Tensor::new(activated, input.shape().to_vec()))
    }
}
