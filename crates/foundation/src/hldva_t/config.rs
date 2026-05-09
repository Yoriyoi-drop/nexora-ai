//! Konfigurasi HLDVA-T
//!
//! Struktur konfigurasi untuk semua komponen HLDVA-T

use serde::{Deserialize, Serialize};

/// Konfigurasi utama HLDVA-T
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HLDVAConfig {
    /// Konfigurasi DiT backbone
    pub dit: DiTConfig,
    
    /// Konfigurasi Cascaded Diffusion Models
    pub cascaded: CascadedConfig,
    
    /// Konfigurasi CLIP conditioning
    pub clip: ClipConfig,
    
    /// Konfigurasi VAE
    pub vae: VAEConfig,
    
    /// Konfigurasi DDPM
    pub ddpm: DDPMConfig,
    
    /// Konfigurasi training
    pub training: TrainingConfig,
}

impl Default for HLDVAConfig {
    fn default() -> Self {
        Self {
            dit: DiTConfig::default(),
            cascaded: CascadedConfig::default(),
            clip: ClipConfig::default(),
            vae: VAEConfig::default(),
            ddpm: DDPMConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

/// Konfigurasi DiT (Diffusion Transformers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiTConfig {
    /// Model size: base, small, large
    pub model_size: DiTModelSize,
    
    /// Jumlah blok transformer
    pub num_blocks: usize,
    
    /// Hidden dimension
    pub hidden_dim: usize,
    
    /// Jumlah attention heads
    pub num_heads: usize,
    
    /// Patch size untuk latent
    pub patch_size: usize,
    
    /// Max sequence length
    pub max_seq_len: usize,
    
    /// Dropout rate
    pub dropout: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiTModelSize {
    Small,
    Base,
    Large,
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            model_size: DiTModelSize::Base,
            num_blocks: 12,
            hidden_dim: 768,
            num_heads: 12,
            patch_size: 2,
            max_seq_len: 256,
            dropout: 0.1,
        }
    }
}

/// Konfigurasi Cascaded Diffusion Models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadedConfig {
    /// Jumlah tahap upsampling
    pub num_stages: usize,
    
    /// Resolusi setiap tahap
    pub resolutions: Vec<(usize, usize)>, // (latent_res, image_res)
    
    /// Konfigurasi upsampler
    pub upsamplers: Vec<UpsamplerConfig>,
    
    /// Noise conditioning augmentation
    pub noise_conditioning: bool,
    
    /// Noise level untuk conditioning
    pub noise_levels: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsamplerConfig {
    /// Resolusi input
    pub input_res: usize,
    
    /// Resolusi output
    pub output_res: usize,
    
    /// Jumlah blok DiT
    pub num_blocks: usize,
    
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl Default for CascadedConfig {
    fn default() -> Self {
        Self {
            num_stages: 2,
            resolutions: vec![(16, 256), (32, 1024)],
            upsamplers: vec![
                UpsamplerConfig {
                    input_res: 8,
                    output_res: 16,
                    num_blocks: 6,
                    hidden_dim: 384,
                },
                UpsamplerConfig {
                    input_res: 16,
                    output_res: 32,
                    num_blocks: 6,
                    hidden_dim: 384,
                },
            ],
            noise_conditioning: true,
            noise_levels: vec![0.1, 0.05],
        }
    }
}

/// Konfigurasi CLIP Conditioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipConfig {
    /// Model CLIP yang digunakan
    pub model_name: String,
    
    /// Maximum token length
    pub max_length: usize,
    
    /// Dimension dari CLIP embedding
    pub embedding_dim: usize,
    
    /// Cross-attention dropout
    pub cross_attention_dropout: f32,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            model_name: "openai/clip-vit-large-patch14".to_string(),
            max_length: 77,
            embedding_dim: 768,
            cross_attention_dropout: 0.1,
        }
    }
}

/// Konfigurasi VAE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAEConfig {
    /// Latent dimension
    pub latent_dim: usize,
    
    /// Compression factor (faktor downsampling)
    pub compression_factor: usize,
    
    /// KL divergence weight
    pub kl_weight: f32,
    
    /// Reconstruction loss weight
    pub recon_weight: f32,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            latent_dim: 4,
            compression_factor: 8,
            kl_weight: 0.00025,
            recon_weight: 1.0,
        }
    }
}

/// Konfigurasi DDPM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDPMConfig {
    /// Jumlah timestep
    pub num_timesteps: usize,
    
    /// Noise schedule type
    pub schedule_type: NoiseScheduleType,
    
    /// Beta start
    pub beta_start: f32,
    
    /// Beta end
    pub beta_end: f32,
    
    /// Cosine schedule parameter
    pub cosine_s: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseScheduleType {
    Linear,
    Cosine,
    ScaledLinear,
}

impl Default for DDPMConfig {
    fn default() -> Self {
        Self {
            num_timesteps: 1000,
            schedule_type: NoiseScheduleType::Cosine,
            beta_start: 0.0001,
            beta_end: 0.02,
            cosine_s: 0.008,
        }
    }
}

/// Konfigurasi Training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Jumlah epoch
    pub num_epochs: usize,
    
    /// Weight decay
    pub weight_decay: f32,
    
    /// Gradient clipping
    pub grad_clip_norm: f32,
    
    /// Warmup steps
    pub warmup_steps: usize,
    
    /// Checkpoint interval
    pub checkpoint_interval: usize,
    
    /// Evaluation interval
    pub eval_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 256,
            num_epochs: 100,
            weight_decay: 0.01,
            grad_clip_norm: 1.0,
            warmup_steps: 1000,
            checkpoint_interval: 5000,
            eval_interval: 1000,
        }
    }
}
