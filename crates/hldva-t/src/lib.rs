//! HLDVA-T: Hierarchical Latent Diffusion with Vision-Text Alignment using Transformers
//!
//! Kerangka generasi gambar terpadu yang mengintegrasikan DDPM, LDM, CLIP, DiT,
//! Cascaded Diffusion, dan VAE ke dalam satu pipeline kohesif dengan strategi
//! pelatihan bertahap serta evaluasi kuantitatif.
//!
//! ## Komponen Utama:
//! - DiT (Diffusion Transformers) - Backbone denoising
//! - CDM (Cascaded Diffusion Models) - Upsampling bertahap
//! - CLIP Conditioning - Vision-text alignment
//! - VAE - Latent compression/decompression
//! - DDPM - Noise schedule dan sampling
//!
//! ## Pipeline Inference:
//! 1. Text prompt → CLIP encoder → conditioning vector
//! 2. Initialize latent noise → DiT denoising loop
//! 3. Cascaded upsampling (64→256→1024)
//! 4. VAE decoder → final high-resolution image

pub mod cascaded;
pub mod clip;
pub mod config;
pub mod ddpm;
pub mod dit;
pub mod evaluation;
pub mod gpu_ops;
pub mod training;
pub mod types;
pub mod vaed;

// Include core module
#[path = "core.rs"]
pub mod core;

// Re-export main components
pub use cascaded::*;
pub use clip::*;
pub use config::*;
pub use core::*;
pub use ddpm::*;
pub use dit::*;
pub use evaluation::*;
pub use training::*;
pub use types::*;
pub use vaed::*;

// Re-export common types
pub use types::HLDVAResult;
