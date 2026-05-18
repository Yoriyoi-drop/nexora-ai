//! Multi-modal encoders for CAFFEINE
//! 
//! Implements Stage 1: Multi-Scale Contrastive Encoding
//! - Regional Contrastive Visual Encoder (dari CLIP)
//! - Audio encoder (Whisper-based)
//! - Video encoder (3D Vision Transformer)
//! - Text encoder (BERT-based)

pub mod image_encoder;
pub mod audio_encoder;
pub mod video_encoder;
pub mod text_encoder;
pub mod regional_alignment;

pub use image_encoder::*;
pub use audio_encoder::*;
pub use video_encoder::*;
pub use text_encoder::*;
pub use regional_alignment::*;

use crate::caffeine::types::*;
use crate::caffeine::error::Result;
use ndarray::ArrayD;

/// Multi-modal encoder collection
pub struct MultiModalEncoders {
    image_encoder: ImageEncoder,
    audio_encoder: AudioEncoder,
    video_encoder: VideoEncoder,
    text_encoder: TextEncoder,
    regional_aligner: RegionalAlignment,
    config: crate::caffeine::config::EncodersConfig,
}

impl MultiModalEncoders {
    /// Create new multi-modal encoders
    pub fn new(config: crate::caffeine::config::EncodersConfig) -> Result<Self> {
        let image_encoder = ImageEncoder::new(config.image_encoder.clone())?;
        let audio_encoder = AudioEncoder::new(config.audio_encoder.clone())?;
        let video_encoder = VideoEncoder::new(config.video_encoder.clone())?;
        let text_encoder = TextEncoder::new(config.text_encoder.clone())?;
        let regional_aligner = RegionalAlignment::new(config.num_regional_patches)?;
        
        Ok(Self {
            image_encoder,
            audio_encoder,
            video_encoder,
            text_encoder,
            regional_aligner,
            config,
        })
    }
    
    /// Encode multi-modal inputs
    pub fn encode(&mut self, inputs: &MultiModalInputs) -> Result<EncodedFeatures> {
        let mut image_features = None;
        let mut audio_features = None;
        let mut video_features = None;
        let mut text_features = None;
        let mut regional_features = None;
        
        // Encode image if present
        if let Some(ref image_input) = inputs.image {
            let features = self.image_encoder.encode(image_input)?;
            image_features = Some(features.clone());
            
            // Extract regional features if enabled
            if self.config.enable_regional_alignment {
                let regional = self.regional_aligner.extract_regional_features(&features, image_input)?;
                regional_features = Some(regional);
            }
        }
        
        // Encode audio if present
        if let Some(ref audio_input) = inputs.audio {
            let features = self.audio_encoder.encode(audio_input)?;
            audio_features = Some(features);
        }
        
        // Encode video if present
        if let Some(ref video_input) = inputs.video {
            let features = self.video_encoder.encode(video_input)?;
            video_features = Some(features);
        }
        
        // Encode text if present
        if let Some(ref text_input) = inputs.text {
            let features = self.text_encoder.encode(text_input)?;
            text_features = Some(features);
        }
        
        Ok(EncodedFeatures {
            image_features,
            audio_features,
            video_features,
            text_features,
            regional_features,
        })
    }
    
    /// Get encoder statistics
    pub fn get_stats(&self) -> EncoderStats {
        EncoderStats {
            image_encoder_loaded: self.image_encoder.is_loaded(),
            audio_encoder_loaded: self.audio_encoder.is_loaded(),
            video_encoder_loaded: self.video_encoder.is_loaded(),
            text_encoder_loaded: self.text_encoder.is_loaded(),
            regional_alignment_enabled: self.config.enable_regional_alignment,
            num_regional_patches: self.config.num_regional_patches,
        }
    }
}

/// Encoder statistics
#[derive(Debug, Clone)]
pub struct EncoderStats {
    pub image_encoder_loaded: bool,
    pub audio_encoder_loaded: bool,
    pub video_encoder_loaded: bool,
    pub text_encoder_loaded: bool,
    pub regional_alignment_enabled: bool,
    pub num_regional_patches: usize,
}
