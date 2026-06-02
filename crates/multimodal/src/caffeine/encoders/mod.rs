//! Multi-modal encoders for CAFFEINE
//!
//! Implements Stage 1: Multi-Scale Contrastive Encoding
//! - Regional Contrastive Visual Encoder (dari CLIP)
//! - Audio encoder (Whisper-based)
//! - Video encoder (3D Vision Transformer)
//! - Text encoder (BERT-based)

pub mod audio_encoder;
pub mod image_encoder;
pub mod regional_alignment;
pub mod text_encoder;
pub mod video_encoder;

pub use audio_encoder::*;
pub use image_encoder::*;
pub use regional_alignment::*;
pub use text_encoder::*;
pub use video_encoder::*;

use crate::caffeine::cache::{CacheConfig, MultiModalCache};
use crate::caffeine::error::Result;
use crate::caffeine::types::*;
use std::sync::{Arc, Mutex};

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

    /// Original encode method (no cache — backward compat)
    pub fn encode(&mut self, inputs: &MultiModalInputs) -> Result<EncodedFeatures> {
        self.encode_inner(inputs, &None)
    }

    /// Encode with cache support (#4 Multimodal Cache)
    pub fn encode_with_cache(
        &mut self,
        inputs: &MultiModalInputs,
        cache: &Option<Arc<Mutex<MultiModalCache>>>,
        _cache_config: &CacheConfig,
    ) -> Result<EncodedFeatures> {
        self.encode_inner(inputs, cache)
    }

    /// Internal encode with optional cache
    fn encode_inner(
        &mut self,
        inputs: &MultiModalInputs,
        cache: &Option<Arc<Mutex<MultiModalCache>>>,
    ) -> Result<EncodedFeatures> {
        let mut image_features = None;
        let mut audio_features = None;
        let mut video_features = None;
        let mut text_features = None;
        let mut regional_features = None;

        // Encode image if present (#4 FIX 2: dedup, FIX 8: adaptive resolution)
        if let Some(ref image_input) = inputs.image {
            let features = self.encode_image_with_cache(image_input, cache)?;
            image_features = Some(features.clone());

            // Extract regional features if enabled
            if self.config.enable_regional_alignment {
                let regional = self
                    .regional_aligner
                    .extract_regional_features(&features, image_input)?;
                regional_features = Some(regional);
            }
        }

        // Encode audio if present (#4 FIX 2: dedup)
        if let Some(ref audio_input) = inputs.audio {
            let features = self.encode_audio_with_cache(audio_input, cache)?;
            audio_features = Some(features);
        }

        // Encode video if present (#4 FIX 2: dedup, FIX 5: streaming)
        if let Some(ref video_input) = inputs.video {
            let features = self.encode_video_with_cache(video_input, cache)?;
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

    /// Encode image with cache (#4 FIX 1, 2, 8)
    fn encode_image_with_cache(
        &mut self,
        input: &ImageInput,
        cache: &Option<Arc<Mutex<MultiModalCache>>>,
    ) -> Result<ndarray::ArrayD<f32>> {
        if let Some(cache_arc) = cache {
            if let Ok(mut cache_guard) = cache_arc.lock() {
                let hash = MultiModalCache::hash_image_input(input);
                // Estimate output shape: ViT typically outputs [1, num_patches+1, 768].
                // Never use raw input dims — they produce 48T+ element counts.
                let estimated_patches = (input.width / 16).max(1) * (input.height / 16).max(1) + 1;
                let shape = vec![1, estimated_patches, 768];

                let (compressed, _hit) = cache_guard.get_or_compute(
                    hash,
                    ModalityType::Image,
                    shape,
                    || {
                        // FIX 1: Hanya simpan hasil akhir encoding, bukan feature map
                        self.image_encoder.encode(input).unwrap_or_else(|_| {
                            ndarray::ArrayD::from_shape_vec(
                                vec![1, 1, 1],
                                vec![0.0f32],
                            )
                            .unwrap()
                        })
                        .iter()
                        .copied()
                        .collect()
                    },
                );

                let recovered = compressed.to_fp32();
                return ndarray::ArrayD::from_shape_vec(
                    vec![1, recovered.len() / 768, 768],
                    recovered,
                )
                .map_err(|e| crate::caffeine::error::CaffeineError::NdArray(e));
            }
        }

        // No cache — encode directly (FIX 1: hasil akhir)
        self.image_encoder.encode(input)
    }

    /// Encode audio with cache (#4 FIX 1, 2)
    fn encode_audio_with_cache(
        &mut self,
        input: &AudioInput,
        cache: &Option<Arc<Mutex<MultiModalCache>>>,
    ) -> Result<ndarray::ArrayD<f32>> {
        if let Some(cache_arc) = cache {
            if let Ok(mut cache_guard) = cache_arc.lock() {
                let hash = MultiModalCache::hash_audio_input(input);
                // Audio encoder output estimate: [1, seq_len, 768]
                let estimated_audio_seq = (input.data.len() / 320).max(1).min(3000);
                let shape = vec![1, estimated_audio_seq, 768];

                let (compressed, _hit) = cache_guard.get_or_compute(
                    hash,
                    ModalityType::Audio,
                    shape,
                    || {
                        self.audio_encoder.encode(input).unwrap_or_else(|_| {
                            ndarray::ArrayD::from_shape_vec(
                                vec![1, 1, 1],
                                vec![0.0f32],
                            )
                            .unwrap()
                        })
                        .iter()
                        .copied()
                        .collect()
                    },
                );

                let recovered = compressed.to_fp32();
                // Estimate shape: [1, seq_len, embed_dim]
                let seq_len = recovered.len() / 768;
                return ndarray::ArrayD::from_shape_vec(
                    vec![1, seq_len.max(1), 768],
                    recovered,
                )
                .map_err(|e| crate::caffeine::error::CaffeineError::NdArray(e));
            }
        }

        self.audio_encoder.encode(input)
    }

    /// Encode video with cache (#4 FIX 1, 2, 5)
    fn encode_video_with_cache(
        &mut self,
        input: &VideoInput,
        cache: &Option<Arc<Mutex<MultiModalCache>>>,
    ) -> Result<ndarray::ArrayD<f32>> {
        if let Some(cache_arc) = cache {
            if let Ok(mut cache_guard) = cache_arc.lock() {
                let hash = MultiModalCache::hash_video_input(input);
                let shape = vec![1, input.frames.len(), 768];

                let (compressed, _hit) = cache_guard.get_or_compute(
                    hash,
                    ModalityType::Video,
                    shape,
                    || {
                        // FIX 5: Sliding window — process per-frame, buang setelah diproses
                        let mut all_embs = Vec::new();
                        for frame in &input.frames {
                            if let Ok(features) = self.video_encoder.encode_frame(frame) {
                                all_embs.extend(features.iter().copied());
                            }
                        }
                        if all_embs.is_empty() {
                            vec![0.0f32; 768]
                        } else {
                            all_embs
                        }
                    },
                );

                let recovered = compressed.to_fp32();
                let total_len = recovered.len();
                let seq_len = (total_len / 768).max(1);
                return ndarray::ArrayD::from_shape_vec(
                    vec![1, seq_len, 768],
                    recovered,
                )
                .map_err(|e| crate::caffeine::error::CaffeineError::NdArray(e));
            }
        }

        self.video_encoder.encode(input)
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
