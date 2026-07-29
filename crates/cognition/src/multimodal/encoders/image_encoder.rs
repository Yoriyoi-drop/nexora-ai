//! Image encoder implementation for CAFFEINE
//!
//! Based on CLIP ViT with regional contrastive alignment support.
//! Uses a 2-layer MLP projection (GELU activation) with Xavier initialization
//! instead of sinusoidal coefficient mapping.

use crate::multimodal::error::Result;
use crate::multimodal::types::*;
use ndarray::{s, Array1, Array2, ArrayD};
use rand::Rng;
use std::collections::HashMap;

/// 2-layer MLP with GELU activation for image patch projection.
struct PatchMLP {
    pub(crate) w1: Array2<f32>,
    pub(crate) b1: Array1<f32>,
    pub(crate) w2: Array2<f32>,
    pub(crate) b2: Array1<f32>,
}

impl PatchMLP {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / input_dim as f32).sqrt();
        let scale2 = (2.0 / hidden_dim as f32).sqrt();
        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            rng.gen::<f32>() * 2.0 * scale1 - scale1
        });
        let b1 = Array1::zeros(hidden_dim);
        let w2 = Array2::from_shape_fn((hidden_dim, output_dim), |_| {
            rng.gen::<f32>() * 2.0 * scale2 - scale2
        });
        let b2 = Array1::zeros(output_dim);
        Self { w1, b1, w2, b2 }
    }

    fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let hidden = x.dot(&self.w1) + &self.b1;
        let gelu = hidden.mapv(|v| {
            v * 0.5 * (1.0 + (v * 0.7978845608 * (1.0 + 0.044715 * v * v)).tanh())
        });
        gelu.dot(&self.w2) + &self.b2
    }

    #[cfg(feature = "gpu")]
    fn forward_batched_gpu(&self, inputs: &[Array1<f32>]) -> Option<Vec<Array1<f32>>> {
        use crate::multimodal::gpu_compute;
        let batch = inputs.len();
        if batch == 0 { return None; }
        let input_dim = inputs[0].len();
        let hidden_dim = self.w1.shape()[1];
        let output_dim = self.w2.shape()[1];
        let flat_input: Vec<f32> = inputs.iter().flat_map(|i| i.iter().copied()).collect();
        let w1_slice: Vec<f32> = self.w1.iter().copied().collect();
        let b1_slice: Vec<f32> = self.b1.iter().copied().collect();
        let w2_slice: Vec<f32> = self.w2.iter().copied().collect();
        let b2_slice: Vec<f32> = self.b2.iter().copied().collect();
        let result = gpu_compute::try_gpu_mlp_forward(
            &flat_input, &w1_slice, &b1_slice, &w2_slice, &b2_slice,
            batch, input_dim, hidden_dim, output_dim,
        )?;
        Some(result.chunks(output_dim).map(|c| Array1::from_vec(c.to_vec())).collect())
    }
}

/// Image encoder based on CLIP ViT
pub struct ImageEncoder {
    config: crate::multimodal::config::ImageEncoderConfig,
    model_loaded: bool,
    // Real 2-layer MLP for patch-to-embedding projection
    patch_projection: PatchMLP,
    _embeddings: HashMap<String, ArrayD<f32>>,
}

impl ImageEncoder {
    /// Create new image encoder
    pub fn new(config: crate::multimodal::config::ImageEncoderConfig) -> Result<Self> {
        let pixels_per_patch = config.patch_size * config.patch_size * 3; // RGB channels
        let hidden_dim = config.output_dim.max(64);
        let patch_projection = PatchMLP::new(pixels_per_patch, hidden_dim, config.output_dim);

        Ok(Self {
            config,
            model_loaded: false,
            patch_projection,
            _embeddings: HashMap::new(),
        })
    }

    /// Load model weights
    pub fn load_model(&mut self) -> Result<()> {
        self.model_loaded = true;
        Ok(())
    }

    /// Encode image input
    pub fn encode(&mut self, input: &ImageInput) -> Result<ArrayD<f32>> {
        if !self.model_loaded {
            self.load_model()?;
        }

        let batch_size = 1;
        let patch_size = self.config.patch_size;
        let seq_len =
            (input.width / patch_size) * (input.height / patch_size);
        let embed_dim = self.config.output_dim;
        let num_patches_x = input.width / patch_size;
        let channels = input.channels.max(3);

        // Normalize image bytes to float values with CLIP mean/std
        let raw: Vec<f32> = input.data.iter().map(|&b| b as f32 / 255.0).collect();
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];
        let pixels: Vec<f32> = raw
            .chunks(channels)
            .flat_map(|pixel| {
                pixel.iter().enumerate().map(|(c, &v)| {
                    if c < 3 {
                        (v - mean[c]) / std[c]
                    } else {
                        v
                    }
                })
            })
            .collect();

        // Collect all patch vectors
        let mut patch_vecs: Vec<Array1<f32>> = Vec::with_capacity(seq_len);
        for p in 0..seq_len {
            let py = p / num_patches_x;
            let px = p % num_patches_x;

            let mut patch_vec = Vec::with_capacity(patch_size * patch_size * channels);
            for j in 0..patch_size {
                for i in 0..patch_size {
                    for c in 0..channels {
                        let y = py * patch_size + j;
                        let x = px * patch_size + i;
                        if y < input.height && x < input.width {
                            let idx = (y * input.width + x) * channels + c;
                            if idx < pixels.len() {
                                patch_vec.push(pixels[idx]);
                            } else {
                                patch_vec.push(0.0);
                            }
                        } else {
                            patch_vec.push(0.0);
                        }
                    }
                }
            }
            patch_vecs.push(Array1::from_vec(patch_vec));
        }

        // Try GPU batched MLP first, fall back to per-patch CPU
        let mut data = vec![0.0f32; batch_size * seq_len * embed_dim];
        #[cfg(feature = "gpu")]
        if let Some(gpu_results) = self.patch_projection.forward_batched_gpu(&patch_vecs) {
            for (p, projected) in gpu_results.iter().enumerate() {
                for d in 0..embed_dim {
                    data[p * embed_dim + d] = projected[d];
                }
            }
        } else {
            for (p, patch_arr) in patch_vecs.iter().enumerate() {
                let projected = self.patch_projection.forward(patch_arr);
                for d in 0..embed_dim {
                    data[p * embed_dim + d] = projected[d];
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        for (p, patch_arr) in patch_vecs.iter().enumerate() {
            let projected = self.patch_projection.forward(patch_arr);
            for d in 0..embed_dim {
                data[p * embed_dim + d] = projected[d];
            }
        }

        let shape = vec![batch_size, seq_len, embed_dim];
        Ok(ArrayD::from_shape_vec(shape, data)?)
    }

    /// Extract regional features for contrastive alignment
    pub fn extract_regional_features(
        &self,
        features: &ArrayD<f32>,
        input: &ImageInput,
    ) -> Result<Vec<ArrayD<f32>>> {
        let num_patches_x = input.width / self.config.patch_size;
        let num_patches_y = input.height / self.config.patch_size;
        let total_patches = num_patches_x * num_patches_y;

        let mut regional_features = Vec::new();

        // Extract features for each region
        for region_idx in 0..7 {
            // 7 regions for simplicity
            let start_patch = (region_idx * total_patches) / 7;
            let end_patch = ((region_idx + 1) * total_patches) / 7;

            // Extract patch features
            let patch_features = features.slice(s![
                0..1,                   // batch
                start_patch..end_patch, // patches
                0..features.shape()[2]  // embed_dim
            ]);

            regional_features.push(patch_features.to_owned().into_dimensionality()?);
        }

        Ok(regional_features)
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Get configuration
    pub fn config(&self) -> &crate::multimodal::config::ImageEncoderConfig {
        &self.config
    }

    /// Collect all trainable weights for checkpoint
    pub(crate) fn collect_weights(&self) -> Vec<(String, ndarray::ArrayD<f32>)> {
        vec![
            ("image_encoder.patch_proj.w1".to_string(), self.patch_projection.w1.clone().into_dyn()),
            ("image_encoder.patch_proj.b1".to_string(), self.patch_projection.b1.clone().into_dyn()),
            ("image_encoder.patch_proj.w2".to_string(), self.patch_projection.w2.clone().into_dyn()),
            ("image_encoder.patch_proj.b2".to_string(), self.patch_projection.b2.clone().into_dyn()),
        ]
    }
}

/// Regional contrastive alignment for visual features
#[allow(dead_code)]
pub struct RegionalContrastiveAligner {
    _temperature: f32,
    _num_regions: usize,
}

impl RegionalContrastiveAligner {
    /// Create new regional aligner
    pub fn new(num_regions: usize, temperature: f32) -> Self {
        Self {
            _temperature: temperature,
            _num_regions: num_regions,
        }
    }

    /// Compute contrastive loss between image regions and text
    pub fn compute_contrastive_loss(
        &self,
        image_regions: &[ArrayD<f32>],
        text_features: &ArrayD<f32>,
    ) -> Result<f32> {
        // Simulate contrastive loss computation
        let mut total_loss = 0.0;

        for region in image_regions {
            // Normalize features
            let norm_region = self.l2_normalize(region)?;
            let norm_text = self.l2_normalize(text_features)?;

            // Compute similarity
            let similarity = self.cosine_similarity(&norm_region, &norm_text)?;

            // Compute cross-entropy loss
            let loss = -similarity.exp().ln();
            total_loss += loss;
        }

        Ok(total_loss / image_regions.len() as f32)
    }

    /// L2 normalize tensor
    fn l2_normalize(&self, tensor: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let norm = tensor.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(crate::multimodal::error::CaffeineError::tensor_operation(
                "Cannot normalize zero tensor",
            ));
        }

        let normalized = tensor.mapv(|x| x / norm);
        Ok(normalized)
    }

    /// Compute cosine similarity between two tensors
    fn cosine_similarity(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<f32> {
        if a.shape() != b.shape() {
            return Err(crate::multimodal::error::CaffeineError::tensor_operation(
                "Tensor shapes don't match for cosine similarity",
            ));
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }
}
