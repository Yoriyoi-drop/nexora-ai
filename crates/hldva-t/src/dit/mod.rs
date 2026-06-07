//! DiT (Diffusion Transformers) Implementation
//!
//! GPU-accelerated DiT backbone for noise prediction.

pub mod attention;
pub mod embedding;
pub mod transformer;

use crate::{config::DiTConfig, gpu_ops, types::*};
use nexora_atqs::Tensor;
use std::collections::HashMap;

/// Main DiT Model
pub struct DiTModel {
    config: DiTConfig,
    transformer_blocks: Vec<TransformerBlock>,
    patch_embedding: PatchEmbedding,
    time_embedding: TimeEmbedding,
    position_embedding: PositionEmbedding,
    final_layer_norm: LayerNorm,
    final_linear: Linear,
    conditioning_projection: Linear,
    parameters: HashMap<String, Tensor>,
}

impl DiTModel {
    pub fn new(config: &DiTConfig) -> HLDVAResult<Self> {
        let patch_embedding = PatchEmbedding::new(config.patch_size, config.hidden_dim)?;
        let time_embedding = TimeEmbedding::new(config.hidden_dim)?;
        let position_embedding = PositionEmbedding::new(config.max_seq_len, config.hidden_dim)?;

        let mut transformer_blocks = Vec::new();
        for i in 0..config.num_blocks {
            transformer_blocks.push(TransformerBlock::new(
                config.hidden_dim,
                config.num_heads,
                format!("dit_block_{}", i),
            )?);
        }

        let final_layer_norm = LayerNorm::new(config.hidden_dim)?;
        let final_linear = Linear::new(config.hidden_dim, 4)?;
        let conditioning_projection = Linear::new(512, config.hidden_dim)?;

        Ok(Self {
            config: config.clone(),
            transformer_blocks,
            patch_embedding,
            time_embedding,
            position_embedding,
            final_layer_norm,
            final_linear,
            conditioning_projection,
            parameters: HashMap::new(),
        })
    }

    pub fn predict_noise(
        &self,
        latent: &Tensor,
        timestep: Timestep,
        clip_embedding: &ClipEmbedding,
        _guidance_scale: f32,
    ) -> HLDVAResult<NoisePrediction> {
        let patches = self.patch_embedding.embed(latent)?;
        let time_emb = self.time_embedding.embed(timestep)?;
        let patches_with_time = self.add_time_embedding(&patches, &time_emb)?;
        let patches_with_pos = self.position_embedding.add_to(&patches_with_time)?;
        let conditioning = self.process_conditioning(clip_embedding)?;

        let mut hidden = patches_with_pos;
        for block in &self.transformer_blocks {
            hidden = block.forward(&hidden, &conditioning)?;
        }

        let hidden_norm = self.final_layer_norm.forward(&hidden)?;
        let noise_pred = self.final_linear.forward(&hidden_norm)?;
        let predicted_noise = self.reshape_to_latent(&noise_pred, latent.shape())?;

        Ok(NoisePrediction::new(predicted_noise, timestep))
    }

    pub fn predict_with_guidance(
        &self,
        latent: &Tensor,
        timestep: Timestep,
        clip_embedding: &ClipEmbedding,
        uncond_embedding: &ClipEmbedding,
        guidance_scale: f32,
    ) -> HLDVAResult<NoisePrediction> {
        let cond_pred = self.predict_noise(latent, timestep, clip_embedding, 1.0)?;
        let uncond_pred = self.predict_noise(latent, timestep, uncond_embedding, 1.0)?;
        let guided_noise = self.apply_guidance(
            &cond_pred.predicted_noise,
            &uncond_pred.predicted_noise,
            guidance_scale,
        )?;
        Ok(NoisePrediction::new(guided_noise, timestep))
    }

    fn add_time_embedding(&self, patches: &Tensor, time_emb: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_add(patches, time_emb);
        }
        let patches_data = patches.data();
        let time_data = time_emb.data();
        let mut result = patches_data.to_vec();
        for i in 0..result.len() {
            result[i] += time_data[i % time_data.len()];
        }
        Ok(Tensor::new(result, patches.shape().to_vec()))
    }

    fn process_conditioning(&self, clip_embedding: &ClipEmbedding) -> HLDVAResult<Tensor> {
        self.conditioning_projection.forward(&clip_embedding.text_features)
    }

    fn apply_guidance(&self, cond_noise: &Tensor, uncond_noise: &Tensor, guidance_scale: f32) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            // guided = uncond + scale * (cond - uncond)
            let diff = gpu_ops::gpu_sub(cond_noise, uncond_noise)?;
            let scaled = gpu_ops::gpu_scale(&diff, guidance_scale)?;
            return gpu_ops::gpu_add(uncond_noise, &scaled);
        }
        let cond_data = cond_noise.data();
        let uncond_data = uncond_noise.data();
        let mut guided = Vec::with_capacity(cond_data.len());
        for i in 0..cond_data.len() {
            guided.push(uncond_data[i] + guidance_scale * (cond_data[i] - uncond_data[i]));
        }
        Ok(Tensor::new(guided, cond_noise.shape().to_vec()))
    }

    fn reshape_to_latent(&self, noise_pred: &Tensor, latent_shape: &[usize]) -> HLDVAResult<Tensor> {
        let pred_data = noise_pred.data();
        let total_elements = latent_shape.iter().product::<usize>();
        let mut reshaped = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            reshaped.push(pred_data[i % pred_data.len()]);
        }
        Ok(Tensor::new(reshaped, latent_shape.to_vec()))
    }

    pub fn parameters(&self) -> &HashMap<String, Tensor> { &self.parameters }

    pub fn collect_parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for block in &self.transformer_blocks {
            params.push(block.self_attention.q_projection().weight().clone());
            params.push(block.self_attention.k_projection().weight().clone());
            params.push(block.self_attention.v_projection().weight().clone());
            params.push(block.self_attention.out_projection().weight().clone());
            params.push(block.cross_attention.q_projection().weight().clone());
            params.push(block.cross_attention.k_projection().weight().clone());
            params.push(block.cross_attention.v_projection().weight().clone());
            params.push(block.cross_attention.out_projection().weight().clone());
            params.push(block.feed_forward.linear1().weight().clone());
            params.push(block.feed_forward.linear2().weight().clone());
        }
        params
    }

    pub fn config(&self) -> &DiTConfig { &self.config }
}

/// Transformer Block with residual connections — GPU accelerated
pub struct TransformerBlock {
    hidden_dim: usize,
    num_heads: usize,
    self_attention: MultiHeadAttention,
    self_attention_norm: LayerNorm,
    cross_attention: MultiHeadAttention,
    cross_attention_norm: LayerNorm,
    feed_forward: FeedForward,
    feed_forward_norm: LayerNorm,
    name: String,
}

impl TransformerBlock {
    pub fn new(hidden_dim: usize, num_heads: usize, name: String) -> HLDVAResult<Self> {
        Ok(Self {
            hidden_dim,
            num_heads,
            self_attention: MultiHeadAttention::new(hidden_dim, num_heads)?,
            self_attention_norm: LayerNorm::new(hidden_dim)?,
            cross_attention: MultiHeadAttention::new(hidden_dim, num_heads)?,
            cross_attention_norm: LayerNorm::new(hidden_dim)?,
            feed_forward: FeedForward::new(hidden_dim)?,
            feed_forward_norm: LayerNorm::new(hidden_dim)?,
            name,
        })
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
        if gpu_ops::gpu_available() {
            return gpu_ops::gpu_add(a, b);
        }
        let a_data = a.data();
        let b_data = b.data();
        let mut sum = Vec::with_capacity(a_data.len().max(b_data.len()));
        for i in 0..a_data.len().max(b_data.len()) {
            let av = if i < a_data.len() { a_data[i] } else { 0.0 };
            let bv = if i < b_data.len() { b_data[i] } else { 0.0 };
            sum.push(av + bv);
        }
        let shape = if a.shape().iter().product::<usize>() >= b.shape().iter().product::<usize>() {
            a.shape().to_vec()
        } else {
            b.shape().to_vec()
        };
        Ok(Tensor::new(sum, shape))
    }

    pub fn forward(&self, hidden: &Tensor, conditioning: &Tensor) -> HLDVAResult<Tensor> {
        let self_attn_out = self.self_attention.forward(hidden, hidden, hidden)?;
        let added = self.add_tensors(hidden, &self_attn_out)?;
        let self_attn_norm = self.self_attention_norm.forward(&added)?;

        let cross_attn_out = self.cross_attention.forward(&self_attn_norm, conditioning, conditioning)?;
        let added2 = self.add_tensors(&self_attn_norm, &cross_attn_out)?;
        let cross_attn_norm = self.cross_attention_norm.forward(&added2)?;

        let ff_out = self.feed_forward.forward(&cross_attn_norm)?;
        let added3 = self.add_tensors(&cross_attn_norm, &ff_out)?;
        let output = self.feed_forward_norm.forward(&added3)?;

        Ok(output)
    }
}

pub use attention::*;
pub use embedding::*;
pub use transformer::*;
