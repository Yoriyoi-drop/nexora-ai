//! Alignment module for CLIP

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Text-Image Alignment for CLIP
pub struct ClipAlignment {
    config: ClipAlignmentConfig,
}

impl ClipAlignment {
    /// Create new alignment module
    pub fn new(config: ClipAlignmentConfig) -> HLDVAResult<Self> {
        Ok(Self { config })
    }
    
    /// Compute alignment score between text and image embeddings
    pub fn compute_alignment(&self, text_embedding: &Tensor, image_embedding: &Tensor) -> HLDVAResult<f32> {
        let text_data = text_embedding.data();
        let image_data = image_embedding.data();
        
        if text_data.len() != image_data.len() {
            return Err(HLDVAError::Model("Embedding dimensions must match".to_string()));
        }
        
        // Compute cosine similarity
        let mut dot_product = 0.0;
        let mut text_norm = 0.0;
        let mut image_norm = 0.0;
        
        for (t, i) in text_data.iter().zip(image_data.iter()) {
            dot_product += t * i;
            text_norm += t * t;
            image_norm += i * i;
        }
        
        text_norm = text_norm.sqrt();
        image_norm = image_norm.sqrt();
        
        if text_norm > 0.0 && image_norm > 0.0 {
            Ok(dot_product / (text_norm * image_norm))
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute alignment scores for batch
    pub fn compute_batch_alignment(&self, text_embeddings: &[Tensor], image_embeddings: &[Tensor]) -> HLDVAResult<Vec<f32>> {
        if text_embeddings.len() != image_embeddings.len() {
            return Err(HLDVAError::Model("Batch sizes must match".to_string()));
        }
        
        let mut scores = Vec::new();
        
        for (text_emb, image_emb) in text_embeddings.iter().zip(image_embeddings.iter()) {
            let score = self.compute_alignment(text_emb, image_emb)?;
            scores.push(score);
        }
        
        Ok(scores)
    }
    
    /// Find best matching image for text
    pub fn find_best_match(&self, text_embedding: &Tensor, image_embeddings: &[Tensor]) -> HLDVAResult<(usize, f32)> {
        let mut best_idx = 0;
        let mut best_score = 0.0;
        
        for (idx, image_emb) in image_embeddings.iter().enumerate() {
            let score = self.compute_alignment(text_embedding, image_emb)?;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        
        Ok((best_idx, best_score))
    }
    
    /// Apply contrastive learning loss
    pub fn contrastive_loss(&self, text_embeddings: &[Tensor], image_embeddings: &[Tensor]) -> HLDVAResult<f32> {
        let mut total_loss = 0.0;
        let temperature = self.config.temperature;
        
        for i in 0..text_embeddings.len() {
            // Positive pair
            let pos_score = self.compute_alignment(&text_embeddings[i], &image_embeddings[i])?;
            let pos_logit = pos_score / temperature;
            
            // Negative pairs
            let mut sum_exp_logits = 0.0;
            for j in 0..image_embeddings.len() {
                if i != j {
                    let neg_score = self.compute_alignment(&text_embeddings[i], &image_embeddings[j])?;
                    let neg_logit = neg_score / temperature;
                    sum_exp_logits += neg_logit.exp();
                }
            }
            
            // Cross-entropy loss
            let loss = -pos_logit + (pos_logit.exp() + sum_exp_logits).ln();
            total_loss += loss;
        }
        
        Ok(total_loss / text_embeddings.len() as f32)
    }
}

/// CLIP Alignment configuration
#[derive(Debug, Clone)]
pub struct ClipAlignmentConfig {
    pub temperature: f32,
    pub max_similarity: f32,
}

impl Default for ClipAlignmentConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            max_similarity: 1.0,
        }
    }
}
