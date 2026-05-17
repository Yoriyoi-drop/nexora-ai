//! Evaluation Metrics for HLDVA-T
//!
//! Implementasi metrik evaluasi untuk mengukur kualitas generasi:
//! - FID (Fréchet Inception Distance)
//! - IS (Inception Score)  
//! - CLIP Score
//! - Precision/Recall
//! - LPIPS (Learned Perceptual Image Patch Similarity)

pub mod fid;
pub mod inception_score;
pub mod clip_score;
pub mod precision_recall;
pub mod lpips;

use crate::hldva_t::types::*;
use crate::atqs::Tensor;
use std::collections::HashMap;

/// Main Evaluator
pub struct HLDAEvaluator {
    metrics: HashMap<String, Box<dyn Metric>>,
    
    // Reference data untuk FID
    reference_features: Option<Vec<Tensor>>,
    
    // Inception network untuk IS
    inception_network: Option<InceptionNetwork>,
    
    // CLIP model untuk CLIP score
    clip_model: Option<ClipModel>,
}

impl HLDAEvaluator {
    /// Create new evaluator
    pub fn new() -> HLDVAResult<Self> {
        let mut metrics: HashMap<String, Box<dyn Metric>> = HashMap::new();
        
        // Register metrics
        metrics.insert("fid".to_string(), Box::new(FIDMetric::new()?));
        metrics.insert("inception_score".to_string(), Box::new(InceptionScoreMetric::new()?));
        metrics.insert("clip_score".to_string(), Box::new(CLIPScoreMetric::new()?));
        metrics.insert("precision_recall".to_string(), Box::new(PrecisionRecallMetric::new()?));
        metrics.insert("lpips".to_string(), Box::new(LPIPSMetric::new()?));
        
        Ok(Self {
            metrics,
            reference_features: None,
            inception_network: None,
            clip_model: None,
        })
    }
    
    /// Evaluate generated images
    pub fn evaluate(
        &mut self,
        generated_images: &[Tensor],
        prompts: &[String],
        reference_images: Option<&[Tensor]>,
    ) -> HLDVAResult<GenerationMetrics> {
        let mut results = GenerationMetrics::default();
        
        // Calculate FID jika ada reference images
        if let Some(ref_imgs) = reference_images {
            let fid_score = self.calculate_fid(generated_images, ref_imgs)?;
            results.fid = Some(fid_score);
        }
        
        // Calculate Inception Score
        let is_score = self.calculate_inception_score(generated_images)?;
        results.inception_score = Some(is_score);
        
        // Calculate CLIP score
        let clip_score = self.calculate_clip_score(generated_images, prompts)?;
        results.clip_score = Some(clip_score);
        
        // Calculate Precision/Recall
        let (precision, recall) = self.calculate_precision_recall(generated_images, reference_images)?;
        results.precision = Some(precision);
        results.recall = Some(recall);
        
        // Calculate LPIPS
        let lpips_score = self.calculate_lpips(generated_images, reference_images)?;
        results.lpips = Some(lpips_score);
        
        Ok(results)
    }
    
    /// Calculate FID score
    fn calculate_fid(&self, generated: &[Tensor], reference: &[Tensor]) -> HLDVAResult<f32> {
        let fid_metric = self.metrics.get("fid").ok_or_else(|| crate::hldva_t::HLDVAError::InvalidInput("FID metric not found".to_string()))?;
        
        // Extract features
        let gen_features = self.extract_inception_features(generated)?;
        let ref_features = self.extract_inception_features(reference)?;
        
        // Calculate statistics
        let gen_stats = self.calculate_statistics(&gen_features)?;
        let ref_stats = self.calculate_statistics(&ref_features)?;
        
        // Calculate FID
        fid_metric.calculate(&[&gen_stats, &ref_stats])
    }
    
    /// Calculate Inception Score
    fn calculate_inception_score(&self, images: &[Tensor]) -> HLDVAResult<f32> {
        let is_metric = self.metrics.get("inception_score").ok_or_else(|| crate::hldva_t::HLDVAError::InvalidInput("Inception Score metric not found".to_string()))?;
        
        // Get predictions from inception network
        let predictions = self.get_inception_predictions(images)?;
        
        // Calculate IS
        is_metric.calculate(&[&predictions])
    }
    
    /// Calculate CLIP score
    fn calculate_clip_score(&self, images: &[Tensor], prompts: &[String]) -> HLDVAResult<f32> {
        let clip_metric = self.metrics.get("clip_score").ok_or_else(|| crate::hldva_t::HLDVAError::InvalidInput("CLIP Score metric not found".to_string()))?;
        
        let mut clip_scores = Vec::new();
        
        for (image, prompt) in images.iter().zip(prompts.iter()) {
            let score = self.calculate_single_clip_score(image, prompt)?;
            clip_scores.push(score);
        }
        
        // Average CLIP score
        let avg_score = clip_scores.iter().sum::<f32>() / clip_scores.len() as f32;
        Ok(avg_score)
    }
    
    /// Calculate Precision/Recall
    fn calculate_precision_recall(
        &self,
        generated: &[Tensor],
        reference: Option<&[Tensor]>,
    ) -> HLDVAResult<(f32, f32)> {
        let pr_metric = self.metrics.get("precision_recall").ok_or_else(|| crate::hldva_t::HLDVAError::InvalidInput("Precision-Recall metric not found".to_string()))?;
        
        if let Some(ref_imgs) = reference {
            let gen_features = self.extract_inception_features(generated)?;
            let ref_features = self.extract_inception_features(ref_imgs)?;
            
            // Flatten feature vectors for precision recall calculation
            let all_features: Vec<Tensor> = gen_features.into_iter().chain(ref_features.into_iter()).collect();
            let feature_refs: Vec<&Tensor> = all_features.iter().collect();
            let pr_score = pr_metric.calculate(&feature_refs)?;
            // Assume the metric returns (precision, recall) tuple
            Ok((pr_score, pr_score)) // Simplified
        } else {
            Ok((0.0, 0.0)) // Cannot calculate without reference
        }
    }
    
    /// Calculate LPIPS
    fn calculate_lpips(
        &self,
        generated: &[Tensor],
        reference: Option<&[Tensor]>,
    ) -> HLDVAResult<f32> {
        let lpips_metric = self.metrics.get("lpips").ok_or_else(|| crate::hldva_t::HLDVAError::InvalidInput("LPIPS metric not found".to_string()))?;
        
        if let Some(ref_imgs) = reference {
            let mut lpips_scores = Vec::new();
            
            for (gen_img, ref_img) in generated.iter().zip(ref_imgs.iter()) {
                let score = self.calculate_single_lpips(gen_img, ref_img)?;
                lpips_scores.push(score);
            }
            
            let avg_lpips = lpips_scores.iter().sum::<f32>() / lpips_scores.len() as f32;
            Ok(avg_lpips)
        } else {
            Ok(0.0) // Cannot calculate without reference
        }
    }
    
    /// Helper methods
    fn extract_inception_features(&self, images: &[Tensor]) -> HLDVAResult<Vec<Tensor>> {
        let mut features = Vec::new();
        
        for image in images {
            // Simplified feature extraction
            let feature = self.extract_image_features(image)?;
            features.push(feature);
        }
        
        Ok(features)
    }
    
    fn extract_image_features(&self, image: &Tensor) -> HLDVAResult<Tensor> {
        // Simplified feature extraction (would use actual Inception network)
        let image_data = image.data();
        let feature_dim = 2048; // Standard Inception v3 feature dimension
        
        let mut features = Vec::with_capacity(feature_dim);
        for i in 0..feature_dim {
            // Simple aggregation of image pixels as features
            let mut feature_val = 0.0;
            for (j, &pixel) in image_data.iter().enumerate() {
                feature_val += pixel * ((i + j) as f32).sin();
            }
            features.push(feature_val / image_data.len() as f32);
        }
        
        Ok(Tensor::new(features, vec![feature_dim]))
    }
    
    fn calculate_statistics(&self, features: &[Tensor]) -> HLDVAResult<Tensor> {
        let mut mean = vec![0.0; features[0].data().len()];
        let mut cov = vec![0.0; features[0].data().len() * features[0].data().len()];
        
        // Calculate mean
        for feature in features {
            let feature_data = feature.data();
            for i in 0..mean.len() {
                if i < feature_data.len() {
                    mean[i] += feature_data[i];
                }
            }
        }
        
        for i in 0..mean.len() {
            mean[i] /= features.len() as f32;
        }
        
        // Calculate covariance (simplified)
        for i in 0..mean.len() {
            for j in 0..mean.len() {
                let mut cov_val = 0.0;
                for feature in features {
                    let feature_data = feature.data();
                    if i < feature_data.len() && j < feature_data.len() {
                        cov_val += (feature_data[i] - mean[i]) * (feature_data[j] - mean[j]);
                    }
                }
                cov[i * mean.len() + j] = cov_val / (features.len() - 1) as f32;
            }
        }
        
        // Combine mean and covariance
        let mut stats = mean;
        stats.extend_from_slice(&cov);
        
        Ok(Tensor::new(stats.clone(), vec![stats.len()]))
    }
    
    fn get_inception_predictions(&self, images: &[Tensor]) -> HLDVAResult<Tensor> {
        // Simplified inception predictions
        let mut predictions = Vec::new();
        
        for image in images {
            let prediction = self.get_single_inception_prediction(image)?;
            predictions.extend_from_slice(&prediction);
        }
        
        Ok(Tensor::new(predictions, vec![images.len() * 1000])) // 1000 ImageNet classes
    }
    
    fn get_single_inception_prediction(&self, image: &Tensor) -> HLDVAResult<Vec<f32>> {
        // Simplified prediction - random softmax
        let mut prediction = vec![0.0; 1000];
        let mut sum = 0.0;
        
        for i in 0..1000 {
            prediction[i] = rand::random::<f32>();
            sum += prediction[i];
        }
        
        // Normalize to probability distribution
        for i in 0..1000 {
            prediction[i] /= sum;
        }
        
        Ok(prediction)
    }
    
    fn calculate_single_clip_score(&self, image: &Tensor, prompt: &str) -> HLDVAResult<f32> {
        // Simplified CLIP score calculation
        let image_features = self.extract_image_features(image)?;
        let text_features = self.extract_text_features(prompt)?;
        
        // Cosine similarity
        let img_data = image_features.data();
        let text_data = text_features.data();
        
        let mut dot_product = 0.0;
        let mut img_norm_sq = 0.0;
        let mut text_norm_sq = 0.0;
        
        for i in 0..img_data.len().min(text_data.len()) {
            dot_product += img_data[i] * text_data[i];
            img_norm_sq += img_data[i] * img_data[i];
            text_norm_sq += text_data[i] * text_data[i];
        }
        
        let img_norm = img_norm_sq.sqrt();
        let text_norm = text_norm_sq.sqrt();
        
        if img_norm > 0.0 && text_norm > 0.0 {
            Ok(dot_product / (img_norm * text_norm))
        } else {
            Ok(0.0)
        }
    }
    
    fn extract_text_features(&self, text: &str) -> HLDVAResult<Tensor> {
        // Simplified text feature extraction
        let text_bytes = text.as_bytes();
        let feature_dim = 512;
        
        let mut features = Vec::with_capacity(feature_dim);
        for i in 0..feature_dim {
            let mut feature_val = 0.0;
            for (j, &byte) in text_bytes.iter().enumerate() {
                feature_val += (byte as f32) * ((i + j) as f32).cos();
            }
            features.push(feature_val / text_bytes.len() as f32);
        }
        
        Ok(Tensor::new(features, vec![feature_dim]))
    }
    
    fn calculate_single_lpips(&self, image1: &Tensor, image2: &Tensor) -> HLDVAResult<f32> {
        // Simplified LPIPS calculation
        let data1 = image1.data();
        let data2 = image2.data();
        
        let mut mse = 0.0;
        let count = data1.len().min(data2.len());
        
        for i in 0..count {
            let diff = data1[i] - data2[i];
            mse += diff * diff;
        }
        
        Ok(mse / count as f32)
    }
}

/// Trait untuk metrik evaluasi
pub trait Metric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32>;
    fn name(&self) -> &str;
}

/// FID Metric
pub struct FIDMetric;

impl FIDMetric {
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self)
    }
}

impl Metric for FIDMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.len() < 2 {
            return Err(HLDVAError::Evaluation("Need at least 2 inputs for FID".to_string()));
        }
        
        let gen_stats = inputs[0].data();
        let ref_stats = inputs[1].data();
        
        // Simplified FID calculation
        let gen_mean = &gen_stats[0..512];
        let ref_mean = &ref_stats[0..512];
        
        let mut mean_diff_sq = 0.0;
        for i in 0..gen_mean.len().min(ref_mean.len()) {
            let diff = gen_mean[i] - ref_mean[i];
            mean_diff_sq += diff * diff;
        }
        
        Ok(mean_diff_sq.sqrt())
    }
    
    fn name(&self) -> &str {
        "fid"
    }
}

/// Inception Score Metric
pub struct InceptionScoreMetric;

impl InceptionScoreMetric {
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self)
    }
}

impl Metric for InceptionScoreMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.is_empty() {
            return Err(HLDVAError::Evaluation("Need inputs for Inception Score".to_string()));
        }
        
        let predictions = inputs[0].data();
        let num_images = predictions.len() / 1000;
        
        // Calculate entropy for each image
        let mut total_entropy = 0.0;
        
        for i in 0..num_images {
            let start = i * 1000;
            let end = start + 1000;
            
            if end <= predictions.len() {
                let image_preds = &predictions[start..end];
                let entropy = self.calculate_entropy(image_preds);
                total_entropy += entropy;
            }
        }
        
        let avg_entropy = total_entropy / num_images as f32;
        
        // Inception Score = exp(average_entropy)
        Ok(avg_entropy.exp())
    }
    
    fn name(&self) -> &str {
        "inception_score"
    }
}

impl InceptionScoreMetric {
    fn calculate_entropy(&self, probs: &[f32]) -> f32 {
        let mut entropy = 0.0;
        
        for &p in probs {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        
        entropy
    }
}

/// CLIP Score Metric
pub struct CLIPScoreMetric;

impl CLIPScoreMetric {
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self)
    }
}

impl Metric for CLIPScoreMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.is_empty() {
            return Err(HLDVAError::Evaluation("Need inputs for CLIP Score".to_string()));
        }
        
        // Simplified - return first value as CLIP score
        let data = inputs[0].data();
        Ok(if data.is_empty() { 0.0 } else { data[0] })
    }
    
    fn name(&self) -> &str {
        "clip_score"
    }
}

/// Precision/Recall Metric
pub struct PrecisionRecallMetric;

impl PrecisionRecallMetric {
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self)
    }
}

impl Metric for PrecisionRecallMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.len() < 2 {
            return Err(HLDVAError::Evaluation("Need at least 2 inputs for Precision/Recall".to_string()));
        }
        
        // Simplified precision/recall calculation
        let gen_features = inputs[0].data();
        let ref_features = inputs[1].data();
        
        // Calculate feature similarity
        let mut similarity = 0.0;
        let count = gen_features.len().min(ref_features.len());
        
        for i in 0..count {
            similarity += gen_features[i] * ref_features[i];
        }
        
        Ok(similarity / count as f32)
    }
    
    fn name(&self) -> &str {
        "precision_recall"
    }
}

/// LPIPS Metric
pub struct LPIPSMetric;

impl LPIPSMetric {
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self)
    }
}

impl Metric for LPIPSMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.is_empty() {
            return Err(HLDVAError::Evaluation("Need inputs for LPIPS".to_string()));
        }
        
        // Simplified LPIPS - return MSE
        let data = inputs[0].data();
        let mut mse = 0.0;
        
        for &val in data.iter() {
            mse += val * val;
        }
        
        Ok(mse / data.len() as f32)
    }
    
    fn name(&self) -> &str {
        "lpips"
    }
}

/// Inception Network untuk feature extraction
pub struct InceptionNetwork {
    feature_dim: usize,
    pooling_strategy: PoolingStrategy,
}

#[derive(Debug, Clone)]
pub enum PoolingStrategy {
    GlobalAvg,
    GlobalMax,
    Adaptive,
}

impl InceptionNetwork {
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self {
            feature_dim: 2048, // Standard Inception v3 feature dimension
            pooling_strategy: PoolingStrategy::GlobalAvg,
        })
    }
    
    pub fn with_pooling(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling_strategy = strategy;
        self
    }
    
    pub fn forward(&self, image: &Tensor) -> HLDVAResult<Tensor> {
        let image_data = image.data();
        let image_shape = image.shape();
        
        // Validate input shape (expecting 3D: H x W x C or 4D: N x H x W x C)
        if image_shape.len() < 3 {
            return Err(crate::hldva_t::HLDVAError::InvalidInput(
                "Image must have at least 3 dimensions".to_string()
            ));
        }
        
        // Extract features menggunakan simplified Inception-like operations
        let features = self.extract_inception_features(image_data, image_shape)?;
        
        Ok(Tensor::new(features, vec![self.feature_dim]))
    }
    
    /// Extract Inception-style features dari image data
    fn extract_inception_features(&self, image_data: &[f32], image_shape: &[usize]) -> HLDVAResult<Vec<f32>> {
        let mut features = Vec::with_capacity(self.feature_dim);
        
        // Simulate multi-scale feature extraction (Inception modules)
        let scales = vec![1, 3, 5, 7]; // Different kernel sizes
        
        for scale_idx in 0..self.feature_dim {
            let scale = scales[scale_idx % scales.len()];
            let feature_value = self.compute_multi_scale_feature(image_data, image_shape, scale);
            features.push(feature_value);
        }
        
        // Apply global pooling
        let pooled_features = self.apply_pooling(&features, image_shape);
        
        // Apply feature normalization
        let normalized_features = self.normalize_features(&pooled_features);
        
        Ok(normalized_features)
    }
    
    /// Compute multi-scale feature untuk given scale
    fn compute_multi_scale_feature(&self, image_data: &[f32], image_shape: &[usize], scale: usize) -> f32 {
        let mut feature_sum = 0.0;
        let mut feature_count = 0;
        
        for (i, &pixel) in image_data.iter().enumerate() {
            // Apply spatial weighting based on scale
            let spatial_weight = self.compute_spatial_weight(i, image_shape, scale);
            let weighted_pixel = pixel * spatial_weight;
            
            // Apply non-linearity (ReLU)
            let activated = weighted_pixel.max(0.0);
            
            feature_sum += activated;
            feature_count += 1;
        }
        
        if feature_count > 0 {
            feature_sum / feature_count as f32
        } else {
            0.0
        }
    }
    
    /// Compute spatial weight untuk given position and scale
    fn compute_spatial_weight(&self, position: usize, image_shape: &[usize], scale: usize) -> f32 {
        // Convert position ke 2D coordinates (assuming H x W x C)
        let height = image_shape[0];
        let width = image_shape[1];
        let channels = image_shape.get(2).unwrap_or(&1);
        
        let channel_idx = position % channels;
        let spatial_idx = position / channels;
        let y = spatial_idx / width;
        let x = spatial_idx % width;
        
        // Gaussian-like weighting based on distance from center and scale
        let center_y = height as f32 / 2.0;
        let center_x = width as f32 / 2.0;
        
        let distance = ((y as f32 - center_y).powi(2) + (x as f32 - center_x).powi(2)).sqrt();
        let sigma = scale as f32 * 2.0;
        
        (-distance.powi(2) / (2.0 * sigma.powi(2))).exp()
    }
    
    /// Apply global pooling strategy
    fn apply_pooling(&self, features: &[f32], _image_shape: &[usize]) -> Vec<f32> {
        match &self.pooling_strategy {
            PoolingStrategy::GlobalAvg => {
                // Average pooling across features
                let avg = features.iter().sum::<f32>() / features.len() as f32;
                vec![avg; self.feature_dim]
            }
            PoolingStrategy::GlobalMax => {
                // Max pooling
                let max_val = features.iter().fold(0.0_f32, |a, b| a.max(*b));
                vec![max_val; self.feature_dim]
            }
            PoolingStrategy::Adaptive => {
                // Adaptive pooling: combination of avg and max
                let avg = features.iter().sum::<f32>() / features.len() as f32;
                let max_val = features.iter().fold(0.0_f32, |a, b| a.max(*b));
                features.iter().map(|&f| 0.7 * f + 0.2 * avg + 0.1 * max_val).collect()
            }
        }
    }
    
    /// Normalize features ke unit range
    fn normalize_features(&self, features: &[f32]) -> Vec<f32> {
        if features.is_empty() {
            return features.to_vec();
        }
        
        let min_val = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let range = max_val - min_val;
        if range < 1e-8 {
            // All features are the same, return zeros
            vec![0.0; features.len()]
        } else {
            features.iter().map(|&f| (f - min_val) / range).collect()
        }
    }
}

/// CLIP Model untuk multimodal embeddings
pub struct ClipModel {
    embedding_dim: usize,
    image_encoder: ImageEncoder,
    text_encoder: TextEncoder,
    temperature: f32,
}

struct ImageEncoder {
    patch_size: usize,
    embed_dim: usize,
    num_layers: usize,
}

struct TextEncoder {
    vocab_size: usize,
    _embed_dim: usize,
    max_length: usize,
    num_layers: usize,
}

impl ClipModel {
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self {
            embedding_dim: 512, // Standard CLIP embedding dimension
            image_encoder: ImageEncoder {
                patch_size: 16,
                embed_dim: 512,
                num_layers: 12,
            },
            text_encoder: TextEncoder {
                vocab_size: 49408, // CLIP vocabulary size
                _embed_dim: 512,
                max_length: 77,
                num_layers: 12,
            },
            temperature: 0.07, // CLIP temperature parameter
        })
    }
    
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    
    pub fn encode_image(&self, image: &Tensor) -> HLDVAResult<Tensor> {
        let image_data = image.data();
        let image_shape = image.shape();
        
        // Validate input shape
        if image_shape.len() < 3 {
            return Err(crate::hldva_t::HLDVAError::InvalidInput(
                "Image must have at least 3 dimensions".to_string()
            ));
        }
        
        // Extract image features menggunakan vision transformer approach
        let features = self.extract_image_features(image_data, image_shape)?;
        
        Ok(Tensor::new(features, vec![self.embedding_dim]))
    }
    
    pub fn encode_text(&self, text: &str) -> HLDVAResult<Tensor> {
        // Tokenize text
        let tokens = self.tokenize_text(text)?;
        
        // Extract text features menggunakan transformer approach
        let features = self.extract_text_features(&tokens)?;
        
        Ok(Tensor::new(features, vec![self.embedding_dim]))
    }
    
    /// Compute similarity antara image dan text embeddings
    pub fn compute_similarity(&self, image: &Tensor, text: &Tensor) -> HLDVAResult<f32> {
        let image_features = self.encode_image(image)?;
        // text is already encoded as a tensor
        let text_features = text.clone();
        
        // Compute cosine similarity
        let similarity = self.cosine_similarity(&image_features, &text_features)?;
        
        Ok(similarity)
    }
    
    /// Tokenize text input
    fn tokenize_text(&self, text: &str) -> HLDVAResult<Vec<usize>> {
        let mut tokens = Vec::new();
        
        // Add start token
        tokens.push(49406); // CLIP start token
        
        // Simple tokenization - dalam implementasi nyata gunakan proper tokenizer
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words.iter().take(self.text_encoder.max_length - 2) {
            // Hash word ke token ID (simplified)
            let token_id = self.simple_word_to_token(word);
            tokens.push(token_id);
        }
        
        // Add end token
        tokens.push(49407); // CLIP end token
        
        // Pad to max length
        while tokens.len() < self.text_encoder.max_length {
            tokens.push(0); // Padding token
        }
        
        Ok(tokens)
    }
    
    /// Simple word to token conversion (simplified)
    fn simple_word_to_token(&self, word: &str) -> usize {
        // Hash function untuk convert word ke token ID
        let mut hash: usize = 0;
        for (i, byte) in word.bytes().enumerate() {
            hash = hash.wrapping_add((byte as usize) * (i + 1));
        }
        
        // Modulo vocabulary size, ensure valid range
        (hash % (self.text_encoder.vocab_size - 1000)) + 1000
    }
    
    /// Extract image features menggunakan vision transformer approach
    fn extract_image_features(&self, image_data: &[f32], image_shape: &[usize]) -> HLDVAResult<Vec<f32>> {
        let mut features = Vec::with_capacity(self.embedding_dim);
        
        // Vision transformer layers
        let patches = self.extract_patches(image_data, image_shape)?;
        
        for i in 0..self.embedding_dim {
            let patch_idx = i % patches.len();
            let layer_idx = i / patches.len();
            
            // Apply transformer attention
            let patch_feature = patches[patch_idx];
            let attention_weight = self.compute_attention_weight(patch_idx, layer_idx);
            let transformed_feature = patch_feature * attention_weight;
            
            // Apply layer normalization and activation
            let normalized_feature = self.layer_normalize(transformed_feature, layer_idx);
            let activated_feature = self.gelu_activation(normalized_feature);
            
            features.push(activated_feature);
        }
        
        // Global average pooling
        let pooled_features = self.global_average_pool(&features);
        
        // Final projection
        let projected_features = self.project_to_embedding_dim(&pooled_features);
        
        Ok(projected_features)
    }
    
    /// Extract patches dari image
    fn extract_patches(&self, image_data: &[f32], image_shape: &[usize]) -> HLDVAResult<Vec<f32>> {
        let height = image_shape[0];
        let width = image_shape[1];
        let channels = image_shape.get(2).unwrap_or(&1);
        
        let mut patches = Vec::new();
        
        for y in (0..height).step_by(self.image_encoder.patch_size) {
            for x in (0..width).step_by(self.image_encoder.patch_size) {
                let mut patch_sum = 0.0;
                let mut patch_count = 0;
                
                for py in y..(y + self.image_encoder.patch_size).min(height) {
                    for px in x..(x + self.image_encoder.patch_size).min(width) {
                        for c in 0..*channels {
                            let idx = ((py * width + px) * *channels + c);
                            if idx < image_data.len() {
                                patch_sum += image_data[idx];
                                patch_count += 1;
                            }
                        }
                    }
                }
                
                let patch_value = if patch_count > 0 {
                    patch_sum / patch_count as f32
                } else {
                    0.0
                };
                
                patches.push(patch_value);
            }
        }
        
        Ok(patches)
    }
    
    /// Extract text features menggunakan transformer approach
    fn extract_text_features(&self, tokens: &[usize]) -> HLDVAResult<Vec<f32>> {
        let mut features = Vec::with_capacity(self.embedding_dim);
        
        // Transformer layers
        for i in 0..self.embedding_dim {
            let token_idx = i % tokens.len();
            let layer_idx = i / tokens.len();
            
            // Token embedding
            let token_embedding = self.get_token_embedding(tokens[token_idx]);
            
            // Positional encoding
            let positional_encoding = self.get_positional_encoding(token_idx);
            
            // Combine embeddings
            let combined_embedding = token_embedding + positional_encoding;
            
            // Apply transformer layers
            let mut hidden_state = combined_embedding;
            for layer in 0..self.text_encoder.num_layers {
                hidden_state = self.transformer_layer(hidden_state, token_idx, layer);
            }
            
            features.push(hidden_state);
        }
        
        // Take [CLS] token representation (first token)
        let cls_features = features.iter().take(self.embedding_dim).copied().collect();
        
        Ok(cls_features)
    }
    
    /// Get token embedding (simplified)
    fn get_token_embedding(&self, token_id: usize) -> f32 {
        // Simplified embedding based on token ID
        (token_id as f32).sin() * 0.1 + (token_id as f32).cos() * 0.05
    }
    
    /// Get positional encoding
    fn get_positional_encoding(&self, position: usize) -> f32 {
        // Simplified positional encoding
        (position as f32 * 0.01).sin()
    }
    
    /// Transformer layer
    fn transformer_layer(&self, input: f32, token_idx: usize, layer_idx: usize) -> f32 {
        // Simplified attention computation
        let attention_score = (input + token_idx as f32 + layer_idx as f32).cos();
        let attention_weight = 1.0 / (1.0 + (-attention_score).exp());
        
        // Feed-forward network
        let ffn_output = (input * 2.0).tanh();
        
        // Layer normalization
        let normalized = (input + attention_weight * ffn_output) * 0.577350269; // 1/sqrt(3)
        
        normalized
    }
    
    /// Compute attention weight untuk patch
    fn compute_attention_weight(&self, patch_idx: usize, layer_idx: usize) -> f32 {
        let combined_idx = patch_idx + layer_idx * 100;
        (combined_idx as f32 * 0.1).sin().abs()
    }
    
    /// Layer normalization
    fn layer_normalize(&self, input: f32, layer_idx: usize) -> f32 {
        let mean = input; // Simplified
        let variance = 0.1; // Simplified
        (input - mean) / (variance + 1e-6_f32).sqrt()
    }
    
    /// GELU activation
    fn gelu_activation(&self, x: f32) -> f32 {
        0.5 * x * (1.0 + (-1.702 * x * x).sqrt().tanh())
    }
    
    /// Global average pooling
    fn global_average_pool(&self, features: &[f32]) -> Vec<f32> {
        if features.is_empty() {
            return Vec::new();
        }
        
        let avg = features.iter().sum::<f32>() / features.len() as f32;
        vec![avg; features.len()]
    }
    
    /// Project ke embedding dimension
    fn project_to_embedding_dim(&self, features: &[f32]) -> Vec<f32> {
        if features.len() != self.embedding_dim {
            // Resize features
            let mut projected = Vec::with_capacity(self.embedding_dim);
            for i in 0..self.embedding_dim {
                let source_idx = (i * features.len()) / self.embedding_dim;
                let value = features.get(source_idx).copied().unwrap_or(0.0);
                projected.push(value);
            }
            projected
        } else {
            features.to_vec()
        }
    }
    
    /// Compute cosine similarity
    fn cosine_similarity(&self, tensor1: &Tensor, tensor2: &Tensor) -> HLDVAResult<f32> {
        let data1 = tensor1.data();
        let data2 = tensor2.data();
        
        if data1.len() != data2.len() {
            return Err(crate::hldva_t::HLDVAError::InvalidInput(
                "Tensors must have same length for cosine similarity".to_string()
            ));
        }
        
        let dot_product: f32 = data1.iter().zip(data2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = data1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = data2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        let similarity = dot_product / (norm1 * norm2);
        
        // Apply temperature scaling
        let scaled_similarity = similarity / self.temperature;
        
        Ok(scaled_similarity)
    }
}
