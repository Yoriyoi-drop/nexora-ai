//! Inception Score metric implementation

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Inception Score metric
pub struct InceptionScoreMetric {
    config: InceptionScoreConfig,
}

impl InceptionScoreMetric {
    /// Create new Inception Score metric
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self {
            config: InceptionScoreConfig::default(),
        })
    }
    
    /// Calculate Inception Score for generated images
    pub fn calculate_inception_score(&self, images: &[Tensor]) -> HLDVAResult<f32> {
        if images.is_empty() {
            return Err(HLDVAError::Evaluation("Need inputs for Inception Score".to_string()));
        }
        
        // Get predictions from Inception network
        let predictions = self.get_inception_predictions(images)?;
        
        // Calculate KL divergence for each image
        let mut kl_divs = Vec::new();
        for pred in &predictions {
            let kl = self.calculate_kl_divergence(pred)?;
            kl_divs.push(kl);
        }
        
        // Calculate average KL divergence
        let avg_kl = kl_divs.iter().sum::<f32>() / kl_divs.len() as f32;
        
        // Inception Score is exp(avg_kl)
        Ok(avg_kl.exp())
    }
    
    /// Get predictions from Inception network
    fn get_inception_predictions(&self, images: &[Tensor]) -> HLDVAResult<Vec<Vec<f32>>> {
        let mut all_predictions = Vec::new();
        
        for image in images {
            let prediction = self.get_single_inception_prediction(image)?;
            all_predictions.push(prediction);
        }
        
        Ok(all_predictions)
    }
    
    /// Get prediction for a single image
    fn get_single_inception_prediction(&self, _image: &Tensor) -> HLDVAResult<Vec<f32>> {
        // Simplified Inception prediction - in reality this would use a neural network
        let num_classes = 1000; // Standard ImageNet classes
        let mut prediction = Vec::with_capacity(num_classes);
        
        // Generate random softmax probabilities (simplified)
        let mut random_values = Vec::with_capacity(num_classes);
        for _ in 0..num_classes {
            random_values.push(rand::random::<f32>());
        }
        
        // Apply softmax
        let sum_exp = random_values.iter().map(|x| x.exp()).sum::<f32>();
        for val in random_values {
            prediction.push(val.exp() / sum_exp);
        }
        
        Ok(prediction)
    }
    
    /// Calculate KL divergence between prediction and marginal distribution
    fn calculate_kl_divergence(&self, prediction: &[f32]) -> HLDVAResult<f32> {
        // Calculate marginal distribution (average prediction across all images)
        // For simplicity, we'll use uniform distribution as marginal
        let uniform_prob = 1.0 / prediction.len() as f32;
        
        let mut kl_div = 0.0;
        for &p in prediction {
            if p > 0.0 {
                kl_div += p * (p / uniform_prob).ln();
            }
        }
        
        Ok(kl_div)
    }
    
    /// Calculate Inception Score with splits
    pub fn calculate_inception_score_with_splits(&self, images: &[Tensor], splits: usize) -> HLDVAResult<f32> {
        if images.len() < splits {
            return Err(HLDVAError::Evaluation("Not enough images for splits".to_string()));
        }
        
        let split_size = images.len() / splits;
        let mut scores = Vec::new();
        
        for i in 0..splits {
            let start = i * split_size;
            let end = if i == splits - 1 { images.len() } else { start + split_size };
            let split_images = &images[start..end];
            
            let score = self.calculate_inception_score(split_images)?;
            scores.push(score);
        }
        
        // Return mean and standard deviation
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        let std_dev = variance.sqrt();
        
        println!("Inception Score: {:.3} ± {:.3}", mean, std_dev);
        
        Ok(mean)
    }
}

/// Inception Score configuration
#[derive(Debug, Clone)]
pub struct InceptionScoreConfig {
    pub num_classes: usize,
    pub splits: usize,
}

impl Default for InceptionScoreConfig {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            splits: 10,
        }
    }
}

/// Inception Score Metric trait implementation
impl super::Metric for InceptionScoreMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.is_empty() {
            return Err(HLDVAError::Evaluation("Need inputs for Inception Score".to_string()));
        }
        
        let tensors: Vec<Tensor> = inputs.iter().map(|&t| t.clone()).collect();
        self.calculate_inception_score(&tensors)
    }
    
    fn name(&self) -> &str {
        "inception_score"
    }
}
