//! Precision/Recall metric implementation

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Precision/Recall metric
pub struct PrecisionRecallMetric {
    config: PrecisionRecallConfig,
}

impl PrecisionRecallMetric {
    /// Create new Precision/Recall metric
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self {
            config: PrecisionRecallConfig::default(),
        })
    }
    
    /// Calculate precision and recall between two sets of features
    pub fn calculate_precision_recall(&self, real_features: &[Tensor], generated_features: &[Tensor]) -> HLDVAResult<(f32, f32)> {
        if real_features.is_empty() || generated_features.is_empty() {
            return Err(HLDVAError::Evaluation("Need at least 2 inputs for Precision/Recall".to_string()));
        }
        
        // Extract feature manifolds
        let real_manifold = self.extract_manifold(real_features)?;
        let generated_manifold = self.extract_manifold(generated_features)?;
        
        // Calculate precision and recall
        let precision = self.calculate_precision(&generated_manifold, &real_manifold)?;
        let recall = self.calculate_recall(&generated_manifold, &real_manifold)?;
        
        Ok((precision, recall))
    }
    
    /// Extract feature manifold from tensors
    fn extract_manifold(&self, features: &[Tensor]) -> HLDVAResult<FeatureManifold> {
        let mut all_feature_vectors = Vec::new();
        
        for feature_tensor in features {
            let feature_data = feature_tensor.data();
            let chunk_size = self.config.feature_dim;
            
            // Split tensor into feature vectors
            for chunk in feature_data.chunks(chunk_size) {
                if chunk.len() == chunk_size {
                    all_feature_vectors.push(chunk.to_vec());
                }
            }
        }
        
        Ok(FeatureManifold {
            features: all_feature_vectors,
            dimension: self.config.feature_dim,
        })
    }
    
    /// Calculate precision: how many generated samples are close to real samples
    fn calculate_precision(&self, generated: &FeatureManifold, real: &FeatureManifold) -> HLDVAResult<f32> {
        let mut close_count = 0;
        let k = self.config.k_nearest;
        
        for gen_feature in &generated.features {
            // Find k nearest neighbors in real manifold
            let mut distances = Vec::new();
            for real_feature in &real.features {
                let dist = self.euclidean_distance(gen_feature, real_feature);
                distances.push(dist);
            }
            
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // Check if any of k nearest are within threshold
            let threshold = self.config.distance_threshold;
            for i in 0..k.min(distances.len()) {
                if distances[i] < threshold {
                    close_count += 1;
                    break;
                }
            }
        }
        
        Ok(close_count as f32 / generated.features.len() as f32)
    }
    
    /// Calculate recall: how many real samples are close to generated samples
    fn calculate_recall(&self, generated: &FeatureManifold, real: &FeatureManifold) -> HLDVAResult<f32> {
        let mut close_count = 0;
        let k = self.config.k_nearest;
        
        for real_feature in &real.features {
            // Find k nearest neighbors in generated manifold
            let mut distances = Vec::new();
            for gen_feature in &generated.features {
                let dist = self.euclidean_distance(real_feature, gen_feature);
                distances.push(dist);
            }
            
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // Check if any of k nearest are within threshold
            let threshold = self.config.distance_threshold;
            for i in 0..k.min(distances.len()) {
                if distances[i] < threshold {
                    close_count += 1;
                    break;
                }
            }
        }
        
        Ok(close_count as f32 / real.features.len() as f32)
    }
    
    /// Calculate Euclidean distance between two feature vectors
    fn euclidean_distance(&self, feature1: &[f32], feature2: &[f32]) -> f32 {
        if feature1.len() != feature2.len() {
            return f32::INFINITY;
        }
        
        let mut distance_sq = 0.0;
        for (val1, val2) in feature1.iter().zip(feature2.iter()) {
            let diff = val1 - val2;
            distance_sq += diff * diff;
        }
        
        distance_sq.sqrt()
    }
    
    /// Calculate F1 score from precision and recall
    pub fn calculate_f1_score(&self, precision: f32, recall: f32) -> f32 {
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }
    
    /// Calculate precision/recall with different distance metrics
    pub fn calculate_with_distance_metric(
        &self,
        real_features: &[Tensor],
        generated_features: &[Tensor],
        metric: DistanceMetric,
    ) -> HLDVAResult<(f32, f32)> {
        // This would use different distance metrics (cosine, Manhattan, etc.)
        // For now, we'll use the default Euclidean distance
        self.calculate_precision_recall(real_features, generated_features)
    }
}

/// Feature manifold representation
#[derive(Debug, Clone)]
pub struct FeatureManifold {
    pub features: Vec<Vec<f32>>,
    pub dimension: usize,
}

/// Precision/Recall configuration
#[derive(Debug, Clone)]
pub struct PrecisionRecallConfig {
    pub feature_dim: usize,
    pub k_nearest: usize,
    pub distance_threshold: f32,
    pub batch_size: usize,
}

impl Default for PrecisionRecallConfig {
    fn default() -> Self {
        Self {
            feature_dim: 512,
            k_nearest: 3,
            distance_threshold: 0.1,
            batch_size: 1000,
        }
    }
}

/// Distance metrics
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Manhattan,
    Chebyshev,
}

/// Precision/Recall Metric trait implementation
impl super::Metric for PrecisionRecallMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.len() < 2 {
            return Err(HLDVAError::Evaluation("Need at least 2 inputs for Precision/Recall".to_string()));
        }
        
        // Split inputs into real and generated
        let mid = inputs.len() / 2;
        let real_features: Vec<Tensor> = inputs[..mid].iter().map(|&t| t.clone()).collect();
        let generated_features: Vec<Tensor> = inputs[mid..].iter().map(|&t| t.clone()).collect();
        
        let (precision, recall) = self.calculate_precision_recall(
            &real_features,
            &generated_features
        )?;
        
        // Return F1 score as the metric value
        Ok(self.calculate_f1_score(precision, recall))
    }
    
    fn name(&self) -> &str {
        "precision_recall"
    }
}
