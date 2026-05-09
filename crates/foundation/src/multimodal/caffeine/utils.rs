//! Utility functions for CAFFEINE (Comprehensive Adaptive Framework for Enhanced Neural Intelligence)

use std::collections::HashMap;
use ndarray::ArrayD;

/// Utility functions for multimodal processing
pub struct MultimodalUtils;

impl MultimodalUtils {
    /// Normalize tensor values to [0, 1] range
    pub fn normalize_tensor(tensor: &mut ArrayD<f32>) {
        let min_val = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        if max_val > min_val {
            tensor.mapv_inplace(|x| (x - min_val) / (max_val - min_val));
        }
    }

    /// Calculate similarity between two tensors
    pub fn tensor_similarity(a: &ArrayD<f32>, b: &ArrayD<f32>) -> f32 {
        if a.shape() != b.shape() {
            return 0.0;
        }

        let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Convert tensor to feature vector
    pub fn tensor_to_features(tensor: &ArrayD<f32>) -> Vec<f32> {
        tensor.iter().cloned().collect()
    }
}

/// Utility functions for model configuration
pub struct ConfigUtils;

impl ConfigUtils {
    /// Validate model configuration
    pub fn validate_config(config: &HashMap<String, String>) -> Result<(), String> {
        let required_fields = vec!["model_type", "input_shape", "output_shape"];
        
        for field in required_fields {
            if !config.contains_key(field) {
                return Err(format!("Missing required field: {}", field));
            }
        }
        
        Ok(())
    }

    /// Parse model dimensions from string
    pub fn parse_dimensions(dim_str: &str) -> Result<Vec<usize>, String> {
        dim_str
            .split(',')
            .map(|s| s.trim().parse::<usize>().map_err(|e| format!("Invalid dimension: {}", e)))
            .collect()
    }
}
