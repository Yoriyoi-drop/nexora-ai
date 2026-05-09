//! Utility functions for ATQS (Adaptive Tensor Quantization System)

use std::collections::HashMap;
use ndarray::ArrayD;

/// Utility functions for tensor operations
pub struct TensorUtils;

impl TensorUtils {
    /// Calculate tensor compression ratio
    pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f32 {
        if compressed_size == 0 {
            return 0.0;
        }
        original_size as f32 / compressed_size as f32
    }

    /// Estimate tensor memory usage
    pub fn estimate_memory_usage(tensor: &ArrayD<f32>) -> usize {
        tensor.len() * std::mem::size_of::<f32>()
    }

    /// Validate tensor dimensions
    pub fn validate_dimensions(tensor: &ArrayD<f32>, expected_dims: &[usize]) -> bool {
        tensor.shape() == expected_dims
    }
}

/// Utility functions for calibration
pub struct CalibrationUtils;

impl CalibrationUtils {
    /// Calculate calibration statistics
    pub fn calculate_stats(data: &[f32]) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        if data.is_empty() {
            return stats;
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();
        
        stats.insert("mean".to_string(), mean);
        stats.insert("std_dev".to_string(), std_dev);
        stats.insert("min".to_string(), data.iter().cloned().fold(f32::INFINITY, f32::min));
        stats.insert("max".to_string(), data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        
        stats
    }
}
