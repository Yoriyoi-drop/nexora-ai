//! Tensor Normalization Operations
//! 
//! Konsolidasi dari CAFFEINE tensor_utils dan ATQS normalization functions.

use crate::tensor::{TensorError, TensorResult};
use ndarray::{ArrayD, s};

/// Tensor normalization utilities
pub struct TensorNormalizer;

impl TensorNormalizer {
    /// Create new normalizer instance
    pub fn new() -> Self {
        Self
    }
    
    /// L2 normalize tensor
    pub fn l2_normalize(tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        let norm = tensor.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(TensorError::InvalidOperation(
                "Cannot normalize zero tensor".to_string()
            ));
        }
        
        let normalized = tensor.mapv(|x| x / norm);
        Ok(normalized)
    }
    
    /// L1 normalize tensor
    pub fn l1_normalize(tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        let norm = tensor.iter().map(|x| x.abs()).sum::<f32>();
        if norm == 0.0 {
            return Err(TensorError::InvalidOperation(
                "Cannot normalize zero tensor".to_string()
            ));
        }
        
        let normalized = tensor.mapv(|x| x / norm);
        Ok(normalized)
    }
    
    /// Apply layer normalization
    pub fn layer_norm(tensor: &ArrayD<f32>, eps: f32) -> TensorResult<ArrayD<f32>> {
        let mean = tensor.iter().sum::<f32>() / tensor.len() as f32;
        let variance = tensor.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / tensor.len() as f32;
        let std_dev = (variance + eps).sqrt();
        
        if std_dev == 0.0 {
            return Ok(tensor.mapv(|_| 0.0));
        }
        
        let normalized = tensor.mapv(|x| (x - mean) / std_dev);
        Ok(normalized)
    }
    
    /// Apply batch normalization (simplified)
    pub fn batch_norm(
        tensor: &ArrayD<f32>,
        mean: &ArrayD<f32>,
        var: &ArrayD<f32>,
        gamma: Option<&ArrayD<f32>>,
        beta: Option<&ArrayD<f32>>,
        eps: f32,
    ) -> TensorResult<ArrayD<f32>> {
        if tensor.shape() != mean.shape() || tensor.shape() != var.shape() {
            return Err(TensorError::ShapeMismatch(
                "Tensor, mean, and variance must have same shape".to_string()
            ));
        }
        
        let std_dev = var.mapv(|x| (x + eps).sqrt());
        let normalized = (tensor - mean) / &std_dev;
        
        let mut result = normalized;
        
        if let Some(g) = gamma {
            if g.shape() != tensor.shape() {
                return Err(TensorError::ShapeMismatch(
                    "Gamma must have same shape as tensor".to_string()
                ));
            }
            result = result * g;
        }
        
        if let Some(b) = beta {
            if b.shape() != tensor.shape() {
                return Err(TensorError::ShapeMismatch(
                    "Beta must have same shape as tensor".to_string()
                ));
            }
            result = result + b;
        }
        
        Ok(result)
    }
    
    /// Apply instance normalization (simplified)
    pub fn instance_norm(tensor: &ArrayD<f32>, eps: f32) -> TensorResult<ArrayD<f32>> {
        if tensor.ndim() < 3 {
            return Err(TensorError::InvalidOperation(
                "Instance normalization requires at least 3D tensor".to_string()
            ));
        }
        
        // For simplicity, apply layer norm along the last two dimensions
        let shape = tensor.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size = shape.iter().skip(2).product();
        
        let mut result = tensor.clone();
        
        for b in 0..batch_size {
            for c in 0..channels {
                let mut slice_view = result.slice_mut(s![b, c, ..]);
                let slice_data = slice_view.to_owned();
                let slice_array = slice_data.into_dyn();
                
                let normalized = Self::layer_norm(&slice_array, eps)?;
                
                // Copy back (simplified - in real implementation would be more efficient)
                for (i, &val) in normalized.iter().enumerate() {
                    if i < slice_view.len() {
                        slice_view[i] = val;
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Apply group normalization (simplified)
    pub fn group_norm(tensor: &ArrayD<f32>, num_groups: usize, eps: f32) -> TensorResult<ArrayD<f32>> {
        if tensor.ndim() < 3 {
            return Err(TensorError::InvalidOperation(
                "Group normalization requires at least 3D tensor".to_string()
            ));
        }
        
        let shape = tensor.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        
        if channels % num_groups != 0 {
            return Err(TensorError::InvalidOperation(
                format!("Number of channels {} must be divisible by num_groups {}", 
                       channels, num_groups)
            ));
        }
        
        let channels_per_group = channels / num_groups;
        
        // For simplicity, return layer norm (real implementation would group channels)
        Self::layer_norm(tensor, eps)
    }
    
    /// Min-max normalization
    pub fn min_max_normalize(tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        if tensor.is_empty() {
            return Ok(tensor.clone());
        }
        
        let min_val = tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = tensor.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            return Ok(tensor.mapv(|_| 0.0));
        }
        
        let normalized = tensor.mapv(|x| (x - min_val) / (max_val - min_val));
        Ok(normalized)
    }
    
    /// Z-score normalization (same as layer norm but without learnable parameters)
    pub fn z_score_normalize(tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Self::layer_norm(tensor, 1e-5)
    }
    
    /// Unit vector normalization (same as L2)
    pub fn unit_vector_normalize(tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Self::l2_normalize(tensor)
    }
}

impl Default for TensorNormalizer {
    fn default() -> Self {
        Self::new()
    }
}
