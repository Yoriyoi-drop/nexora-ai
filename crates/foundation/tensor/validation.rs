//! Tensor Validation Operations
//! 
//! Konsolidasi dari ATQS validation_utils dan CAFFEINE validation.

use crate::tensor::{TensorError, TensorResult};
use ndarray::{ArrayD, ArrayView};

/// Tensor validation utilities
pub struct TensorValidator;

impl TensorValidator {
    /// Create new validator instance
    pub fn new() -> Self {
        Self
    }
    
    /// Validate tensor contains no NaN values
    pub fn validate_no_nan(tensor: &ArrayD<f32>) -> TensorResult<()> {
        if tensor.iter().any(|x| x.is_nan()) {
            return Err(TensorError::InvalidOperation(
                "Tensor contains NaN values".to_string()
            ));
        }
        Ok(())
    }
    
    /// Validate tensor contains no infinite values
    pub fn validate_no_inf(tensor: &ArrayD<f32>) -> TensorResult<()> {
        if tensor.iter().any(|x| x.is_infinite()) {
            return Err(TensorError::InvalidOperation(
                "Tensor contains infinite values".to_string()
            ));
        }
        Ok(())
    }
    
    /// Validate tensor contains only finite values
    pub fn validate_finite(tensor: &ArrayD<f32>) -> TensorResult<()> {
        Self::validate_no_nan(tensor)?;
        Self::validate_no_inf(tensor)?;
        Ok(())
    }
    
    /// Validate tensor shape
    pub fn validate_shape(tensor: &ArrayD<f32>, expected_shape: &[usize]) -> TensorResult<()> {
        if tensor.shape() != expected_shape {
            return Err(TensorError::ShapeMismatch(
                format!("Expected shape {:?}, got {:?}", expected_shape, tensor.shape())
            ));
        }
        Ok(())
    }
    
    /// Validate tensor rank (number of dimensions)
    pub fn validate_rank(tensor: &ArrayD<f32>, expected_rank: usize) -> TensorResult<()> {
        if tensor.ndim() != expected_rank {
            return Err(TensorError::DimensionError(
                format!("Expected rank {}, got {}", expected_rank, tensor.ndim())
            ));
        }
        Ok(())
    }
    
    /// Validate tensor is not empty
    pub fn validate_non_empty(tensor: &ArrayD<f32>) -> TensorResult<()> {
        if tensor.is_empty() {
            return Err(TensorError::InvalidOperation(
                "Tensor is empty".to_string()
            ));
        }
        Ok(())
    }
    
    /// Validate tensor size matches expected
    pub fn validate_size(tensor: &ArrayD<f32>, expected_size: usize) -> TensorResult<()> {
        if tensor.len() != expected_size {
            return Err(TensorError::ShapeMismatch(
                format!("Expected size {}, got {}", expected_size, tensor.len())
            ));
        }
        Ok(())
    }
    
    /// Validate tensor values are within range
    pub fn validate_range(tensor: &ArrayD<f32>, min: f32, max: f32) -> TensorResult<()> {
        if tensor.iter().any(|x| *x < min || *x > max) {
            return Err(TensorError::InvalidOperation(
                format!("Tensor contains values outside range [{}, {}]", min, max)
            ));
        }
        Ok(())
    }
    
    /// Validate tensor is symmetric (for matrices)
    pub fn validate_symmetric(tensor: &ArrayD<f32>) -> TensorResult<()> {
        if tensor.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "Symmetry check requires 2D tensor".to_string()
            ));
        }
        
        let shape = tensor.shape();
        if shape[0] != shape[1] {
            return Err(TensorError::ShapeMismatch(
                "Symmetry check requires square matrix".to_string()
            ));
        }
        
        let tensor_2d = tensor.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TensorError::DimensionError(format!("Cannot convert to 2D: {:?}", e)))?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                if (tensor_2d[[i, j]] - tensor_2d[[j, i]]).abs() > 1e-6 {
                    return Err(TensorError::InvalidOperation(
                        "Tensor is not symmetric".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate tensor is positive definite (simplified check)
    pub fn validate_positive_definite(tensor: &ArrayD<f32>) -> TensorResult<()> {
        Self::validate_symmetric(tensor)?;
        
        // Simplified check: all diagonal elements must be positive
        let tensor_2d = tensor.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TensorError::DimensionError(format!("Cannot convert to 2D: {:?}", e)))?;
        
        for i in 0..tensor_2d.shape()[0] {
            if tensor_2d[[i, i]] <= 0.0 {
                return Err(TensorError::InvalidOperation(
                    "Tensor is not positive definite".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate tensor is orthogonal (simplified check)
    pub fn validate_orthogonal(tensor: &ArrayD<f32>) -> TensorResult<()> {
        if tensor.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "Orthogonality check requires 2D tensor".to_string()
            ));
        }
        
        let tensor_2d = tensor.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TensorError::DimensionError(format!("Cannot convert to 2D: {:?}", e)))?;
        
        let transpose = tensor_2d.t();
        let product = tensor_2d.dot(&transpose);
        
        // Check if product is close to identity
        let identity = ndarray::Array2::eye(tensor_2d.shape()[0]);
        for i in 0..product.shape()[0] {
            for j in 0..product.shape()[1] {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (product[[i, j]] - expected).abs() > 1e-6 {
                    return Err(TensorError::InvalidOperation(
                        "Tensor is not orthogonal".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Comprehensive validation
    pub fn validate_comprehensive(
        tensor: &ArrayD<f32>,
        expected_shape: Option<&[usize]>,
        expected_rank: Option<usize>,
        check_finite: bool,
    ) -> TensorResult<()> {
        Self::validate_non_empty(tensor)?;
        
        if let Some(shape) = expected_shape {
            Self::validate_shape(tensor, shape)?;
        }
        
        if let Some(rank) = expected_rank {
            Self::validate_rank(tensor, rank)?;
        }
        
        if check_finite {
            Self::validate_finite(tensor)?;
        }
        
        Ok(())
    }
}

impl Default for TensorValidator {
    fn default() -> Self {
        Self::new()
    }
}
