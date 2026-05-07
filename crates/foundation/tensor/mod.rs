//! Foundation Tensor Operations
//! 
//! Shared tensor utilities untuk semua AI frameworks di Nexora.
//! Konsolidasi dari duplicate tensor utils di ATQS, CAFFEINE, dan core.

use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::Standard;
use std::collections::HashMap;

pub mod ops;
pub mod reshape;
pub mod normalize;
pub mod validation;

// Re-export main components
pub use ops::*;
pub use reshape::*;
pub use normalize::*;
pub use validation::*;

/// Foundation tensor error type
#[derive(Debug, Clone)]
pub enum TensorError {
    ShapeMismatch(String),
    InvalidOperation(String),
    DimensionError(String),
    AllocationError(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            TensorError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            TensorError::DimensionError(msg) => write!(f, "Dimension error: {}", msg),
            TensorError::AllocationError(msg) => write!(f, "Allocation error: {}", msg),
        }
    }
}

impl std::error::Error for TensorError {}

/// Result type untuk tensor operations
pub type TensorResult<T> = Result<T, TensorError>;

/// Main tensor utilities struct
pub struct TensorUtils;

impl TensorUtils {
    /// Create new tensor utilities instance
    pub fn new() -> Self {
        Self
    }
    
    /// Validate tensor shape compatibility
    pub fn validate_shape(&self, tensor: &ArrayD<f32>, expected_shape: &[usize]) -> TensorResult<()> {
        if tensor.shape() != expected_shape {
            return Err(TensorError::ShapeMismatch(
                format!("Expected shape {:?}, got {:?}", expected_shape, tensor.shape())
            ));
        }
        Ok(())
    }
    
    /// Get tensor total elements
    pub fn total_elements(&self, tensor: &ArrayD<f32>) -> usize {
        tensor.len()
    }
    
    /// Check if tensor is empty
    pub fn is_empty(&self, tensor: &ArrayD<f32>) -> bool {
        tensor.len() == 0
    }
}

impl Default for TensorUtils {
    fn default() -> Self {
        Self::new()
    }
}
