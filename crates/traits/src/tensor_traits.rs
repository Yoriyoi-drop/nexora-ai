// Tensor Traits for Foundation Components
//
// Traits specific to tensor operations and manipulations

use nexora_atqs::{Tensor, TensorError, TensorResult};
use std::fmt;

/// Trait for basic tensor operations
pub trait TensorOps {
    /// Add two tensors
    fn add(&self, other: &Tensor) -> TensorResult<Tensor>;
    
    /// Subtract two tensors
    fn sub(&self, other: &Tensor) -> TensorResult<Tensor>;
    
    /// Multiply two tensors element-wise
    fn mul(&self, other: &Tensor) -> TensorResult<Tensor>;
    
    /// Divide two tensors element-wise
    fn div(&self, other: &Tensor) -> TensorResult<Tensor>;
    
    /// Get tensor dimensions
    fn ndim(&self) -> usize;
    
    /// Get total number of elements
    fn size(&self) -> usize;
    
    /// Reshape tensor
    fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Tensor>;
}

/// Trait for tensor mathematical operations
pub trait TensorMath {
    /// Compute sum of all elements
    fn sum(&self) -> f32;
    
    /// Compute mean of all elements
    fn mean(&self) -> f32;
    
    /// Compute standard deviation
    fn std(&self) -> f32;
    
    /// Find minimum value
    fn min(&self) -> f32;
    
    /// Find maximum value
    fn max(&self) -> f32;
    
    /// Find index of minimum value
    fn argmin(&self) -> usize;
    
    /// Find index of maximum value
    fn argmax(&self) -> usize;
}

/// Trait for tensor transformations
pub trait TensorTransform {
    /// Transpose tensor
    fn transpose(&self) -> TensorResult<Tensor>;
    
    /// Apply function to each element
    fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f32) -> f32;
    
    /// Apply function to each element with index
    fn enumerate_map<F>(&self, f: F) -> Tensor
    where
        F: Fn(usize, f32) -> f32;
    
    /// Slice tensor along specified dimensions
    fn slice(&self, ranges: &[(usize, usize)]) -> TensorResult<Tensor>;
    
    /// Concatenate tensors along specified axis
    fn concat(&self, other: &Tensor, axis: usize) -> TensorResult<Tensor>;
}

/// Trait for tensor compression operations
pub trait TensorCompression {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Compress tensor data
    fn compress(&mut self) -> Result<(), Self::Error>;
    
    /// Decompress tensor data
    fn decompress(&mut self) -> Result<(), Self::Error>;
    
    /// Check if tensor is compressed
    fn is_compressed(&self) -> bool;
    
    /// Get compression ratio
    fn compression_ratio(&self) -> Option<f32>;
}

/// Trait for tensor validation
pub trait TensorValidation {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Validate tensor shape
    fn validate_shape(&self) -> Result<(), Self::Error>;
    
    /// Validate tensor data
    fn validate_data(&self) -> Result<(), Self::Error>;
    
    /// Check for NaN values
    fn has_nan(&self) -> bool;
    
    /// Check for infinite values
    fn has_inf(&self) -> bool;
    
    /// Check if tensor is finite (no NaN or Inf)
    fn is_finite(&self) -> bool;
}

/// Trait for tensor I/O operations
pub trait TensorIO {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Save tensor to file
    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Self::Error>;
    
    /// Load tensor from file
    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Tensor, Self::Error>;
    
    /// Serialize tensor to bytes
    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error>;
    
    /// Deserialize tensor from bytes
    fn from_bytes(data: &[u8]) -> Result<Tensor, Self::Error>;
}

/// Trait for tensor comparison operations
pub trait TensorCompare {
    /// Check if two tensors are approximately equal
    fn approx_eq(&self, other: &Tensor, epsilon: f32) -> bool;
    
    /// Check if two tensors are exactly equal
    fn eq(&self, other: &Tensor) -> bool;
    
    /// Compute element-wise equality
    fn elementwise_eq(&self, other: &Tensor) -> TensorResult<Tensor>;
    
    /// Compute element-wise approximate equality
    fn elementwise_approx_eq(&self, other: &Tensor, epsilon: f32) -> TensorResult<Tensor>;
}

/// Trait for tensor aggregation operations
pub trait TensorAggregate {
    /// Sum along specified axis
    fn sum_axis(&self, axis: usize) -> TensorResult<Tensor>;
    
    /// Mean along specified axis
    fn mean_axis(&self, axis: usize) -> TensorResult<Tensor>;
    
    /// Min along specified axis
    fn min_axis(&self, axis: usize) -> TensorResult<Tensor>;
    
    /// Max along specified axis
    fn max_axis(&self, axis: usize) -> TensorResult<Tensor>;
    
    /// Argmin along specified axis
    fn argmin_axis(&self, axis: usize) -> TensorResult<Tensor>;
    
    /// Argmax along specified axis
    fn argmax_axis(&self, axis: usize) -> TensorResult<Tensor>;
}
