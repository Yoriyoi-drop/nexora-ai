// ATQS Tensor Implementation
//
// Core tensor structure for Advanced Tensor Quantization and Compression system

use ndarray::ArrayD;
use std::fmt;

/// Tensor structure using ATQS compression
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    _compression_engine: Option<crate::atqs::compression::CompressionEngine>,
}

impl Tensor {
    /// Create a new tensor with the given shape and data
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            _compression_engine: None,
        }
    }

    /// Get the tensor data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable tensor data
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get the tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Tensor-specific errors
#[derive(Debug)]
pub enum TensorError {
    /// No compression engine available
    NoCompressionEngine,
    /// ATQS-related error
    Atqs(crate::atqs::error::ATQSError),
    /// Invalid tensor shape
    InvalidShape,
    /// Data size mismatch
    DataSizeMismatch,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::NoCompressionEngine => write!(f, "No compression engine available"),
            TensorError::Atqs(err) => write!(f, "ATQS error: {}", err),
            TensorError::InvalidShape => write!(f, "Invalid tensor shape"),
            TensorError::DataSizeMismatch => write!(f, "Data size does not match shape"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<crate::atqs::error::ATQSError> for TensorError {
    fn from(err: crate::atqs::error::ATQSError) -> Self {
        TensorError::Atqs(err)
    }
}

/// Result type for tensor operations
pub type TensorResult<T> = Result<T, TensorError>;

impl std::ops::Add for &Tensor {
    type Output = TensorResult<Tensor>;
    
    fn add(self, rhs: &Tensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(TensorError::InvalidShape);
        }
        
        if self.data.len() != rhs.data.len() {
            return Err(TensorError::DataSizeMismatch);
        }
        
        let result_data: Vec<f32> = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Ok(Tensor::new(result_data, self.shape.clone()))
    }
}

impl std::ops::Add for Tensor {
    type Output = TensorResult<Tensor>;
    
    fn add(self, rhs: Tensor) -> Self::Output {
        &self + &rhs
    }
}
