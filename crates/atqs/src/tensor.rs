// ATQS Tensor Implementation
//
// Core tensor structure for Advanced Tensor Quantization and Compression system

use std::fmt;

/// Tensor structure using ATQS compression
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    frozen: bool,
    _compression_engine: Option<crate::compression::CompressionEngine>,
}

impl Tensor {
    /// Create a new tensor with the given shape and data
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            frozen: false,
            _compression_engine: None,
        }
    }

    /// Create a new frozen tensor (excluded from optimizer updates)
    pub fn new_frozen(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            frozen: true,
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

    /// Check if tensor is frozen (excluded from optimizer updates)
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Set frozen status
    pub fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }
}

/// Tensor-specific errors
#[derive(Debug)]
pub enum TensorError {
    /// No compression engine available
    NoCompressionEngine,
    /// ATQS-related error
    Atqs(crate::error::ATQSError),
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

impl From<crate::error::ATQSError> for TensorError {
    fn from(err: crate::error::ATQSError) -> Self {
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

        let result_data: Vec<f32> = self
            .data
            .iter()
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
