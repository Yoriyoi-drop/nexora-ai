//! Tensor operations for Nexora AI
//! 
//! Basic tensor data structures and operations

use std::fmt;

/// Simple tensor structure
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor data
    pub data: Vec<f32>,
    /// Tensor shape
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Get tensor size
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape: {:?}, size: {})", self.shape, self.size())
    }
}
