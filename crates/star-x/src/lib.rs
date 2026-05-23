pub mod aca;
pub mod agr;
pub mod asc;
pub mod blas_backend;
pub mod core;
pub mod emr;
pub mod fused_ops;
pub mod hte;
pub mod kv_cache;
pub mod quantization;
pub mod sca;
pub mod sliding_window;
pub mod ssu;
pub mod tensor_pool;
pub mod tgh;

pub use aca::*;
pub use agr::*;
pub use asc::*;
pub use blas_backend::{
    get_blas_operations, init_blas_with_backend, BlasBackend, BlasBackendInfo, BlasFeatures,
    BlasOperations,
};
pub use core::*;
pub use emr::*;
pub use fused_ops::*;
pub use hte::*;
pub use kv_cache::*;
pub use quantization::*;
pub use sca::*;
pub use sliding_window::*;
pub use ssu::*;
pub use tensor_pool::*;
pub use tgh::*;

pub type DLResult<T> = std::result::Result<T, DeepLearningError>;

#[derive(Debug, thiserror::Error)]
pub enum DeepLearningError {
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Invalid input dimension: {dim}")]
    InvalidDimension { dim: usize },
    #[error("Memory allocation failed: {reason}")]
    MemoryAllocation { reason: String },
    #[error("Computation error: {reason}")]
    Computation { reason: String },
    #[error("Configuration error: {reason}")]
    Configuration { reason: String },
}

impl From<ndarray::ShapeError> for DeepLearningError {
    fn from(_err: ndarray::ShapeError) -> Self {
        DeepLearningError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
        }
    }
}

/// Convert an `Option` (from ndarray contiguity check) into a `DLResult`.
/// Used to avoid `.unwrap()` on hot paths.
pub fn require_contiguous<'a, T>(slice: Option<&'a [T]>) -> DLResult<&'a [T]> {
    slice.ok_or_else(|| DeepLearningError::Computation {
        reason: "tensor not contiguous".to_string(),
    })
}

/// Convert an `Option` mutable slice into a `DLResult`.
pub fn require_contiguous_mut<'a, T>(slice: Option<&'a mut [T]>) -> DLResult<&'a mut [T]> {
    slice.ok_or_else(|| DeepLearningError::Computation {
        reason: "tensor not contiguous".to_string(),
    })
}

pub mod traits {
    use super::*;
    pub trait Forward {
        type Input;
        type Output;
        fn forward(&self, input: &Self::Input) -> DLResult<Self::Output>;
    }

}
