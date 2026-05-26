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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_deep_learning_error_shape_mismatch() {
        let err = DeepLearningError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 3],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Shape mismatch"));
    }

    #[test]
    fn test_deep_learning_error_invalid_dimension() {
        let err = DeepLearningError::InvalidDimension { dim: 42 };
        assert_eq!(format!("{}", err), "Invalid input dimension: 42");
    }

    #[test]
    fn test_deep_learning_error_memory_allocation() {
        let err = DeepLearningError::MemoryAllocation {
            reason: "OOM".into(),
        };
        assert_eq!(format!("{}", err), "Memory allocation failed: OOM");
    }

    #[test]
    fn test_deep_learning_error_computation() {
        let err = DeepLearningError::Computation {
            reason: "overflow".into(),
        };
        assert_eq!(format!("{}", err), "Computation error: overflow");
    }

    #[test]
    fn test_deep_learning_error_configuration() {
        let err = DeepLearningError::Configuration {
            reason: "bad param".into(),
        };
        assert_eq!(format!("{}", err), "Configuration error: bad param");
    }

    #[test]
    fn test_require_contiguous_ok() {
        let arr = ndarray::Array1::<f32>::zeros(10);
        let result = require_contiguous(arr.as_slice());
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_contiguous_err() {
        let result = require_contiguous::<f32>(None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not contiguous"));
    }

    #[test]
    fn test_require_contiguous_mut_ok() {
        let mut arr = ndarray::Array1::<f32>::zeros(10);
        let result = require_contiguous_mut(arr.as_slice_mut());
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_contiguous_mut_err() {
        let result = require_contiguous_mut::<f32>(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dl_result_type_alias() {
        let ok_val: DLResult<i32> = Ok(42);
        assert_eq!(ok_val.unwrap(), 42);
    }

    #[test]
    fn test_from_shape_error() {
        let shape_err = ndarray::Array2::<f32>::zeros((2, 3))
            .into_shape((2, 4))
            .unwrap_err();
        let dl_err: DeepLearningError = shape_err.into();
        let msg = format!("{}", dl_err);
        assert!(msg.contains("Shape mismatch"));
    }
}
