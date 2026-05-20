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

pub mod traits {
    use super::*;
    pub trait Forward {
        type Input;
        type Output;
        fn forward(&self, input: &Self::Input) -> DLResult<Self::Output>;
    }
    pub trait Backward {
        type Gradient;
        fn backward(&self, grad: &Self::Gradient) -> DLResult<Self::Gradient>;
    }
    pub trait Stateful {
        type State;
        fn reset_state(&mut self);
        fn get_state(&self) -> &Self::State;
        fn set_state(&mut self, state: Self::State);
    }
    pub trait Trainable {
        fn parameters(&self) -> Vec<&[f32]>;
        fn parameters_mut(&mut self) -> Vec<&mut [f32]>;
        fn gradients(&self) -> Vec<&[f32]>;
        fn gradients_mut(&mut self) -> Vec<&mut [f32]>;
    }
}
