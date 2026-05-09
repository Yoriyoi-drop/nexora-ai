//! Prelude module for ATQS (Attention Tensor Quantum System)

pub use crate::atqs::calibration::*;
pub use crate::atqs::compression::*;
pub use crate::atqs::config::*;
pub use crate::atqs::core::*;
pub use crate::atqs::error::*;
pub use crate::atqs::foundation::*;
pub use crate::atqs::profiling::*;
pub use crate::atqs::types::*;
pub use crate::atqs::utils::*;

// Re-export common types
pub use ndarray::{Array, ArrayD, ArrayView};
pub use std::error::Error;
pub use std::fmt::{Debug, Display};

// Re-export common types for easier access
pub use crate::atqs::types::{LayerType, TensorFormat};