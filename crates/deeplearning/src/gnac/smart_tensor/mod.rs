//! SmartTensor: Visual Tensor Intelligence
//!
//! Seluruh edge pada graf merupakan SmartTensor aktif dengan:
//! - shape metadata, gradient state, entropy score
//! - activation distribution, bandwidth estimate, memory cost
//! - Visualisasi real-time: ketebalan kabel = bandwidth, warna = gradien, pola = throughput

pub mod metadata;
pub mod propagation;
pub mod visualization;

pub use metadata::*;
pub use propagation::*;
pub use visualization::*;

use crate::DLResult;
use crate::gnac::TensorDesc;

/// Entri kalkulasi shape propagation yang telah di-cache
#[derive(Debug, Clone)]
pub struct ShapePropEntry {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub compatible: bool,
    pub error_message: Option<String>,
}
