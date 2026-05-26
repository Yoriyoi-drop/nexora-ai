//! Compression algorithms for ATQS-Compress

pub mod adaptive_rank;
pub mod quantum_sparse;
pub mod sparse_augmentation;

pub use adaptive_rank::*;
pub use quantum_sparse::*;
pub use sparse_augmentation::*;

use crate::core::tensor_ops::TensorDecomposition;
use crate::core::tensor_ops::{tucker_decompose, ATQSCompressor, TensorCompressor};
use ndarray::ArrayD;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct CompressionResult {
    pub compression_ratio: f32,
    pub compressed_size: usize,
}

#[derive(Debug, Clone)]
pub struct AtqsCompression {
    compressor: Arc<TensorCompressor>,
}

impl Default for AtqsCompression {
    fn default() -> Self {
        Self::new()
    }
}

impl AtqsCompression {
    pub fn new() -> Self {
        Self {
            compressor: Arc::new(TensorCompressor::default()),
        }
    }

    pub fn with_compressor(compressor: TensorCompressor) -> Self {
        Self {
            compressor: Arc::new(compressor),
        }
    }

    pub async fn compress(&self, data: &[u8]) -> Result<CompressionResult, crate::ATQSError> {
        let original_size = data.len();
        if original_size == 0 {
            return Ok(CompressionResult {
                compression_ratio: 1.0,
                compressed_size: 0,
            });
        }

        // Convert bytes to f32 tensor — interpret as a 4D tensor
        let total_floats = original_size / 4;
        if total_floats < 4 {
            return Ok(CompressionResult {
                compression_ratio: 1.0,
                compressed_size: original_size,
            });
        }

        let mut float_data = Vec::with_capacity(total_floats);
        for chunk in data.chunks_exact(4).take(total_floats) {
            float_data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        // Reshape into a 4D tensor with roughly equal dimensions
        let dim = (total_floats as f64).powf(0.25).ceil() as usize;
        let shape = vec![dim, dim, dim, dim.max(1)];
        let tensor_len: usize = shape.iter().product();
        if tensor_len > total_floats {
            let mut padded = float_data.clone();
            padded.resize(tensor_len, 0.0);
            float_data = padded;
        } else {
            float_data.truncate(tensor_len);
        }

        let tensor = ArrayD::from_shape_vec(shape.clone(), float_data)
            .map_err(|e| crate::ATQSError::TensorError(e.to_string()))?;

        // Apply Tucker decomposition
        let ranks = vec![
            (shape[0] / 2).max(2),
            (shape[1] / 2).max(2),
            (shape[2] / 2).max(2),
            (shape[3] / 2).max(2),
        ];

        let original_params: usize = shape.iter().product();

        let compressed_params = match tucker_decompose(&tensor, &ranks) {
            Ok(decomposition) => {
                let core_params: usize = decomposition.core.shape().iter().product();
                let factor_params: usize = decomposition.factors.iter().map(|f| f.len()).sum();
                core_params + factor_params
            }
            Err(_) => {
                // Fallback: use TensorCompressor trait
                let tensor_2d = tensor
                    .into_shape(vec![tensor_len, 1])
                    .map_err(|e| crate::ATQSError::TensorError(e.to_string()))?;
                let decomp = self.compressor.compress(&tensor_2d.into_dyn())?;
                match &decomp {
                    TensorDecomposition::Tucker(t) => {
                        let core: usize = t.core.shape().iter().product();
                        let factors: usize = t.factors.iter().map(|f| f.len()).sum();
                        core + factors
                    }
                    TensorDecomposition::TensorTrain(tt) => tt.cores.iter().map(|c| c.len()).sum(),
                }
            }
        };

        let ratio = if compressed_params > 0 && compressed_params < original_params {
            original_params as f32 / compressed_params as f32
        } else {
            1.0
        };

        Ok(CompressionResult {
            compression_ratio: ratio,
            compressed_size: compressed_params * 4,
        })
    }
}
