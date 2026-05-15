//! Compression algorithms for ATQS-Compress

pub mod sparse_augmentation;
pub mod adaptive_rank;
pub mod quantum_sparse;

pub use sparse_augmentation::*;
pub use adaptive_rank::*;
pub use quantum_sparse::*;

#[derive(Debug, Clone, Default)]
pub struct CompressionResult {
    pub compression_ratio: f32,
    pub compressed_size: usize,
}

#[derive(Debug, Clone, Default)]
pub struct AtqsCompression;

impl AtqsCompression {
    pub fn new() -> Self {
        Self
    }

    pub async fn compress(&self, data: &[u8]) -> Result<CompressionResult, crate::atqs::ATQSError> {
        Ok(CompressionResult {
            compression_ratio: 0.0,
            compressed_size: data.len(),
        })
    }
}
