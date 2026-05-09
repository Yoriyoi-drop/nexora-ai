//! Compression Engine Trait
//!
//! Defines the interface for compression implementations (ATQS, etc.)

use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::FoundationResult;

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub target_ratio: f32,
    pub max_accuracy_drop: f32,
    pub method: CompressionMethod,
}

#[derive(Debug, Clone)]
pub enum CompressionMethod {
    Quantization { bits: usize },
    Sparsification { ratio: f32 },
    LowRank { rank: usize },
    Hybrid,
}

/// Compression result
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub compressed_data: ArrayD<f32>,
    pub compression_ratio: f32,
    pub accuracy_preserved: f32,
    pub metadata: CompressionMetadata,
}

#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub method: String,
    pub parameters: HashMap<String, String>,
    pub original_size: usize,
    pub compressed_size: usize,
}

/// Core compression engine trait
#[async_trait]
pub trait CompressionEngine: Send + Sync {
    /// Compress tensor data
    async fn compress(&self, data: &ArrayD<f32>, config: CompressionConfig) -> FoundationResult<CompressionResult>;
    
    /// Decompress tensor data
    async fn decompress(&self, compressed: &CompressionResult) -> FoundationResult<ArrayD<f32>>;
    
    /// Get supported compression methods
    fn supported_methods(&self) -> Vec<CompressionMethod>;
    
    /// Estimate compression ratio without actual compression
    fn estimate_ratio(&self, data: &ArrayD<f32>, method: &CompressionMethod) -> f32;
}
