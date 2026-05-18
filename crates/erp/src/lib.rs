//! Entropic Resonance Pruning (ERP) - Strengthened Architecture
//! 
//! ERP tidak menghapus neuron secara permanen seperti pruning tradisional.
//! Sebaliknya, ERP mendeteksi neuron dengan distribusi informasi yang sangat mirip,
//! lalu mengelompokkannya ke dalam Resonance Groups (RG), merepresentasikannya
//! sebagai superposed latent representation, dan hanya merekonstruksi bagian yang
//! relevan berdasarkan konteks inferensi.

pub mod core;
pub mod resonance;
pub mod compression;
pub mod reconstruction;
pub mod cache;
pub mod training;
pub mod utils;

// Re-export main components
pub use core::*;
pub use resonance::*;
pub use compression::*;
pub use reconstruction::*;
pub use cache::*;
pub use training::*;
pub use utils::*;

/// ERP Configuration
#[derive(Debug, Clone)]
pub struct ERPConfig {
    /// Threshold untuk resonance distance
    pub resonance_threshold: f32,
    /// Maximum size per resonance group
    pub max_group_size: usize,
    /// Stability constraint variance
    pub stability_variance: f32,
    /// Sparse activation regularization
    pub sparse_regularization: f32,
    /// Cache size for inference patterns
    pub cache_size: usize,
    /// Compression mode
    pub compression_mode: CompressionMode,
}

#[derive(Debug, Clone)]
pub enum CompressionMode {
    Conservative,
    Balanced,
    Aggressive,
}

impl Default for ERPConfig {
    fn default() -> Self {
        Self {
            resonance_threshold: 0.1,
            max_group_size: 8,
            stability_variance: 0.05,
            sparse_regularization: 0.01,
            cache_size: 1000,
            compression_mode: CompressionMode::Balanced,
        }
    }
}

/// Main ERP Engine
pub struct ERPEngine {
    _config: ERPConfig,
    resonance_mapper: ResonanceMapper,
    compressor: SuperpositionCompressor,
    reconstructor: ContextReconstructor,
    cache: InferenceCache,
}

impl ERPEngine {
    pub fn new(config: ERPConfig) -> Self {
        Self {
            _config: config.clone(),
            resonance_mapper: ResonanceMapper::new(config.clone()),
            compressor: SuperpositionCompressor::new(config.clone()),
            reconstructor: ContextReconstructor::new(config.clone()),
            cache: InferenceCache::new(config.cache_size),
        }
    }

    /// Apply ERP pruning to neural network weights
    pub fn apply_pruning(&mut self, weights: &[ndarray::Array2<f32>]) -> Result<Vec<CompressedLayer>, ERPError> {
        // Phase 1: Information resonance mapping
        let resonance_groups = self.resonance_mapper.map_resonance(weights)?;
        
        // Phase 2: Superposition compression
        let compressed_layers = self.compressor.compress_weights(weights, &resonance_groups)?;
        
        Ok(compressed_layers)
    }

    /// Inference with ERP reconstruction
    pub fn forward(&mut self, compressed_layers: &[CompressedLayer], input: &ndarray::Array1<f32>) -> Result<ndarray::Array1<f32>, ERPError> {
        // Check cache first
        let context_hash = self.cache.hash_context(input);
        if let Some(cached_pattern) = self.cache.get(context_hash) {
            return Ok(self.reconstructor.reconstruct_with_gates(compressed_layers, input, &cached_pattern.gates)?);
        }

        // Compute gates and cache
        let gates = self.reconstructor.compute_gates(compressed_layers, input)?;
        self.cache.insert(context_hash, gates.clone());
        
        self.reconstructor.reconstruct_with_gates(compressed_layers, input, &gates)
    }
}

/// Custom error types for ERP
#[derive(Debug, thiserror::Error)]
pub enum ERPError {
    #[error("Resonance mapping failed: {0}")]
    ResonanceMappingError(String),
    #[error("Compression failed: {0}")]
    CompressionError(String),
    #[error("Reconstruction failed: {0}")]
    ReconstructionError(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}
