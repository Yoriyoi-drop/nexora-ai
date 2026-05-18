//! STAR-X (Selective Temporal Adaptive Resonance Network)
//!
//! Evolusi dari STAR-RNN untuk foundation model skala besar dengan:
//! - Hierarchical Temporal Processing
//! - Sparse Causal Reasoning  
//! - Adaptive Compute Routing
//! - Resonant State Dynamics
//! - Episodic Memory Retention
//! - Parallel Associative Recurrence

pub mod core;
pub mod tgh;      // Temporal Gating Hierarchy
pub mod sca;      // Sparse Causal Attention
pub mod hte;      // Harmonic Temporal Encoding
pub mod ssu;      // Selective State Update
pub mod agr;      // Adaptive Gradient Resonance
pub mod emr;      // Episodic Memory Retention
pub mod asc;      // Associative State Composition
pub mod aca;      // Adaptive Compute Allocation
pub mod kv_cache; // KV Cache for efficient inference
pub mod tensor_pool;
pub mod fused_ops;
pub mod blas_backend;
pub mod sliding_window;
pub mod quantization; // Tensor Pool untuk optimasi alokasi

// Re-export semua komponen
pub use core::*;
pub use tgh::*;
pub use sca::*;
pub use hte::*;
pub use ssu::*;
pub use agr::*;
pub use emr::*;
pub use asc::*;
pub use aca::*;
pub use kv_cache::*;
pub use tensor_pool::*;
pub use fused_ops::*;
pub use blas_backend::{
    BlasBackend, BlasOperations, BlasFeatures, BlasBackendInfo,
    get_blas_operations, init_blas_with_backend,
};
pub use sliding_window::*;
pub use quantization::*;

use crate::DLResult;
use ndarray::ArrayD;

/// Konfigurasi STAR-X
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StarXConfig {
    // Model dimensions
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    
    // Temporal Gating Hierarchy
    pub micro_gate_size: usize,
    pub meso_gate_size: usize,
    pub macro_gate_size: usize,
    pub chunk_size: usize,  // Untuk meso level
    
    // Sparse Causal Attention
    pub attention_heads: usize,
    pub max_sparse_connections: usize,
    pub entropy_regularization: f32,
    
    // Harmonic Temporal Encoding
    pub harmonic_frequencies: usize,
    pub temporal_embedding_dim: usize,
    
    // Selective State Update
    pub update_threshold: f32,
    pub relevance_alpha: f32,
    
    // Episodic Memory
    pub memory_size: usize,
    pub memory_write_threshold: f32,
    
    // Adaptive Compute
    pub compute_levels: usize,
    pub compute_thresholds: Vec<f32>,
    
    // Training
    pub learning_rate: f32,
    pub gradient_clip: f32,
    pub resonance_factor: f32,
}

impl Default for StarXConfig {
    fn default() -> Self {
        Self {
            input_size: 512,
            hidden_size: 1024,
            output_size: 512,
            micro_gate_size: 256,
            meso_gate_size: 512,
            macro_gate_size: 1024,
            chunk_size: 8,
            attention_heads: 8,
            max_sparse_connections: 64,
            entropy_regularization: 0.1,
            harmonic_frequencies: 16,
            temporal_embedding_dim: 64,
            update_threshold: 0.1,
            relevance_alpha: 0.5,
            memory_size: 10000,
            memory_write_threshold: 0.7,
            compute_levels: 3,
            compute_thresholds: vec![0.3, 0.6, 0.9],
            learning_rate: 1e-4,
            gradient_clip: 1.0,
            resonance_factor: 0.1,
        }
    }
}

/// State komplit untuk STAR-X
#[derive(Debug, Clone)]
pub struct StarXState {
    // Hidden states
    pub hidden_state: ArrayD<f32>,
    pub micro_state: ArrayD<f32>,
    pub meso_state: ArrayD<f32>,
    pub macro_state: ArrayD<f32>,
    
    // Temporal states
    pub temporal_position: usize,
    pub chunk_states: Vec<ArrayD<f32>>,
    
    // Attention states
    pub attention_weights: ArrayD<f32>,
    pub sparse_mask: ArrayD<f32>,
    
    // Memory states
    pub episodic_memory: ArrayD<f32>,
    pub memory_priorities: Vec<f32>,
    
    // Compute allocation
    pub current_compute_level: usize,
    pub relevance_scores: Vec<f32>,
    
    // Resonance states
    pub resonance_factor: f32,
    pub previous_norm: f32,
}

impl StarXState {
    pub fn new(config: &StarXConfig) -> DLResult<Self> {
        let hidden_state = ArrayD::zeros(vec![config.hidden_size]);
        let micro_state = ArrayD::zeros(vec![config.micro_gate_size]);
        let meso_state = ArrayD::zeros(vec![config.meso_gate_size]);
        let macro_state = ArrayD::zeros(vec![config.macro_gate_size]);
        
        Ok(Self {
            hidden_state,
            micro_state,
            meso_state,
            macro_state,
            temporal_position: 0,
            chunk_states: vec![],
            attention_weights: ArrayD::zeros(vec![config.attention_heads, config.max_sparse_connections]),
            sparse_mask: ArrayD::zeros(vec![config.attention_heads, config.max_sparse_connections]),
            episodic_memory: ArrayD::zeros(vec![config.memory_size, config.hidden_size]),
            memory_priorities: vec![0.0; config.memory_size],
            current_compute_level: 0,
            relevance_scores: vec![],
            resonance_factor: config.resonance_factor,
            previous_norm: 0.0,
        })
    }
    
    pub fn reset(&mut self) {
        self.hidden_state.fill(0.0);
        self.micro_state.fill(0.0);
        self.meso_state.fill(0.0);
        self.macro_state.fill(0.0);
        self.temporal_position = 0;
        self.chunk_states.clear();
        self.attention_weights.fill(0.0);
        self.sparse_mask.fill(0.0);
        self.episodic_memory.fill(0.0);
        self.memory_priorities.fill(0.0);
        self.current_compute_level = 0;
        self.relevance_scores.clear();
        self.resonance_factor = 0.1;
        self.previous_norm = 0.0;
    }
}

/// Metrics untuk monitoring STAR-X
#[derive(Debug, Clone)]
pub struct StarXMetics {
    pub compute_efficiency: f32,
    pub memory_utilization: f32,
    pub sparsity_ratio: f32,
    pub update_frequency: f32,
    pub resonance_stability: f32,
    pub throughput_tokens_per_second: f32,
    pub attention_entropy: f32,
}

impl Default for StarXMetics {
    fn default() -> Self {
        Self {
            compute_efficiency: 0.0,
            memory_utilization: 0.0,
            sparsity_ratio: 0.0,
            update_frequency: 0.0,
            resonance_stability: 0.0,
            throughput_tokens_per_second: 0.0,
            attention_entropy: 0.0,
        }
    }
}
