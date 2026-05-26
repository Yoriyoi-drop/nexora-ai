//! Core traits dan struktur dasar untuk STAR-X

use crate::DLResult;
use ndarray::ArrayD;
use std::collections::HashMap;

/// Trait untuk komponen STAR-X yang memiliki temporal processing
pub trait TemporalProcessor {
    /// Process input dengan temporal context
    fn process_temporal(&self, input: &ArrayD<f32>, temporal_pos: usize) -> DLResult<ArrayD<f32>>;

    /// Update internal temporal state
    fn update_temporal_state(&mut self, new_state: ArrayD<f32>) -> DLResult<()>;

    /// Get current temporal state
    fn get_temporal_state(&self) -> &ArrayD<f32>;
}

/// Trait untuk sparse attention mechanisms
pub trait SparseAttention {
    /// Compute sparse attention scores
    fn compute_sparse_attention(
        &mut self,
        query: &ArrayD<f32>,
        key: &ArrayD<f32>,
        value: &ArrayD<f32>,
        temporal_encoding: &ArrayD<f32>,
    ) -> DLResult<(ArrayD<f32>, ArrayD<f32>)>; // (output, sparse_mask)

    /// Get current sparsity ratio
    fn get_sparsity_ratio(&self) -> f32;

    /// Set sparsity level
    fn set_sparsity_level(&mut self, ratio: f32) -> DLResult<()>;
}

/// Trait untuk hierarchical gating
pub trait HierarchicalGating {
    /// Process melalui multi-level gates
    fn process_hierarchical(
        &self,
        input: &ArrayD<f32>,
        hidden_state: &ArrayD<f32>,
        chunk_context: &ArrayD<f32>,
        episodic_memory: &ArrayD<f32>,
    ) -> DLResult<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)>; // (micro, meso, macro)

    /// Fusion dari hierarchical outputs
    fn fuse_hierarchical(
        &self,
        micro: &ArrayD<f32>,
        meso: &ArrayD<f32>,
        macro_out: &ArrayD<f32>,
        weights: (f32, f32, f32),
    ) -> DLResult<ArrayD<f32>>;
}

/// Trait untuk selective state updates
pub trait SelectiveUpdate {
    /// Compute relevance scores untuk selective update
    fn compute_relevance(
        &self,
        tgh_output: &ArrayD<f32>,
        sca_output: &ArrayD<f32>,
    ) -> DLResult<ArrayD<f32>>;

    /// Apply selective state update
    fn selective_update(
        &self,
        previous_state: &ArrayD<f32>,
        candidate_state: &ArrayD<f32>,
        relevance: &ArrayD<f32>,
        threshold: f32,
    ) -> DLResult<ArrayD<f32>>;

    /// Get update frequency statistics
    fn get_update_frequency(&self) -> f32;
}

/// Trait untuk episodic memory management
pub trait EpisodicMemory {
    /// Write ke episodic memory dengan priority
    fn write_memory(
        &mut self,
        state: &ArrayD<f32>,
        gradient: &ArrayD<f32>,
        relevance: f32,
        threshold: f32,
    ) -> DLResult<bool>; // Returns true if written

    /// Read dari episodic memory
    fn read_memory(&self, query: &ArrayD<f32>) -> DLResult<ArrayD<f32>>;

    /// Get memory utilization
    fn get_memory_utilization(&self) -> f32;

    /// Cleanup old/low-priority memories
    fn cleanup_memory(&mut self) -> DLResult<usize>; // Returns number of cleaned entries
}

/// Trait untuk adaptive compute allocation
pub trait AdaptiveCompute {
    /// Determine compute level untuk input tertentu
    fn determine_compute_level(
        &self,
        input: &ArrayD<f32>,
        hidden_state: &ArrayD<f32>,
    ) -> DLResult<usize>;

    /// Get compute statistics
    fn get_compute_stats(&self) -> (f32, f32, f32); // (efficiency, utilization, cost)

    /// Set compute thresholds
    fn set_compute_thresholds(&mut self, thresholds: Vec<f32>) -> DLResult<()>;
}

/// Trait untuk gradient resonance stabilization
pub trait GradientResonance {
    /// Compute resonance factor
    fn compute_resonance(
        &self,
        current_state: &ArrayD<f32>,
        previous_state: &ArrayD<f32>,
    ) -> DLResult<f32>;

    /// Apply resonance stabilization
    fn apply_resonance(
        &self,
        candidate_state: &ArrayD<f32>,
        previous_state: &ArrayD<f32>,
        resonance_factor: f32,
    ) -> DLResult<ArrayD<f32>>;

    /// Check resonance stability
    fn check_stability(&self, resonance_history: &[f32]) -> bool;
}

/// Core struct untuk parameter management
#[derive(Debug, Clone)]
pub struct StarXParameters {
    pub weights: HashMap<String, ArrayD<f32>>,
    pub biases: HashMap<String, ArrayD<f32>>,
    pub gradients: HashMap<String, ArrayD<f32>>,
}

impl StarXParameters {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            biases: HashMap::new(),
            gradients: HashMap::new(),
        }
    }

    pub fn register_parameter(&mut self, name: String, shape: Vec<usize>) -> DLResult<()> {
        let weight = ArrayD::zeros(shape.clone());
        let bias = ArrayD::zeros(shape.clone());
        let gradient = ArrayD::zeros(shape.clone());

        self.weights.insert(name.clone(), weight);
        self.biases.insert(name.clone(), bias);
        self.gradients.insert(name, gradient);

        Ok(())
    }

    pub fn get_parameter(&self, name: &str) -> Option<&ArrayD<f32>> {
        self.weights.get(name)
    }

    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut ArrayD<f32>> {
        self.weights.get_mut(name)
    }

    pub fn get_gradient(&self, name: &str) -> Option<&ArrayD<f32>> {
        self.gradients.get(name)
    }

    pub fn get_gradient_mut(&mut self, name: &str) -> Option<&mut ArrayD<f32>> {
        self.gradients.get_mut(name)
    }

    pub fn zero_gradients(&mut self) {
        for gradient in self.gradients.values_mut() {
            gradient.fill(0.0);
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.weights.values().map(|w| w.len()).sum()
    }

    pub fn gradient_norm(&self) -> f32 {
        let mut sum_sq = 0.0;
        for gradient in self.gradients.values() {
            for &val in gradient.iter() {
                sum_sq += val * val;
            }
        }
        sum_sq.sqrt()
    }
}

impl Default for StarXParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions untuk STAR-X
pub mod core_utils {
    use super::*;
    use crate::require_contiguous;
    use ndarray::Array1;

    /// Harmonic temporal encoding function
    pub fn harmonic_encoding(
        position: usize,
        frequencies: usize,
        embedding_dim: usize,
    ) -> ArrayD<f32> {
        let mut encoding = Array1::zeros(embedding_dim);

        for i in 0..frequencies.min(embedding_dim / 2) {
            let freq = 2.0 * std::f32::consts::PI * (i as f32 + 1.0);
            let pos = position as f32;

            encoding[2 * i] = (freq * pos).sin();
            if 2 * i + 1 < embedding_dim {
                encoding[2 * i + 1] = (freq * pos).cos();
            }
        }

        encoding.into_dyn()
    }

    /// Top-K sparse selection
    pub fn top_k_sparse(scores: &ArrayD<f32>, k: usize) -> ArrayD<f32> {
        let mut mask = ArrayD::zeros(scores.shape());
        let flat_scores = require_contiguous(scores.as_slice())
            .expect("scores tensor is contiguous by construction");

        let mut indexed_scores: Vec<(usize, f32)> = flat_scores
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, _) in indexed_scores.iter().take(k) {
            let mut indices = vec![0; scores.ndim()];
            let mut remaining = *i;
            for (dim, &size) in scores.shape().iter().enumerate() {
                indices[dim] = remaining % size;
                remaining /= size;
            }

            if let Some(val) = mask.get_mut(indices.as_slice()) {
                *val = 1.0;
            }
        }

        mask
    }

    /// Compute entropy untuk regularization
    pub fn compute_entropy(probabilities: &[f32]) -> f32 {
        let mut entropy = 0.0;
        for &p in probabilities {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Normalize tensor
    pub fn normalize(tensor: &mut ArrayD<f32>) -> f32 {
        let mut norm = 0.0;
        for &val in tensor.iter() {
            norm += val * val;
        }
        norm = norm.sqrt();

        if norm > 1e-8 {
            for val in tensor.iter_mut() {
                *val /= norm;
            }
        }

        norm
    }
}

/// Configuration untuk STAR-X model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StarXConfig {
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_sequence_length: usize,
    pub sparsity_ratio: f32,
    pub compute_levels: usize,
    pub memory_size: usize,
    pub learning_rate: f32,
    pub max_sparse_connections: usize,
    pub memory_threshold: f32,
    pub gradient_clip: f32,
    pub max_position: usize,
    pub update_threshold: f32,
    pub compute_thresholds: Vec<f32>,
    pub harmonic_frequencies: usize,
    pub temporal_embedding_dim: usize,
    pub hidden_size: usize,
    pub chunk_size: usize,
    pub input_size: usize,
    pub resonance_factor: f32,
    pub relevance_alpha: f32,
    pub attention_heads: usize,
    pub entropy_regularization: f32,
    pub micro_gate_size: usize,
    pub meso_gate_size: usize,
    pub macro_gate_size: usize,
}

impl Default for StarXConfig {
    fn default() -> Self {
        Self {
            model_dim: 512,
            num_layers: 6,
            num_heads: 8,
            max_sequence_length: 2048,
            sparsity_ratio: 0.1,
            compute_levels: 3,
            memory_size: 10000,
            learning_rate: 1e-3,
            max_sparse_connections: 1000,
            memory_threshold: 0.8,
            gradient_clip: 1.0,
            max_position: 2048,
            update_threshold: 0.05,
            compute_thresholds: vec![0.2, 0.5, 0.8],
            harmonic_frequencies: 32,
            temporal_embedding_dim: 128,
            hidden_size: 512,
            chunk_size: 512,
            input_size: 512,
            resonance_factor: 0.95,
            relevance_alpha: 0.1,
            attention_heads: 8,
            entropy_regularization: 0.01,
            micro_gate_size: 64,
            meso_gate_size: 128,
            macro_gate_size: 256,
        }
    }
}

/// State untuk STAR-X model
#[derive(Debug, Clone)]
pub struct StarXState {
    pub hidden_state: ArrayD<f32>,
    pub temporal_position: usize,
    pub memory_state: ArrayD<f32>,
    pub compute_level: usize,
    pub current_compute_level: usize,
    pub relevance_scores: Vec<f32>,
    pub resonance_factor: f32,
}

impl StarXState {
    pub fn new(config: &StarXConfig) -> Self {
        Self {
            hidden_state: ArrayD::zeros(vec![config.model_dim]),
            temporal_position: 0,
            memory_state: ArrayD::zeros(vec![config.memory_size, config.model_dim]),
            compute_level: 0,
            current_compute_level: 0,
            relevance_scores: vec![0.0; config.max_sequence_length],
            resonance_factor: config.resonance_factor,
        }
    }

    pub fn reset(&mut self) {
        self.temporal_position = 0;
        self.compute_level = 0;
        self.current_compute_level = 0;
        self.hidden_state.fill(0.0);
        self.memory_state.fill(0.0);
        self.relevance_scores.fill(0.0);
    }
}

/// Metrics untuk STAR-X model
#[derive(Debug, Clone)]
pub struct StarXMetics {
    pub loss: f32,
    pub accuracy: f32,
    pub compute_efficiency: f32,
    pub memory_utilization: f32,
    pub sparsity_ratio: f32,
    pub resonance_stability: f32,
    pub attention_entropy: f32,
    pub update_frequency: f32,
}

impl Default for StarXMetics {
    fn default() -> Self {
        Self {
            loss: 0.0,
            accuracy: 0.0,
            compute_efficiency: 0.0,
            memory_utilization: 0.0,
            sparsity_ratio: 0.0,
            resonance_stability: 0.0,
            attention_entropy: 0.0,
            update_frequency: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_x_parameters_new() {
        let params = StarXParameters::new();
        assert_eq!(params.parameter_count(), 0);
        assert_eq!(params.gradient_norm(), 0.0);
    }

    #[test]
    fn test_star_x_parameters_register() -> DLResult<()> {
        let mut params = StarXParameters::new();
        params.register_parameter("weight1".into(), vec![2, 3])?;
        assert_eq!(params.parameter_count(), 6);
        assert!(params.get_parameter("weight1").is_some());
        assert!(params.get_gradient("weight1").is_some());
        Ok(())
    }

    #[test]
    fn test_star_x_parameters_zero_gradients() -> DLResult<()> {
        let mut params = StarXParameters::new();
        params.register_parameter("w1".into(), vec![2, 2])?;
        if let Some(g) = params.get_gradient_mut("w1") {
            g.fill(1.0);
        }
        let grad_norm_before = params.gradient_norm();
        assert!(grad_norm_before > 0.0);
        params.zero_gradients();
        assert_eq!(params.gradient_norm(), 0.0);
        Ok(())
    }

    #[test]
    fn test_star_x_parameters_default() {
        let params: StarXParameters = Default::default();
        assert_eq!(params.parameter_count(), 0);
    }

    #[test]
    fn test_star_x_parameters_get_mut() -> DLResult<()> {
        let mut params = StarXParameters::new();
        params.register_parameter("w1".into(), vec![3])?;
        assert!(params.get_parameter_mut("w1").is_some());
        assert!(params.get_parameter_mut("nonexistent").is_none());
        Ok(())
    }

    #[test]
    fn test_star_x_config_default() {
        let config = StarXConfig::default();
        assert_eq!(config.model_dim, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.max_sequence_length, 2048);
        assert_eq!(config.sparsity_ratio, 0.1);
        assert_eq!(config.compute_levels, 3);
    }

    #[test]
    fn test_star_x_state_new() {
        let config = StarXConfig::default();
        let state = StarXState::new(&config);
        assert_eq!(state.temporal_position, 0);
        assert_eq!(state.hidden_state.shape(), &[512]);
        assert_eq!(state.compute_level, 0);
    }

    #[test]
    fn test_star_x_state_reset() {
        let config = StarXConfig::default();
        let mut state = StarXState::new(&config);
        state.temporal_position = 42;
        state.compute_level = 99;
        state.reset();
        assert_eq!(state.temporal_position, 0);
        assert_eq!(state.compute_level, 0);
    }

    #[test]
    fn test_star_x_metrics_default() {
        let metrics = StarXMetics::default();
        assert_eq!(metrics.loss, 0.0);
        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.sparsity_ratio, 0.0);
    }

    #[test]
    fn test_harmonic_encoding() {
        let encoding = core_utils::harmonic_encoding(0, 16, 64);
        assert_eq!(encoding.shape(), &[64]);
    }

    #[test]
    fn test_harmonic_encoding_first_position() {
        let encoding = core_utils::harmonic_encoding(0, 8, 32);
        assert_eq!(encoding.shape(), &[32]);
        assert!((encoding[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_sparse() {
        let scores =
            ndarray::ArrayD::from_shape_vec(vec![5], vec![0.1, 0.5, 0.3, 0.9, 0.2]).unwrap();
        let mask = core_utils::top_k_sparse(&scores, 2);
        let count: usize = mask.iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_top_k_sparse_all() {
        let scores =
            ndarray::ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let mask = core_utils::top_k_sparse(&scores, 5);
        let count: usize = mask.iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_compute_entropy_uniform() {
        let probs = vec![0.5, 0.5];
        let entropy = core_utils::compute_entropy(&probs);
        assert!((entropy - (2.0 * 0.5_f32 * 0.5_f32.ln()).abs()).abs() < 1e-6);
    }

    #[test]
    fn test_compute_entropy_certain() {
        let probs = vec![1.0, 0.0];
        let entropy = core_utils::compute_entropy(&probs);
        assert!((entropy - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut tensor =
            ndarray::ArrayD::from_shape_vec(vec![3], vec![3.0, 4.0, 0.0]).unwrap();
        let norm = core_utils::normalize(&mut tensor);
        assert!((norm - 5.0).abs() < 1e-6);
        let sum_sq: f32 = tensor.iter().map(|x| x * x).sum();
        assert!((sum_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero() {
        let mut tensor = ndarray::ArrayD::from_shape_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let norm = core_utils::normalize(&mut tensor);
        assert!((norm - 0.0).abs() < 1e-6);
        assert!(tensor.iter().all(|&x| (x - 0.0).abs() < 1e-6));
    }
}
