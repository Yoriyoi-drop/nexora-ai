//! ERP Context-Gated Partial Reconstruction
//!
//! Implementasi dari context-gated partial reconstruction untuk
//! efficient runtime inference dengan sparse activation objectives.

use crate::{CompressedLayer, ERPConfig, ERPError};
use ndarray::{Array1, Array2};
use rand::Rng;

/// Context-gated reconstructor untuk ERP inference
pub struct ContextReconstructor {
    config: ERPConfig,
    gate_network: GateNetwork,
    reconstruction_method: ReconstructionMethod,
}

#[derive(Debug, Clone)]
pub enum ReconstructionMethod {
    FullDecompression,
    SparseGated { target_sparsity: f32 },
    AdaptiveReconstruction { budget: usize },
}

impl ContextReconstructor {
    pub fn new(config: ERPConfig) -> Self {
        let reconstruction_method = match config.compression_mode {
            crate::CompressionMode::Conservative => ReconstructionMethod::FullDecompression,
            crate::CompressionMode::Balanced => ReconstructionMethod::SparseGated {
                target_sparsity: 0.3,
            },
            crate::CompressionMode::Aggressive => {
                ReconstructionMethod::AdaptiveReconstruction { budget: 64 }
            }
        };

        Self {
            config: config.clone(),
            gate_network: GateNetwork::new(config.clone()),
            reconstruction_method,
        }
    }

    /// Compute gates untuk input konteks
    pub fn compute_gates(
        &self,
        compressed_layers: &[CompressedLayer],
        input: &Array1<f32>,
    ) -> Result<Vec<GatePattern>, ERPError> {
        let mut all_gates = Vec::new();

        for layer in compressed_layers {
            let layer_gates = self.compute_layer_gates(layer, input)?;
            all_gates.push(layer_gates);
        }

        Ok(all_gates)
    }

    /// Compute gates untuk individual layer
    fn compute_layer_gates(
        &self,
        layer: &CompressedLayer,
        input: &Array1<f32>,
    ) -> Result<GatePattern, ERPError> {
        let mut gates = Array1::zeros(layer.original_weights.nrows());
        let mut active_neurons = Vec::new();

        // Compute context embedding
        let context_embedding = self.compute_context_embedding(input, layer.layer_idx);

        // Compute gate scores untuk setiap neuron/resonance group
        for resonance_rep in layer.resonance_representations.iter() {
            let gate_score = self
                .gate_network
                .compute_gate_score(&context_embedding, resonance_rep);

            // Apply sparsity constraints
            let is_active =
                self.apply_sparsity_constraint(gate_score, &resonance_rep.importance_coeffs);

            for &neuron_idx in &resonance_rep.group_neurons {
                gates[neuron_idx] = if is_active { gate_score } else { 0.0 };
                if is_active {
                    active_neurons.push(neuron_idx);
                }
            }
        }

        // Handle original neurons (non-compressed)
        for (i, status) in layer.neuron_status.iter().enumerate() {
            if matches!(status, crate::compression::NeuronStatus::Original) {
                gates[i] = 1.0; // Always active for original neurons
                active_neurons.push(i);
            }
        }

        Ok(GatePattern {
            layer_idx: layer.layer_idx,
            gates: gates.clone(),
            active_neurons,
            sparsity_ratio: self.compute_sparsity_ratio(&gates),
        })
    }

    /// Reconstruct output dengan pre-computed gates
    pub fn reconstruct_with_gates(
        &self,
        compressed_layers: &[CompressedLayer],
        input: &Array1<f32>,
        gates: &[GatePattern],
    ) -> Result<Array1<f32>, ERPError> {
        let mut current_input = input.clone();

        for (layer, layer_gates) in compressed_layers.iter().zip(gates.iter()) {
            current_input = self.reconstruct_layer_output(layer, &current_input, layer_gates)?;
        }

        Ok(current_input)
    }

    /// Reconstruct output untuk individual layer
    fn reconstruct_layer_output(
        &self,
        layer: &CompressedLayer,
        input: &Array1<f32>,
        gates: &GatePattern,
    ) -> Result<Array1<f32>, ERPError> {
        match &self.reconstruction_method {
            ReconstructionMethod::FullDecompression => self.full_decompression(layer, input),
            ReconstructionMethod::SparseGated { target_sparsity } => {
                self.sparse_gated_reconstruction(layer, input, gates, *target_sparsity)
            }
            ReconstructionMethod::AdaptiveReconstruction { budget } => {
                self.adaptive_reconstruction(layer, input, gates, *budget)
            }
        }
    }

    /// Full decompression reconstruction
    fn full_decompression(
        &self,
        layer: &CompressedLayer,
        input: &Array1<f32>,
    ) -> Result<Array1<f32>, ERPError> {
        // Try GPU matvec first
        #[cfg(feature = "gpu")]
        {
            use crate::gpu;
            if let Some(Ok(result)) = gpu::try_matvec(&layer.compressed_weights, input) {
                return Ok(result);
            }
        }

        // Gunakan compressed weights langsung
        let output = layer.compressed_weights.dot(input);
        Ok(output)
    }

    /// Sparse gated reconstruction
    fn sparse_gated_reconstruction(
        &self,
        layer: &CompressedLayer,
        input: &Array1<f32>,
        gates: &GatePattern,
        target_sparsity: f32,
    ) -> Result<Array1<f32>, ERPError> {
        let nrows = layer.compressed_weights.nrows();

        // Try GPU full matvec then apply gates
        #[cfg(feature = "gpu")]
        {
            use crate::gpu;
            if let Some(Ok(full_output)) = gpu::try_matvec(&layer.compressed_weights, input) {
                let mut output = full_output;
                for i in 0..nrows {
                    output[i] *= gates.gates[i];
                }
                self.apply_sparse_regularization(&mut output, target_sparsity);
                return Ok(output);
            }
        }

        let mut output = Array1::zeros(nrows);

        // Hanya proses active neurons
        for &neuron_idx in &gates.active_neurons {
            let neuron_output = layer.compressed_weights.row(neuron_idx).dot(input);
            output[neuron_idx] = neuron_output * gates.gates[neuron_idx];
        }

        // Apply sparse activation regularization
        self.apply_sparse_regularization(&mut output, target_sparsity);

        Ok(output)
    }

    /// Adaptive reconstruction dengan compute budget
    fn adaptive_reconstruction(
        &self,
        layer: &CompressedLayer,
        input: &Array1<f32>,
        gates: &GatePattern,
        budget: usize,
    ) -> Result<Array1<f32>, ERPError> {
        let mut output = Array1::zeros(layer.compressed_weights.nrows());

        // Prioritize neurons berdasarkan importance scores
        let mut prioritized_neurons: Vec<_> = gates
            .active_neurons
            .iter()
            .map(|&neuron_idx| {
                let importance = self.compute_neuron_importance(neuron_idx, layer);
                (neuron_idx, importance)
            })
            .collect();

        prioritized_neurons
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Proses top-k neurons sesuai budget
        let num_active = std::cmp::min(budget, prioritized_neurons.len());

        // Try GPU matvec for active neurons only (extract sub-matrix)
        #[cfg(feature = "gpu")]
        {
            use crate::gpu;
            use ndarray::Array2;
            let active: Vec<usize> = prioritized_neurons
                .iter()
                .take(num_active)
                .map(|(idx, _)| *idx)
                .collect();
            if !active.is_empty() {
                let dim = input.len();
                let mut sub_weights = Array2::zeros((active.len(), dim));
                for (row, &neuron_idx) in active.iter().enumerate() {
                    for col in 0..dim {
                        sub_weights[[row, col]] = layer.compressed_weights[[neuron_idx, col]];
                    }
                }
                if let Some(Ok(sub_output)) = gpu::try_matvec(&sub_weights, input) {
                    for (row, &neuron_idx) in active.iter().enumerate() {
                        if row < sub_output.len() {
                            output[neuron_idx] = sub_output[row] * gates.gates[neuron_idx];
                        }
                    }
                    return Ok(output);
                }
            }
        }

        for (neuron_idx, _) in prioritized_neurons.iter().take(num_active) {
            let neuron_output = layer.compressed_weights.row(*neuron_idx).dot(input);
            output[*neuron_idx] = neuron_output * gates.gates[*neuron_idx];
        }

        Ok(output)
    }

    /// Compute context embedding dari input
    fn compute_context_embedding(&self, input: &Array1<f32>, layer_idx: usize) -> Array1<f32> {
        // Simplified context embedding - dalam implementasi nyata gunakan learned embedding
        let mut embedding = Array1::zeros(64); // Fixed embedding size

        // Hash input ke embedding space
        for &value in input.iter() {
            let hash_idx = (value.abs() * 1000.0) as usize % embedding.len();
            embedding[hash_idx] += value;
        }

        // Add layer information
        embedding[0] = layer_idx as f32;

        embedding
    }

    /// Apply sparsity constraint
    fn apply_sparsity_constraint(&self, gate_score: f32, importance_coeffs: &Array1<f32>) -> bool {
        // Combine gate score dengan importance
        let avg_importance = importance_coeffs.mean().unwrap_or(0.0);
        let combined_score = gate_score * (1.0 + avg_importance);

        // Threshold untuk activation
        combined_score > 0.5
    }

    /// Compute sparsity ratio
    fn compute_sparsity_ratio(&self, gates: &Array1<f32>) -> f32 {
        let active_count = gates.iter().filter(|&&x| x > 0.0).count();
        let total_count = gates.len();

        if total_count > 0 {
            1.0 - (active_count as f32 / total_count as f32)
        } else {
            1.0
        }
    }

    /// Apply sparse activation regularization
    fn apply_sparse_regularization(&self, output: &mut Array1<f32>, target_sparsity: f32) {
        // L1 regularization untuk sparsity
        let l1_penalty = self.config.sparse_regularization;

        for i in 0..output.len() {
            if output[i].abs() < l1_penalty {
                output[i] = 0.0;
            } else if output[i] > 0.0 {
                output[i] -= l1_penalty;
            } else {
                output[i] += l1_penalty;
            }
        }

        // Adjust untuk target sparsity
        let current_sparsity = self.compute_output_sparsity(output);
        if current_sparsity < target_sparsity {
            // Increase sparsity
            let threshold = self.compute_sparsity_threshold(output, target_sparsity);
            for i in 0..output.len() {
                if output[i].abs() < threshold {
                    output[i] = 0.0;
                }
            }
        }
    }

    /// Compute output sparsity
    fn compute_output_sparsity(&self, output: &Array1<f32>) -> f32 {
        let zero_count = output.iter().filter(|&&x| x == 0.0).count();
        let total_count = output.len();

        zero_count as f32 / total_count as f32
    }

    /// Compute sparsity threshold
    fn compute_sparsity_threshold(&self, output: &Array1<f32>, target_sparsity: f32) -> f32 {
        let mut sorted_values: Vec<f32> = output.iter().map(|&x| x.abs()).collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let target_zeros = (target_sparsity * output.len() as f32) as usize;
        if target_zeros < sorted_values.len() {
            sorted_values[target_zeros]
        } else {
            sorted_values[sorted_values.len() - 1]
        }
    }

    /// Compute neuron importance untuk adaptive reconstruction
    fn compute_neuron_importance(&self, neuron_idx: usize, layer: &CompressedLayer) -> f32 {
        // Cari neuron dalam resonance representations
        for resonance_rep in &layer.resonance_representations {
            if let Some(pos) = resonance_rep
                .group_neurons
                .iter()
                .position(|&x| x == neuron_idx)
            {
                return resonance_rep.importance_coeffs[pos];
            }
        }

        // Default importance untuk original neurons
        1.0
    }

    pub fn gate_network_mut(&mut self) -> &mut GateNetwork {
        &mut self.gate_network
    }

    pub fn gate_network(&self) -> &GateNetwork {
        &self.gate_network
    }
}

/// Gate network untuk context-gated reconstruction
pub struct GateNetwork {
    _config: ERPConfig,
    gate_weights: Array2<f32>,
    gate_bias: Array1<f32>,
}

impl GateNetwork {
    pub fn new(config: ERPConfig) -> Self {
        // Initialize gate network weights
        let input_dim = 64; // Context embedding size
        let mut rng = rand::thread_rng();
        let gate_weights = Array2::from_shape_fn((input_dim, 1), |_| rng.gen());
        let gate_bias = Array1::zeros(1);

        Self {
            _config: config,
            gate_weights,
            gate_bias,
        }
    }

    /// Compute gate score untuk resonance group
    pub fn compute_gate_score(
        &self,
        context_embedding: &Array1<f32>,
        resonance_rep: &crate::compression::ResonanceRepresentation,
    ) -> f32 {
        // Compute weighted context signal
        let context_signal = {
            #[cfg(feature = "gpu")]
            {
                use crate::gpu;
                let flat: Vec<f32> = self.gate_weights.iter().copied().collect();
                let w = match ndarray::Array2::from_shape_vec((flat.len(), 1), flat) {
                    Ok(w) => w,
                    Err(_) => self.gate_weights.clone(),
                };
                let col: Vec<f32> = (0..self.gate_weights.nrows())
                    .map(|i| self.gate_weights[[i, 0]])
                    .collect();
                let col_arr = match ndarray::Array2::from_shape_vec(
                    (self.gate_weights.nrows(), 1),
                    col,
                ) {
                    Ok(w) => w,
                    Err(_) => self.gate_weights.clone(),
                };
                let ctx_dot = crate::gpu_try!(gpu::try_matvec(
                    &col_arr,
                    context_embedding,
                ));
                if let Some(val) = ctx_dot {
                    val[0]
                } else {
                    context_embedding.dot(&self.gate_weights)[0] + self.gate_bias[0]
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                context_embedding.dot(&self.gate_weights)[0] + self.gate_bias[0]
            }
        };

        // Apply sigmoid activation
        let gate_score = 1.0 / (1.0 + (-context_signal).exp());

        // Modulate dengan importance coefficients
        let avg_importance = resonance_rep.importance_coeffs.mean().unwrap_or(0.0);

        gate_score * (1.0 + avg_importance * 0.5) // Importance modulation
    }

    /// Update gate weights (training)
    pub fn update_weights(&mut self, gradient: &Array2<f32>, learning_rate: f32) {
        self.gate_weights -= &(gradient * learning_rate);

        // Apply weight decay untuk prevent overfitting
        let weight_decay = 1e-4;
        self.gate_weights *= 1.0 - weight_decay;
    }

    /// Get current gate weights
    pub fn get_weights(&self) -> &Array2<f32> {
        &self.gate_weights
    }

    /// Get current gate bias
    pub fn get_bias(&self) -> &Array1<f32> {
        &self.gate_bias
    }

    pub fn set_weights(&mut self, weights: Array2<f32>) {
        self.gate_weights = weights;
    }

    pub fn set_bias(&mut self, bias: Array1<f32>) {
        self.gate_bias = bias;
    }
}

/// Gate pattern untuk layer
#[derive(Debug, Clone)]
pub struct GatePattern {
    pub layer_idx: usize,
    pub gates: Array1<f32>,
    pub active_neurons: Vec<usize>,
    pub sparsity_ratio: f32,
}

/// Sparse activation objective
pub struct SparseActivationObjective {
    pub lambda_l1: f32,
    pub target_sparsity: f32,
}

impl SparseActivationObjective {
    pub fn new(lambda_l1: f32, target_sparsity: f32) -> Self {
        Self {
            lambda_l1,
            target_sparsity,
        }
    }

    /// Compute sparse activation loss
    pub fn compute_loss(&self, gates: &Array1<f32>) -> f32 {
        // L1 penalty
        let l1_loss = self.lambda_l1 * gates.iter().map(|x| x.abs()).sum::<f32>();

        // Sparsity penalty
        let current_sparsity =
            gates.iter().filter(|&&x| x == 0.0).count() as f32 / gates.len() as f32;
        let sparsity_loss = (current_sparsity - self.target_sparsity).powi(2);

        l1_loss + sparsity_loss
    }

    /// Compute gradient untuk sparse activation
    pub fn compute_gradient(&self, gates: &Array1<f32>) -> Array1<f32> {
        let mut gradient = Array1::zeros(gates.len());

        for i in 0..gates.len() {
            // L1 gradient
            gradient[i] = self.lambda_l1 * if gates[i] > 0.0 { 1.0 } else { -1.0 };

            // Add small epsilon untuk avoid zero gradient
            if gates[i].abs() < 1e-8 {
                gradient[i] = 0.0;
            }
        }

        gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::{NeuronStatus, ResonanceRepresentation};

    fn cfg() -> ERPConfig {
        ERPConfig::default()
    }

    fn layer(layer_idx: usize, nrows: usize, ncols: usize) -> CompressedLayer {
        CompressedLayer {
            layer_idx,
            original_weights: Array2::zeros((nrows, ncols)),
            compressed_weights: Array2::zeros((nrows, ncols)),
            resonance_representations: vec![],
            neuron_status: vec![NeuronStatus::Original; nrows],
            compression_ratio: 0.0,
        }
    }

    fn rep() -> ResonanceRepresentation {
        ResonanceRepresentation {
            group_neurons: vec![0, 1],
            superposed_weights: Array2::zeros((1, 4)),
            importance_coeffs: Array1::from_vec(vec![0.6, 0.4]),
            residual: crate::compression::ResidualRepresentation::FullResidual(Array2::zeros((
                2, 4,
            ))),
        }
    }

    // ── ContextReconstructor ──

    #[test]
    fn test_new_conservative() {
        let r = ContextReconstructor::new(ERPConfig {
            compression_mode: crate::CompressionMode::Conservative,
            ..cfg()
        });
        assert!(matches!(
            r.reconstruction_method,
            ReconstructionMethod::FullDecompression
        ));
    }

    #[test]
    fn test_new_balanced() {
        let r = ContextReconstructor::new(ERPConfig {
            compression_mode: crate::CompressionMode::Balanced,
            ..cfg()
        });
        assert!(matches!(
            r.reconstruction_method,
            ReconstructionMethod::SparseGated {
                target_sparsity: 0.3
            }
        ));
    }

    #[test]
    fn test_new_aggressive() {
        let r = ContextReconstructor::new(ERPConfig {
            compression_mode: crate::CompressionMode::Aggressive,
            ..cfg()
        });
        assert!(matches!(
            r.reconstruction_method,
            ReconstructionMethod::AdaptiveReconstruction { budget: 64 }
        ));
    }

    #[test]
    fn test_compute_gates_no_resonance() {
        let r = ContextReconstructor::new(cfg());
        let layers = vec![layer(0, 3, 4)];
        let input = Array1::from_vec(vec![1.0; 4]);
        let gates = r.compute_gates(&layers, &input).unwrap();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gates.len(), 3);
        // All original neurons have gate 1.0
        assert!(gates[0].gates.iter().all(|&g| (g - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_compute_gates_with_resonance() {
        let r = ContextReconstructor::new(cfg());
        let mut l = layer(0, 3, 4);
        l.resonance_representations = vec![rep()];
        l.neuron_status = vec![
            NeuronStatus::Compressed,
            NeuronStatus::Compressed,
            NeuronStatus::Original,
        ];
        let layers = vec![l];
        let input = Array1::from_vec(vec![1.0; 4]);
        let gates = r.compute_gates(&layers, &input).unwrap();
        assert_eq!(gates.len(), 1);
        // Neuron idx 2 is original, should have gate 1.0
        assert!((gates[0].gates[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_reconstruct_with_gates() {
        let r = ContextReconstructor::new(cfg());
        let mut l = layer(0, 2, 2);
        l.compressed_weights = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let input = Array1::from_vec(vec![2.0, 3.0]);
        let gate = GatePattern {
            layer_idx: 0,
            gates: Array1::from_vec(vec![1.0, 1.0]),
            active_neurons: vec![0, 1],
            sparsity_ratio: 0.0,
        };
        let result = r.reconstruct_with_gates(&[l], &input, &[gate]).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_full_decompression() {
        let r = ContextReconstructor::new(cfg());
        let l = layer(0, 2, 3);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = r.full_decompression(&l, &input).unwrap();
        assert_eq!(result.len(), 2);
        // All zeros because compressed_weights is zeros
        assert!(result.iter().all(|&x| (x - 0.0).abs() < 1e-5));
    }

    #[test]
    fn test_sparse_gated_reconstruction() {
        let r = ContextReconstructor::new(cfg());
        let l = layer(0, 3, 2);
        let input = Array1::from_vec(vec![1.0, 2.0]);
        let gates = GatePattern {
            layer_idx: 0,
            gates: Array1::from_vec(vec![0.5, 0.0, 0.8]),
            active_neurons: vec![0, 2],
            sparsity_ratio: 1.0 / 3.0,
        };
        let result = r
            .sparse_gated_reconstruction(&l, &input, &gates, 0.5)
            .unwrap();
        assert_eq!(result.len(), 3);
        // Neuron 1 should be 0 (gate = 0.0)
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn test_compute_context_embedding() {
        let r = ContextReconstructor::new(cfg());
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let embedding = r.compute_context_embedding(&input, 0);
        assert_eq!(embedding.len(), 64);
        assert_eq!(embedding[0], 0.0); // layer_idx
    }

    #[test]
    fn test_apply_sparsity_constraint() {
        let r = ContextReconstructor::new(cfg());
        let imp = Array1::from_vec(vec![0.5, 0.5]);
        // gate_score=1.0, avg_importance=0.5
        // combined = 1.0 * (1 + 0.5) = 1.5 > 0.5 => true
        assert!(r.apply_sparsity_constraint(1.0, &imp));
        // gate_score=0.0, avg_importance=0.5
        // combined = 0.0 * 1.5 = 0.0 <= 0.5 => false
        assert!(!r.apply_sparsity_constraint(0.0, &imp));
    }

    #[test]
    fn test_compute_sparsity_ratio() {
        let r = ContextReconstructor::new(cfg());
        let gates = Array1::from_vec(vec![1.0, 0.0, 0.5, 0.0]);
        let ratio = r.compute_sparsity_ratio(&gates);
        assert!((ratio - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_compute_sparsity_ratio_empty() {
        let r = ContextReconstructor::new(cfg());
        let gates = Array1::<f32>::from_vec(vec![]);
        assert_eq!(r.compute_sparsity_ratio(&gates), 1.0);
    }

    #[test]
    fn test_adaptive_reconstruction() {
        let r = ContextReconstructor::new(ERPConfig {
            compression_mode: crate::CompressionMode::Aggressive,
            ..cfg()
        });
        let l = layer(0, 3, 2);
        let input = Array1::from_vec(vec![1.0, 2.0]);
        let gates = GatePattern {
            layer_idx: 0,
            gates: Array1::from_vec(vec![0.5, 0.5, 0.5]),
            active_neurons: vec![0, 1, 2],
            sparsity_ratio: 0.0,
        };
        let result = r.adaptive_reconstruction(&l, &input, &gates, 2).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_compute_output_sparsity() {
        let r = ContextReconstructor::new(cfg());
        let out = Array1::from_vec(vec![0.0, 1.5, 0.0, 0.3]);
        assert!((r.compute_output_sparsity(&out) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_compute_sparsity_threshold() {
        let r = ContextReconstructor::new(cfg());
        let out = Array1::from_vec(vec![0.1, 0.5, 0.0, 1.0, 0.2]);
        let threshold = r.compute_sparsity_threshold(&out, 0.6);
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_compute_neuron_importance_default() {
        let r = ContextReconstructor::new(cfg());
        let l = layer(0, 3, 2);
        let importance = r.compute_neuron_importance(0, &l);
        assert_eq!(importance, 1.0);
    }

    // ── GateNetwork ──

    #[test]
    fn test_gate_network_compute_gate_score() {
        let gn = GateNetwork::new(cfg());
        let ctx = Array1::from_vec(vec![0.5; 64]);
        let rep = rep();
        let score = gn.compute_gate_score(&ctx, &rep);
        assert!(score > 0.0 && score < 2.0);
    }

    #[test]
    fn test_gate_network_update_weights() {
        let mut gn = GateNetwork::new(cfg());
        let grad = Array2::from_shape_fn((64, 1), |_| 0.01);
        let old_weights = gn.gate_weights.clone();
        gn.update_weights(&grad, 0.1);
        let diff = (&old_weights - &gn.gate_weights).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_gate_network_get_weights_and_bias() {
        let gn = GateNetwork::new(cfg());
        assert_eq!(gn.get_weights().shape(), &[64, 1]);
        assert_eq!(gn.get_bias().len(), 1);
    }

    // ── SparseActivationObjective ──

    #[test]
    fn test_objective_compute_loss_all_zeros() {
        let obj = SparseActivationObjective::new(0.1, 0.5);
        let gates = Array1::zeros(10);
        let loss = obj.compute_loss(&gates);
        // sparsity = 1.0, target = 0.5 => sparsity_loss = (1.0 - 0.5)^2 = 0.25
        // l1_loss = 0.1 * 0 = 0
        assert!((loss - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_objective_compute_loss_all_active() {
        let obj = SparseActivationObjective::new(0.1, 0.5);
        let gates = Array1::from_vec(vec![1.0; 10]);
        let loss = obj.compute_loss(&gates);
        // sparsity = 0.0, target = 0.5 => sparsity_loss = (0.0 - 0.5)^2 = 0.25
        // l1_loss = 0.1 * 10 = 1.0
        assert!((loss - 1.25).abs() < 1e-5);
    }

    #[test]
    fn test_objective_compute_gradient() {
        let obj = SparseActivationObjective::new(0.1, 0.5);
        let gates = Array1::from_vec(vec![1.0, -2.0, 0.0]);
        let grad = obj.compute_gradient(&gates);
        assert_eq!(grad.len(), 3);
        // g[0] > 0 => lambda_l1 * 1 = 0.1
        assert!((grad[0] - 0.1).abs() < 1e-5);
        // g[1] < 0 => lambda_l1 * -1 = -0.1
        assert!((grad[1] + 0.1).abs() < 1e-5);
        // g[2] == 0 => grad[2] = 0.0
        assert_eq!(grad[2], 0.0);
    }
}
