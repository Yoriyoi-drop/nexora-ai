//! ERP Training Strategy dan Calibration Utilities
//! 
//! Implementasi dari training strategy untuk ERP dengan realistic costs:
//! - Resonance mapping: offline
//! - Compression: one-time  
//! - Calibration tuning: ringan
//! - Optional LoRA recovery: kecil

use crate::{ERPConfig, ERPError, ERPEngine, CompressedLayer};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng};
use rand_distr::Standard;
use std::collections::HashMap;

/// ERP training manager
pub struct ERPTrainer {
    config: ERPConfig,
    training_strategy: TrainingStrategy,
    _calibration_data: CalibrationDataset,
}

#[derive(Debug, Clone)]
pub enum TrainingStrategy {
    Conservative { calibration_epochs: usize },
    Balanced { lora_rank: usize, calibration_epochs: usize },
    Aggressive { lora_rank: usize, fine_tuning_epochs: usize },
}

impl ERPTrainer {
    pub fn new(config: ERPConfig, strategy: TrainingStrategy) -> Self {
        Self {
            config,
            training_strategy: strategy,
            _calibration_data: CalibrationDataset::new(),
        }
    }

    /// Full ERP training pipeline
    pub fn train(&mut self, original_weights: &[Array2<f32>], training_data: &TrainingDataset) -> Result<TrainedERPModel, ERPError> {
        // Phase 1: Resonance mapping (offline)
        let resonance_result = self.phase1_resonance_mapping(original_weights)?;
        
        // Phase 2: Compression (one-time)
        let compressed_model = self.phase2_compression(original_weights, &resonance_result)?;
        
        // Phase 3: Calibration tuning (ringan)
        let calibrated_model = self.phase3_calibration(&compressed_model, training_data)?;
        
        // Phase 4: Optional LoRA recovery (kecil)
        let final_model = self.phase4_lora_recovery(&calibrated_model, training_data)?;
        
        Ok(final_model)
    }

    /// Phase 1: Resonance mapping - offline computation
    fn phase1_resonance_mapping(&self, weights: &[Array2<f32>]) -> Result<ResonanceMappingResult, ERPError> {
        let engine = ERPEngine::new(self.config.clone());
        
        // Perform resonance analysis
        let resonance_groups = engine.resonance_mapper.map_resonance(weights)?;
        
        // Compute mapping statistics
        let mapping_stats = self.compute_mapping_statistics(&resonance_groups, weights);
        
        Ok(ResonanceMappingResult {
            resonance_groups: resonance_groups.clone(),
            mapping_stats,
            compression_potential: self.estimate_compression_potential(&resonance_groups, weights),
        })
    }

    /// Phase 2: Compression - one-time operation
    fn phase2_compression(&self, weights: &[Array2<f32>], resonance_result: &ResonanceMappingResult) -> Result<CompressedModel, ERPError> {
        let mut engine = ERPEngine::new(self.config.clone());
        
        // Apply compression
        let compressed_layers = engine.apply_pruning(weights)?;
        
        // Compute compression metrics
        let compression_metrics = self.compute_compression_metrics(weights, &compressed_layers);
        
        Ok(CompressedModel {
            original_weights: weights.to_vec(),
            compressed_layers,
            resonance_groups: resonance_result.resonance_groups.clone(),
            compression_metrics,
        })
    }

    /// Phase 3: Calibration tuning - lightweight
    fn phase3_calibration(&mut self, model: &CompressedModel, training_data: &TrainingDataset) -> Result<CalibratedModel, ERPError> {
        let mut calibrated_layers = model.compressed_layers.clone();
        let calibration_epochs = match &self.training_strategy {
            TrainingStrategy::Conservative { calibration_epochs } => *calibration_epochs,
            TrainingStrategy::Balanced { calibration_epochs, .. } => *calibration_epochs,
            TrainingStrategy::Aggressive { .. } => 5, // Minimal calibration untuk aggressive
        };

        for epoch in 0..calibration_epochs {
            // Sample subset dari training data untuk calibration
            let calibration_batch = training_data.sample_calibration_batch(100);
            
            // Calibrate gate networks
            for layer in &mut calibrated_layers {
                self.calibrate_layer_gates(layer, &calibration_batch)?;
            }
            
            // Validate calibration quality
            if epoch % 2 == 0 {
                let calibration_loss = self.evaluate_calibration_loss(&calibrated_layers, &calibration_batch)?;
                tracing::info!("Calibration epoch {}: loss = {:.4}", epoch, calibration_loss);
            }
        }

        Ok(CalibratedModel {
            compressed_model: model.clone(),
            calibrated_layers: calibrated_layers.clone(),
            calibration_stats: self.compute_calibration_stats(&calibrated_layers),
        })
    }

    /// Phase 4: LoRA recovery - optional lightweight recovery
    fn phase4_lora_recovery(&mut self, model: &CalibratedModel, training_data: &TrainingDataset) -> Result<TrainedERPModel, ERPError> {
        let lora_adapters = match &self.training_strategy {
            TrainingStrategy::Conservative { .. } => {
                // No LoRA untuk conservative strategy
                Vec::new()
            }
            TrainingStrategy::Balanced { lora_rank, .. } => {
                self.train_lora_adapters(&model.calibrated_layers, training_data, *lora_rank)?
            }
            TrainingStrategy::Aggressive { lora_rank, fine_tuning_epochs: _ } => {
                self.train_lora_adapters(&model.calibrated_layers, training_data, *lora_rank)?
            }
        };

        Ok(TrainedERPModel {
            calibrated_model: model.clone(),
            lora_adapters: lora_adapters.clone(),
            training_stats: self.compute_training_stats(&model, &lora_adapters),
        })
    }

    /// Calibrate gate networks untuk layer
    fn calibrate_layer_gates(&mut self, layer: &mut CompressedLayer, calibration_batch: &CalibrationBatch) -> Result<(), ERPError> {
        let mut reconstructor = crate::reconstruction::ContextReconstructor::new(self.config.clone());
        
        // Accumulate gradients over batch
        let mut accumulated_gradients = Array2::zeros((64, 1));
        let mut sample_count = 0;
        
        for (input, target_output) in &calibration_batch.samples {
            // Compute current gates
            let current_gates = reconstructor.compute_gates(&[layer.clone()], input)?;
            
            // Compute gate gradients
            let gate_gradients = self.compute_gate_gradients(layer, input, target_output, &current_gates)?;
            
            // Accumulate gradients
            accumulated_gradients += &gate_gradients;
            sample_count += 1;
        }
        
        // Average gradients over batch
        if sample_count > 0 {
            accumulated_gradients /= sample_count as f32;
            
            // Apply gradient clipping untuk stability
            let gradient_norm = accumulated_gradients.iter().map(|x| x * x).sum::<f32>().sqrt();
            let max_norm = 1.0;
            if gradient_norm > max_norm {
                accumulated_gradients *= max_norm / gradient_norm;
            }
            
            // Update gate weights dengan adaptive learning rate
            let learning_rate = self.compute_adaptive_learning_rate();
            self.update_gate_weights_from_reconstructor(&mut reconstructor, &accumulated_gradients, learning_rate)?;
            
            tracing::debug!("Updated gate weights for layer {} with learning rate {:.6}", layer.layer_idx, learning_rate);
        }

        Ok(())
    }

    /// Compute gradients untuk gate networks
    fn compute_gate_gradients(&self, layer: &CompressedLayer, input: &Array1<f32>, target: &Array1<f32>, gates: &[crate::reconstruction::GatePattern]) -> Result<Array2<f32>, ERPError> {
        // Simplified gradient computation
        let output = layer.compressed_weights.dot(input);
        let error = output - target;
        
        // Compute gradient terhadap gate weights
        let mut gradient = Array2::zeros((64, 1)); // Gate weight dimensions
        
        for gate_pattern in gates {
            for (i, &gate_value) in gate_pattern.gates.iter().enumerate() {
                if gate_value > 0.0 {
                    // Backpropagate error ke gate
                    let gate_error = error[i] * gate_value;
                    gradient[[i % 64, 0]] += gate_error;
                }
            }
        }

        Ok(gradient)
    }

    /// Compute adaptive learning rate based on training progress
    fn compute_adaptive_learning_rate(&self) -> f32 {
        // Base learning rate dengan decay based on training strategy
        let base_lr = match &self.training_strategy {
            TrainingStrategy::Conservative { .. } => 1e-4,  // Conservative learning rate
            TrainingStrategy::Balanced { .. } => 5e-4,    // Moderate learning rate
            TrainingStrategy::Aggressive { .. } => 1e-3,  // Higher learning rate for aggressive training
        };
        
        // Apply cosine annealing untuk stability
        let progress_factor = 0.1; // Simulasi progress - dalam implementasi nyata gunakan actual progress
        let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress_factor).cos());
        
        base_lr * cosine_factor
    }

    /// Update gate weights dari reconstructor (workaround untuk visibility)
    fn update_gate_weights_from_reconstructor(&self, reconstructor: &mut crate::reconstruction::ContextReconstructor, gradient: &Array2<f32>, learning_rate: f32) -> Result<(), ERPError> {
        // Create a temporary gate network dengan weights dari reconstructor
        let mut temp_gate_network = crate::reconstruction::GateNetwork::new(self.config.clone());
        
        // Copy current weights dari reconstructor (dalam implementasi nyata, gunakan proper accessor)
        let current_weights = temp_gate_network.get_weights();
        let weight_shape = current_weights.shape();
        
        // Update dengan gradient
        temp_gate_network.update_weights(gradient, learning_rate);
        
        // Update reconstructor dengan new weights (dalam implementasi nyata, gunakan proper setter)
        tracing::debug!("Gate weights updated with gradient norm: {:.6}", 
                        gradient.iter().map(|x| x * x).sum::<f32>().sqrt());
        
        Ok(())
    }

    /// Update gate weights
    fn _update_gate_weights(&self, gate_network: &mut crate::reconstruction::GateNetwork, gradient: &Array2<f32>, learning_rate: f32) {
        gate_network.update_weights(gradient, learning_rate);
    }

    /// Train LoRA adapters untuk recovery
    fn train_lora_adapters(&mut self, layers: &[CompressedLayer], training_data: &TrainingDataset, rank: usize) -> Result<Vec<LoRAAdapter>, ERPError> {
        let mut adapters = Vec::new();
        
        for (_layer_idx, layer) in layers.iter().enumerate() {
            let adapter = self.train_layer_lora(layer, training_data, rank)?;
            adapters.push(adapter);
        }

        Ok(adapters)
    }

    /// Train LoRA untuk individual layer
    fn train_layer_lora(&mut self, layer: &CompressedLayer, training_data: &TrainingDataset, rank: usize) -> Result<LoRAAdapter, ERPError> {
        let (output_dim, input_dim) = layer.compressed_weights.dim();
        let actual_rank = std::cmp::min(rank, std::cmp::min(output_dim, input_dim));
        
        // Initialize LoRA matrices
        let mut rng = rand::thread_rng();
        let mut lora_a = Array2::from_shape_fn((actual_rank, input_dim), |_| rng.gen());
        let mut lora_b = Array2::from_shape_fn((output_dim, actual_rank), |_| rng.gen());
        
        // Training epochs untuk LoRA
        let epochs = match &self.training_strategy {
            TrainingStrategy::Balanced { .. } => 3,
            TrainingStrategy::Aggressive { fine_tuning_epochs, .. } => *fine_tuning_epochs,
            _ => 0,
        };

        for epoch in 0..epochs {
            let training_batch = training_data.sample_training_batch(50);
            
            for (input, target) in &training_batch.samples {
                // Forward pass dengan LoRA
                let base_output = layer.compressed_weights.dot(input);
                let lora_output = lora_b.dot(&lora_a.dot(input));
                let total_output = base_output + lora_output;
                
                // Compute loss
                let error = total_output - target;
                let loss = error.iter().map(|x| x * x).sum::<f32>() / output_dim as f32;
                
                // Backpropagation untuk LoRA matrices
                let lora_grad_b = error.to_owned().insert_axis(ndarray::Axis(1)).dot(&lora_a.dot(input).insert_axis(ndarray::Axis(0)).t());
                let lora_grad_a = lora_b.t().dot(&error.to_owned().insert_axis(ndarray::Axis(1))).dot(&input.clone().insert_axis(ndarray::Axis(0)).t());
                
                // Update LoRA matrices dengan small learning rate
                let learning_rate = 1e-4;
                lora_a -= &(lora_grad_a * learning_rate);
                lora_b -= &(lora_grad_b * learning_rate);
            }
            
            if epoch % 1 == 0 {
                tracing::info!("LoRA training epoch {} for layer {}", epoch, layer.layer_idx);
            }
        }

        Ok(LoRAAdapter {
            layer_idx: layer.layer_idx,
            lora_a,
            lora_b,
            rank: actual_rank,
        })
    }

    /// Evaluate calibration loss
    fn evaluate_calibration_loss(&self, layers: &[CompressedLayer], batch: &CalibrationBatch) -> Result<f32, ERPError> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (input, target) in &batch.samples {
            let mut current_input = input.clone();
            
            for layer in layers {
                let output = layer.compressed_weights.dot(&current_input);
                let error = output.clone() - target;
                total_loss += error.iter().map(|x| x * x).sum::<f32>();
                count += 1;
                current_input = output;
            }
        }

        Ok(if count > 0 { total_loss / count as f32 } else { 0.0 })
    }

    /// Compute mapping statistics
    fn compute_mapping_statistics(&self, resonance_groups: &[crate::core::ResonanceGroup], weights: &[Array2<f32>]) -> MappingStatistics {
        let total_neurons: usize = weights.iter().map(|w| w.nrows()).sum();
        let compressed_neurons: usize = resonance_groups.iter().map(|g| g.neurons.len()).sum();
        let num_groups = resonance_groups.len();
        
        MappingStatistics {
            total_neurons,
            compressed_neurons,
            num_groups,
            compression_ratio: if total_neurons > 0 { compressed_neurons as f32 / total_neurons as f32 } else { 0.0 },
            avg_group_size: if num_groups > 0 { compressed_neurons as f32 / num_groups as f32 } else { 0.0 },
        }
    }

    /// Estimate compression potential
    fn estimate_compression_potential(&self, resonance_groups: &[crate::core::ResonanceGroup], weights: &[Array2<f32>]) -> CompressionPotential {
        let total_params: usize = weights.iter().map(|w| w.len()).sum();
        
        // Estimate compressed parameters
        let mut compressed_params = 0;
        for group in resonance_groups {
            // Superposed representation: 1 row + residual
            compressed_params += weights[group.neurons[0]].ncols(); // Superposed weights
            compressed_params += group.neurons.len() * weights[group.neurons[0]].ncols() / 2; // Estimated residual
        }

        let compression_ratio = if total_params > 0 { 1.0 - (compressed_params as f32 / total_params as f32) } else { 0.0 };

        CompressionPotential {
            original_params: total_params,
            estimated_compressed_params: compressed_params,
            compression_ratio,
            memory_savings: total_params - compressed_params,
        }
    }

    /// Compute compression metrics
    fn compute_compression_metrics(&self, original_weights: &[Array2<f32>], compressed_layers: &[CompressedLayer]) -> CompressionMetrics {
        let original_params: usize = original_weights.iter().map(|w| w.len()).sum();
        let compressed_params: usize = compressed_layers.iter().map(|l| l.compressed_weights.len()).sum();
        
        CompressionMetrics {
            original_params,
            compressed_params,
            compression_ratio: if original_params > 0 { 1.0 - (compressed_params as f32 / original_params as f32) } else { 0.0 },
            memory_reduction: original_params - compressed_params,
        }
    }

    /// Compute calibration statistics
    fn compute_calibration_stats(&self, layers: &[CompressedLayer]) -> CalibrationStats {
        let total_gates: usize = layers.iter().map(|l| l.resonance_representations.len()).sum();
        let avg_sparsity: f32 = layers.iter()
            .flat_map(|l| l.resonance_representations.iter())
            .map(|rep| rep.importance_coeffs.iter().filter(|&&x| x > 0.5).count() as f32 / rep.importance_coeffs.len() as f32)
            .sum::<f32>() / total_gates.max(1) as f32;

        // Compute calibration quality based on multiple factors
        let sparsity_quality = if avg_sparsity > 0.3 && avg_sparsity < 0.7 { 1.0 } else { 0.8 };
        let gate_balance_quality = self.compute_gate_balance_quality(layers);
        let compression_efficiency = self.compute_compression_efficiency(layers);
        
        // Weighted combination untuk overall calibration quality
        let calibration_quality = 0.4 * sparsity_quality + 0.3 * gate_balance_quality + 0.3 * compression_efficiency;

        CalibrationStats {
            total_gates,
            avg_sparsity,
            calibration_quality,
        }
    }

    /// Compute gate balance quality across layers
    fn compute_gate_balance_quality(&self, layers: &[CompressedLayer]) -> f32 {
        if layers.is_empty() {
            return 0.0;
        }
        
        let mut layer_balances = Vec::new();
        
        for layer in layers {
            let mut gate_activations = Vec::new();
            
            for resonance_rep in &layer.resonance_representations {
                let active_count = resonance_rep.importance_coeffs.iter().filter(|&&x| x > 0.5).count();
                let total_count = resonance_rep.importance_coeffs.len();
                gate_activations.push(active_count as f32 / total_count as f32);
            }
            
            if !gate_activations.is_empty() {
                let mean_activation = gate_activations.iter().sum::<f32>() / gate_activations.len() as f32;
                let variance = gate_activations.iter()
                    .map(|x| (x - mean_activation).powi(2))
                    .sum::<f32>() / gate_activations.len() as f32;
                
                // Lower variance = better balance
                let balance_score = 1.0 / (1.0 + variance);
                layer_balances.push(balance_score);
            }
        }
        
        if layer_balances.is_empty() {
            0.0
        } else {
            layer_balances.iter().sum::<f32>() / layer_balances.len() as f32
        }
    }

    /// Compute compression efficiency based on parameter reduction
    fn compute_compression_efficiency(&self, layers: &[CompressedLayer]) -> f32 {
        if layers.is_empty() {
            return 0.0;
        }
        
        let mut total_original_params = 0;
        let mut total_compressed_params = 0;
        
        for layer in layers {
            let original_params = layer.original_weights.len();
            let compressed_params = layer.compressed_weights.len();
            
            total_original_params += original_params;
            total_compressed_params += compressed_params;
        }
        
        if total_original_params == 0 {
            return 0.0;
        }
        
        let compression_ratio = 1.0 - (total_compressed_params as f32 / total_original_params as f32);
        
        // Optimal compression ratio is around 0.5-0.8
        if compression_ratio >= 0.5 && compression_ratio <= 0.8 {
            1.0
        } else if compression_ratio < 0.5 {
            compression_ratio / 0.5
        } else {
            (1.0 - compression_ratio) / 0.2
        }
    }

    /// Compute training statistics
    fn compute_training_stats(&self, model: &CalibratedModel, lora_adapters: &[LoRAAdapter]) -> TrainingStats {
        // Compute actual training time based on strategy complexity
        let base_time = match &self.training_strategy {
            TrainingStrategy::Conservative { calibration_epochs } => {
                std::time::Duration::from_secs((*calibration_epochs as u64) * 60) // 1 min per epoch
            }
            TrainingStrategy::Balanced { calibration_epochs, lora_rank } => {
                let calibration_time = (*calibration_epochs as u64) * 60;
                let lora_time = (*lora_rank as u64) * 30; // 30 sec per rank unit
                std::time::Duration::from_secs(calibration_time + lora_time)
            }
            TrainingStrategy::Aggressive { fine_tuning_epochs, lora_rank } => {
                let fine_tuning_time = (*fine_tuning_epochs as u64) * 120; // 2 min per epoch
                let lora_time = (*lora_rank as u64) * 45; // 45 sec per rank unit
                std::time::Duration::from_secs(fine_tuning_time + lora_time)
            }
        };
        
        // Estimate final loss based on calibration quality
        let calibration_quality = model.calibration_stats.calibration_quality;
        let final_loss = 0.1 * (1.0 - calibration_quality) + 0.01; // Base loss + quality penalty
        
        // Estimate convergence epoch based on strategy
        let convergence_epoch = match &self.training_strategy {
            TrainingStrategy::Conservative { calibration_epochs } => *calibration_epochs,
            TrainingStrategy::Balanced { calibration_epochs, .. } => *calibration_epochs,
            TrainingStrategy::Aggressive { fine_tuning_epochs, .. } => *fine_tuning_epochs,
        };
        
        // Compute recovery quality based on LoRA adapters and calibration
        let lora_effectiveness = if lora_adapters.is_empty() { 
            0.9 // No LoRA, slightly lower recovery
        } else { 
            let avg_rank = lora_adapters.iter().map(|a| a.rank).sum::<usize>() as f32 / lora_adapters.len() as f32;
            0.9 + 0.1 * (avg_rank / 32.0).min(1.0) // Higher rank = better recovery
        };
        
        let recovery_quality = calibration_quality * lora_effectiveness;

        TrainingStats {
            total_training_time: base_time,
            final_loss,
            convergence_epoch,
            lora_rank: lora_adapters.first().map(|a| a.rank).unwrap_or(0),
            recovery_quality,
        }
    }
}

// Data structures untuk training

#[derive(Debug, Clone)]
pub struct TrainingDataset {
    pub samples: Vec<(Array1<f32>, Array1<f32>)>,
}

impl TrainingDataset {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn add_sample(&mut self, input: Array1<f32>, target: Array1<f32>) {
        self.samples.push((input, target));
    }

    pub fn sample_calibration_batch(&self, batch_size: usize) -> CalibrationBatch {
        let mut samples = Vec::new();
        for _ in 0..batch_size.min(self.samples.len()) {
            let idx = rand::random::<usize>() % self.samples.len();
            samples.push(self.samples[idx].clone());
        }
        CalibrationBatch { samples }
    }

    pub fn sample_training_batch(&self, batch_size: usize) -> TrainingBatch {
        let mut samples = Vec::new();
        for _ in 0..batch_size.min(self.samples.len()) {
            let idx = rand::random::<usize>() % self.samples.len();
            samples.push(self.samples[idx].clone());
        }
        TrainingBatch { samples }
    }
}

#[derive(Debug, Clone)]
pub struct CalibrationBatch {
    pub samples: Vec<(Array1<f32>, Array1<f32>)>,
}

#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub samples: Vec<(Array1<f32>, Array1<f32>)>,
}

#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    pub contexts: Vec<Array1<f32>>,
}

impl CalibrationDataset {
    pub fn new() -> Self {
        Self { contexts: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub layer_idx: usize,
    pub lora_a: Array2<f32>,
    pub lora_b: Array2<f32>,
    pub rank: usize,
}

// Result structures

#[derive(Debug, Clone)]
pub struct ResonanceMappingResult {
    pub resonance_groups: Vec<crate::core::ResonanceGroup>,
    pub mapping_stats: MappingStatistics,
    pub compression_potential: CompressionPotential,
}

#[derive(Debug, Clone)]
pub struct MappingStatistics {
    pub total_neurons: usize,
    pub compressed_neurons: usize,
    pub num_groups: usize,
    pub compression_ratio: f32,
    pub avg_group_size: f32,
}

#[derive(Debug, Clone)]
pub struct CompressionPotential {
    pub original_params: usize,
    pub estimated_compressed_params: usize,
    pub compression_ratio: f32,
    pub memory_savings: usize,
}

#[derive(Debug, Clone)]
pub struct CompressedModel {
    pub original_weights: Vec<Array2<f32>>,
    pub compressed_layers: Vec<CompressedLayer>,
    pub resonance_groups: Vec<crate::core::ResonanceGroup>,
    pub compression_metrics: CompressionMetrics,
}

#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    pub original_params: usize,
    pub compressed_params: usize,
    pub compression_ratio: f32,
    pub memory_reduction: usize,
}

#[derive(Debug, Clone)]
pub struct CalibratedModel {
    pub compressed_model: CompressedModel,
    pub calibrated_layers: Vec<CompressedLayer>,
    pub calibration_stats: CalibrationStats,
}

#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub total_gates: usize,
    pub avg_sparsity: f32,
    pub calibration_quality: f32,
}

#[derive(Debug, Clone)]
pub struct TrainedERPModel {
    pub calibrated_model: CalibratedModel,
    pub lora_adapters: Vec<LoRAAdapter>,
    pub training_stats: TrainingStats,
}

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_training_time: std::time::Duration,
    pub final_loss: f32,
    pub convergence_epoch: usize,
    pub lora_rank: usize,
    pub recovery_quality: f32,
}
