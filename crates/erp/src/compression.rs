//! ERP Superposition Compression Engine
//! 
//! Implementasi dari superposition compression untuk resonance groups
//! dengan adaptive importance coefficients dan residual preservation.

use crate::{ERPConfig, ERPError, ResonanceGroup};
use ndarray::{Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng};
use rand_distr::Standard;
use std::collections::HashMap;

/// Superposition compression engine untuk resonance groups
pub struct SuperpositionCompressor {
    _config: ERPConfig,
    compression_method: CompressionMethod,
}

#[derive(Debug, Clone)]
pub enum CompressionMethod {
    WeightedSuperposition,
    LowRankApproximation { rank: usize },
    SparseResidual { sparsity: f32 },
}

impl SuperpositionCompressor {
    pub fn new(config: ERPConfig) -> Self {
        let compression_method = match config.compression_mode {
            crate::CompressionMode::Conservative => CompressionMethod::WeightedSuperposition,
            crate::CompressionMode::Balanced => CompressionMethod::LowRankApproximation { rank: 16 },
            crate::CompressionMode::Aggressive => CompressionMethod::SparseResidual { sparsity: 0.1 },
        };

        Self {
            _config: config,
            compression_method,
        }
    }

    /// Compress weights menggunakan resonance groups
    pub fn compress_weights(&self, weights: &[Array2<f32>], resonance_groups: &[ResonanceGroup]) -> Result<Vec<CompressedLayer>, ERPError> {
        let mut compressed_layers = Vec::new();
        let total_neurons: usize = weights.iter().map(|w| w.shape()[0]).sum();
        let mut global_neuron_map = HashMap::with_capacity(total_neurons);
        let mut neuron_counter = 0;

        // Build global neuron mapping
        for (layer_idx, layer_weights) in weights.iter().enumerate() {
            let (output_dim, _) = layer_weights.dim();
            for neuron_idx in 0..output_dim {
                global_neuron_map.insert((layer_idx, neuron_idx), neuron_counter);
                neuron_counter += 1;
            }
        }

        for (layer_idx, layer_weights) in weights.iter().enumerate() {
            let compressed_layer = self.compress_layer(layer_idx, layer_weights, resonance_groups, &global_neuron_map)?;
            compressed_layers.push(compressed_layer);
        }

        Ok(compressed_layers)
    }

    /// Compress individual layer
    fn compress_layer(&self, layer_idx: usize, weights: &Array2<f32>, resonance_groups: &[ResonanceGroup], neuron_map: &HashMap<(usize, usize), usize>) -> Result<CompressedLayer, ERPError> {
        let (output_dim, input_dim) = weights.dim();
        let mut resonance_representations = Vec::new();
        let mut compressed_weights = Array2::zeros((output_dim, input_dim));
        let mut neuron_status = vec![NeuronStatus::Original; output_dim];

        // Find resonance groups untuk layer ini
        let layer_groups: Vec<_> = resonance_groups.iter()
            .filter(|group| {
                group.neurons.iter().any(|&global_neuron_idx| {
                    // Check jika neuron ini ada di layer ini
                    self.is_neuron_in_layer(global_neuron_idx, layer_idx, neuron_map, output_dim)
                })
            })
            .collect();

        // Process setiap resonance group
        for group in &layer_groups {
            let layer_neurons: Vec<_> = group.neurons.iter()
                .filter_map(|&global_neuron_idx| {
                    self.get_layer_neuron_index(global_neuron_idx, layer_idx, neuron_map, output_dim)
                })
                .collect();

            if layer_neurons.len() > 1 {
                let representation = self.compress_resonance_group(weights, &layer_neurons, &group.importance_scores)?;
                
                // Update compressed weights
                for (i, &neuron_idx) in layer_neurons.iter().enumerate() {
                    compressed_weights.row_mut(neuron_idx).assign(&representation.reconstructed_weights.row(i));
                    neuron_status[neuron_idx] = NeuronStatus::Compressed;
                }

                resonance_representations.push(ResonanceRepresentation {
                    group_neurons: layer_neurons,
                    superposed_weights: representation.superposed_weights,
                    importance_coeffs: representation.importance_coeffs,
                    residual: representation.residual,
                });
            }
        }

        // Apply energy stability regularization
        self.apply_energy_stability(&mut compressed_weights, weights)?;

        Ok(CompressedLayer {
            layer_idx,
            original_weights: weights.clone(),
            compressed_weights: compressed_weights.clone(),
            resonance_representations,
            neuron_status,
            compression_ratio: self.compute_compression_ratio(weights, &compressed_weights),
        })
    }

    /// Compress resonance group menjadi superposition representation
    fn compress_resonance_group(&self, weights: &Array2<f32>, group_neurons: &[usize], importance_scores: &[f32]) -> Result<GroupCompression, ERPError> {
        let group_weights = if group_neurons.is_empty() {
            Array2::zeros((0, weights.ncols()))
        } else {
            let first_row = weights.row(group_neurons[0]).to_owned();
            let mut group_weights = Array2::zeros((group_neurons.len(), weights.ncols()));
            for (i, &neuron_idx) in group_neurons.iter().enumerate() {
                group_weights.row_mut(i).assign(&weights.row(neuron_idx));
            }
            group_weights
        };

        let importance_coeffs = self.compute_adaptive_importance_coefficients(importance_scores);
        
        // Compute superposed weights
        let superposed_weights = self.compute_superposed_weights(&group_weights, &importance_coeffs);
        
        // Compute residual
        let residual = self.compute_residual(&group_weights, &superposed_weights, &importance_coeffs);
        
        // Reconstruct individual weights
        let reconstructed_weights = self.reconstruct_group_weights(&superposed_weights, &residual, &importance_coeffs);

        Ok(GroupCompression {
            superposed_weights,
            importance_coeffs,
            residual,
            reconstructed_weights,
        })
    }

    /// Compute adaptive importance coefficients
    fn compute_adaptive_importance_coefficients(&self, importance_scores: &[f32]) -> Array1<f32> {
        // Apply softmax pada importance scores
        let max_score = importance_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = importance_scores.iter()
            .map(|&score| (score - max_score).exp())
            .collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        
        Array1::from_vec(exp_scores.iter().map(|&x| x / sum_exp).collect())
    }

    /// Compute superposed weights representation
    fn compute_superposed_weights(&self, group_weights: &Array2<f32>, importance_coeffs: &Array1<f32>) -> Array2<f32> {
        let (k, input_dim) = group_weights.dim();
        let mut superposed = Array2::zeros((1, input_dim));

        for i in 0..k {
            let weighted_row = &group_weights.row(i) * importance_coeffs[i];
            let current = superposed.row(0).to_owned();
superposed.row_mut(0).assign(&(current + &weighted_row));
        }

        superposed
    }

    /// Compute residual preservation
    fn compute_residual(&self, group_weights: &Array2<f32>, superposed_weights: &Array2<f32>, importance_coeffs: &Array1<f32>) -> ResidualRepresentation {
        match &self.compression_method {
            CompressionMethod::WeightedSuperposition => {
                // Full residual preservation
                let mut residual = Array2::zeros(group_weights.dim());
                for i in 0..group_weights.nrows() {
                    residual.row_mut(i).assign(&(group_weights.row(i).to_owned() - superposed_weights.row(0).to_owned()));
                }
                
                ResidualRepresentation::FullResidual(residual)
            }
            CompressionMethod::LowRankApproximation { rank } => {
                // Low-rank residual approximation
                let residual_matrix = self.compute_low_rank_residual(group_weights, superposed_weights, *rank);
                ResidualRepresentation::LowRankResidual(residual_matrix)
            }
            CompressionMethod::SparseResidual { sparsity } => {
                // Sparse residual preservation
                let sparse_residual = self.compute_sparse_residual(group_weights, superposed_weights, *sparsity);
                ResidualRepresentation::SparseResidual(sparse_residual)
            }
        }
    }

    /// Compute low-rank residual approximation
    fn compute_low_rank_residual(&self, group_weights: &Array2<f32>, superposed_weights: &Array2<f32>, rank: usize) -> LowRankResidual {
        let (k, input_dim) = group_weights.dim();
        let actual_rank = std::cmp::min(rank, std::cmp::min(k, input_dim));
        
        // Simplified SVD - dalam implementasi nyata gunakan proper SVD
        let mut rng = rand::thread_rng();
        let u = Array2::from_shape_fn((k, actual_rank), |_| rng.gen());
        let v = Array2::from_shape_fn((input_dim, actual_rank), |_| rng.gen());
        
        LowRankResidual { u, v }
    }

    /// Compute sparse residual
    fn compute_sparse_residual(&self, group_weights: &Array2<f32>, superposed_weights: &Array2<f32>, sparsity: f32) -> SparseResidual {
        let (k, input_dim) = group_weights.dim();
        let mut residual = Array2::zeros((k, input_dim));
        
        for i in 0..k {
            residual.row_mut(i).assign(&(group_weights.row(i).to_owned() - superposed_weights.row(0).to_owned()));
        }
        
        // Apply sparsity mask
        let mut mask = Array2::zeros((k, input_dim));
        let num_elements = (k * input_dim) as f32 * sparsity;
        
        for _ in 0..num_elements as usize {
            let i = rand::random::<usize>() % k;
            let j = rand::random::<usize>() % input_dim;
            mask[[i, j]] = 1.0;
        }
        
        SparseResidual {
            residual: residual * &mask,
            mask,
        }
    }

    /// Reconstruct individual weights dari superposed representation
    fn reconstruct_group_weights(&self, superposed_weights: &Array2<f32>, residual: &ResidualRepresentation, importance_coeffs: &Array1<f32>) -> Array2<f32> {
        let (k, input_dim) = (importance_coeffs.len(), superposed_weights.ncols());
        let mut reconstructed = Array2::zeros((k, input_dim));

        for i in 0..k {
            // Base reconstruction dari superposed weights
            reconstructed.row_mut(i).assign(&superposed_weights.row(0));
            
            // Add contribution dari residual
            match residual {
                ResidualRepresentation::FullResidual(full_residual) => {
                    let current = reconstructed.row(i).to_owned();
                    reconstructed.row_mut(i).assign(&(current + &full_residual.row(i)));
                }
                ResidualRepresentation::LowRankResidual(low_rank) => {
                    // Simplified reconstruction dari low-rank
                    let contribution = low_rank.u.row(i).dot(&low_rank.v.t());
                    let current = reconstructed.row(i).to_owned();
                    reconstructed.row_mut(i).assign(&(current + &contribution));
                }
                ResidualRepresentation::SparseResidual(sparse) => {
                    let current = reconstructed.row(i).to_owned();
                    reconstructed.row_mut(i).assign(&(current + &sparse.residual.row(i)));
                }
            }
        }

        reconstructed
    }

    /// Apply energy stability regularization
    fn apply_energy_stability(&self, compressed_weights: &mut Array2<f32>, original_weights: &Array2<f32>) -> Result<(), ERPError> {
        let (output_dim, input_dim) = original_weights.dim();
        
        // Compute original energy
        let original_energy: f32 = original_weights.iter().map(|&x| x * x).sum();
        
        // Compute current compressed energy
        let compressed_energy: f32 = compressed_weights.iter().map(|&x| x * x).sum();
        
        if compressed_energy > 0.0 {
            // Energy normalization untuk stability
            let energy_ratio = (original_energy / compressed_energy).sqrt();
            *compressed_weights *= energy_ratio;
        }

        Ok(())
    }

    /// Compute compression ratio
    fn compute_compression_ratio(&self, original: &Array2<f32>, compressed: &Array2<f32>) -> f32 {
        let original_params = original.len();
        let compressed_params = compressed.len();
        
        if original_params > 0 {
            1.0 - (compressed_params as f32 / original_params as f32)
        } else {
            0.0
        }
    }

    /// Helper: check jika neuron ada di layer tertentu
    fn is_neuron_in_layer(&self, global_neuron_idx: usize, layer_idx: usize, neuron_map: &HashMap<(usize, usize), usize>, output_dim: usize) -> bool {
        for local_neuron_idx in 0..output_dim {
            if let Some(&mapped_global_idx) = neuron_map.get(&(layer_idx, local_neuron_idx)) {
                if mapped_global_idx == global_neuron_idx {
                    return true;
                }
            }
        }
        false
    }

    /// Helper: get layer neuron index dari global neuron index
    fn get_layer_neuron_index(&self, global_neuron_idx: usize, layer_idx: usize, neuron_map: &HashMap<(usize, usize), usize>, output_dim: usize) -> Option<usize> {
        for local_neuron_idx in 0..output_dim {
            if let Some(&mapped_global_idx) = neuron_map.get(&(layer_idx, local_neuron_idx)) {
                if mapped_global_idx == global_neuron_idx {
                    return Some(local_neuron_idx);
                }
            }
        }
        None
    }
}

/// Compressed layer representation
#[derive(Debug, Clone)]
pub struct CompressedLayer {
    pub layer_idx: usize,
    pub original_weights: Array2<f32>,
    pub compressed_weights: Array2<f32>,
    pub resonance_representations: Vec<ResonanceRepresentation>,
    pub neuron_status: Vec<NeuronStatus>,
    pub compression_ratio: f32,
}

/// Resonance representation untuk compressed group
#[derive(Debug, Clone)]
pub struct ResonanceRepresentation {
    pub group_neurons: Vec<usize>,
    pub superposed_weights: Array2<f32>,
    pub importance_coeffs: Array1<f32>,
    pub residual: ResidualRepresentation,
}

/// Residual representation variants
#[derive(Debug, Clone)]
pub enum ResidualRepresentation {
    FullResidual(Array2<f32>),
    LowRankResidual(LowRankResidual),
    SparseResidual(SparseResidual),
}

/// Low-rank residual representation
#[derive(Debug, Clone)]
pub struct LowRankResidual {
    pub u: Array2<f32>,
    pub v: Array2<f32>,
}

/// Sparse residual representation
#[derive(Debug, Clone)]
pub struct SparseResidual {
    pub residual: Array2<f32>,
    pub mask: Array2<f32>,
}

/// Neuron status dalam compressed layer
#[derive(Debug, Clone)]
pub enum NeuronStatus {
    Original,
    Compressed,
}

/// Internal group compression result
struct GroupCompression {
    superposed_weights: Array2<f32>,
    importance_coeffs: Array1<f32>,
    residual: ResidualRepresentation,
    reconstructed_weights: Array2<f32>,
}
