//! ERP Core - Information Signature dan Resonance Mapping
//! 
//! Implementasi dari Information Resonance Mapping untuk mendeteksi neuron
//! dengan distribusi informasi yang sangat mirip.

use crate::{ERPConfig, ERPError};
use ndarray::{Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng};
use rand_distr::Standard;
use std::collections::HashMap;

/// Information signature untuk neuron
#[derive(Debug, Clone)]
pub struct NeuronSignature {
    /// ID neuron
    pub neuron_id: usize,
    /// Distribusi informasi p_i
    pub information_distribution: Array1<f32>,
    /// Proyeksi low-dimensional z_i
    pub projection: Array1<f32>,
    /// Fisher information untuk importance
    pub fisher_info: f32,
    /// Gradient norm untuk importance
    pub gradient_norm: f32,
}

/// Resonance mapper untuk menemukan neuron dengan distribusi informasi mirip
pub struct ResonanceMapper {
    config: ERPConfig,
    projection_method: ProjectionMethod,
}

#[derive(Debug, Clone)]
pub enum ProjectionMethod {
    RandomProjection { dim: usize },
    PCA { components: usize },
    LowRankEmbedding { rank: usize },
}

impl ResonanceMapper {
    pub fn new(config: ERPConfig) -> Self {
        let projection_method = match config.compression_mode {
            crate::CompressionMode::Conservative => ProjectionMethod::PCA { components: 64 },
            crate::CompressionMode::Balanced => ProjectionMethod::LowRankEmbedding { rank: 32 },
            crate::CompressionMode::Aggressive => ProjectionMethod::RandomProjection { dim: 16 },
        };

        Self {
            config,
            projection_method,
        }
    }

    /// Map informasi resonance untuk semua neuron dalam weights
    pub fn map_resonance(&self, weights: &[Array2<f32>]) -> Result<Vec<ResonanceGroup>, ERPError> {
        // Step 1: Extract neuron signatures
        let signatures = self.extract_neuron_signatures(weights)?;
        
        // Step 2: Two-stage filtering untuk resonance detection
        let resonance_pairs = self.two_stage_filtering(&signatures)?;
        
        // Step 3: Build resonance graph
        let graph = self.build_resonance_graph(signatures.len(), &resonance_pairs);
        
        // Step 4: Cluster graph into resonance groups
        let groups = self.cluster_resonance_graph(graph, &signatures)?;
        
        Ok(groups)
    }

    /// Extract neuron signatures dari weights
    fn extract_neuron_signatures(&self, weights: &[Array2<f32>]) -> Result<Vec<NeuronSignature>, ERPError> {
        let mut signatures = Vec::new();
        let mut neuron_id = 0;

        for (layer_idx, layer_weights) in weights.iter().enumerate() {
            let (output_dim, input_dim) = layer_weights.dim();
            
            for i in 0..output_dim {
                // Extract activation distribution untuk neuron i
                let neuron_weights = layer_weights.row(i).to_owned();
                let information_dist = self.compute_information_distribution(&neuron_weights);
                
                // Compute projection
                let projection = self.compute_projection(&information_dist);
                
                // Compute importance metrics
                let fisher_info = self.compute_fisher_information(&neuron_weights);
                let gradient_norm = neuron_weights.mapv(|x| x * x).sum().sqrt();
                
                signatures.push(NeuronSignature {
                    neuron_id,
                    information_distribution: information_dist,
                    projection,
                    fisher_info,
                    gradient_norm,
                });
                
                neuron_id += 1;
            }
        }

        Ok(signatures)
    }

    /// Compute information distribution dari neuron weights
    fn compute_information_distribution(&self, weights: &Array1<f32>) -> Array1<f32> {
        // Normalize weights untuk mendapatkan distribusi probabilitas
        let abs_weights = weights.mapv(|x| x.abs());
        let sum = abs_weights.sum();
        
        if sum > 0.0 {
            abs_weights / sum
        } else {
            Array1::zeros(weights.len())
        }
    }

    /// Compute low-dimensional projection
    fn compute_projection(&self, distribution: &Array1<f32>) -> Array1<f32> {
        match &self.projection_method {
            ProjectionMethod::RandomProjection { dim } => {
                // Random projection matrix
                let mut rng = rand::thread_rng();
                let projection_matrix = Array2::from_shape_fn((*dim, distribution.len()), |_| rng.gen());
                
                projection_matrix.dot(distribution)
            }
            ProjectionMethod::PCA { components } => {
                // Simplified PCA - dalam implementasi nyata gunakan proper PCA
                let dim = std::cmp::min(*components, distribution.len());
                distribution.slice(ndarray::s![..dim]).to_owned()
            }
            ProjectionMethod::LowRankEmbedding { rank } => {
                // Low-rank approximation
                let dim = std::cmp::min(*rank, distribution.len());
                distribution.slice(ndarray::s![..dim]).to_owned()
            }
        }
    }

    /// Compute Fisher information approximation
    fn compute_fisher_information(&self, weights: &Array1<f32>) -> f32 {
        // Simplified Fisher information - squared gradient magnitude
        weights.mapv(|w| w * w).sum()
    }

    /// Two-stage filtering untuk scalable resonance detection
    fn two_stage_filtering(&self, signatures: &[NeuronSignature]) -> Result<Vec<(usize, usize, f32)>, ERPError> {
        let mut resonance_pairs = Vec::new();
        let n = signatures.len();

        // Stage 1: Fast similarity menggunakan cosine similarity
        let mut candidates = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&signatures[i].projection, &signatures[j].projection);
                if sim > 0.7 { // Threshold untuk fast filtering
                    candidates.push((i, j, sim));
                }
            }
        }

        // Stage 2: Exact resonance computation dengan KL divergence
        for (i, j, _) in candidates {
            let kl_div = symmetric_kl_divergence(
                &signatures[i].information_distribution,
                &signatures[j].information_distribution,
            );
            
            if kl_div < self.config.resonance_threshold {
                resonance_pairs.push((i, j, kl_div));
            }
        }

        Ok(resonance_pairs)
    }

    /// Build resonance graph dari resonance pairs
    fn build_resonance_graph(&self, n_neurons: usize, pairs: &[(usize, usize, f32)]) -> ResonanceGraph {
        let mut adjacency = vec![vec![]; n_neurons];
        
        for &(i, j, weight) in pairs {
            adjacency[i].push((j, weight));
            adjacency[j].push((i, weight));
        }

        ResonanceGraph { adjacency }
    }

    /// Cluster resonance graph menggunakan Louvain clustering
    fn cluster_resonance_graph(&self, graph: ResonanceGraph, signatures: &[NeuronSignature]) -> Result<Vec<ResonanceGroup>, ERPError> {
        // Simplified Louvain clustering - dalam implementasi nyata gunakan proper algorithm
        let mut groups = Vec::new();
        let mut visited = vec![false; signatures.len()];

        for i in 0..signatures.len() {
            if !visited[i] {
                let mut group_neurons = vec![i];
                visited[i] = true;

                // Find connected neurons dengan stability constraints
                for &(neighbor, _) in &graph.adjacency[i] {
                    if !visited[neighbor] && self.check_stability_constraints(&group_neurons, neighbor, signatures) {
                        group_neurons.push(neighbor);
                        visited[neighbor] = true;
                        
                        if group_neurons.len() >= self.config.max_group_size {
                            break;
                        }
                    }
                }

                if group_neurons.len() > 1 {
                    let group = ResonanceGroup {
                        neurons: group_neurons.clone(),
                        stability_variance: self.compute_group_variance(&group_neurons, signatures),
                        importance_scores: group_neurons.iter()
                            .map(|&idx| signatures[idx].fisher_info + signatures[idx].gradient_norm)
                            .collect(),
                    };
                    groups.push(group);
                }
            }
        }

        Ok(groups)
    }

    /// Check stability constraints untuk clustering
    fn check_stability_constraints(&self, group: &[usize], new_neuron: usize, signatures: &[NeuronSignature]) -> bool {
        if group.len() >= self.config.max_group_size {
            return false;
        }

        // Compute variance jika neuron ditambahkan
        let mut extended_group = group.to_vec();
        extended_group.push(new_neuron);
        let variance = self.compute_group_variance(&extended_group, signatures);
        
        variance < self.config.stability_variance
    }

    /// Compute variance dalam resonance group
    fn compute_group_variance(&self, group: &[usize], signatures: &[NeuronSignature]) -> f32 {
        if group.len() <= 1 {
            return 0.0;
        }

        // Compute mean projection
        let mut mean_projection = Array1::zeros(signatures[group[0]].projection.len());
        for &idx in group {
            mean_projection += &signatures[idx].projection;
        }
        mean_projection /= group.len() as f32;

        // Compute variance
        let mut variance = 0.0;
        for &idx in group {
            let diff = &signatures[idx].projection - &mean_projection;
            variance += diff.mapv(|x| x * x).sum();
        }

        variance / group.len() as f32
    }
}

/// Resonance graph structure
#[derive(Debug)]
pub struct ResonanceGraph {
    pub adjacency: Vec<Vec<(usize, f32)>>, // (neighbor_index, weight)
}

/// Resonance group yang berisi neuron-neuron resonan
#[derive(Debug, Clone)]
pub struct ResonanceGroup {
    pub neurons: Vec<usize>,
    pub stability_variance: f32,
    pub importance_scores: Vec<f32>,
}

/// Cosine similarity antara dua vektor
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Symmetric KL divergence
fn symmetric_kl_divergence(p: &Array1<f32>, q: &Array1<f32>) -> f32 {
    let eps = 1e-8;
    let kl_pq: f32 = p.iter().zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > eps && qi > eps {
                pi * ((pi / qi).ln())
            } else {
                0.0
            }
        })
        .sum();
    
    let kl_qp: f32 = q.iter().zip(p.iter())
        .map(|(&qi, &pi)| {
            if qi > eps && pi > eps {
                qi * ((qi / pi).ln())
            } else {
                0.0
            }
        })
        .sum();
    
    kl_pq + kl_qp
}
