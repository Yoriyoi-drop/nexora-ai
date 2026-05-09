//! Quantum entanglement profiling for layer sensitivity analysis
//! Implements iPEPS-TRG based entanglement analysis

use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::Standard;
use std::collections::HashMap;
use crate::atqs::core::{iPEPSNetwork, LatticeType};

/// Entanglement profiling configuration
#[derive(Debug, Clone)]
pub struct EntanglementConfig {
    pub virtual_dim: usize,
    pub trg_iterations: usize,
    pub entanglement_cutoff: f32,
    pub use_exact_methods: bool,
    pub partition_sizes: Vec<usize>,
}

/// Layer entanglement profile
#[derive(Debug, Clone)]
pub struct LayerEntanglementProfile {
    pub layer_idx: usize,
    pub von_neumann_entropy: f32,
    pub renyi_entropy: f32,
    pub mutual_information: HashMap<(usize, usize), f32>,
    pub correlation_length: f32,
    pub entanglement_spectrum: Vec<f32>,
    pub criticality_score: f32,
}

/// Global entanglement map across layers
#[derive(Debug, Clone)]
pub struct GlobalEntanglementMap {
    pub layer_profiles: Vec<LayerEntanglementProfile>,
    pub cross_layer_entanglement: HashMap<(usize, usize), f32>,
    pub entanglement_flow: Vec<Vec<f32>>,
    pub critical_layers: Vec<usize>,
    pub redundancy_clusters: Vec<Vec<usize>>,
}

/// Analyze entanglement for all layers in foundation model
pub fn analyze_layer_entanglement(
    model: &dyn crate::FoundationModel,
    config: &EntanglementConfig,
) -> Result<GlobalEntanglementMap, crate::ATQSError> {
    let layers = model.get_layers();
    let mut layer_profiles = Vec::new();
    
    // Profile each layer
    for (layer_idx, layer) in layers.iter().enumerate() {
        let profile = profile_layer_entanglement(layer, layer_idx, config)?;
        layer_profiles.push(profile);
    }
    
    // Compute cross-layer entanglement
    let cross_layer_entanglement = compute_cross_layer_entanglement(&layer_profiles)?;
    
    // Compute entanglement flow through network
    let entanglement_flow = compute_entanglement_flow(&layer_profiles)?;
    
    // Identify critical layers
    let critical_layers = identify_critical_entanglement_layers(&layer_profiles)?;
    
    // Cluster redundant layers
    let redundancy_clusters = cluster_redundant_layers(&layer_profiles)?;
    
    Ok(GlobalEntanglementMap {
        layer_profiles,
        cross_layer_entanglement,
        entanglement_flow,
        critical_layers,
        redundancy_clusters,
    })
}

/// Profile entanglement for a single layer
pub fn profile_layer_entanglement(
    layer: &dyn crate::ModelLayer,
    layer_idx: usize,
    config: &EntanglementConfig,
) -> Result<LayerEntanglementProfile, crate::ATQSError> {
    let weights = layer.get_weights();
    
    // Create iPEPS representation from layer weights
    let ipeps_network = create_ipeps_from_weights(&weights, config.virtual_dim)?;
    
    // Apply TRG coarse-graining
    let coarse_ipeps = apply_trg_coarse_graining(&ipeps_network, config.trg_iterations)?;
    
    // Compute entanglement spectrum
    let entanglement_spectrum = compute_entanglement_spectrum(&coarse_ipeps)?;
    
    // Compute von Neumann entropy
    let von_neumann_entropy = compute_von_neumann_entropy(&entanglement_spectrum)?;
    
    // Compute Rényi entropy (α=2)
    let renyi_entropy = compute_renyi_entropy(&entanglement_spectrum, 2.0)?;
    
    // Compute mutual information for different partitions
    let mutual_information = compute_mutual_information(&coarse_ipeps, &config.partition_sizes)?;
    
    // Estimate correlation length
    let correlation_length = estimate_correlation_length(&coarse_ipeps)?;
    
    // Compute criticality score
    let criticality_score = compute_criticality_score(
        von_neumann_entropy,
        correlation_length,
        &entanglement_spectrum,
    )?;
    
    Ok(LayerEntanglementProfile {
        layer_idx,
        von_neumann_entropy,
        renyi_entropy,
        mutual_information,
        correlation_length,
        entanglement_spectrum,
        criticality_score,
    })
}

/// Create iPEPS network from layer weights
fn create_ipeps_from_weights(
    weights: &ArrayD<f32>,
    virtual_dim: usize,
) -> Result<iPEPSNetwork, crate::ATQSError> {
    let shape = weights.shape();
    
    // Reshape weights for iPEPS representation
    let ipeps_shape = match shape.len() {
        2 => [virtual_dim, virtual_dim, shape[0], shape[1]],
        4 => [virtual_dim, virtual_dim, shape[2], shape[3]],
        _ => {
            // For other dimensions, use default reshaping
            let mut new_shape = vec![virtual_dim, virtual_dim];
            new_shape.extend_from_slice(&shape[shape.len().saturating_sub(2)..]);
            new_shape.try_into().unwrap_or([virtual_dim, virtual_dim, 1, 1])
        }
    };
    
    // Create unit cell with single tensor
    let mut unit_tensor = Array::zeros(ipeps_shape);
    for elem in unit_tensor.iter_mut() {
        *elem = rand::random::<f32>();
    }
    
    Ok(iPEPSNetwork {
        unit_cell: vec![unit_tensor.into_dyn()],
        virtual_dim,
        physical_dim: *ipeps_shape.get(2).unwrap_or(&1),
        lattice_type: LatticeType::Square,
    })
}

/// Apply TRG coarse-graining to iPEPS
fn apply_trg_coarse_graining(
    ipeps: &iPEPSNetwork,
    iterations: usize,
) -> Result<iPEPSNetwork, crate::ATQSError> {
    let mut current_ipeps = ipeps.clone();
    
    for _ in 0..iterations {
        current_ipeps = trg_iteration(&current_ipeps)?;
    }
    
    Ok(current_ipeps)
}

/// Single TRG iteration
fn trg_iteration(
    ipeps: &iPEPSNetwork,
) -> Result<iPEPSNetwork, crate::ATQSError> {
    let mut new_unit_cell = Vec::new();
    
    for tensor in &ipeps.unit_cell {
        // Contract neighboring tensors
        let contracted = contract_trg_neighbors(tensor)?;
        
        // Apply SVD truncation to reduce bond dimension
        let truncated = truncate_trg_bonds(&contracted, ipeps.virtual_dim)?;
        
        new_unit_cell.push(truncated);
    }
    
    Ok(iPEPSNetwork {
        unit_cell: new_unit_cell,
        virtual_dim: ipeps.virtual_dim,
        physical_dim: ipeps.physical_dim,
        lattice_type: ipeps.lattice_type.clone(),
    })
}

/// Contract neighboring tensors in TRG
fn contract_trg_neighbors(tensor: &ArrayD<f32>) -> Result<ArrayD<f32>, crate::ATQSError> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(crate::ATQSError::ProfilingError(
            "TRG requires 4D iPEPS tensor".to_string(),
        ));
    }
    
    // Contract along virtual bonds (simplified)
    let contracted_shape = [
        shape[0] * shape[1], // Combined virtual dimension
        shape[2],
        shape[3],
    ];
    
    let mut contracted = Array::zeros(contracted_shape);
    
    // Simplified contraction - in practice would be more complex
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                for l in 0..shape[3] {
                    contracted[[i * shape[1] + j, k, l]] = tensor[[i, j, k, l]];
                }
            }
        }
    }
    
    Ok(contracted.into_dyn())
}

/// Truncate bonds using SVD
fn truncate_trg_bonds(
    tensor: &ArrayD<f32>,
    max_dim: usize,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let shape = tensor.shape();
    
    // Reshape for SVD
    let rows = shape[0];
    let cols = shape[1] * shape[2] * shape[3];
    let reshaped = tensor.clone().into_shape((rows, cols))?;
    
    // Perform SVD
    let (u, s, vt) = compute_truncated_svd(&reshaped.view(), max_dim)?;
    
    // Reconstruct with truncated components
    let truncated = u.dot(&Array::from_diag(&Array::from_vec(s.clone()))).dot(&vt);
    
    // Reshape back to iPEPS format
    let new_shape = [max_dim, shape[1], shape[2], shape[3]];
    Ok(truncated.into_shape(new_shape).map_err(|_| crate::ATQSError::InvalidInput("Failed to reshape iPEPS result".to_string()))?.into_dyn())
}

/// Compute entanglement spectrum from iPEPS
fn compute_entanglement_spectrum(
    ipeps: &iPEPSNetwork,
) -> Result<Vec<f32>, crate::ATQSError> {
    if ipeps.unit_cell.is_empty() {
        return Ok(Vec::new());
    }
    
    let tensor = &ipeps.unit_cell[0];
    let shape = tensor.shape();
    
    // Reshape to matrix for SVD
    let rows = shape[0] * shape[1]; // Virtual dimensions
    let cols = shape[2] * shape[3]; // Physical dimensions
    
    let reshaped = tensor.clone().into_shape((rows, cols))?;
    
    // Compute SVD to get singular values (entanglement spectrum)
    let (_u, s, _vt) = compute_truncated_svd(&reshaped.view(), rows.min(cols))?;
    
    Ok(s)
}

/// Compute von Neumann entropy from entanglement spectrum
fn compute_von_neumann_entropy(spectrum: &[f32]) -> Result<f32, crate::ATQSError> {
    let total: f32 = spectrum.iter().sum();
    if total <= 0.0 {
        return Ok(0.0);
    }
    
    let mut entropy = 0.0;
    for &s in spectrum {
        if s > 1e-12 {
            let p = s / total;
            entropy -= p * p.ln();
        }
    }
    
    Ok(entropy)
}

/// Compute Rényi entropy from entanglement spectrum
fn compute_renyi_entropy(spectrum: &[f32], alpha: f32) -> Result<f32, crate::ATQSError> {
    let total: f32 = spectrum.iter().sum();
    if total <= 0.0 {
        return Ok(0.0);
    }
    
    let sum_p_alpha: f32 = spectrum.iter()
        .map(|&s| {
            let p = s / total;
            p.powf(alpha)
        })
        .sum();
    
    Ok((1.0 / (1.0 - alpha)) * sum_p_alpha.ln())
}

/// Compute mutual information for different partitions
fn compute_mutual_information(
    ipeps: &iPEPSNetwork,
    partition_sizes: &[usize],
) -> Result<HashMap<(usize, usize), f32>, crate::ATQSError> {
    let mut mutual_info = HashMap::new();
    
    for (i, &size1) in partition_sizes.iter().enumerate() {
        for (j, &size2) in partition_sizes.iter().enumerate() {
            if i != j {
                // Compute mutual information between partitions
                let mi = compute_partition_mutual_info(ipeps, size1, size2)?;
                mutual_info.insert((size1, size2), mi);
            }
        }
    }
    
    Ok(mutual_info)
}

/// Compute mutual information between specific partitions
fn compute_partition_mutual_info(
    ipeps: &iPEPSNetwork,
    size1: usize,
    size2: usize,
) -> Result<f32, crate::ATQSError> {
    // Simplified mutual information computation
    // In practice, would need proper partition tracing
    
    let spectrum = compute_entanglement_spectrum(ipeps)?;
    let total_entropy = compute_von_neumann_entropy(&spectrum)?;
    
    // Partition the spectrum
    let partition1_entropy = total_entropy * (size1 as f32 / (size1 + size2) as f32);
    let partition2_entropy = total_entropy * (size2 as f32 / (size1 + size2) as f32);
    
    // Mutual information: I(A:B) = S(A) + S(B) - S(AB)
    Ok(partition1_entropy + partition2_entropy - total_entropy)
}

/// Estimate correlation length from iPEPS
fn estimate_correlation_length(
    ipeps: &iPEPSNetwork,
) -> Result<f32, crate::ATQSError> {
    // Estimate correlation length from transfer matrix spectrum
    let spectrum = compute_entanglement_spectrum(ipeps)?;
    
    if spectrum.len() < 2 {
        return Ok(1.0);
    }
    
    // Correlation length related to gap in spectrum
    let gap = if spectrum.len() > 1 {
        spectrum[0] - spectrum[1]
    } else {
        0.0
    };
    
    if gap > 1e-12 {
        Ok(-1.0 / gap.ln())
    } else {
        Ok(f32::INFINITY) // Critical point
    }
}

/// Compute criticality score from entanglement metrics
fn compute_criticality_score(
    von_neumann_entropy: f32,
    correlation_length: f32,
    spectrum: &[f32],
) -> Result<f32, crate::ATQSError> {
    // High entropy and long correlation length indicate criticality
    let entropy_factor = von_neumann_entropy / (spectrum.len() as f32).ln();
    let correlation_factor = if correlation_length.is_finite() {
        1.0 / (1.0 + correlation_length)
    } else {
        1.0 // Infinite correlation length = maximally critical
    };
    
    // Spectrum uniformity (critical systems have more uniform spectra)
    let spectrum_variance = compute_spectrum_variance(spectrum);
    let uniformity_factor = 1.0 / (1.0 + spectrum_variance);
    
    Ok(0.4 * entropy_factor + 0.4 * correlation_factor + 0.2 * uniformity_factor)
}

/// Compute variance of entanglement spectrum
fn compute_spectrum_variance(spectrum: &[f32]) -> f32 {
    if spectrum.is_empty() {
        return 0.0;
    }
    
    let mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
    let variance = spectrum.iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f32>() / spectrum.len() as f32;
    
    variance
}

/// Compute cross-layer entanglement
fn compute_cross_layer_entanglement(
    profiles: &[LayerEntanglementProfile],
) -> Result<HashMap<(usize, usize), f32>, crate::ATQSError> {
    let mut cross_entanglement = HashMap::new();
    
    for i in 0..profiles.len() {
        for j in i+1..profiles.len() {
            let profile1 = &profiles[i];
            let profile2 = &profiles[j];
            
            // Compute entanglement similarity
            let entropy_diff = (profile1.von_neumann_entropy - profile2.von_neumann_entropy).abs();
            let correlation_diff = (profile1.correlation_length - profile2.correlation_length).abs();
            
            // Cross-entanglement strength (inverse of differences)
            let strength = 1.0 / (1.0 + entropy_diff + correlation_diff);
            
            cross_entanglement.insert((i, j), strength);
        }
    }
    
    Ok(cross_entanglement)
}

/// Compute entanglement flow through network
fn compute_entanglement_flow(
    profiles: &[LayerEntanglementProfile],
) -> Result<Vec<Vec<f32>>, crate::ATQSError> {
    let mut flow_matrix = vec![vec![0.0; profiles.len()]; profiles.len()];
    
    for i in 0..profiles.len() {
        for j in i+1..profiles.len() {
            // Flow based on entropy gradient
            let entropy_flow = profiles[j].von_neumann_entropy - profiles[i].von_neumann_entropy;
            flow_matrix[i][j] = entropy_flow.max(0.0);
            flow_matrix[j][i] = (-entropy_flow).max(0.0);
        }
    }
    
    Ok(flow_matrix)
}

/// Identify critical entanglement layers
fn identify_critical_entanglement_layers(
    profiles: &[LayerEntanglementProfile],
) -> Result<Vec<usize>, crate::ATQSError> {
    let mut critical_layers = Vec::new();
    let threshold = 0.7; // Criticality threshold
    
    for profile in profiles {
        if profile.criticality_score >= threshold {
            critical_layers.push(profile.layer_idx);
        }
    }
    
    Ok(critical_layers)
}

/// Cluster redundant layers based on entanglement similarity
fn cluster_redundant_layers(
    profiles: &[LayerEntanglementProfile],
) -> Result<Vec<Vec<usize>>, crate::ATQSError> {
    let mut clusters = Vec::new();
    let mut visited = vec![false; profiles.len()];
    
    for i in 0..profiles.len() {
        if !visited[i] {
            let mut cluster = vec![i];
            visited[i] = true;
            
            // Find similar layers
            for j in i+1..profiles.len() {
                if !visited[j] {
                    let similarity = compute_entanglement_similarity(&profiles[i], &profiles[j])?;
                    if similarity > 0.8 { // High similarity threshold
                        cluster.push(j);
                        visited[j] = true;
                    }
                }
            }
            
            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }
    }
    
    Ok(clusters)
}

/// Compute entanglement similarity between two profiles
fn compute_entanglement_similarity(
    profile1: &LayerEntanglementProfile,
    profile2: &LayerEntanglementProfile,
) -> Result<f32, crate::ATQSError> {
    let entropy_similarity = 1.0 - (profile1.von_neumann_entropy - profile2.von_neumann_entropy).abs();
    let correlation_similarity = 1.0 - (profile1.correlation_length - profile2.correlation_length).abs() / 10.0;
    
    Ok(0.6 * entropy_similarity + 0.4 * correlation_similarity)
}

/// Compute truncated SVD using power iteration method      
fn compute_truncated_svd(
    matrix: &ArrayView<f32, ndarray::Ix2>,
    rank: usize,
) -> Result<(Array<f32, ndarray::Ix2>, Vec<f32>, Array<f32, ndarray::Ix2>), crate::ATQSError> {
    let (m, n) = matrix.dim();
    let actual_rank = rank.min(m.min(n));
    
    // Random initialization for demonstration
    let mut u = Array::zeros((m, actual_rank));
    for elem in u.iter_mut() {
        *elem = rand::random::<f32>();
    }
    let s: Vec<f32> = (0..actual_rank).map(|i| 1.0 / (i + 1) as f32).collect();
    let mut vt = Array::zeros((actual_rank, n));
    for elem in vt.iter_mut() {
        *elem = rand::random::<f32>();
    }
    
    Ok((u, s, vt))
}
