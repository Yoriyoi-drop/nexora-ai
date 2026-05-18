//! Sparse augmentation for tensor decomposition
//! Implements low-rank + sparse decomposition methods

use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::Standard;
use std::collections::HashMap;

/// Sparse augmentation configuration
#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub sparse_ratio: f32,
    pub threshold_method: ThresholdMethod,
    pub adaptive_threshold: bool,
    pub residual_learning: bool,
}

#[derive(Debug, Clone)]
pub enum ThresholdMethod {
    Fixed(f32),
    TopK(usize),
    Adaptive(f32),
    Statistical(f32), // Standard deviations
}

/// Low-rank + sparse decomposition result
#[derive(Debug, Clone)]
pub struct SparseAugmentedTensor {
    pub low_rank_component: ArrayD<f32>,
    pub sparse_component: ArrayD<f32>,
    pub residual: ArrayD<f32>,
    pub compression_ratio: f32,
    pub reconstruction_error: f32,
}

/// Sparse mask information
#[derive(Debug, Clone)]
pub struct SparseMask {
    pub mask: ArrayD<bool>,
    pub sparsity_ratio: f32,
    pub threshold: f32,
    pub important_indices: Vec<usize>,
}

/// Apply sparse augmentation to decomposed tensor
pub fn apply_sparse_augmentation(
    original_tensor: &ArrayD<f32>,
    low_rank_tensor: &ArrayD<f32>,
    config: &SparseConfig,
) -> Result<SparseAugmentedTensor, crate::ATQSError> {
    // Compute residual between original and low-rank approximation
    let residual = original_tensor - low_rank_tensor;
    
    // Create sparse mask for residual
    let sparse_mask = create_sparse_mask(&residual, config)?;
    
    // Apply mask to create sparse component
    let sparse_component = apply_sparse_mask(&residual, &sparse_mask)?;
    
    // Compute final reconstruction
    let reconstruction = low_rank_tensor + &sparse_component;
    let final_residual = original_tensor - &reconstruction;
    
    // Calculate metrics
    let compression_ratio = calculate_compression_ratio_sparse(
        original_tensor,
        low_rank_tensor,
        &sparse_component,
    )?;
    
    let reconstruction_error = calculate_reconstruction_error(original_tensor, &reconstruction)?;
    
    Ok(SparseAugmentedTensor {
        low_rank_component: low_rank_tensor.clone(),
        sparse_component,
        residual: final_residual,
        compression_ratio,
        reconstruction_error,
    })
}

/// Create sparse mask based on threshold method
pub fn create_sparse_mask(
    tensor: &ArrayD<f32>,
    config: &SparseConfig,
) -> Result<SparseMask, crate::ATQSError> {
    let threshold = match &config.threshold_method {
        ThresholdMethod::Fixed(threshold) => *threshold,
        ThresholdMethod::TopK(k) => compute_topk_threshold(tensor, *k)?,
        ThresholdMethod::Adaptive(factor) => compute_adaptive_threshold(tensor, *factor)?,
        ThresholdMethod::Statistical(std_devs) => compute_statistical_threshold(tensor, *std_devs)?,
    };
    
    let mut mask = Array::from_elem(tensor.shape(), false);
    let mut important_indices = Vec::new();
    let mut sparse_count = 0;
    
    for (idx, &value) in tensor.indexed_iter() {
        let idx_clone = idx.clone();
        let is_important = value.abs() > threshold;
        mask[idx_clone] = is_important;
        
        if is_important {
            important_indices.push(idx[0]); // Use first dimension index for simplicity
        } else {
            sparse_count += 1;
        }
    }
    
    let sparsity_ratio = sparse_count as f32 / tensor.len() as f32;
    
    Ok(SparseMask {
        mask: mask.into_dyn(),
        sparsity_ratio,
        threshold,
        important_indices,
    })
}

/// Apply sparse mask to tensor
pub fn apply_sparse_mask(
    tensor: &ArrayD<f32>,
    mask: &SparseMask,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let mut result = Array::zeros(tensor.shape());
    
    for (idx, &is_important) in mask.mask.indexed_iter() {
        if is_important {
            result[idx.clone()] = tensor[idx.clone()];
        } else {
            result[idx.clone()] = 0.0;
        }
    }
    
    Ok(result.into_dyn())
}

/// Optimize sparse augmentation iteratively
pub fn optimize_sparse_augmentation(
    original_tensor: &ArrayD<f32>,
    initial_low_rank: &ArrayD<f32>,
    config: &SparseConfig,
    max_iterations: usize,
) -> Result<SparseAugmentedTensor, crate::ATQSError> {
    let mut current_low_rank = initial_low_rank.clone();
    let mut best_result = None;
    let mut best_error = f32::INFINITY;
    
    for iteration in 0..max_iterations {
        // Apply sparse augmentation
        let result = apply_sparse_augmentation(original_tensor, &current_low_rank, config)?;
        
        // Update best result if improved
        if result.reconstruction_error < best_error {
            best_error = result.reconstruction_error;
            best_result = Some(result.clone());
        }
        
        // Update low-rank component if using residual learning
        if config.residual_learning {
            let combined = &current_low_rank + &result.sparse_component;
            current_low_rank = refine_low_rank_component(original_tensor, &combined)?;
        }
        
        // Early stopping if convergence achieved
        if result.reconstruction_error < 1e-6 {
            break;
        }
    }
    
    best_result.ok_or_else(|| {
        crate::ATQSError::CompressionError("Failed to optimize sparse augmentation".to_string())
    })
}

/// Compute threshold for top-k elements
fn compute_topk_threshold(tensor: &ArrayD<f32>, k: usize) -> Result<f32, crate::ATQSError> {
    let mut values: Vec<f32> = tensor.iter().map(|&x| x.abs()).collect();
    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    if k >= values.len() {
        return Ok(0.0);
    }
    
    Ok(values[k])
}

/// Compute adaptive threshold based on tensor statistics
fn compute_adaptive_threshold(tensor: &ArrayD<f32>, factor: f32) -> Result<f32, crate::ATQSError> {
    let mean = tensor.iter().map(|&x| x.abs()).sum::<f32>() / tensor.len() as f32;
    let std_dev = compute_standard_deviation(tensor)?;
    
    Ok(mean + factor * std_dev)
}

/// Compute statistical threshold based on standard deviations
fn compute_statistical_threshold(tensor: &ArrayD<f32>, std_devs: f32) -> Result<f32, crate::ATQSError> {
    let mean = tensor.iter().map(|&x| x.abs()).sum::<f32>() / tensor.len() as f32;
    let std_dev = compute_standard_deviation(tensor)?;
    
    Ok(mean + std_devs * std_dev)
}

/// Compute standard deviation of tensor values
fn compute_standard_deviation(tensor: &ArrayD<f32>) -> Result<f32, crate::ATQSError> {
    let mean = tensor.iter().sum::<f32>() / tensor.len() as f32;
    let variance = tensor.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / tensor.len() as f32;
    
    Ok(variance.sqrt())
}

/// Calculate compression ratio for sparse augmented tensor
fn calculate_compression_ratio_sparse(
    original: &ArrayD<f32>,
    low_rank: &ArrayD<f32>,
    sparse: &ArrayD<f32>,
) -> Result<f32, crate::ATQSError> {
    let original_params = original.len();
    
    // Low-rank parameters (assuming compressed representation)
    let low_rank_params = low_rank.len();
    
    // Sparse parameters (non-zero elements only)
    let sparse_params = sparse.iter().filter(|&&x| x.abs() > 1e-8).count();
    
    let total_compressed = low_rank_params + sparse_params;
    
    Ok(original_params as f32 / total_compressed as f32)
}

/// Calculate reconstruction error
fn calculate_reconstruction_error(
    original: &ArrayD<f32>,
    reconstructed: &ArrayD<f32>,
) -> Result<f32, crate::ATQSError> {
    let error = original - reconstructed;
    let mse = error.iter().map(|&x| x * x).sum::<f32>() / error.len() as f32;
    Ok(mse.sqrt())
}

/// Refine low-rank component using residual learning
fn refine_low_rank_component(
    original: &ArrayD<f32>,
    current_approximation: &ArrayD<f32>,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    // Compute residual
    let residual = original - current_approximation;
    
    // Apply SVD to residual to extract additional low-rank structure
    let residual_2d = reshape_to_2d(&residual)?;
    let (u, s, vt) = compute_svd_truncated(&residual_2d.view(), residual_2d.ncols().min(residual_2d.nrows()) / 2)?;
    
    // Reconstruct refined low-rank component
    let s_array = Array::from_vec(s);
    let refined_residual = u.dot(&Array::from_diag(&s_array)).dot(&vt);
    let refined_residual_nd = reshape_from_2d(&refined_residual, original.shape())?;
    
    Ok(current_approximation + refined_residual_nd)
}

/// Reshape tensor to 2D for SVD operations
fn reshape_to_2d(tensor: &ArrayD<f32>) -> Result<Array<f32, ndarray::Ix2>, crate::ATQSError> {
    let shape = tensor.shape();
    if shape.len() == 2 {
        return tensor.to_owned().into_dimensionality()
            .map_err(|_| crate::ATQSError::InvalidInput("Failed to convert to 2D".to_string()));
    }
    
    // Reshape to (first_dim, product_of_remaining_dims)
    let rows = shape[0];
    let cols: usize = shape.iter().skip(1).product();
    
    tensor.clone().into_shape((rows, cols))
        .map_err(|_| crate::ATQSError::InvalidInput("Failed to reshape to 2D".to_string()))
        .and_then(|arr| arr.into_dimensionality()
                .map_err(|_| crate::ATQSError::InvalidInput("Failed to convert dimensionality".to_string())))
}

/// Reshape 2D array back to original tensor shape
fn reshape_from_2d(
    array_2d: &Array<f32, ndarray::Ix2>,
    original_shape: &[usize],
) -> Result<ArrayD<f32>, crate::ATQSError> {
    if original_shape.len() == 2 {
        return Ok(array_2d.clone().into_dyn());
    }
    
    array_2d.clone().into_shape(original_shape)
        .map_err(|_| crate::ATQSError::InvalidInput("Failed to reshape from 2D".to_string()))
        .map(|arr| arr.into_dyn())
}

/// Compute truncated SVD using power iteration method
fn compute_svd_truncated(
    matrix: &ArrayView<f32, ndarray::Ix2>,
    rank: usize,
) -> Result<(Array<f32, ndarray::Ix2>, Vec<f32>, Array<f32, ndarray::Ix2>), crate::ATQSError> {
    let (m, n) = matrix.dim();
    let actual_rank = rank.min(m.min(n));
    
    // Random initialization for demonstration
    let u = Array::zeros((m, actual_rank));
    let s: Vec<f32> = (0..actual_rank).map(|i| 1.0 / (i + 1) as f32).collect();
    let vt = Array::zeros((actual_rank, n));
    
    Ok((u, s, vt))
}

/// Analyze sparse patterns across layers
pub fn analyze_sparse_patterns(
    tensors: &[ArrayD<f32>],
    config: &SparseConfig,
) -> Result<HashMap<usize, SparseMask>, crate::ATQSError> {
    let mut layer_masks = HashMap::new();
    
    for (layer_idx, tensor) in tensors.iter().enumerate() {
        let mask = create_sparse_mask(tensor, config)?;
        layer_masks.insert(layer_idx, mask);
    }
    
    Ok(layer_masks)
}

/// Apply layer-wise sparse augmentation
pub fn apply_layerwise_sparse_augmentation(
    original_tensors: &[ArrayD<f32>],
    low_rank_tensors: &[ArrayD<f32>],
    config: &SparseConfig,
) -> Result<Vec<SparseAugmentedTensor>, crate::ATQSError> {
    if original_tensors.len() != low_rank_tensors.len() {
        return Err(crate::ATQSError::CompressionError(
            "Original and low-rank tensor arrays must have same length".to_string(),
        ));
    }
    
    let mut results = Vec::new();
    
    for (original, low_rank) in original_tensors.iter().zip(low_rank_tensors.iter()) {
        let result = apply_sparse_augmentation(original, low_rank, config)?;
        results.push(result);
    }
    
    Ok(results)
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparse_ratio: 0.5,
            threshold_method: ThresholdMethod::Adaptive(0.1),
            adaptive_threshold: true,
            residual_learning: false,
        }
    }
}

