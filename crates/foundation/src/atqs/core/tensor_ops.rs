//! Tensor operations for ATQS-Compress
//! Implements Tucker and Tensor-Train decomposition methods

use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::Standard;
use std::collections::HashMap;
use crate::atqs::error::ATQSError;

/// Trait for ATQS compression operations
pub trait ATQSCompressor: Send + Sync {
    fn compress(&self, tensor: &ArrayD<f32>) -> Result<TensorDecomposition, ATQSError>;
    fn decompress(&self, decomposition: &TensorDecomposition) -> Result<ArrayD<f32>, ATQSError>;
    fn compression_ratio(&self) -> f32;
}

/// Basic tensor compressor implementation
#[derive(Debug, Clone)]
pub struct TensorCompressor {
    pub method: CompressionMethod,
    pub target_ratio: f32,
}

#[derive(Debug, Clone)]
pub enum CompressionMethod {
    Tucker,
    TensorTrain,
    Hybrid,
}

impl Default for TensorCompressor {
    fn default() -> Self {
        Self {
            method: CompressionMethod::Tucker,
            target_ratio: 0.5,
        }
    }
}

impl ATQSCompressor for TensorCompressor {
    fn compress(&self, tensor: &ArrayD<f32>) -> Result<TensorDecomposition, ATQSError> {
        match self.method {
            CompressionMethod::Tucker => {
                let ranks = vec![4; tensor.ndim()];
                tucker_decompose(tensor, &ranks).map(TensorDecomposition::Tucker)
            }
            CompressionMethod::TensorTrain => {
                let ranks = vec![4; tensor.ndim() - 1];
                tensor_train_decompose(tensor, &ranks).map(TensorDecomposition::TensorTrain)
            }
            CompressionMethod::Hybrid => {
                // Simple hybrid: try Tucker first, fallback to TT
                let ranks = vec![4; tensor.ndim()];
                tucker_decompose(tensor, &ranks).map(TensorDecomposition::Tucker).or_else(|_| {
                    let ranks = vec![4; tensor.ndim() - 1];
                    tensor_train_decompose(tensor, &ranks).map(TensorDecomposition::TensorTrain)
                })
            }
        }
    }

    fn decompress(&self, decomposition: &TensorDecomposition) -> Result<ArrayD<f32>, ATQSError> {
        match decomposition {
            TensorDecomposition::Tucker(tucker) => tucker_reconstruct(tucker),
            TensorDecomposition::TensorTrain(tt) => tensor_train_reconstruct(tt),
        }
    }

    fn compression_ratio(&self) -> f32 {
        self.target_ratio
    }
}

/// Tensor decomposition formats
#[derive(Debug, Clone)]
pub enum TensorDecomposition {
    Tucker(TuckerDecomposition),
    TensorTrain(TensorTrainDecomposition),
}

/// Tucker decomposition: A tensor ≈ G ×₁U₁ ×₂U₂ ... ×ₙUₙ
#[derive(Debug, Clone)]
pub struct TuckerDecomposition {
    /// Core tensor G
    pub core: ArrayD<f32>,
    /// Factor matrices U₁, U₂, ..., Uₙ
    pub factors: Vec<Array<f32, ndarray::Ix2>>,
    /// Original tensor shape
    pub original_shape: Vec<usize>,
}

/// Tensor-Train (TT) decomposition
#[derive(Debug, Clone)]
pub struct TensorTrainDecomposition {
    /// TT cores G₁, G₂, ..., Gₙ
    pub cores: Vec<ArrayD<f32>>,
    /// TT ranks
    pub ranks: Vec<usize>,
    /// Original tensor shape
    pub original_shape: Vec<usize>,
}

/// Perform Tucker decomposition on a tensor
pub fn tucker_decompose(
    tensor: &ArrayD<f32>,
    ranks: &[usize],
) -> Result<TuckerDecomposition, crate::ATQSError> {
    let ndim = tensor.ndim();
    if ranks.len() != ndim {
        return Err(crate::ATQSError::CompressionError(
            "Rank dimensions must match tensor dimensions".to_string(),
        ));
    }

    let mut factors = Vec::new();
    
    // Compute factor matrices for each mode
    for mode in 0..ndim {
        let unfolded = unfold_tensor(tensor, mode)?;
        let (u, _, _) = compute_svd_truncated(&unfolded.view(), ranks[mode])?;
        factors.push(u);
    }
    
    // Compute core tensor
    let core = compute_core_tensor(tensor, &factors)?;

    Ok(TuckerDecomposition {
        core,
        factors,
        original_shape: tensor.shape().to_vec(),
    })
}

/// Perform Tensor-Train decomposition
pub fn tensor_train_decompose(
    tensor: &ArrayD<f32>,
    ranks: &[usize],
) -> Result<TensorTrainDecomposition, crate::ATQSError> {
    let shape = tensor.shape();
    let order = shape.len();
    
    if ranks.len() != order - 1 {
        return Err(crate::ATQSError::CompressionError(
            "TT ranks length must be tensor order - 1".to_string(),
        ));
    }

    let mut cores = Vec::new();
    let mut current_tensor = tensor.clone();

    // Sequential unfolding and SVD
    for k in 0..order {
        let (core, next_tensor) = if k == order - 1 {
            // Last core
            let core_shape = vec![ranks[k-1], shape[k]];
            let reshaped = current_tensor.into_shape(core_shape)?;
            (reshaped.into_dyn(), ArrayD::zeros(IxDyn(&[])))
        } else {
            unfold_and_svd(&current_tensor, k, shape[k], ranks[k])?
        };
        
        cores.push(core);
        current_tensor = next_tensor;
    }

    Ok(TensorTrainDecomposition {
        cores,
        ranks: ranks.to_vec(),
        original_shape: shape.to_vec(),
    })
}

/// Compute compression ratio
pub fn compute_compression_ratio(
    original_shape: &[usize],
    decomp: &TensorDecomposition,
) -> f32 {
    let original_params: usize = original_shape.iter().product();
    
    let compressed_params = match decomp {
        TensorDecomposition::Tucker(tucker) => {
            let core_params: usize = tucker.core.shape().iter().product();
            let factor_params: usize = tucker.factors.iter()
                .map(|f| f.len())
                .sum();
            core_params + factor_params
        }
        TensorDecomposition::TensorTrain(tt) => {
            tt.cores.iter().map(|c| c.len()).sum()
        }
    };
    
    original_params as f32 / compressed_params as f32
}

/// Unfold tensor along specified mode
fn unfold_tensor(
    tensor: &ArrayD<f32>,
    mode: usize,
) -> Result<Array<f32, ndarray::Ix2>, crate::ATQSError> {
    let shape = tensor.shape();
    let ndim = shape.len();
    
    if mode >= ndim {
        return Err(crate::ATQSError::CompressionError(
            format!("Mode {} exceeds tensor dimension {}", mode, ndim)
        ));
    }
    
    // Compute dimensions for unfolding
    let rows = shape.iter().take(mode + 1).product();
    let cols = shape.iter().skip(mode + 1).product();
    
    // Create unfolded matrix
    let mut unfolded = Array::zeros((rows, cols));
    
    // Fill unfolded matrix
    for i in 0..rows {
        for j in 0..cols {
            // Map (i, j) to multi-dimensional indices
            let mut indices = Vec::new();
            let mut temp_i = i;
            let mut temp_j = j;
            
            // Extract indices for modes before and after the unfolding mode
            for m in 0..=mode {
                indices.push(temp_i % shape[m]);
                temp_i /= shape[m];
            }
            for m in (mode + 1)..ndim {
                indices.push(temp_j % shape[m]);
                temp_j /= shape[m];
            }
            
            unfolded[[i, j]] = tensor[indices.as_slice()];
        }
    }
    
    Ok(unfolded)
}

/// Compute factor matrix for specific mode using SVD
fn compute_factor_matrix(
    tensor: &ArrayD<f32>,
    mode: usize,
    rank: usize,
) -> Result<Array<f32, ndarray::Ix2>, crate::ATQSError> {
    // Mode-n unfolding
    let unfolded = unfold_tensor(tensor, mode)?;
    
    // Compute SVD
    let (u, _s, _vt) = compute_svd_truncated(&unfolded.view(), rank)?;
    
    Ok(u)
}

/// Mode-n product of tensor with matrix
fn mode_n_product(
    tensor: &ArrayD<f32>,
    matrix: &ArrayView<f32, ndarray::Ix2>,
    mode: usize,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let shape = tensor.shape();
    let n = shape[mode];
    
    if matrix.shape()[0] != n {
        return Err(crate::ATQSError::CompressionError(
            "Matrix dimensions incompatible for mode-n product".to_string(),
        ));
    }

    // Reshape tensor for matrix multiplication
    let mut new_shape = shape.to_vec();
    new_shape[mode] = matrix.shape()[1];
    
    // Perform the contraction
    let tensor_view = tensor.view().into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| crate::ATQSError::InvalidInput("Failed to convert tensor to 2D view".to_string()))?;
    let result = tensor_view.dot(matrix);
    
    Ok(result.into_dyn())
}

/// Mode-n unfolding of tensor
fn mode_n_unfold(tensor: &ArrayD<f32>, mode: usize) -> Result<Array<f32, ndarray::Ix2>, crate::ATQSError> {
    let shape = tensor.shape();
    let ndim = tensor.ndim();
    
    if mode >= ndim {
        return Err(crate::ATQSError::CompressionError(
            "Mode exceeds tensor dimensions".to_string(),
        ));
    }

    // Compute unfolded dimensions
    let rows = shape[mode];
    let cols: usize = shape.iter().enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, s)| *s)
        .product();

    // Create unfolded matrix
    let mut unfolded = Array::zeros((rows, cols));
    
    // Fill unfolded matrix (simplified implementation)
    for i in 0..rows {
        for j in 0..cols {
            // Convert linear index to multi-dimensional index
            let mut indices = vec![0; ndim];
            indices[mode] = i;
            
            // Compute remaining indices from j
            let mut temp = j;
            for (dim_idx, &dim_size) in shape.iter().enumerate() {
                if dim_idx != mode {
                    indices[dim_idx] = temp % dim_size;
                    temp /= dim_size;
                }
            }
            
            unfolded[[i, j]] = tensor[&indices[..]];
        }
    }

    Ok(unfolded)
}

/// Compute core tensor for Tucker decomposition
fn compute_core_tensor(
    tensor: &ArrayD<f32>,
    factors: &[Array<f32, ndarray::Ix2>],
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let mut result = tensor.clone();
    
    for (mode, factor) in factors.iter().enumerate() {
        result = mode_n_product(&result, &factor.t().view(), mode)?;
    }
    
    Ok(result)
}

/// Unfold tensor and perform SVD for TT decomposition
fn unfold_and_svd(
    tensor: &ArrayD<f32>,
    mode: usize,
    dim_size: usize,
    rank: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>), crate::ATQSError> {
    // Reshape for unfolding
    let mut shape = tensor.shape().to_vec();
    let rows = shape.iter().take(mode + 1).product();
    let cols = shape.iter().skip(mode + 1).product();
    
    let reshaped = tensor.clone().into_shape((rows, cols))?;
    
    // Perform SVD
    let (u, s, vt) = compute_svd_truncated(&reshaped.view(), rank)?;
    
    // Create core
    let core_shape = (rank, dim_size, cols / dim_size);
    let mut core = Array::zeros(core_shape);
    
    // Extract core from U and S
    for i in 0..rank {
        for j in 0..dim_size {
            for k in 0..(cols / dim_size) {
                core[[i, j, k]] = u[[i, j * (cols / dim_size) + k]] * s[i];
            }
        }
    }
    
    // Create next tensor for recursion
    let next_shape = (rank, cols);
    let mut next_tensor = Array::zeros(next_shape);
    
    for i in 0..rank {
        for j in 0..cols {
            next_tensor[[i, j]] = vt[[i, j]];
        }
    }
    
    Ok((core.into_dyn(), next_tensor.into_dyn()))
}

/// Contract two TT cores
fn contract_cores(core1: &ArrayD<f32>, core2: &ArrayD<f32>) -> ArrayD<f32> {
    // Simplified contraction - would need proper implementation
    let shape1 = core1.shape();
    let shape2 = core2.shape();
    
    // Result shape computation
    let mut result_shape = Vec::new();
    result_shape.extend_from_slice(&shape1[..shape1.len()-1]);
    result_shape.extend_from_slice(&shape2[1..]);
    
    // Create result tensor with actual computation
    let mut result = Array::zeros(result_shape.clone());
    
    // Perform batch matrix multiplication
    let batch_size = shape1[0];
    let m = shape1[shape1.len() - 1];
    let n = shape2[shape2.len() - 1];
    let k = shape2[0];
    
    for b in 0..batch_size {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    // Use linear indexing for simplicity
                    let idx1 = b * m * k + i * k + l;
                    let idx2 = l * n + j;
                    if idx1 < core1.len() && idx2 < core2.len() {
                        let val1 = core1.as_slice().map(|slice| slice.get(idx1).copied()).unwrap_or(Some(0.0)).unwrap_or(0.0);
                        let val2 = core2.as_slice().map(|slice| slice.get(idx2).copied()).unwrap_or(Some(0.0)).unwrap_or(0.0);
                        sum += val1 * val2;
                    }
                }
                let result_idx = b * m * n + i * n + j;
                if result_idx < result.len() {
                    if let Some(slice) = result.as_slice_mut() {
                        if let Some(val) = slice.get_mut(result_idx) {
                            *val = sum;
                        }
                    }
                }
            }
        }
    }
    
    result.into_dyn()
}

/// Compute truncated SVD using power iteration method
fn compute_svd_truncated(
    matrix: &ArrayView<f32, ndarray::Ix2>,
    rank: usize,
) -> Result<(Array<f32, ndarray::Ix2>, Vec<f32>, Array<f32, ndarray::Ix2>), crate::ATQSError> {
    let (m, n) = matrix.dim();
    let actual_rank = rank.min(m.min(n));
    
    if actual_rank == 0 {
        return Err(crate::ATQSError::InvalidInput("Rank must be greater than 0".to_string()));
    }
    
    // Initialize U, S, Vt matrices
    let mut u = Array::zeros((m, actual_rank));
    let mut s = Vec::with_capacity(actual_rank);
    let mut vt = Array::zeros((actual_rank, n));
    
    // Use power iteration to compute dominant singular vectors
    let mut temp_matrix = matrix.to_owned();
    
    for i in 0..actual_rank {
        // Power iteration for singular value
        let mut v = Array::zeros(n);
        for i in 0..n {
            v[i] = rand::random::<f32>() * 2.0 - 1.0; // Random value between -1 and 1
        }
        let mut sigma = 0.0;
        
        // Normalize v
        let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if v_norm > 1e-10 {
            v.mapv_inplace(|x| x / v_norm);
        }
        
        // Power iteration
        for _ in 0..50 {
            // v = A^T * u
            let u_vec = temp_matrix.dot(&v);
            sigma = u_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            if sigma > 1e-10 {
                v = temp_matrix.t().dot(&u_vec) / sigma;
                
                // Normalize v
                let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if v_norm > 1e-10 {
                    v.mapv_inplace(|x| x / v_norm);
                }
            }
        }
        
        // Compute u vector
        let u_vec = temp_matrix.dot(&v);
        sigma = u_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if sigma > 1e-10 {
            let mut u_normalized = u_vec / sigma;
            
            // Store results
            for j in 0..m {
                u[[j, i]] = u_normalized[j];
            }
            s.push(sigma);
            for j in 0..n {
                vt[[i, j]] = v[j];
            }
            
            // Deflate matrix for next iteration
            let mut outer_product = Array::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    outer_product[[i, j]] = u_normalized[i] * v[j];
                }
            }
            temp_matrix = temp_matrix - &(outer_product * sigma);
        } else {
            // If sigma is too small, use random values
            for j in 0..m {
                u[[j, i]] = rand::random::<f32>() * 0.01;
            }
            s.push(0.01);
            for j in 0..n {
                vt[[i, j]] = rand::random::<f32>() * 0.01;
            }
        }
    }
    
    Ok((u, s, vt))
}


/// Tucker reconstruction
pub fn tucker_reconstruct(
    decomposition: &TuckerDecomposition,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let mut result = decomposition.core.clone();
    
    for (mode, factor) in decomposition.factors.iter().enumerate() {
        result = mode_n_product(&result, &factor.view(), mode)?;
    }
    
    Ok(result)
}

/// Tensor-Train reconstruction
pub fn tensor_train_reconstruct(
    decomposition: &TensorTrainDecomposition,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    if decomposition.cores.is_empty() {
        return Err(crate::ATQSError::CompressionError(
            "No cores found in TT decomposition".to_string()
        ));
    }
    
    let mut result = decomposition.cores[0].clone();
    
    for core in &decomposition.cores[1..] {
        result = contract_cores(&result, core);
    }
    
    // Reshape to original shape
    result.into_shape(decomposition.original_shape.clone())
        .map_err(|e| crate::ATQSError::CompressionError(
            format!("Failed to reshape to original dimensions: {}", e)
        ))
}
