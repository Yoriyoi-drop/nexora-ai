//! Tensor operations for ATQS-Compress
//! Implements Tucker and Tensor-Train decomposition methods

use crate::error::{ATQSError, ATQSResult};
use ndarray::{Array, ArrayD, ArrayView, IxDyn};

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
                tucker_decompose(tensor, &ranks)
                    .map(TensorDecomposition::Tucker)
                    .or_else(|_| {
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TuckerDecomposition {
    /// Core tensor G
    pub core: ArrayD<f32>,
    /// Factor matrices U₁, U₂, ..., Uₙ
    pub factors: Vec<Array<f32, ndarray::Ix2>>,
    /// Original tensor shape
    pub original_shape: Vec<usize>,
}

/// Tensor-Train (TT) decomposition
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
            let core_shape = vec![ranks[k - 1], shape[k]];
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
pub fn compute_compression_ratio(original_shape: &[usize], decomp: &TensorDecomposition) -> f32 {
    let original_params: usize = original_shape.iter().product();

    let compressed_params = match decomp {
        TensorDecomposition::Tucker(tucker) => {
            let core_params: usize = tucker.core.shape().iter().product();
            let factor_params: usize = tucker.factors.iter().map(|f| f.len()).sum();
            core_params + factor_params
        }
        TensorDecomposition::TensorTrain(tt) => tt.cores.iter().map(|c| c.len()).sum(),
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
        return Err(crate::ATQSError::CompressionError(format!(
            "Mode {} exceeds tensor dimension {}",
            mode, ndim
        )));
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
            let mut row_index = i;
            let mut col_index = j;

            // Extract indices for modes before and after the unfolding mode
            for m in 0..=mode {
                indices.push(row_index % shape[m]);
                row_index /= shape[m];
            }
            for m in (mode + 1)..ndim {
                indices.push(col_index % shape[m]);
                col_index /= shape[m];
            }

            unfolded[[i, j]] = tensor[indices.as_slice()];
        }
    }

    Ok(unfolded)
}

/// Compute factor matrix for specific mode using SVD
fn _compute_factor_matrix(
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
    let tensor_view = tensor
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| {
            crate::ATQSError::InvalidInput("Failed to convert tensor to 2D view".to_string())
        })?;
    let result = tensor_view.dot(matrix);

    Ok(result.into_dyn())
}

/// Mode-n unfolding of tensor
fn _mode_n_unfold(
    tensor: &ArrayD<f32>,
    mode: usize,
) -> Result<Array<f32, ndarray::Ix2>, crate::ATQSError> {
    let shape = tensor.shape();
    let ndim = tensor.ndim();

    if mode >= ndim {
        return Err(crate::ATQSError::CompressionError(
            "Mode exceeds tensor dimensions".to_string(),
        ));
    }

    // Compute unfolded dimensions
    let rows = shape[mode];
    let cols: usize = shape
        .iter()
        .enumerate()
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
    let shape = tensor.shape().to_vec();
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

/// Contract two TT cores via batched GEMM.
/// core1 shape: [batch, ..., m, k]; core2 shape: [k, ..., n]
/// Contracts over the shared axis k and produces shape [batch, ..., m, n].
fn contract_cores(core1: &ArrayD<f32>, core2: &ArrayD<f32>) -> ATQSResult<ArrayD<f32>> {
    let shape1 = core1.shape();
    let shape2 = core2.shape();
    if shape1.len() < 2 || shape2.len() < 2 {
        return Err(ATQSError::DimensionMismatch(
            "TT cores must have at least 2 dimensions".to_string(),
        ));
    }
    // Verify contraction axis alignment: core1's last dim == core2's first dim
    if shape1[shape1.len() - 1] != shape2[0] {
        return Err(ATQSError::DimensionMismatch(format!(
            "TT core contraction dimension mismatch: core1[-1]={} vs core2[0]={}",
            shape1[shape1.len() - 1],
            shape2[0]
        )));
    }

    let mut result_shape: Vec<usize> = shape1[..shape1.len() - 1].to_vec();
    result_shape.extend_from_slice(&shape2[1..]);

    let mut result = Array::zeros(result_shape.clone());

    let batch_size = shape1[0];
    let m = shape1[shape1.len() - 1];
    let n = shape2[shape2.len() - 1];
    let k = shape2[0];

    let c1_slice = core1.as_slice().unwrap_or(&[]);
    let c2_slice = core2.as_slice().unwrap_or(&[]);
    let r_slice = result
        .as_slice_mut()
        .ok_or_else(|| ATQSError::TensorError("Cannot get mutable slice of result".to_string()))?;

    for b in 0..batch_size {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let idx1 = b * m * k + i * k + l;
                    let idx2 = l * n + j;
                    if idx1 < c1_slice.len() && idx2 < c2_slice.len() {
                        sum += c1_slice[idx1] * c2_slice[idx2];
                    }
                }
                let result_idx = b * m * n + i * n + j;
                if let Some(val) = r_slice.get_mut(result_idx) {
                    *val = sum;
                }
            }
        }
    }

    Ok(result.into_dyn())
}

/// Compute truncated SVD using power iteration method.
/// Uses GPU acceleration when `gpu` feature is enabled and context is available.
pub fn compute_svd_truncated(
    matrix: &ArrayView<f32, ndarray::Ix2>,
    rank: usize,
) -> Result<(Array<f32, ndarray::Ix2>, Vec<f32>, Array<f32, ndarray::Ix2>), crate::ATQSError> {
    let (m, n) = matrix.dim();
    let actual_rank = rank.min(m.min(n));

    if actual_rank == 0 {
        return Err(crate::ATQSError::InvalidInput(
            "Rank must be greater than 0".to_string(),
        ));
    }

    // Try GPU path first (feature-gated)
    #[cfg(feature = "gpu")]
    {
        let owned = matrix.to_owned();
        match crate::gpu_ops::compute_svd_truncated_gpu(&owned, rank) {
            Ok(result) => return Ok(result),
            Err(e) => {
                tracing::debug!("GPU SVD failed, falling back to CPU: {e}");
            }
        }
    }

    // CPU fallback: power iteration with ndarray
    let mut u = Array::zeros((m, actual_rank));
    let mut s = Vec::with_capacity(actual_rank);
    let mut vt = Array::zeros((actual_rank, n));
    let mut working_matrix = matrix.to_owned();

    for i in 0..actual_rank {
        let mut v = Array::zeros(n);
        for j in 0..n {
            v[j] = rand::random::<f32>() * 2.0 - 1.0;
        }
        let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if v_norm > 1e-10 {
            v.mapv_inplace(|x| x / v_norm);
        }

        for _ in 0..50 {
            let u_vec = working_matrix.dot(&v);
            let sigma = u_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

            if sigma > 1e-10 {
                v = working_matrix.t().dot(&u_vec) / sigma;
                let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if v_norm > 1e-10 {
                    v.mapv_inplace(|x| x / v_norm);
                }
            }
        }

        let u_vec = working_matrix.dot(&v);
        let sigma = u_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        if sigma > 1e-10 {
            let u_normalized = u_vec / sigma;
            for j in 0..m {
                u[[j, i]] = u_normalized[j];
            }
            s.push(sigma);
            for j in 0..n {
                vt[[i, j]] = v[j];
            }

            let mut outer_product = Array::zeros((m, n));
            for r in 0..m {
                for c in 0..n {
                    outer_product[[r, c]] = u_normalized[r] * v[c];
                }
            }
            working_matrix = working_matrix - &(outer_product * sigma);
        } else {
            for r in 0..m {
                u[[r, i]] = rand::random::<f32>() * 0.01;
            }
            s.push(0.01);
            for r in 0..n {
                vt[[i, r]] = rand::random::<f32>() * 0.01;
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
            "No cores found in TT decomposition".to_string(),
        ));
    }

    let mut result = decomposition.cores[0].clone();

    for core in &decomposition.cores[1..] {
        result = contract_cores(&result, core)?;
    }

    // Reshape to original shape
    result
        .into_shape(decomposition.original_shape.clone())
        .map_err(|e| {
            crate::ATQSError::CompressionError(format!(
                "Failed to reshape to original dimensions: {}",
                e
            ))
        })
}
