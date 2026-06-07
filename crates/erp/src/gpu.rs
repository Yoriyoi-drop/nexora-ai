use ndarray::{Array1, Array2};

/// Try to compute batch cosine similarity matrix on GPU.
/// Returns None if GPU unavailable.
pub fn try_batch_cosine_similarity(
    vectors: &[Array1<f32>],
) -> Option<Result<Vec<Vec<f32>>, String>> {
    try_gpu_batch_cosine_similarity(vectors)
}

#[cfg(feature = "gpu")]
fn try_gpu_batch_cosine_similarity(
    vectors: &[Array1<f32>],
) -> Option<Result<Vec<Vec<f32>>, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(_) => return None,
    };
    let n = vectors.len();
    if n == 0 {
        return Some(Ok(Vec::new()));
    }
    let dim = vectors[0].len();

    // Stack all vectors into a single [n, dim] matrix
    let mut flat = Vec::with_capacity(n * dim);
    for v in vectors {
        if v.len() != dim {
            return Some(Err("Dimension mismatch in batch cosine similarity".into()));
        }
        flat.extend_from_slice(v.as_slice().unwrap());
    }

    let shape = vec![n, dim];
    let matrix = match GpuTensor::from_slice(shape, &flat) {
        Ok(t) => t,
        Err(_) => return None,
    };

    // Compute L2 norm per row: sqrt(sum(x^2)) along axis=1
    let norms = compute_row_norms_gpu(ctx, &matrix)?;

    // Normalize rows: x / max(norm, eps)
    let eps = GpuTensor::from_slice(vec![1], &[1e-10f32]).ok()?;
    let norm_safe = ctx.add(&norms, &eps).ok()?;
    let normalized = ctx.div(&matrix, &norm_safe).ok()?;

    // Cosine similarity = normalized * normalized^T
    let normalized_t = ctx.transpose(&normalized).ok()?;
    let sim_matrix = ctx.matmul(&normalized, &normalized_t).ok()?;

    // Readback
    let sim_flat = sim_matrix.to_cpu().ok()?;
    let sim_slice = sim_flat.as_slice().unwrap_or(&[]);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * n;
        let end = start + n;
        if end <= sim_slice.len() {
            result.push(sim_slice[start..end].to_vec());
        }
    }

    Some(Ok(result))
}

#[cfg(feature = "gpu")]
fn compute_row_norms_gpu(ctx: &nexora_autograd::gpu::GpuContext, matrix: &nexora_autograd::gpu::GpuTensor) -> Option<nexora_autograd::gpu::GpuTensor> {
    use nexora_autograd::gpu::GpuTensor;
    let sq = ctx.mul(matrix, matrix).ok()?;
    let rows = matrix.shape()[0];
    let dim = matrix.shape()[1];

    // Sum along axis=1: [rows, dim] sum -> [rows, 1]
    // wgpu sum reduces full tensor, so we compute column-wise sum
    let sum_result = ctx.sum(&sq).ok()?;

    // Broadcast scalar sum back
    let sum_val = sum_result.to_cpu_first_element().ok()?;
    let per_row = sum_val / rows as f32; // approximate, not correct
    // Actually we need per-row sum, not total sum
    // Better approach: compute elementwise in loop
    drop(sum_result);

    // Manual row-wise sum using reshape + matmul
    // [rows, dim] * [dim, 1] = [rows, 1]
    let ones = GpuTensor::from_slice(vec![dim, 1], &vec![1.0f32; dim]).ok()?;
    let row_sums = ctx.matmul(&sq, &ones).ok()?;
    let row_norms = ctx.sqrt(&row_sums).ok()?;

    Some(row_norms)
}

#[cfg(not(feature = "gpu"))]
fn try_gpu_batch_cosine_similarity(
    _vectors: &[Array1<f32>],
) -> Option<Result<Vec<Vec<f32>>, String>> {
    None
}

/// Try to compute softmax on GPU.
pub fn try_softmax(values: &[f32]) -> Option<Result<Vec<f32>, String>> {
    try_gpu_softmax(values)
}

#[cfg(feature = "gpu")]
fn try_gpu_softmax(values: &[f32]) -> Option<Result<Vec<f32>, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    let ctx = GpuContext::global().ok()?;
    let n = values.len();
    if n == 0 {
        return Some(Ok(Vec::new()));
    }

    // Softmax expects 2D: [1, n]
    let tensor = GpuTensor::from_slice(vec![1, n], values).ok()?;
    let result = ctx.softmax(&tensor).ok()?;
    let flat = result.to_cpu().ok()?;
    let slice = flat.as_slice().unwrap_or(&[]);

    Some(Ok(slice.to_vec()))
}

#[cfg(not(feature = "gpu"))]
fn try_gpu_softmax(_values: &[f32]) -> Option<Result<Vec<f32>, String>> {
    None
}

/// Try to compute matrix-vector multiply on GPU: weights * input → output
pub fn try_matvec(
    weights: &Array2<f32>,
    input: &Array1<f32>,
) -> Option<Result<Array1<f32>, String>> {
    try_gpu_matvec(weights, input)
}

#[cfg(feature = "gpu")]
fn try_gpu_matvec(
    weights: &Array2<f32>,
    input: &Array1<f32>,
) -> Option<Result<Array1<f32>, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    let ctx = GpuContext::global().ok()?;
    let (rows, cols) = weights.dim();

    let w_flat: Vec<f32> = weights.iter().copied().collect();
    let w_gpu = GpuTensor::from_slice(vec![rows, cols], &w_flat).ok()?;

    let i_gpu = GpuTensor::from_slice(vec![cols, 1], input.as_slice().unwrap_or(&[])).ok()?;

    let out_gpu = ctx.matmul(&w_gpu, &i_gpu).ok()?;
    let out_flat = out_gpu.to_cpu().ok()?;
    let out_slice = out_flat.as_slice().unwrap_or(&[]);

    Some(Ok(Array1::from_vec(out_slice.to_vec())))
}

#[cfg(not(feature = "gpu"))]
fn try_gpu_matvec(
    _weights: &Array2<f32>,
    _input: &Array1<f32>,
) -> Option<Result<Array1<f32>, String>> {
    None
}

/// Try to compute sum of squares on GPU.
pub fn try_sum_squares(values: &Array1<f32>) -> Option<Result<f32, String>> {
    try_gpu_sum_squares(values)
}

#[cfg(feature = "gpu")]
fn try_gpu_sum_squares(values: &Array1<f32>) -> Option<Result<f32, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    let ctx = GpuContext::global().ok()?;
    let n = values.len();
    let tensor = GpuTensor::from_slice(vec![1, n], values.as_slice().unwrap_or(&[])).ok()?;
    let sq = ctx.mul(&tensor, &tensor).ok()?;
    let sum_t = sq.sum_gpu().ok()?;

    Some(Ok(sum_t))
}

#[cfg(not(feature = "gpu"))]
fn try_gpu_sum_squares(_values: &Array1<f32>) -> Option<Result<f32, String>> {
    None
}

/// Try to compute matrix-matrix multiply on GPU.
pub fn try_matmul(
    a: &Array2<f32>,
    b: &Array2<f32>,
) -> Option<Result<Array2<f32>, String>> {
    try_gpu_matmul(a, b)
}

#[cfg(feature = "gpu")]
fn try_gpu_matmul(
    a: &Array2<f32>,
    b: &Array2<f32>,
) -> Option<Result<Array2<f32>, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    let ctx = GpuContext::global().ok()?;
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();

    let a_flat: Vec<f32> = a.iter().copied().collect();
    let b_flat: Vec<f32> = b.iter().copied().collect();

    let a_gpu = GpuTensor::from_slice(vec![a_rows, a_cols], &a_flat).ok()?;
    let b_gpu = GpuTensor::from_slice(vec![b_rows, b_cols], &b_flat).ok()?;

    let out_gpu = ctx.matmul(&a_gpu, &b_gpu).ok()?;
    let out_flat = out_gpu.to_cpu().ok()?;
    let out_slice = out_flat.as_slice().unwrap_or(&[]);

    Some(Ok(Array2::from_shape_vec((a_rows, b_cols), out_slice.to_vec()).unwrap()))
}

#[cfg(not(feature = "gpu"))]
fn try_gpu_matmul(
    _a: &Array2<f32>,
    _b: &Array2<f32>,
) -> Option<Result<Array2<f32>, String>> {
    None
}

/// Try to compute dot product on GPU.
pub fn try_dot(a: &Array1<f32>, b: &Array1<f32>) -> Option<Result<f32, String>> {
    try_gpu_dot(a, b)
}

#[cfg(feature = "gpu")]
fn try_gpu_dot(a: &Array1<f32>, b: &Array1<f32>) -> Option<Result<f32, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    if a.len() != b.len() {
        return Some(Err("Dot product dimension mismatch".into()));
    }
    let ctx = GpuContext::global().ok()?;
    let n = a.len();

    let a_gpu = GpuTensor::from_slice(vec![1, n], a.as_slice().unwrap_or(&[])).ok()?;
    let b_gpu = GpuTensor::from_slice(vec![n, 1], b.as_slice().unwrap_or(&[])).ok()?;

    let out_gpu = ctx.matmul(&a_gpu, &b_gpu).ok()?;
    let val = out_gpu.to_cpu_first_element().ok()?;

    Some(Ok(val))
}

#[cfg(not(feature = "gpu"))]
fn try_gpu_dot(_a: &Array1<f32>, _b: &Array1<f32>) -> Option<Result<f32, String>> {
    None
}

/// Try to compute L2 norm on GPU.
pub fn try_l2_norm(v: &Array1<f32>) -> Option<Result<f32, String>> {
    try_gpu_l2_norm(v)
}

#[cfg(feature = "gpu")]
fn try_gpu_l2_norm(v: &Array1<f32>) -> Option<Result<f32, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    let ctx = GpuContext::global().ok()?;
    let n = v.len();
    let tensor = GpuTensor::from_slice(vec![1, n], v.as_slice().unwrap_or(&[])).ok()?;
    let sq = ctx.mul(&tensor, &tensor).ok()?;
    let sum_t = sq.sum_gpu().ok()?;
    let norm = sum_t.sqrt();

    Some(Ok(norm))
}

#[cfg(not(feature = "gpu"))]
fn try_gpu_l2_norm(_v: &Array1<f32>) -> Option<Result<f32, String>> {
    None
}

/// Try to compute elementwise sum on GPU: a + b
pub fn try_elem_add(a: &Array1<f32>, b: &Array1<f32>) -> Option<Result<Array1<f32>, String>> {
    try_gpu_elem_binary(a, b, ElemOp::Add)
}

/// Try to compute elementwise difference on GPU: a - b
pub fn try_elem_sub(a: &Array1<f32>, b: &Array1<f32>) -> Option<Result<Array1<f32>, String>> {
    try_gpu_elem_binary(a, b, ElemOp::Sub)
}

/// Try to compute elementwise multiply on GPU: a * b
pub fn try_elem_mul(a: &Array1<f32>, b: &Array1<f32>) -> Option<Result<Array1<f32>, String>> {
    try_gpu_elem_binary(a, b, ElemOp::Mul)
}

/// Try to compute elementwise divide on GPU: a / b
pub fn try_elem_div(a: &Array1<f32>, b: &Array1<f32>) -> Option<Result<Array1<f32>, String>> {
    try_gpu_elem_binary(a, b, ElemOp::Div)
}

#[cfg(feature = "gpu")]
enum ElemOp { Add, Sub, Mul, Div }

#[cfg(feature = "gpu")]
fn try_gpu_elem_binary(
    a: &Array1<f32>,
    b: &Array1<f32>,
    op: ElemOp,
) -> Option<Result<Array1<f32>, String>> {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};

    if a.len() != b.len() {
        return Some(Err("Elementwise dimension mismatch".into()));
    }
    let ctx = GpuContext::global().ok()?;
    let n = a.len();

    let a_gpu = GpuTensor::from_slice(vec![1, n], a.as_slice().unwrap_or(&[])).ok()?;
    let b_gpu = GpuTensor::from_slice(vec![1, n], b.as_slice().unwrap_or(&[])).ok()?;

    let out_gpu = match op {
        ElemOp::Add => ctx.add(&a_gpu, &b_gpu).ok()?,
        ElemOp::Sub => ctx.sub(&a_gpu, &b_gpu).ok()?,
        ElemOp::Mul => ctx.mul(&a_gpu, &b_gpu).ok()?,
        ElemOp::Div => ctx.div(&a_gpu, &b_gpu).ok()?,
    };

    let flat = out_gpu.to_cpu().ok()?;
    let slice = flat.as_slice().unwrap_or(&[]);

    Some(Ok(Array1::from_vec(slice.to_vec())))
}

#[cfg(not(feature = "gpu"))]
enum ElemOp { Add, Sub, Mul, Div }

#[cfg(not(feature = "gpu"))]
fn try_gpu_elem_binary(
    _a: &Array1<f32>,
    _b: &Array1<f32>,
    _op: ElemOp,
) -> Option<Result<Array1<f32>, String>> {
    None
}

/// Helper: try any GPU op, returning None on failure (fallback to CPU).
#[macro_export]
macro_rules! gpu_try {
    ($expr:expr) => {{
        match $expr {
            Some(Ok(v)) => Some(v),
            Some(Err(e)) => {
                tracing::debug!("ERP GPU op failed (fallback to CPU): {}", e);
                None
            }
            None => None,
        }
    }};
}
