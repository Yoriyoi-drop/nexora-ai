use ndarray::{Array, ArrayD, IxDyn};
use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};

use crate::ATQSError;

/// Compute truncated SVD using GPU-accelerated power iteration.
///
/// Strategy: upload matrix once, do all mat-vec + l2-norm + deflation on GPU,
/// only read back results at the end. Each rank's power iteration runs 50
/// iterations of (A·v, Aᵀ·u, normalize, deflate), all GPU-resident.
pub fn compute_svd_truncated_gpu(
    matrix: &Array<f32, ndarray::Ix2>,
    rank: usize,
) -> Result<(Array<f32, ndarray::Ix2>, Vec<f32>, Array<f32, ndarray::Ix2>), ATQSError> {
    let ctx = GpuContext::global()
        .map_err(|e| ATQSError::TensorError(format!("GPU context not initialized: {e}")))?;

    let (m, n) = matrix.dim();
    let actual_rank = rank.min(m.min(n));
    if actual_rank == 0 {
        return Err(ATQSError::InvalidInput("Rank must be > 0".into()));
    }

    // Upload matrix once
    let flat: Vec<f32> = matrix.iter().copied().collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[m, n]), flat)
        .map_err(|e| ATQSError::TensorError(e.to_string()))?;
    let mut a_gpu = GpuTensor::from_cpu(&arr).map_err(|e| ATQSError::TensorError(e.to_string()))?;
    let a_t_gpu = ctx
        .transpose(&a_gpu)
        .map_err(|e| ATQSError::TensorError(e.to_string()))?;

    let mut u_out = Array::<f32, _>::zeros((m, actual_rank));
    let mut s_out = Vec::with_capacity(actual_rank);
    let mut vt_out = Array::<f32, _>::zeros((actual_rank, n));

    for i in 0..actual_rank {
        // Random init v on CPU, upload
        let mut v_host: Vec<f32> = (0..n).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let v_norm: f32 = v_host.iter().map(|x| x * x).sum::<f32>().sqrt();
        if v_norm > 1e-10 {
            for x in v_host.iter_mut() {
                *x /= v_norm;
            }
        }
        let v_arr = ArrayD::from_shape_vec(IxDyn(&[n, 1]), v_host)
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        let mut v_gpu =
            GpuTensor::from_cpu(&v_arr).map_err(|e| ATQSError::TensorError(e.to_string()))?;

        // Power iteration: 50 rounds of A·v / Aᵀ·u
        for _ in 0..50 {
            let u_gpu = ctx
                .matmul(&a_gpu, &v_gpu)
                .map_err(|e| ATQSError::TensorError(e.to_string()))?;
            let sigma_sq_gpu = ctx
                .l2_norm(&u_gpu)
                .map_err(|e| ATQSError::TensorError(e.to_string()))?;
            let sigma_sq: f32 = sigma_sq_gpu
                .to_cpu_first_element()
                .map_err(|e| ATQSError::TensorError(e.to_string()))?;
            if sigma_sq > 1e-20 {
                let sigma = sigma_sq.sqrt();
                ctx.scale_inplace(&u_gpu, 1.0 / sigma)
                    .map_err(|e| ATQSError::TensorError(e.to_string()))?;
                v_gpu = ctx
                    .matmul(&a_t_gpu, &u_gpu)
                    .map_err(|e| ATQSError::TensorError(e.to_string()))?;
                let vn_gpu = ctx
                    .l2_norm(&v_gpu)
                    .map_err(|e| ATQSError::TensorError(e.to_string()))?;
                let vn: f32 = vn_gpu
                    .to_cpu_first_element()
                    .map_err(|e| ATQSError::TensorError(e.to_string()))?;
                if vn > 1e-10 {
                    ctx.scale_inplace(&v_gpu, 1.0 / vn)
                        .map_err(|e| ATQSError::TensorError(e.to_string()))?;
                }
            }
        }

        // Final u = A·v and σ
        let u_gpu = ctx
            .matmul(&a_gpu, &v_gpu)
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        let sigma_sq_gpu = ctx
            .l2_norm(&u_gpu)
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        let sigma: f32 = sigma_sq_gpu
            .to_cpu_first_element()
            .map_err(|e| ATQSError::TensorError(e.to_string()))?
            .sqrt();

        if sigma > 1e-10 {
            ctx.scale_inplace(&u_gpu, 1.0 / sigma)
                .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        }

        // Read back u and v
        let u_cpu = u_gpu
            .to_cpu()
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        let v_cpu = v_gpu
            .to_cpu()
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        let u_data: Vec<f32> = u_cpu.iter().copied().collect();
        let v_data: Vec<f32> = v_cpu.iter().copied().collect();

        for j in 0..m.min(u_data.len()) {
            u_out[[j, i]] = u_data[j];
        }
        s_out.push(sigma);
        for j in 0..n.min(v_data.len()) {
            vt_out[[i, j]] = v_data[j];
        }

        // Deflate: A -= σ · u · vᵀ  (all on GPU)
        let v_row = ctx
            .transpose(&v_gpu)
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        let outer = ctx
            .matmul(&u_gpu, &v_row)
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        ctx.scale_inplace(&outer, -sigma)
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
        ctx.add_inplace(&mut a_gpu, &outer)
            .map_err(|e| ATQSError::TensorError(e.to_string()))?;
    }

    Ok((u_out, s_out, vt_out))
}
