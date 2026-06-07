//! GPU-accelerated quantized matrix multiply.
//!
//! Dequantizes Q4/INT8 weights to FP32 on CPU (once), then runs batched
//! matmul on GPU via `nexora-autograd`. Falls back to `None` if GPU
//! unavailable, leaving callers to fall through to CPU `gemm::*`.

/// GPU-accelerated INT4 matmul: `C[M, N] = A[M, K] @ B_q4[K/2, N]`
///
/// Dequantizes weights to FP32 on CPU first (fast, bandwidth-bound loop),
/// then does the matmul on GPU via cuBLAS/wgpu.
///
/// Returns `None` if GPU is unavailable (graceful CPU fallback).
pub fn try_matmul_int4(
    a: &[f32],
    b_packed: &[u8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Option<Vec<f32>> {
    #[cfg(not(feature = "gpu"))]
    { let _ = (a, b_packed, scales, m, n, k, group_size); None }

    #[cfg(feature = "gpu")]
    {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global().ok()?;

        // Dequantize Q4 → FP32 on CPU (fast, once per call)
        let w_fp32 = dequant_q4_to_fp32(b_packed, scales, n, k, group_size);

        // GPU matmul: C[M,N] = A[M,K] @ W[K,N]
        let a_t = GpuTensor::from_slice(vec![m, k], a).ok()?;
        let w_t = GpuTensor::from_slice(vec![k, n], &w_fp32).ok()?;
        let c = ctx.matmul(&a_t, &w_t).ok()?;
        let arr = c.to_cpu().ok()?;
        Some(arr.into_iter().collect())
    }
}

/// GPU-accelerated INT8 matmul: `C[M, N] = A[M, K] @ B_int8[K, N]`
pub fn try_matmul_int8(
    a: &[f32],
    b: &[i8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Option<Vec<f32>> {
    #[cfg(not(feature = "gpu"))]
    { let _ = (a, b, scales, m, n, k, group_size); None }

    #[cfg(feature = "gpu")]
    {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global().ok()?;

        // Dequantize INT8 → FP32 on CPU (fast, once per call)
        let w_fp32 = dequant_int8_to_fp32(b, scales, n, k, group_size);

        // GPU matmul
        let a_t = GpuTensor::from_slice(vec![m, k], a).ok()?;
        let w_t = GpuTensor::from_slice(vec![k, n], &w_fp32).ok()?;
        let c = ctx.matmul(&a_t, &w_t).ok()?;
        let arr = c.to_cpu().ok()?;
        Some(arr.into_iter().collect())
    }
}

// --- Helper: Dequantize Q4 packed → FP32 flat matrix ---
fn dequant_q4_to_fp32(packed: &[u8], scales: &[f32], n: usize, k: usize, group_size: usize) -> Vec<f32> {
    let num_groups = k.div_ceil(group_size);
    let groups = num_groups.max(1);
    let mut out = vec![0.0f32; k * n];
    for col in 0..n {
        for g in 0..groups {
            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(k);
            let scale = scales[g * n + col];
            for kk in (g_start..g_end).step_by(2) {
                let byte_idx = (kk / 2) * n + col;
                let (b0, b1) = if byte_idx < packed.len() {
                    crate::gemm::unpack_q4(packed[byte_idx])
                } else {
                    (0, 0)
                };
                out[kk * n + col] = (b0 as f32 - 8.0) * scale;
                if kk + 1 < k {
                    out[(kk + 1) * n + col] = (b1 as f32 - 8.0) * scale;
                }
            }
        }
    }
    out
}

// --- Helper: Dequantize INT8 → FP32 flat matrix ---
fn dequant_int8_to_fp32(b: &[i8], scales: &[f32], n: usize, k: usize, group_size: usize) -> Vec<f32> {
    let num_groups = k.div_ceil(group_size);
    let groups = num_groups.max(1);
    let mut out = vec![0.0f32; k * n];
    for col in 0..n {
        for g in 0..groups {
            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(k);
            let scale = scales[g * n + col];
            for kk in g_start..g_end {
                let idx = kk * n + col;
                out[idx] = if idx < b.len() { b[idx] as f32 * scale } else { 0.0 };
            }
        }
    }
    out
}
