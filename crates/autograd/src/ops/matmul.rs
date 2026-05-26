use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::super::tensor::Tensor;
#[cfg(feature = "gpu")]
use crate::Storage;
use ndarray::ArrayD;
use tracing::warn;

pub(crate) static GPU_MATMUL_FALLBACKS: AtomicU64 = AtomicU64::new(0);
const GPU_FALLBACK_THRESHOLD: u64 = 50;
static GPU_CIRCUIT_BROKEN: AtomicBool = AtomicBool::new(false);

pub fn gpu_matmul_fallback_count() -> u64 {
    GPU_MATMUL_FALLBACKS.load(Ordering::Relaxed)
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 {
        tracing::error!("MatMul: a must be 2D, got {}D", a_shape.len());
        return Tensor::new(ArrayD::zeros(vec![0]));
    }
    if b_shape.len() != 2 {
        tracing::error!("MatMul: b must be 2D, got {}D", b_shape.len());
        return Tensor::new(ArrayD::zeros(vec![0]));
    }
    if a_shape[1] != b_shape[0] {
        tracing::error!(
            "MatMul: inner dims must match, got {} and {}",
            a_shape[1],
            b_shape[0]
        );
        return Tensor::new(ArrayD::zeros(vec![0]));
    }

    #[cfg(feature = "gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(_)) && matches!(&b_storage, Storage::Gpu(_));
        if on_gpu && !GPU_CIRCUIT_BROKEN.load(Ordering::Acquire) {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga), Storage::Gpu(gb)) => {
                    if let Ok(ctx) = crate::gpu::GpuContext::global() {
                        match ctx.matmul(ga, gb) {
                            Ok(gpu_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    // Full GPU residency: return tensor with Storage::Gpu
                                    let id = crate::tensor::next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                // Training path: GPU matmul with GPU-native backward
                                let a_gpu = ga.clone();
                                let b_gpu = gb.clone();
                                // Saved GPU tensors needed for backward
                                let saved_gpu = vec![a_gpu.clone(), b_gpu.clone()];
                                // CPU backward fallback
                                let cpu_backward: Box<
                                    dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>,
                                > = Box::new(move |grad, saved| {
                                    let a_val = &saved[0];
                                    let b_val = &saved[1];
                                    let grad_arr = grad.clone();
                                    let grad_mat =
                                        match grad_arr.view().into_dimensionality::<ndarray::Ix2>()
                                        {
                                            Ok(m) => m,
                                            Err(e) => {
                                                tracing::error!("matmul backward grad: {e}");
                                                return vec![
                                                    ArrayD::zeros(vec![0]),
                                                    ArrayD::zeros(vec![0]),
                                                ];
                                            }
                                        };
                                    let a_mat =
                                        match a_val.view().into_dimensionality::<ndarray::Ix2>() {
                                            Ok(m) => m,
                                            Err(e) => {
                                                tracing::error!("matmul backward a: {e}");
                                                return vec![
                                                    ArrayD::zeros(vec![0]),
                                                    ArrayD::zeros(vec![0]),
                                                ];
                                            }
                                        };
                                    let b_mat =
                                        match b_val.view().into_dimensionality::<ndarray::Ix2>() {
                                            Ok(m) => m,
                                            Err(e) => {
                                                tracing::error!("matmul backward b: {e}");
                                                return vec![
                                                    ArrayD::zeros(vec![0]),
                                                    ArrayD::zeros(vec![0]),
                                                ];
                                            }
                                        };
                                    let da = grad_mat.dot(&b_mat.t()).into_dyn();
                                    let db = a_mat.t().dot(&grad_mat).into_dyn();
                                    vec![da, db]
                                });
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a.data(), b.data()],
                                    saved_gpu,
                                    cpu_backward,
                                    Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                        let (da, db) = crate::gpu_backward::matmul_backward(
                                            ctx,
                                            &saved_gpu[0],
                                            &saved_gpu[1],
                                            grad_gpu,
                                        )
                                        .map_err(|e| format!("GPU matmul backward failed: {e}"))?;
                                        Ok(vec![da, db])
                                    })),
                                );
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "GPU matmul failed, falling back to CPU");
                                GPU_MATMUL_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                if GPU_MATMUL_FALLBACKS.load(Ordering::Relaxed)
                                    >= GPU_FALLBACK_THRESHOLD
                                {
                                    GPU_CIRCUIT_BROKEN.store(true, Ordering::Release);
                                    tracing::warn!(
                                        "GPU_MATMUL circuit breaker tripped after {} fallbacks",
                                        GPU_FALLBACK_THRESHOLD
                                    );
                                }
                            }
                        }
                    }
                }
                _ => {
                    warn!("MatMul: non-GPU storage despite on_gpu check — falling back to CPU");
                }
            }
        }
    }

    let a_data = a.data();
    let b_data = b.data();

    let a_view = a_data.view();
    let b_view = b_data.view();

    let a_mat = match a_view.into_dimensionality::<ndarray::Ix2>() {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("MatMul: a must be 2D: {e}");
            return Tensor::new(ArrayD::zeros(vec![0]));
        }
    };
    let b_mat = match b_view.into_dimensionality::<ndarray::Ix2>() {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("MatMul: b must be 2D: {e}");
            return Tensor::new(ArrayD::zeros(vec![0]));
        }
    };

    let result = a_mat.dot(&b_mat).into_dyn();

    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let saved = vec![a_data.clone(), b_data.clone()];

    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        saved,
        Box::new(|grad, saved| {
            let a_val = &saved[0];
            let b_val = &saved[1];
            let grad_arr = grad.clone();

            let grad_mat = match grad_arr.view().into_dimensionality::<ndarray::Ix2>() {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("matmul backward: grad must be 2D: {e}");
                    return vec![ArrayD::zeros(vec![0]), ArrayD::zeros(vec![0])];
                }
            };

            let a_mat = match a_val.view().into_dimensionality::<ndarray::Ix2>() {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("matmul backward: a must be 2D: {e}");
                    return vec![ArrayD::zeros(vec![0]), ArrayD::zeros(vec![0])];
                }
            };

            let b_mat = match b_val.view().into_dimensionality::<ndarray::Ix2>() {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("matmul backward: b must be 2D: {e}");
                    return vec![ArrayD::zeros(vec![0]), ArrayD::zeros(vec![0])];
                }
            };

            let da = grad_mat.dot(&b_mat.t()).into_dyn();
            let db = a_mat.t().dot(&grad_mat).into_dyn();

            vec![da, db]
        }),
    )
}
