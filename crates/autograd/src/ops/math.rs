use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{ArrayD, IxDyn};
use tracing::debug;

use super::super::broadcast;
use super::super::tensor::Tensor;
#[cfg(feature = "device-gpu")]
use crate::gpu::gpu_recovery::RECOVERY_MANAGER;
#[cfg(feature = "device-gpu")]
use crate::gpu::{ElemOp, GpuContext, GpuTensor};
#[cfg(feature = "device-gpu")]
use crate::gpu::GpuError;
#[cfg(feature = "device-gpu")]
use crate::{tensor::next_tensor_id, Storage};

pub(crate) static GPU_MATH_FALLBACKS: AtomicU64 = AtomicU64::new(0);

pub fn gpu_math_fallback_count() -> u64 {
    GPU_MATH_FALLBACKS.load(Ordering::Relaxed)
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        #[cfg(feature = "device-cuda")]
        if matches!(&a_storage, Storage::Cuda(..))
            && matches!(&b_storage, Storage::Cuda(..))
        {
            match (&a_storage, &b_storage) {
                #[cfg(feature = "device-cuda")]
                (Storage::Cuda(ca, _), Storage::Cuda(cb, _)) => {
                    if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                        match ctx.add(ca, cb) {
                            Ok(cuda_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_cuda(cuda_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let a_shape_saved = ArrayD::from_shape_vec(
                                    vec![a_shape.len()],
                                    a_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                let b_shape_saved = ArrayD::from_shape_vec(
                                    vec![b_shape.len()],
                                    b_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                return Tensor::from_cuda_with_grad_fn(
                                    cuda_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> =
                                            saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> =
                                            saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, db]
                                    }),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!(error = %e, "CUDA add failed, falling back");
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(..)) && matches!(&b_storage, Storage::Gpu(..));
        if on_gpu && !RECOVERY_MANAGER.is_circuit_open() {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga, _), Storage::Gpu(gb, _)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.add(ga, gb) {
                            Ok(gpu_result) => {
                                RECOVERY_MANAGER.record_recovery();
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let a_shape_saved = ArrayD::from_shape_vec(
                                    vec![a_shape.len()],
                                    a_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                let b_shape_saved = ArrayD::from_shape_vec(
                                    vec![b_shape.len()],
                                    b_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    vec![ga.clone(), gb.clone()],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> =
                                            saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> =
                                            saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, db]
                                    }),
                                    Some(Box::new(move |_saved_gpu, grad_gpu, ctx| {
                                        let a_shape = a_shape.to_vec();
                                        let b_shape = b_shape.to_vec();
                                        let (da, db) = crate::gpu_backward::add_backward(
                                            ctx, &a_shape, &b_shape, grad_gpu,
                                        )
                                        .map_err(|e| format!("add_backward: {e}"))?;
                                        Ok(vec![da, db])
                                    })),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                    "GPU add failed: {e}"
                                )));
                                tracing::warn!(error = %e, "GPU add failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                            }
                        }
                    }
                }
                _ => {
                    tracing::warn!(
                        "add: non-GPU storage despite on_gpu check — falling back to CPU"
                    );
                    GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    RECOVERY_MANAGER.record_failure(&GpuError::Device(
                        "add: non-GPU storage despite on_gpu check".into(),
                    ));
                }
            }
        }
    }
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc + &b_bc;
    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }
    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();
    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            ArrayD::from_shape_vec(
                vec![a_shape.len()],
                a_shape.iter().map(|&x| x as f32).collect(),
            )
            .unwrap_or_else(|e| {
                debug!("shape encoding failed (infallible): {e}");
                ArrayD::zeros(vec![0])
            }),
            ArrayD::from_shape_vec(
                vec![b_shape.len()],
                b_shape.iter().map(|&x| x as f32).collect(),
            )
            .unwrap_or_else(|e| {
                debug!("shape encoding failed (infallible): {e}");
                ArrayD::zeros(vec![0])
            }),
            a_bc,
            b_bc,
        ],
        Box::new(|grad, saved| {
            let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
            let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
            vec![da, db]
        }),
    )
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        #[cfg(feature = "device-cuda")]
        if matches!(&a_storage, Storage::Cuda(..)) && matches!(&b_storage, Storage::Cuda(..)) {
            match (&a_storage, &b_storage) {
                #[cfg(feature = "device-cuda")]
                (Storage::Cuda(ca, _), Storage::Cuda(cb, _)) => {
                    if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                        match ctx.sub(ca, cb) {
                            Ok(cuda_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_cuda(cuda_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let a_shape_saved = ArrayD::from_shape_vec(
                                    vec![a_shape.len()],
                                    a_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                let b_shape_saved = ArrayD::from_shape_vec(
                                    vec![b_shape.len()],
                                    b_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                return Tensor::from_cuda_with_grad_fn(
                                    cuda_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> =
                                            saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> =
                                            saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, -db]
                                    }),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!(error = %e, "CUDA sub failed, falling back");
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(..)) && matches!(&b_storage, Storage::Gpu(..));
        if on_gpu && !RECOVERY_MANAGER.is_circuit_open() {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga, _), Storage::Gpu(gb, _)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.sub(ga, gb) {
                            Ok(gpu_result) => {
                                RECOVERY_MANAGER.record_recovery();
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let a_shape_saved = ArrayD::from_shape_vec(
                                    vec![a_shape.len()],
                                    a_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                let b_shape_saved = ArrayD::from_shape_vec(
                                    vec![b_shape.len()],
                                    b_shape.iter().map(|&x| x as f32).collect(),
                                )
                                .unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                });
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    vec![ga.clone(), gb.clone()],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> =
                                            saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> =
                                            saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, -db]
                                    }),
                                    Some(Box::new(move |_saved_gpu, grad_gpu, ctx| {
                                        let a_shape = a_shape.clone();
                                        let b_shape = b_shape.clone();
                                        let (da, db) = crate::gpu_backward::sub_backward(
                                            ctx, &a_shape, &b_shape, grad_gpu,
                                        )
                                        .map_err(|e| format!("sub_backward: {e}"))?;
                                        Ok(vec![da, db])
                                    })),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                    "GPU sub failed: {e}"
                                )));
                                tracing::warn!(error = %e, "GPU sub failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                            }
                        }
                    }
                }
                _ => {
                    tracing::warn!(
                        "sub: non-GPU storage despite on_gpu check — falling back to CPU"
                    );
                    GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    RECOVERY_MANAGER.record_failure(&GpuError::Device(
                        "sub: non-GPU storage despite on_gpu check".into(),
                    ));
                }
            }
        }
    }
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc - &b_bc;
    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }
    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();
    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            ArrayD::from_shape_vec(
                vec![a_shape.len()],
                a_shape.iter().map(|&x| x as f32).collect(),
            )
            .unwrap_or_else(|e| {
                debug!("shape encoding failed (infallible): {e}");
                ArrayD::zeros(vec![0])
            }),
            ArrayD::from_shape_vec(
                vec![b_shape.len()],
                b_shape.iter().map(|&x| x as f32).collect(),
            )
            .unwrap_or_else(|e| {
                debug!("shape encoding failed (infallible): {e}");
                ArrayD::zeros(vec![0])
            }),
            a_bc,
            b_bc,
        ],
        Box::new(|grad, saved| {
            let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
            let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
            vec![da, -db]
        }),
    )
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        #[cfg(feature = "device-cuda")]
        if matches!(&a_storage, Storage::Cuda(..)) && matches!(&b_storage, Storage::Cuda(..)) {
            match (&a_storage, &b_storage) {
                #[cfg(feature = "device-cuda")]
                (Storage::Cuda(ca, _), Storage::Cuda(cb, _)) => {
                    if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                        match ctx.mul(ca, cb) {
                            Ok(cuda_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_cuda(cuda_result, id, false);
                                }
                                let a_cpu = a.data();
                                let b_cpu = b.data();
                                return Tensor::from_cuda_with_grad_fn(
                                    cuda_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_cpu, b_cpu],
                                    Box::new(|grad, saved| {
                                        let a_bc = &saved[0];
                                        let b_bc = &saved[1];
                                        let da = grad * b_bc;
                                        let db = grad * a_bc;
                                        vec![da, db]
                                    }),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!(error = %e, "CUDA mul failed, falling back");
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(..)) && matches!(&b_storage, Storage::Gpu(..));
        if on_gpu && !RECOVERY_MANAGER.is_circuit_open() {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga, _), Storage::Gpu(gb, _)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.mul(ga, gb) {
                            Ok(gpu_result) => {
                                RECOVERY_MANAGER.record_recovery();
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }

                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![],
                                    vec![ga.clone(), gb.clone()],
                                    Box::new(|grad, _saved| {
                                        let ga = grad.clone();
                                        let gb = grad.clone();
                                        vec![ga, gb]
                                    }),
                                    Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                        let (da, db) = crate::gpu_backward::mul_backward(
                                            ctx,
                                            &saved_gpu[0],
                                            &saved_gpu[1],
                                            grad_gpu,
                                        )
                                        .map_err(|e| format!("mul_backward: {e}"))?;
                                        Ok(vec![da, db])
                                    })),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                    "GPU mul failed: {e}"
                                )));
                                tracing::warn!(error = %e, "GPU mul failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                            }
                        }
                    }
                }
                _ => {
                    tracing::warn!(
                        "mul: non-GPU storage despite on_gpu check — falling back to CPU"
                    );
                    GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    RECOVERY_MANAGER.record_failure(&GpuError::Device(
                        "mul: non-GPU storage despite on_gpu check".into(),
                    ));
                }
            }
        }
    }
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc * &b_bc;
    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![a_bc, b_bc],
        Box::new(|grad, saved| {
            let a_bc = &saved[0];
            let b_bc = &saved[1];

            let da = grad * b_bc;
            let db = grad * a_bc;
            vec![da, db]
        }),
    )
}

pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        #[cfg(feature = "device-cuda")]
        if matches!(&a_storage, Storage::Cuda(..)) && matches!(&b_storage, Storage::Cuda(..)) {
            match (&a_storage, &b_storage) {
                #[cfg(feature = "device-cuda")]
                (Storage::Cuda(ca, _), Storage::Cuda(cb, _)) => {
                    if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                        match ctx.div(ca, cb) {
                            Ok(cuda_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_cuda(cuda_result, id, false);
                                }
                                let a_cpu = a.data();
                                let b_cpu = b.data();
                                let result_cpu = &a_cpu / &b_cpu;
                                return Tensor::from_cuda_with_grad_fn(
                                    cuda_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_cpu, b_cpu, result_cpu],
                                    Box::new(|grad, saved| {
                                        let _a_bc = &saved[0];
                                        let b_bc = &saved[1];
                                        let result_val = &saved[2];
                                        let da = grad / b_bc;
                                        let db = -grad * result_val / b_bc;
                                        vec![da, db]
                                    }),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!(error = %e, "CUDA div failed, falling back");
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(..)) && matches!(&b_storage, Storage::Gpu(..));
        if on_gpu && !RECOVERY_MANAGER.is_circuit_open() {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga, _), Storage::Gpu(gb, _)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.div(ga, gb) {
                            Ok(gpu_result) => {
                                RECOVERY_MANAGER.record_recovery();
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }

                                let gpu_result_for_saved = gpu_result.clone();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![],
                                    vec![ga.clone(), gb.clone(), gpu_result_for_saved],
                                    Box::new(|grad, _saved| {
                                        let ga = grad.clone();
                                        let gb = -grad;
                                        vec![ga, gb]
                                    }),
                                    Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                        let (da, db) = crate::gpu_backward::div_backward(
                                            ctx,
                                            &saved_gpu[1],
                                            &saved_gpu[2],
                                            grad_gpu,
                                        )
                                        .map_err(|e| format!("div_backward: {e}"))?;
                                        Ok(vec![da, db])
                                    })),
                                );
                            }
                            Err(e) => {
                                GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                                RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                    "GPU div failed: {e}"
                                )));
                                tracing::warn!(error = %e, "GPU div failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                            }
                        }
                    }
                }
                _ => {
                    tracing::warn!(
                        "div: non-GPU storage despite on_gpu check — falling back to CPU"
                    );
                    GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    RECOVERY_MANAGER.record_failure(&GpuError::Device(
                        "div: non-GPU storage despite on_gpu check".into(),
                    ));
                }
            }
        }
    }
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc / &b_bc;
    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }
    let result_for_saved = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![a_bc, b_bc, result_for_saved],
        Box::new(|grad, saved| {
            let _a_bc = &saved[0];
            let b_bc = &saved[1];
            let result_val = &saved[2];
            let da = grad / b_bc;
            let db = -grad * result_val / b_bc;
            vec![da, db]
        }),
    )
}

pub fn exp(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let input_storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &input_storage {
            if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                match ctx.exp(cu_input) {
                    Ok(cuda_result) => {
                        let requires_grad = input.requires_grad();
                        if !requires_grad {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let result_cpu = input.data().mapv(|x| x.exp());
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![result_cpu],
                            Box::new(|grad, saved| {
                                let result_val = &saved[0];
                                let da = grad * result_val;
                                vec![da]
                            }),
                        );
                    }
                    Err(e) => {
                        GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(error = %e, "CUDA exp failed, falling back");
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let input_storage = input.storage();
        if !RECOVERY_MANAGER.is_circuit_open() {
            if let Storage::Gpu(gpu_input, _) = &input_storage {
                if let Ok(ctx) = GpuContext::global() {
                    match ctx.exp(gpu_input) {
                        Ok(gpu_result) => {
                            RECOVERY_MANAGER.record_recovery();
                            let requires_grad = input.requires_grad();
                            if !requires_grad {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let result_cpu = input.data().mapv(|x| x.exp());
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![result_cpu],
                                vec![gpu_input.clone()],
                                Box::new(|grad, saved| {
                                    let result_val = &saved[0];
                                    let da = grad * result_val;
                                    vec![da]
                                }),
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    let da = crate::gpu_backward::exp_backward(
                                        ctx,
                                        &saved_gpu[0],
                                        grad_gpu,
                                    )
                                    .map_err(|e| format!("exp_backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                        Err(e) => {
                            GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                "GPU exp failed: {e}"
                            )));
                            tracing::warn!(error = %e, "GPU exp failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                        }
                    }
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| x.exp());
    let requires_grad = input.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let result_for_saved = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![result_for_saved],
        Box::new(|grad, saved| {
            let result_val = &saved[0];
            let da = grad * result_val;
            vec![da]
        }),
    )
}

pub fn ln(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let input_storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &input_storage {
            if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                match ctx.ln(cu_input) {
                    Ok(cuda_result) => {
                        let requires_grad = input.requires_grad();
                        if !requires_grad {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let input_cpu = input.data();
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![input_cpu],
                            Box::new(|grad, saved| {
                                let input_val = &saved[0];
                                let da = grad / input_val;
                                vec![da]
                            }),
                        );
                    }
                    Err(e) => {
                        GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(error = %e, "CUDA ln failed, falling back");
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let input_storage = input.storage();
        if !RECOVERY_MANAGER.is_circuit_open() {
            if let Storage::Gpu(gpu_input, _) = &input_storage {
                if let Ok(ctx) = GpuContext::global() {
                    match ctx.elementwise_unary(gpu_input, ElemOp::Ln) {
                        Ok(gpu_result) => {
                            RECOVERY_MANAGER.record_recovery();
                            let requires_grad = input.requires_grad();
                            if !requires_grad {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let input_cpu = input.data();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![input_cpu],
                                vec![gpu_input.clone()],
                                Box::new(|grad, saved| {
                                    let input_val = &saved[0];
                                    let da = grad / input_val;
                                    vec![da]
                                }),
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    let da = crate::gpu_backward::ln_backward(
                                        ctx,
                                        &saved_gpu[0],
                                        grad_gpu,
                                    )
                                    .map_err(|e| format!("ln_backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                        Err(e) => {
                            GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                "GPU ln failed: {e}"
                            )));
                            tracing::warn!(error = %e, "GPU ln failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                        }
                    }
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| x.ln());
    let requires_grad = input.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![input.data()],
        Box::new(|grad, saved| {
            let input_val = &saved[0];
            let da = grad / input_val;
            vec![da]
        }),
    )
}

pub fn powf(input: &Tensor, exponent: f32) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let input_storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &input_storage {
            if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                match ctx.powf(cu_input, exponent) {
                    Ok(cuda_result) => {
                        let requires_grad = input.requires_grad();
                        if !requires_grad {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let exponent_saved = ArrayD::from_elem(IxDyn(&[]), exponent);
                        let input_cpu = input.data();
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![exponent_saved, input_cpu],
                            Box::new(|grad, saved| {
                                let exp = saved[0][IxDyn(&[])];
                                let input_val = &saved[1];
                                let da = grad * exp * input_val.mapv(|x| x.powf(exp - 1.0));
                                vec![da]
                            }),
                        );
                    }
                    Err(e) => {
                        GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(error = %e, "CUDA powf failed, falling back");
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let input_storage = input.storage();
        if !RECOVERY_MANAGER.is_circuit_open() {
            if let Storage::Gpu(gpu_input, _) = &input_storage {
                if let Ok(_ctx) = GpuContext::global() {
                    let result_arr = input.data().mapv(|x| x.powf(exponent));
                    match GpuTensor::from_cpu(&result_arr) {
                        Ok(gpu_result) => {
                            RECOVERY_MANAGER.record_recovery();
                            let requires_grad = input.requires_grad();
                            if !requires_grad {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let exponent_saved = ArrayD::from_elem(IxDyn(&[]), exponent);
                            let input_cpu = input.data();
                            let exponent_val = exponent;
                            let gpu_result_for_saved = gpu_result.clone();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![exponent_saved, input_cpu],
                                vec![gpu_input.clone(), gpu_result_for_saved],
                                Box::new(|grad, saved| {
                                    let exp = saved[0][IxDyn(&[])];
                                    let input_val = &saved[1];
                                    let da = grad * exp * input_val.mapv(|x| x.powf(exp - 1.0));
                                    vec![da]
                                }),
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    let exponent = exponent_val;
                                    let da = crate::gpu_backward::powf_backward(
                                        ctx,
                                        &saved_gpu[0],
                                        &saved_gpu[1],
                                        exponent,
                                        grad_gpu,
                                    )
                                    .map_err(|e| format!("powf_backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                        Err(e) => {
                            GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                "GPU powf failed: {e}"
                            )));
                            tracing::warn!(error = %e, "GPU powf failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                        }
                    }
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| x.powf(exponent));
    let requires_grad = input.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![ArrayD::from_elem(IxDyn(&[]), exponent), data],
        Box::new(|grad, saved| {
            let exp = saved[0][IxDyn(&[])];
            let input_val = &saved[1];
            let da = grad * exp * input_val.mapv(|x| x.powf(exp - 1.0));
            vec![da]
        }),
    )
}

pub fn sqrt(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let input_storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &input_storage {
            if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                match ctx.sqrt(cu_input) {
                    Ok(cuda_result) => {
                        let requires_grad = input.requires_grad();
                        if !requires_grad {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let result_cpu = input.data().mapv(|x| x.sqrt());
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![result_cpu.clone()],
                            Box::new(|grad, saved| {
                                let result_val = &saved[0];
                                let da = grad / (2.0 * result_val);
                                vec![da]
                            }),
                        );
                    }
                    Err(e) => {
                        GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(error = %e, "CUDA sqrt failed, falling back");
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let input_storage = input.storage();
        if !RECOVERY_MANAGER.is_circuit_open() {
            if let Storage::Gpu(gpu_input, _) = &input_storage {
                if let Ok(ctx) = GpuContext::global() {
                    match ctx.sqrt(gpu_input) {
                        Ok(gpu_result) => {
                            RECOVERY_MANAGER.record_recovery();
                            let requires_grad = input.requires_grad();
                            if !requires_grad {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let result_cpu = input.data().mapv(|x| x.sqrt());
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![result_cpu.clone()],
                                vec![gpu_input.clone()],
                                Box::new(|grad, saved| {
                                    let result_val = &saved[0];
                                    let da = grad / (2.0 * result_val);
                                    vec![da]
                                }),
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    let da = crate::gpu_backward::sqrt_backward(
                                        ctx,
                                        &saved_gpu[0],
                                        grad_gpu,
                                    )
                                    .map_err(|e| format!("sqrt_backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                        Err(e) => {
                            GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                "GPU sqrt failed: {e}"
                            )));
                            tracing::warn!(error = %e, "GPU sqrt failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                        }
                    }
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| x.sqrt());
    let requires_grad = input.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let result_for_saved = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![result_for_saved],
        Box::new(|grad, saved| {
            let result_val = &saved[0];
            let da = grad / (2.0 * result_val);
            vec![da]
        }),
    )
}

pub fn neg(a: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let a_storage = a.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(ca, _) = &a_storage {
            if let Ok(ctx) = crate::gpu::CudaRuntime::global() {
                match ctx.neg(ca) {
                    Ok(cuda_result) => {
                        let requires_grad = a.requires_grad();
                        if !requires_grad {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![a.clone()],
                            vec![],
                            Box::new(|grad, _saved| vec![-grad]),
                        );
                    }
                    Err(e) => {
                        GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(error = %e, "CUDA neg failed, falling back");
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let a_storage = a.storage();
        if !RECOVERY_MANAGER.is_circuit_open() {
            if let Storage::Gpu(ga, _) = &a_storage {
                if let Ok(ctx) = GpuContext::global() {
                    match ctx.elementwise_unary(ga, ElemOp::Neg) {
                        Ok(gpu_result) => {
                            RECOVERY_MANAGER.record_recovery();
                            let requires_grad = a.requires_grad();
                            if !requires_grad {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![a.clone()],
                                vec![],
                                vec![ga.clone()],
                                Box::new(|grad, _saved| vec![-grad]),
                                Some(Box::new(move |_saved_gpu, grad_gpu, ctx| {
                                    let da = crate::gpu_backward::neg_backward(ctx, grad_gpu)
                                        .map_err(|e| format!("neg_backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                        Err(e) => {
                            GPU_MATH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                "GPU neg failed: {e}"
                            )));
                            tracing::warn!(error = %e, "GPU neg failed, falling back to CPU (fallback #{})", GPU_MATH_FALLBACKS.load(Ordering::Relaxed));
                        }
                    }
                }
            }
        }
    }
    let data = a.data();
    let result = -&data;
    let requires_grad = a.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    Tensor::with_grad_fn(
        result,
        vec![a.clone()],
        vec![],
        Box::new(|grad, _saved| vec![-grad]),
    )
}
