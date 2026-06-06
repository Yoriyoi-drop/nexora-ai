use ndarray::ArrayD;
use tracing::{debug, warn};

use super::super::tensor::Tensor;
#[cfg(feature = "device-gpu")]
use crate::gpu::{GpuContext, GpuTensor, ReduceOp};
#[cfg(feature = "device-gpu")]
use crate::{tensor::next_tensor_id, Storage};
#[cfg(feature = "device-cuda")]
use crate::gpu::CudaRuntime;

pub fn sum(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let storage = input.storage();
        if let Storage::Cuda(cu_input, _) = &storage {
            if let Ok(ctx) = CudaRuntime::global() {
                match ctx.sum(cu_input) {
                    Ok(cuda_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let orig_shape = input.shape();
                        let shape_saved = ArrayD::from_shape_vec(
                            vec![orig_shape.len()],
                            orig_shape.iter().map(|&x| x as f32).collect(),
                        ).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) });
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![shape_saved],
                            Box::new(|grad, saved| {
                                let shape_data: Vec<f32> = saved[0].iter().copied().collect();
                                let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
                                let grad_val = grad.iter().copied().next().unwrap_or(1.0);
                                vec![ArrayD::from_elem(orig_shape, grad_val)]
                            }),
                        );
                    }
                    Err(e) => warn!("CUDA sum failed: {e}"),
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.reduce(gpu_input, ReduceOp::Sum) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let orig_shape = input.shape();
                        let shape_saved = ArrayD::from_shape_vec(
                            vec![orig_shape.len()],
                            orig_shape.iter().map(|&x| x as f32).collect(),
                        )
                        .unwrap_or_else(|e| {
                            debug!("shape encoding failed (infallible): {e}");
                            ArrayD::zeros(vec![0])
                        });
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![shape_saved],
                            vec![],
                            Box::new(|grad, saved| {
                                let shape_data: Vec<f32> = saved[0].iter().copied().collect();
                                let orig_shape: Vec<usize> =
                                    shape_data.iter().map(|&x| x as usize).collect();
                                let grad_val = grad.iter().copied().next().unwrap_or(1.0);
                                vec![ArrayD::from_elem(orig_shape, grad_val)]
                            }),
                            Some(Box::new(move |_saved_gpu, grad_gpu, ctx| {
                                let da =
                                    crate::gpu_backward::sum_backward(ctx, &orig_shape, grad_gpu)
                                        .map_err(|e| format!("sum_backward: {e}"))?;
                                Ok(vec![da])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd reduce backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let result = ArrayD::from_elem(vec![1], data.sum());
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let orig_shape = data.shape().to_vec();
    let shape_saved = ArrayD::from_shape_vec(
        vec![orig_shape.len()],
        orig_shape.iter().map(|&x| x as f32).collect(),
    )
    .unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![shape_saved],
        Box::new(|grad, saved| {
            let shape_data: Vec<f32> = saved[0].iter().copied().collect();
            let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
            let grad_val = grad.iter().copied().next().unwrap_or(1.0);
            vec![ArrayD::from_elem(orig_shape, grad_val)]
        }),
    )
}

/// CPU fallback for mean (used when GPU path fails)
#[inline]
fn fallback_mean(input: &Tensor) -> Tensor {
    let data = input.data();
    let numel = data.len() as f32;
    let result = ArrayD::from_elem(vec![1], data.sum() / numel);
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let orig_shape = data.shape().to_vec();
    let shape_saved = ArrayD::from_shape_vec(
        vec![orig_shape.len()],
        orig_shape.iter().map(|&x| x as f32).collect(),
    )
    .unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![shape_saved, ArrayD::from_elem(vec![1], numel)],
        Box::new(|grad, saved| {
            let shape_data: Vec<f32> = saved[0].iter().copied().collect();
            let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
            let n = saved[1].iter().copied().next().unwrap_or(1.0);
            let grad_val = grad.iter().copied().next().unwrap_or(1.0);
            vec![ArrayD::from_elem(orig_shape, grad_val / n)]
        }),
    )
}

pub fn mean(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let storage = input.storage();
        if let Storage::Cuda(cu_input, _) = &storage {
            if let Ok(ctx) = CudaRuntime::global() {
                match ctx.mean(cu_input) {
                    Ok(cuda_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let orig_shape = input.shape();
                        let numel = input.numel() as f32;
                        let shape_saved = ArrayD::from_shape_vec(
                            vec![orig_shape.len()],
                            orig_shape.iter().map(|&x| x as f32).collect(),
                        ).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) });
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![shape_saved, ArrayD::from_elem(vec![1], numel)],
                            Box::new(|grad, saved| {
                                let shape_data: Vec<f32> = saved[0].iter().copied().collect();
                                let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
                                let n = saved[1].iter().copied().next().unwrap_or(1.0);
                                let grad_val = grad.iter().copied().next().unwrap_or(1.0);
                                vec![ArrayD::from_elem(orig_shape, grad_val / n)]
                            }),
                        );
                    }
                    Err(e) => warn!("CUDA mean failed: {e}"),
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.reduce(gpu_input, ReduceOp::Sum) {
                    Ok(gpu_sum) => {
                        let numel = input.numel() as f32;
                        let numel_tensor =
                            match GpuTensor::from_cpu(&ArrayD::from_elem(vec![1], numel)) {
                                Ok(t) => t,
                                Err(e) => {
                                    warn!("autograd reduce numel_tensor failed: {e}");
                                    return fallback_mean(input);
                                }
                            };
                        if let Ok(gpu_result) = ctx.div(&gpu_sum, &numel_tensor) {
                            if !input.requires_grad() {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let orig_shape = input.shape();
                            let shape_saved = ArrayD::from_shape_vec(
                                vec![orig_shape.len()],
                                orig_shape.iter().map(|&x| x as f32).collect(),
                            )
                            .unwrap_or_else(|e| {
                                debug!("shape encoding failed (infallible): {e}");
                                ArrayD::zeros(vec![0])
                            });
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![shape_saved, ArrayD::from_elem(vec![1], numel)],
                                vec![],
                                Box::new(|grad, saved| {
                                    let shape_data: Vec<f32> = saved[0].iter().copied().collect();
                                    let orig_shape: Vec<usize> =
                                        shape_data.iter().map(|&x| x as usize).collect();
                                    let n = saved[1].iter().copied().next().unwrap_or(1.0);
                                    let grad_val = grad.iter().copied().next().unwrap_or(1.0);
                                    vec![ArrayD::from_elem(orig_shape, grad_val / n)]
                                }),
                                Some(Box::new(move |_saved_gpu, grad_gpu, ctx| {
                                    let da = crate::gpu_backward::mean_backward(
                                        ctx,
                                        &orig_shape,
                                        numel,
                                        grad_gpu,
                                    )
                                    .map_err(|e| format!("mean_backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                    }
                    Err(e) => warn!("autograd reduce backward failed: {e}"),
                }
            }
        }
    }
    fallback_mean(input)
}
