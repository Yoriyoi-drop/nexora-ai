use ndarray::ArrayD;
use tracing::debug;

use super::super::tensor::Tensor;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuTensor, ReduceOp};
#[cfg(feature = "gpu")]
use crate::{tensor::next_tensor_id, Storage};

pub fn sum(input: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
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
                        .expect("shape data fits in array");
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
                            None,
                        );
                    }
                    Err(e) => debug!("autograd reduce backward failed: {e}")
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
    .expect("shape data fits vector");
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
    ).expect("shape data fits vector");
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
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.reduce(gpu_input, ReduceOp::Sum) {
                    Ok(gpu_sum) => {
                        let numel = input.numel() as f32;
                        let numel_tensor = match
                            GpuTensor::from_cpu(&ArrayD::from_elem(vec![1], numel))
                        {
                            Ok(t) => t,
                            Err(e) => {
                                debug!("autograd reduce numel_tensor failed: {e}");
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
                            .expect("shape data fits in array");
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
                                None,
                            );
                        }
                    }
                    Err(e) => debug!("autograd reduce backward failed: {e}")
                }
            }
        }
    }
    fallback_mean(input)
}
