use ndarray::ArrayD;

use super::super::tensor::Tensor;
#[cfg(feature = "gpu")]
use crate::{Storage, tensor::next_tensor_id};
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuTensor, ReduceOp};

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
                        ).unwrap();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![shape_saved],
                            vec![],
                            Box::new(|grad, saved| {
                                let shape_data: Vec<f32> = saved[0].iter().copied().collect();
                                let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
                                let grad_val = grad.iter().copied().next().unwrap_or(1.0);
                                vec![ArrayD::from_elem(orig_shape, grad_val)]
                            }),
                            None,
                        );
                    }
                    Err(_) => {}
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
    ).expect("shape data fits vector");
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

pub fn mean(input: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.reduce(gpu_input, ReduceOp::Sum) {
                    Ok(gpu_sum) => {
                        let numel = input.numel() as f32;
                        let numel_tensor = GpuTensor::from_cpu(&ArrayD::from_elem(vec![1], numel)).unwrap();
                        if let Ok(gpu_result) = ctx.div(&gpu_sum, &numel_tensor) {
                            if !input.requires_grad() {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let orig_shape = input.shape();
                            let shape_saved = ArrayD::from_shape_vec(
                                vec![orig_shape.len()],
                                orig_shape.iter().map(|&x| x as f32).collect(),
                            ).unwrap();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![shape_saved, ArrayD::from_elem(vec![1], numel)],
                                vec![],
                                Box::new(|grad, saved| {
                                    let shape_data: Vec<f32> = saved[0].iter().copied().collect();
                                    let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
                                    let n = saved[1].iter().copied().next().unwrap_or(1.0);
                                    let grad_val = grad.iter().copied().next().unwrap_or(1.0);
                                    vec![ArrayD::from_elem(orig_shape, grad_val / n)]
                                }),
                                None,
                            );
                        }
                    }
                    Err(_) => {}
                }
            }
        }
    }
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
