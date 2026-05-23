use ndarray::ArrayD;
use tracing::debug;

use super::super::tensor::Tensor;
#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;
#[cfg(feature = "gpu")]
use crate::{tensor::next_tensor_id, Storage};

pub fn reshape(input: &Tensor, new_shape: &[usize]) -> Tensor {
    let data = input.data();
    let new_len: usize = new_shape.iter().product();
    if data.len() != new_len {
        panic!(
            "Reshape: total elements must match. Have {}, need {:?}",
            data.len(),
            new_shape
        );
    }
    let result = data
        .clone()
        .into_shape(new_shape.to_vec())
        .unwrap_or_else(|e| panic!("Reshape failed: {e}"));
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let orig_shape = data.shape().to_vec();
    let shape_saved = ArrayD::from_shape_vec(
        vec![orig_shape.len()],
        orig_shape.iter().map(|&x| x as f32).collect(),
    )
    .unwrap_or_else(|e| panic!("shape data fits vector: {e}"));
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![shape_saved],
        Box::new(|grad, saved| {
            let shape_data: Vec<f32> = saved[0].iter().copied().collect();
            let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
            let reshaped = grad
                .clone()
                .into_shape(orig_shape.clone());
            match reshaped {
                Ok(t) => vec![t],
                Err(e) => {
                    tracing::error!("Reshape backward failed: {e}");
                    vec![ArrayD::zeros(orig_shape)]
                }
            }
        }),
    )
}

pub fn transpose(input: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.transpose(gpu_input) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_clone = gpu_result.clone();                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![],
                                vec![gpu_clone],
                                Box::new(|grad, _| {
                                    let grad_mat = match grad
                                        .view()
                                        .into_dimensionality::<ndarray::Ix2>()
                                    {
                                        Ok(m) => m,
                                        Err(e) => {
                                            tracing::error!("transpose backward: grad must be 2D: {e}");
                                            return vec![ArrayD::zeros(vec![0])];
                                        }
                                    };
                                    vec![grad_mat.t().to_owned().into_dyn()]
                                }),
                                Some(Box::new(move |_saved_gpu, grad_gpu, ctx| {
                                    match ctx.transpose(grad_gpu) {
                                        Ok(t) => vec![t],
                                        Err(e) => {
                                            tracing::error!("GPU transpose backward failed: {e}");
                                            vec![grad_gpu.clone()]
                                        }
                                    }
                                })),
                            );
                    }
                    Err(e) => debug!("autograd shape backward failed: {e}")
                }
            }
        }
    }
    let data = input.data();
    if data.ndim() != 2 {
        panic!("Transpose: input must be 2D, got {}D", data.ndim());
    }
    let result = data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap_or_else(|e| panic!("Transpose: must be 2D: {e}"))
        .t()
        .to_owned()
        .into_dyn();
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![],
        Box::new(|grad, _| {
            let grad_mat = match grad
                .view()
                .into_dimensionality::<ndarray::Ix2>()
            {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("transpose backward: grad must be 2D: {e}");
                    return vec![ArrayD::zeros(vec![0])];
                }
            };
            vec![grad_mat.t().to_owned().into_dyn()]
        }),
    )
}
