use ndarray::ArrayD;
use tracing::warn;

use super::super::tensor::Tensor;
#[cfg(feature = "device-gpu")]
use crate::gpu::GpuContext;
#[cfg(feature = "device-gpu")]
use crate::gpu::gpu_recovery::RECOVERY_MANAGER;
#[cfg(feature = "device-gpu")]
use crate::{tensor::next_tensor_id, Storage};

pub fn reshape(input: &Tensor, new_shape: &[usize]) -> Tensor {
    let new_vec = new_shape.to_vec();
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            match gpu_input.reshape(new_vec.clone()) {
                Ok(gpu_result) => {
                    RECOVERY_MANAGER.record_recovery();
                    let requires_grad = input.requires_grad();
                    if !requires_grad {
                        let id = next_tensor_id();
                        return Tensor::from_gpu(gpu_result, id, false);
                    }
                    let orig_shape = input.shape();
                    let gpu_clone = gpu_result.clone();
                    return Tensor::from_gpu_with_grad_fn(
                        gpu_result,
                        vec![input.clone()],
                        vec![ArrayD::from_shape_vec(
                            vec![orig_shape.len()],
                            orig_shape.iter().map(|&x| x as f32).collect(),
                        )
                        .unwrap_or_else(|_| ArrayD::zeros(vec![0]))],
                        vec![gpu_clone],
                        Box::new(|grad, saved| {
                            let shape_data: Vec<f32> = saved[0].iter().copied().collect();
                            let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
                            let reshaped = grad.clone().into_shape_with_order(orig_shape.clone());
                            match reshaped {
                                Ok(t) => vec![t],
                                Err(e) => {
                                    tracing::error!("Reshape backward failed: {e}");
                                    vec![ArrayD::zeros(orig_shape)]
                                }
                            }
                        }),
                        Some(Box::new(move |_saved_gpu, grad_gpu, _ctx| {
                            grad_gpu
                                .reshape(orig_shape.clone())
                                .map(|t| vec![t])
                                .map_err(|e| format!("GPU reshape backward failed: {e}"))
                        })),
                    );
                }
                Err(e) => {
                    tracing::warn!("GPU reshape failed, falling back to CPU: {e}");
                }
            }
        }
    }
    let data = input.data();
    let new_len: usize = new_shape.iter().product();
    if data.len() != new_len {
        tracing::error!(
            "Reshape: total elements must match. Have {}, need {:?}",
            data.len(),
            new_shape
        );
        return Tensor::new(ArrayD::zeros(new_shape.to_vec()));
    }
    let result = match data.clone().into_shape_with_order(new_shape.to_vec()) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Reshape failed: {e}");
            return Tensor::new(ArrayD::zeros(new_shape.to_vec()));
        }
    };
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let orig_shape = data.shape().to_vec();
    let shape_saved = match ArrayD::from_shape_vec(
        vec![orig_shape.len()],
        orig_shape.iter().map(|&x| x as f32).collect(),
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("shape data fits vector: {e}");
            return Tensor::new(result);
        }
    };
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![shape_saved],
        Box::new(|grad, saved| {
            let shape_data: Vec<f32> = saved[0].iter().copied().collect();
            let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
            let reshaped = grad.clone().into_shape_with_order(orig_shape.clone());
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
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.transpose(gpu_input) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_clone = gpu_result.clone();
                        return Tensor::from_gpu_with_grad_fn(
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
                                ctx.transpose(grad_gpu)
                                    .map(|t| vec![t])
                                    .map_err(|e| format!("GPU transpose backward failed: {e}"))
                            })),
                        );
                    }
                    Err(e) => warn!("autograd shape backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    if data.ndim() != 2 {
        tracing::error!("Transpose: input must be 2D, got {}D", data.ndim());
        return Tensor::new(ArrayD::zeros(vec![0]));
    }
    let result = match data.view().into_dimensionality::<ndarray::Ix2>() {
        Ok(v) => v.t().to_owned().into_dyn(),
        Err(e) => {
            tracing::error!("Transpose: must be 2D: {e}");
            return Tensor::new(ArrayD::zeros(vec![0]));
        }
    };
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![],
        Box::new(|grad, _| {
            let grad_mat = match grad.view().into_dimensionality::<ndarray::Ix2>() {
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
