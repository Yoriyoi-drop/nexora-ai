use ndarray::{s, ArrayD};
use tracing::{debug, warn};

use super::super::tensor::Tensor;
use super::math;
#[cfg(feature = "device-gpu")]
use crate::{tensor::next_tensor_id, Storage};
#[cfg(feature = "device-cuda")]
use crate::gpu::cuda::CudaTensor;
#[cfg(feature = "device-cuda")]
use crate::gpu::CudaRuntime;

pub fn softmax(input: &Tensor, axis: usize) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &storage {
            if axis == input.ndim() - 1 {
                let orig_shape = input.shape().to_vec();
                let batch: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
                let dim = orig_shape[orig_shape.len() - 1];
                if let Ok(cu_2d) = cu_input.reshape(vec![batch, dim]) {
                    if let Ok(ctx) = CudaRuntime::global() {
                        match ctx.softmax(&cu_2d) {
                            Ok(cuda_result_2d) => {
                                let cu_result = match cuda_result_2d.reshape(orig_shape.clone()) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        warn!("softmax CUDA reshape back failed: {e}");
                                        return Tensor::new(input.data());
                                    }
                                };
                                if !input.requires_grad() {
                                    let id = next_tensor_id();
                                    return Tensor::from_cuda(cu_result, id, false);
                                }
                                let result_2d_cpu = cu_2d.clone();
                                return Tensor::from_cuda_with_grad_fn(
                                    cu_result,
                                    vec![input.clone()],
                                    vec![],
                                    Box::new(move |grad, _saved| {
                                        let soft = result_2d_cpu.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                        let g_vec: Vec<f32> = grad.iter().copied().collect();
                                        let batch = orig_shape.iter().take(orig_shape.len() - 1).product::<usize>();
                                        let dim = orig_shape[orig_shape.len() - 1];
                                        let n = grad.len();
                                        let mut dx = vec![0.0f32; n];
                                        for b in 0..batch {
                                            let base = b * dim;
                                            let mut sum_sg = 0.0;
                                            for j in 0..dim {
                                                sum_sg += soft[base + j] * g_vec[base + j];
                                            }
                                            for j in 0..dim {
                                                let idx = base + j;
                                                dx[idx] = soft[idx] * (g_vec[idx] - sum_sg);
                                            }
                                        }
                                        vec![ArrayD::from_shape_vec(orig_shape.clone(), dx)
                                            .unwrap_or_else(|e| {
                                                debug!("shape encoding failed: {e}");
                                                ArrayD::zeros(vec![0])
                                            })]
                                    }),
                                );
                            }
                            Err(e) => warn!("CUDA softmax failed: {e}"),
                        }
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if axis == input.ndim() - 1 {
                let orig_shape = input.shape().to_vec();
                let batch: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
                let dim = orig_shape[orig_shape.len() - 1];
                if let Ok(gpu_2d) = gpu_input.reshape(vec![batch, dim]) {
                    if let Ok(ctx) = crate::gpu::GpuContext::global() {
                        match ctx.softmax(&gpu_2d) {
                            Ok(gpu_result_2d) => {
                                let gpu_2d_for_cpu = gpu_result_2d.clone();
                                let gpu_result = gpu_result_2d.reshape(orig_shape.clone());
                                let gpu_result = match gpu_result {
                                    Ok(r) => r,
                                    Err(e) => {
                                        warn!("softmax reshape back failed: {e}");
                                        return Tensor::new(input.data());
                                    }
                                };
                                if !input.requires_grad() {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let gpu_saved = gpu_result.clone();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![input.clone()],
                                    vec![],
                                    vec![gpu_saved],
                                    Box::new(move |grad, _saved| {
                                        let soft = gpu_2d_for_cpu.to_cpu().unwrap_or_else(|e| {
                                            tracing::warn!(
                                                "GPU CPU-backward readback failed in softmax: {e}"
                                            );
                                            ndarray::ArrayD::zeros(gpu_2d_for_cpu.shape())
                                        });
                                        let g_vec: Vec<f32> = grad.iter().copied().collect();
                                        let n = soft.len();
                                        let batch = orig_shape
                                            .iter()
                                            .take(orig_shape.len() - 1)
                                            .product::<usize>();
                                        let dim = orig_shape[orig_shape.len() - 1];
                                        let mut dx = vec![0.0f32; n];
                                        for b in 0..batch {
                                            let base = b * dim;
                                            let mut sum_sg = 0.0;
                                            for j in 0..dim {
                                                sum_sg += soft[base + j] * g_vec[base + j];
                                            }
                                            for j in 0..dim {
                                                let idx = base + j;
                                                dx[idx] = soft[idx] * (g_vec[idx] - sum_sg);
                                            }
                                        }
                                        vec![ArrayD::from_shape_vec(orig_shape.clone(), dx)
                                            .unwrap_or_else(|e| {
                                                debug!("shape encoding failed (infallible): {e}");
                                                ArrayD::zeros(vec![0])
                                            })]
                                    }),
                                    Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                        let da = crate::gpu_backward::softmax_backward_last_dim(
                                            ctx,
                                            &saved_gpu[0],
                                            grad_gpu,
                                        )
                                        .map_err(|e| format!("softmax_backward: {e}"))?;
                                        Ok(vec![da])
                                    })),
                                );
                            }
                            Err(e) => warn!("autograd nn backward failed: {e}"),
                        }
                    }
                }
            }
        }
    }
    let data = input.data();
    assert!(axis < data.ndim(), "Softmax: axis out of bounds");

    let soft_shape = data.shape().to_vec();
    let n = data.len();
    let flat: Vec<f32> = data.iter().copied().collect();

    // Correct group calculation: softmax along axis=k produces
    // outer * inner groups, each with `dim` elements spaced by `inner` stride.
    let outer: usize = soft_shape.iter().take(axis).product();
    let dim = soft_shape[axis];
    let inner: usize = soft_shape.iter().skip(axis + 1).product();

    let mut result_data = vec![0.0f32; n];
    for o in 0..outer {
        for i in 0..inner {
            let group_base = o * dim * inner + i;
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim {
                max_val = max_val.max(flat[group_base + d * inner]);
            }
            let mut sum_exp = 0.0f32;
            for d in 0..dim {
                sum_exp += (flat[group_base + d * inner] - max_val).exp();
            }
            for d in 0..dim {
                result_data[group_base + d * inner] =
                    (flat[group_base + d * inner] - max_val).exp() / sum_exp;
            }
        }
    }

    let result = ArrayD::from_shape_vec(soft_shape, result_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let soft_shape = input.shape().to_vec();
    let saved_data = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved_data],
        Box::new(move |grad, saved| {
            let soft = &saved[0];
            let s = soft.iter().copied().collect::<Vec<f32>>();
            let g = grad.iter().copied().collect::<Vec<f32>>();

            let outer: usize = soft_shape.iter().take(axis).product();
            let dim = soft_shape[axis];
            let inner: usize = soft_shape.iter().skip(axis + 1).product();

            let mut sum_sg = vec![0.0f32; outer * inner];
            for o in 0..outer {
                for i in 0..inner {
                    let group_base = o * dim * inner + i;
                    let mut total = 0.0f32;
                    for d in 0..dim {
                        total += s[group_base + d * inner] * g[group_base + d * inner];
                    }
                    sum_sg[o * inner + i] = total;
                }
            }

            let mut dx_data = vec![0.0f32; n];
            for o in 0..outer {
                for i in 0..inner {
                    let group_base = o * dim * inner + i;
                    let sg = sum_sg[o * inner + i];
                    for d in 0..dim {
                        let idx = group_base + d * inner;
                        dx_data[idx] = s[idx] * g[idx] - s[idx] * sg;
                    }
                }
            }
            let dx = ArrayD::from_shape_vec(soft_shape.clone(), dx_data).unwrap_or_else(|e| {
                debug!("shape encoding failed (infallible): {e}");
                ArrayD::zeros(vec![0])
            });
            vec![dx]
        }),
    )
}

pub fn log_softmax(input: &Tensor, axis: usize) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &storage {
            if axis == input.ndim() - 1 {
                let orig_shape = input.shape().to_vec();
                let batch: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
                let dim = orig_shape[orig_shape.len() - 1];
                if let Ok(cu_2d) = cu_input.reshape(vec![batch, dim]) {
                    if let Ok(ctx) = CudaRuntime::global() {
                        match ctx.softmax(&cu_2d) {
                            Ok(soft_2d) => {
                                let soft_for_cpu = soft_2d.clone();
                                match ctx.ln(&soft_2d) {
                                    Ok(cuda_result_2d) => {
                                        let cu_result = match cuda_result_2d.reshape(orig_shape.clone()) {
                                            Ok(r) => r,
                                            Err(e) => {
                                                warn!("log_softmax CUDA reshape back failed: {e}");
                                                return Tensor::new(input.data());
                                            }
                                        };
                                        if !input.requires_grad() {
                                            let id = next_tensor_id();
                                            return Tensor::from_cuda(cu_result, id, false);
                                        }
                                        let soft_shape = soft_for_cpu.shape.clone();
                                        let soft_cpu = soft_for_cpu.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                        let soft_arr = ArrayD::from_shape_vec(soft_shape, soft_cpu).unwrap_or_else(|e| {
                                            debug!("log_softmax saved shape mismatch (infallible): {e}");
                                            ArrayD::zeros(vec![0])
                                        });
                                        return Tensor::from_cuda_with_grad_fn(
                                            cu_result,
                                            vec![input.clone()],
                                            vec![soft_arr],
                                            Box::new(move |grad, saved| {
                                                let soft = &saved[0];
                                                let batch = orig_shape.iter().take(orig_shape.len() - 1).product::<usize>();
                                                let dim = orig_shape[orig_shape.len() - 1];
                                                let g_vec: Vec<f32> = grad.iter().copied().collect();
                                                let s_vec: Vec<f32> = soft.iter().copied().collect();
                                                let mut sum_g = vec![0.0f32; batch];
                                                for b in 0..batch {
                                                    let base = b * dim;
                                                    for j in 0..dim {
                                                        sum_g[b] += g_vec[base + j];
                                                    }
                                                }
                                                let mut dx = vec![0.0f32; grad.len()];
                                                for b in 0..batch {
                                                    let base = b * dim;
                                                    for j in 0..dim {
                                                        let idx = base + j;
                                                        dx[idx] = g_vec[idx] - s_vec[idx] * sum_g[b];
                                                    }
                                                }
                                                vec![ArrayD::from_shape_vec(orig_shape.clone(), dx).unwrap_or_else(|e| {
                                                    debug!("shape encoding failed (infallible): {e}");
                                                    ArrayD::zeros(vec![0])
                                                })]
                                            }),
                                        );
                                    }
                                    Err(e) => warn!("CUDA log_softmax ln failed: {e}"),
                                }
                            }
                            Err(e) => warn!("CUDA log_softmax softmax failed: {e}"),
                        }
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if axis == input.ndim() - 1 {
                let orig_shape = input.shape().to_vec();
                let batch: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
                let dim = orig_shape[orig_shape.len() - 1];
                if let Ok(gpu_2d) = gpu_input.reshape(vec![batch, dim]) {
                    if let Ok(ctx) = crate::gpu::GpuContext::global() {
                        // GPU: softmax(input) then ln = log_softmax
                        match ctx.softmax(&gpu_2d) {
                            Ok(soft_2d) => {
                                let gpu_soft_for_cpu = soft_2d.clone();
                                match ctx.elementwise_unary(&soft_2d, crate::gpu::ElemOp::Ln) {
                                    Ok(gpu_result_2d) => {
                                        let gpu_result = gpu_result_2d.reshape(orig_shape.clone());
                                        let gpu_result = match gpu_result {
                                            Ok(r) => r,
                                            Err(e) => {
                                                warn!("log_softmax reshape back failed: {e}");
                                                return Tensor::new(input.data());
                                            }
                                        };
                                        let requires_grad = input.requires_grad();
                                        if !requires_grad {
                                            let id = crate::tensor::next_tensor_id();
                                            return Tensor::from_gpu(gpu_result, id, false);
                                        }
                                        let gpu_soft_saved =
                                            soft_2d.clone().reshape(orig_shape.clone());
                                        let gpu_soft_saved = match gpu_soft_saved {
                                            Ok(r) => r,
                                            Err(e) => {
                                                warn!("log_softmax soft reshape back failed: {e}");
                                                return Tensor::new(input.data());
                                            }
                                        };
                                        return Tensor::from_gpu_with_grad_fn(
                                            gpu_result,
                                            vec![input.clone()],
                                            vec![],
                                            vec![gpu_soft_saved],
                                            Box::new(move |grad, _saved| {
                                                let soft = gpu_soft_for_cpu.to_cpu().unwrap_or_else(|e| { tracing::warn!("GPU CPU-backward readback failed in log_softmax: {e}"); ndarray::ArrayD::zeros(gpu_soft_for_cpu.shape()) });
                                                let g_vec: Vec<f32> =
                                                    grad.iter().copied().collect();
                                                let s_vec: Vec<f32> =
                                                    soft.iter().copied().collect();
                                                let batch = orig_shape
                                                    .iter()
                                                    .take(orig_shape.len() - 1)
                                                    .product::<usize>();
                                                let dim = orig_shape[orig_shape.len() - 1];
                                                let mut sum_g = vec![0.0f32; batch];
                                                for b in 0..batch {
                                                    let base = b * dim;
                                                    for j in 0..dim {
                                                        sum_g[b] += g_vec[base + j];
                                                    }
                                                }
                                                let mut dx = vec![0.0f32; grad.len()];
                                                for b in 0..batch {
                                                    let base = b * dim;
                                                    for j in 0..dim {
                                                        let idx = base + j;
                                                        dx[idx] =
                                                            g_vec[idx] - s_vec[idx] * sum_g[b];
                                                    }
                                                }
                                                vec![ArrayD::from_shape_vec(orig_shape.clone(), dx).unwrap_or_else(|e| {
                                                    debug!("shape encoding failed (infallible): {e}");
                                                    ArrayD::zeros(vec![0])
                                                })]
                                            }),
                                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                                let soft = &saved_gpu[0];
                                                let shape = soft.shape();
                                                let last_dim = shape[shape.len() - 1];
                                                let rest: usize =
                                                    shape[..shape.len() - 1].iter().product();
                                                let s_2d = soft.reshape(vec![rest, last_dim])
                                                    .map_err(|e| format!("log_softmax_backward reshape soft: {e}"))?;
                                                let g_2d = grad_gpu
                                                    .reshape(vec![rest, last_dim])
                                                    .map_err(|e| {
                                                    format!(
                                                        "log_softmax_backward reshape grad: {e}"
                                                    )
                                                })?;
                                                let ones_col = crate::gpu::GpuTensor::ones(
                                                    &[last_dim, 1],
                                                )
                                                .map_err(|e| {
                                                    format!("log_softmax_backward ones: {e}")
                                                })?;
                                                let sum_g =
                                                    ctx.matmul(&g_2d, &ones_col).map_err(|e| {
                                                        format!("log_softmax_backward matmul: {e}")
                                                    })?;
                                                let soft_times_sum =
                                                    ctx.mul(&s_2d, &sum_g).map_err(|e| {
                                                        format!("log_softmax_backward mul: {e}")
                                                    })?;
                                                let dx_2d = ctx
                                                    .sub(&g_2d, &soft_times_sum)
                                                    .map_err(|e| {
                                                        format!("log_softmax_backward sub: {e}")
                                                    })?;
                                                let da =
                                                    dx_2d.reshape(shape.to_vec()).map_err(|e| {
                                                        format!(
                                                            "log_softmax_backward reshape out: {e}"
                                                        )
                                                    })?;
                                                Ok(vec![da])
                                            })),
                                        );
                                    }
                                    Err(e) => warn!("autograd nn backward failed: {e}"),
                                }
                            }
                            Err(e) => warn!("autograd nn backward failed: {e}"),
                        }
                    }
                }
            }
        }
    }
    let sm = softmax(input, axis);
    let data = sm.data();
    let result = data.mapv(|x| x.max(1e-38).ln());

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![data],
        Box::new(move |grad, saved| {
            let soft = &saved[0];
            let sum_g = grad.sum_axis(ndarray::Axis(axis)).into_dyn();
            let dx = grad - soft * &sum_g;
            vec![dx]
        }),
    )
}

pub fn dropout(input: &Tensor, rate: f32, training: bool) -> Tensor {
    assert!((0.0..1.0).contains(&rate), "Dropout rate must be in [0, 1)");
    if !training || rate == 0.0 {
        return Tensor::new(input.data());
    }

    let scale = 1.0 / (1.0 - rate);

    #[cfg(feature = "device-cuda")]
    {
        let storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &storage {
            if let Ok(ctx) = CudaRuntime::global() {
                let shape = cu_input.shape().to_vec();
                if let Ok(mask) = CudaTensor::zeros(&ctx.stream, shape.clone(), ctx.device_id) {
                    let seed = rand::random::<u32>();
                    if ctx.dropout_mask(&mask, rate, scale, seed).is_ok() {
                        if let Ok(cu_result) = ctx.mul(cu_input, &mask) {
                            if !input.requires_grad() {
                                return Tensor::from_cuda(cu_result, next_tensor_id(), false);
                            }
                            let mask_shape = cu_input.shape.clone();
                            let mask_cpu = mask.to_cpu_vec(&ctx.stream).unwrap_or_default();
                            let mask_arr = ArrayD::from_shape_vec(mask_shape, mask_cpu).unwrap_or_else(|e| {
                                debug!("dropout saved shape mismatch (infallible): {e}");
                                ArrayD::zeros(vec![0])
                            });
                            return Tensor::from_cuda_with_grad_fn(
                                cu_result,
                                vec![input.clone()],
                                vec![mask_arr],
                                Box::new(|grad, saved| {
                                    vec![grad * &saved[0]]
                                }),
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                let shape = gpu_input.shape().to_vec();
                let seed = rand::random::<u32>();
                if let Ok(mask) = crate::gpu::GpuTensor::zeros(&shape) {
                    if ctx.generate_dropout_mask(&mask, rate, scale, seed).is_ok() {
                        if let Ok(gpu_result) = ctx.mul(gpu_input, &mask) {
                            if !input.requires_grad() {
                                return Tensor::from_gpu(gpu_result, next_tensor_id(), false);
                            }
                            let mask_for_cpu = mask.clone();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![],
                                vec![mask],
                                Box::new(move |grad, _saved| {
                                    let m = mask_for_cpu.to_cpu().unwrap_or_else(|e| {
                                        tracing::warn!("GPU CPU-backward readback failed in dropout: {e}");
                                        ndarray::ArrayD::zeros(shape.clone())
                                    });
                                    vec![grad * &m]
                                }),
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    let m = &saved_gpu[0];
                                    let da = ctx.mul(grad_gpu, m).map_err(|e| format!("dropout backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                    }
                }
            }
        }
    }

    let data = input.data();
    let mut rng = rand::thread_rng();
    let mask: Vec<f32> = (0..data.len())
        .map(|_| {
            if rand::Rng::gen::<f32>(&mut rng) >= rate {
                scale
            } else {
                0.0
            }
        })
        .collect();
    let mask_arr = ArrayD::from_shape_vec(data.shape(), mask).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });
    let result = &data * &mask_arr;

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let saved = mask_arr;

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| vec![grad * &saved[0]]),
    )
}

pub fn layer_norm_2d(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let in_storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_in, _) = &in_storage {
            if let (Some(w), Some(b)) = (weight, bias) {
                #[cfg(feature = "device-cuda")]
                if let (Storage::Cuda(cu_w, _), Storage::Cuda(cu_b, _)) = (&w.storage(), &b.storage()) {
                    if let Ok(ctx) = CudaRuntime::global() {
                        match ctx.layer_norm(cu_in, cu_w, cu_b, eps) {
                            Ok(cuda_result) => {
                                let requires_grad = input.requires_grad() || w.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_cuda(cuda_result, id, false);
                                }
                                let orig = input.data();
                                let mean_arr = orig.outer_iter().map(|row| row.mean().unwrap_or(0.0)).collect::<Vec<_>>();
                                let std_arr = orig.outer_iter().zip(mean_arr.iter()).map(|(row, &m)| {
                                    let v = row.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / orig.shape()[1] as f32;
                                    (v + eps).sqrt()
                                }).collect::<Vec<_>>();
                                let n = orig.shape()[1] as f32;
                                let _gpu_in_saved = cu_in.clone();
                                let _gpu_w_saved = cu_w.clone();
                                let _gpu_b_saved = cu_b.clone();
                                return Tensor::from_cuda_with_grad_fn(
                                    cuda_result,
                                    vec![input.clone(), w.clone(), b.clone()],
                                    vec![
                                        orig,
                                        ArrayD::from_shape_vec(vec![input.shape()[0]], mean_arr).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) }),
                                        ArrayD::from_shape_vec(vec![input.shape()[0]], std_arr).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) }),
                                        ArrayD::from_elem(vec![1], n),
                                    ],
                                    Box::new(move |grad, saved| {
                                        let x = &saved[0];
                                        let mean = &saved[1];
                                        let std = &saved[2];
                                        let n_val = saved[3].iter().copied().next().unwrap_or(1.0);
                                        let batch = x.shape()[0];
                                        let dim = x.shape()[1];
                                        let gs: Vec<f32> = grad.as_slice().map(|s| s.to_vec()).unwrap_or_else(|| grad.iter().copied().collect());
                                        let xs: Vec<f32> = x.as_slice().map(|s| s.to_vec()).unwrap_or_else(|| x.iter().copied().collect());
                                        let mut dx = grad.clone();
                                        for b in 0..batch {
                                            let m = mean[b];
                                            let s = std[b];
                                            let mut sum_dy = 0.0;
                                            let mut sum_dy_xhat = 0.0;
                                            for j in 0..dim {
                                                let idx = b * dim + j;
                                                let xhat = (xs[idx] - m) / s;
                                                sum_dy += gs[idx];
                                                sum_dy_xhat += gs[idx] * xhat;
                                            }
                                            let inv_s = 1.0 / s;
                                            if let Some(dx_s) = dx.as_slice_mut() {
                                                for j in 0..dim {
                                                    let idx = b * dim + j;
                                                    let xhat = (xs[idx] - m) / s;
                                                    dx_s[idx] = inv_s * (gs[idx] - sum_dy / n_val - xhat * sum_dy_xhat / n_val);
                                                }
                                            }
                                        }
                                        vec![dx]
                                    }),
                                );
                            }
                            Err(e) => warn!("CUDA layer_norm failed: {e}"),
                        }
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let in_storage = input.storage();
        if let Storage::Gpu(gpu_in, _) = &in_storage {
            let has_gpu_weight = weight.is_none_or(|w| matches!(w.storage(), Storage::Gpu(..)));
            let has_gpu_bias = bias.is_none_or(|b| matches!(b.storage(), Storage::Gpu(..)));
            if has_gpu_weight && has_gpu_bias {
                if let (Some(w), Some(b)) = (weight, bias) {
                    if let (Storage::Gpu(gpu_w, _), Storage::Gpu(gpu_b, _)) = (&w.storage(), &b.storage())
                    {
                        if let Ok(ctx) = crate::gpu::GpuContext::global() {
                            match ctx.layer_norm(gpu_in, gpu_w, gpu_b, eps) {
                                Ok(gpu_result) => {
                                    let requires_grad = input.requires_grad()
                                        || w.requires_grad()
                                        || b.requires_grad();
                                    if !requires_grad {
                                        let id = next_tensor_id();
                                        return Tensor::from_gpu(gpu_result, id, false);
                                    }
                                    let orig = input.data();
                                    let mean_arr = orig
                                        .outer_iter()
                                        .map(|row| row.mean().unwrap_or(0.0))
                                        .collect::<Vec<_>>();
                                    let std_arr = orig
                                        .outer_iter()
                                        .zip(mean_arr.iter())
                                        .map(|(row, &m)| {
                                            let v =
                                                row.iter().map(|&x| (x - m).powi(2)).sum::<f32>()
                                                    / orig.shape()[1] as f32;
                                            (v + eps).sqrt()
                                        })
                                        .collect::<Vec<_>>();
                                    let n = orig.shape()[1] as f32;
                                    return Tensor::from_gpu_with_grad_fn(
                                        gpu_result,
                                        vec![input.clone(), w.clone(), b.clone()],
                                        vec![
                                            orig,
                                            ArrayD::from_shape_vec(
                                                vec![input.shape()[0]],
                                                mean_arr,
                                            )
                                            .unwrap_or_else(|e| {
                                                debug!("shape encoding failed (infallible): {e}");
                                                ArrayD::zeros(vec![0])
                                            }),
                                            ArrayD::from_shape_vec(vec![input.shape()[0]], std_arr)
                                                .unwrap_or_else(|e| {
                                                    debug!(
                                                        "shape encoding failed (infallible): {e}"
                                                    );
                                                    ArrayD::zeros(vec![0])
                                                }),
                                            ArrayD::from_elem(vec![1], n),
                                        ],
                                        vec![gpu_in.clone(), gpu_w.clone(), gpu_b.clone()],
                                        Box::new(move |grad, saved| {
                                            let x = &saved[0];
                                            let mean = &saved[1];
                                            let std = &saved[2];
                                            let n_val =
                                                saved[3].iter().copied().next().unwrap_or(1.0);
                                            let batch = x.shape()[0];
                                            let dim = x.shape()[1];
                                            let mut dx = grad.clone();
                                            let gs: Vec<f32> = grad
                                                .as_slice()
                                                .map(|s| s.to_vec())
                                                .unwrap_or_else(|| grad.iter().copied().collect());
                                            let xs: Vec<f32> = x
                                                .as_slice()
                                                .map(|s| s.to_vec())
                                                .unwrap_or_else(|| x.iter().copied().collect());
                                            for b in 0..batch {
                                                let m = mean[b];
                                                let s = std[b];
                                                let mut sum_dy = 0.0;
                                                let mut sum_dy_xhat = 0.0;
                                                for j in 0..dim {
                                                    let idx = b * dim + j;
                                                    let gv = gs[idx];
                                                    let xv = xs[idx];
                                                    let xhat = (xv - m) / s;
                                                    sum_dy += gv;
                                                    sum_dy_xhat += gv * xhat;
                                                }
                                                let inv_s = 1.0 / s;
                                                if let Some(dx_slice) = dx.as_slice_mut() {
                                                    for j in 0..dim {
                                                        let idx = b * dim + j;
                                                        let gv = gs[idx];
                                                        let xv = xs[idx];
                                                        let xhat = (xv - m) / s;
                                                        dx_slice[idx] = inv_s
                                                            * (gv
                                                                - sum_dy / n_val
                                                                - xhat * sum_dy_xhat / n_val);
                                                    }
                                                } else {
                                                    let mut v: Vec<f32> =
                                                        dx.iter().copied().collect();
                                                    for j in 0..dim {
                                                        let idx = b * dim + j;
                                                        let gv = gs[idx];
                                                        let xv = xs[idx];
                                                        let xhat = (xv - m) / s;
                                                        v[idx] = inv_s
                                                            * (gv
                                                                - sum_dy / n_val
                                                                - xhat * sum_dy_xhat / n_val);
                                                    }
                                                    let shape = dx.shape();
                                                    dx = ArrayD::from_shape_vec(shape, v).unwrap_or_else(|e| {
                                                        debug!("shape encoding failed (infallible): {e}");
                                                        ArrayD::zeros(vec![0])
                                                    });
                                                }
                                            }
                                            vec![dx]
                                        }),
                                        Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                            use crate::gpu_backward::layer_norm_backward_gpu;
                                            let input_gpu = &saved_gpu[0];
                                            let weight_gpu = &saved_gpu[1];
                                            let bias_gpu = &saved_gpu[2];
                                            let (dx, dw, db) = layer_norm_backward_gpu(
                                                ctx, input_gpu, weight_gpu, bias_gpu, grad_gpu, eps,
                                            )
                                            .map_err(|e| {
                                                format!("layer_norm gpu backward failed: {e}")
                                            })?;
                                            // db is not needed by the op's output (only grad w.r.t dx matters for loss)
                                            Ok(vec![dx, dw, db])
                                        })),
                                    );
                                }
                                Err(e) => warn!("autograd nn backward failed: {e}"),
                            }
                        }
                    }
                }
            }
        }
        // fall through — CPU will handle None weight/bias or mixed storage
    }

    let data = input.data();
    let shape = data.shape().to_vec();
    assert_eq!(
        shape.len(),
        2,
        "LayerNorm2D: input must be [batch, features]"
    );

    let last_dim = shape[1];
    let n = last_dim as f32;

    let mean: Vec<f32> = data
        .outer_iter()
        .map(|row| row.mean().unwrap_or(0.0))
        .collect();
    let var: Vec<f32> = data
        .outer_iter()
        .zip(mean.iter())
        .map(|(row, &m)| row.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / n)
        .collect();
    let std: Vec<f32> = var.iter().map(|v| (v + eps).sqrt()).collect();

    let mut result_data = vec![0.0f32; data.len()];
    for (i, (row, (&m, &s))) in data
        .outer_iter()
        .zip(mean.iter().zip(std.iter()))
        .enumerate()
    {
        for (j, &x) in row.iter().enumerate() {
            result_data[i * last_dim + j] = (x - m) / s;
        }
    }
    let mut normalized = ArrayD::from_shape_vec(shape.clone(), result_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    if let Some(w) = weight {
        let w_data = w.data();
        normalized = &normalized * &w_data;
    }
    let result = if let Some(b) = bias {
        &normalized + &b.data()
    } else {
        normalized.clone()
    };

    let requires_grad = input.requires_grad()
        || weight.is_some_and(|w| w.requires_grad())
        || bias.is_some_and(|b| b.requires_grad());

    if !requires_grad {
        return Tensor::new(result);
    }

    let mut inputs = vec![input.clone()];
    if let Some(w) = weight {
        inputs.push(w.clone());
    }
    if let Some(b) = bias {
        inputs.push(b.clone());
    }

    let mean_arr = ArrayD::from_shape_vec(vec![shape[0]], mean).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });
    let std_arr = ArrayD::from_shape_vec(vec![shape[0]], std).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    Tensor::with_grad_fn(
        result,
        inputs,
        vec![data, mean_arr, std_arr, ArrayD::from_elem(vec![1], n)],
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let mean = &saved[1];
            let std = &saved[2];
            let n = saved[3].iter().copied().next().unwrap_or(1.0);
            let batch = x.shape()[0];
            let dim = x.shape()[1];

            // dL/dx_i = (1/N * sigma) * (N * dL/dy_i - sum(dL/dy) - x_hat_i * sum(dL/dy * x_hat))
            let g_slice = match grad.as_slice() {
                Some(s) => s,
                None => {
                    tracing::error!("layernorm backward: grad not contiguous");
                    return vec![ArrayD::zeros(x.shape())];
                }
            };
            let x_slice = match x.as_slice() {
                Some(s) => s,
                None => {
                    tracing::error!("layernorm backward: x not contiguous");
                    return vec![ArrayD::zeros(x.shape())];
                }
            };
            let mut dx = ArrayD::zeros(x.shape());
            for b in 0..batch {
                let m = mean[b];
                let s = std[b];
                let mut sum_dy = 0.0;
                let mut sum_dy_xhat = 0.0;
                for j in 0..dim {
                    let idx = b * dim + j;
                    let gv = g_slice[idx];
                    let xv = x_slice[idx];
                    let xhat = (xv - m) / s;
                    sum_dy += gv;
                    sum_dy_xhat += gv * xhat;
                }
                let inv_s = 1.0 / s;
                let dx_slice = match dx.as_slice_mut() {
                    Some(s) => s,
                    None => {
                        tracing::error!("layernorm backward: dx not contiguous");
                        return vec![ArrayD::zeros(x.shape())];
                    }
                };
                for j in 0..dim {
                    let idx = b * dim + j;
                    let gv = g_slice[idx];
                    let xv = x_slice[idx];
                    let xhat = (xv - m) / s;
                    dx_slice[idx] = inv_s * (gv - sum_dy / n - xhat * sum_dy_xhat / n);
                }
            }
            vec![dx]
        }),
    )
}

pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let in_storage = input.storage();
        let t_storage = target.storage();
        #[cfg(feature = "device-cuda")]
        if let (Storage::Cuda(cu_in, _), Storage::Cuda(cu_t, _)) = (&in_storage, &t_storage) {
            if let Ok(ctx) = CudaRuntime::global() {
                match ctx.binary_cross_entropy(cu_in, cu_t) {
                    Ok(cuda_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let in_shape = cu_in.shape.clone();
                        let t_shape = cu_t.shape.clone();
                        let in_cpu = cu_in.to_cpu_vec(&ctx.stream).unwrap_or_default();
                        let t_cpu = cu_t.to_cpu_vec(&ctx.stream).unwrap_or_default();
                        let in_arr = ArrayD::from_shape_vec(in_shape, in_cpu).unwrap_or_else(|e| {
                            debug!("BCE saved in shape mismatch (infallible): {e}");
                            ArrayD::zeros(vec![0])
                        });
                        let t_arr = ArrayD::from_shape_vec(t_shape, t_cpu).unwrap_or_else(|e| {
                            debug!("BCE saved t shape mismatch (infallible): {e}");
                            ArrayD::zeros(vec![0])
                        });
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![in_arr, t_arr],
                            Box::new(move |grad, saved| {
                                let x = &saved[0];
                                let t = &saved[1];
                                let mut dx_data = vec![0.0f32; x.len()];
                                for (i, (&g, (&xv, &tv))) in grad.iter().zip(x.iter().zip(t.iter())).enumerate() {
                                    let p = xv.clamp(1e-7, 1.0 - 1e-7);
                                    dx_data[i] = g * (p - tv) / (p * (1.0 - p)).max(1e-12);
                                }
                                vec![ArrayD::from_shape_vec(x.shape(), dx_data).unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                })]
                            }),
                        );
                    }
                    Err(e) => warn!("CUDA binary_cross_entropy failed: {e}"),
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let in_storage = input.storage();
        let t_storage = target.storage();
        if let (Storage::Gpu(gpu_in, _), Storage::Gpu(gpu_t, _)) = (&in_storage, &t_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.elementwise_binary(gpu_in, gpu_t, crate::gpu::ElemOp::BinaryCrossEntropy)
                {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = crate::tensor::next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_in_for_cpu = gpu_in.clone();
                        let gpu_t_for_cpu = gpu_t.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_in.clone(), gpu_t.clone()],
                            Box::new(move |grad, _saved| {
                                let x = gpu_in_for_cpu.to_cpu().unwrap_or_else(|e| { tracing::warn!("GPU CPU-backward readback failed in binary_cross_entropy: {e}"); ndarray::ArrayD::zeros(gpu_in_for_cpu.shape()) });
                                let t = gpu_t_for_cpu.to_cpu().unwrap_or_else(|e| { tracing::warn!("GPU CPU-backward readback failed in binary_cross_entropy: {e}"); ndarray::ArrayD::zeros(gpu_t_for_cpu.shape()) });
                                let mut dx_data = vec![0.0f32; x.len()];
                                for (i, (&g, (&xv, &tv))) in
                                    grad.iter().zip(x.iter().zip(t.iter())).enumerate()
                                {
                                    let p = xv.clamp(1e-7, 1.0 - 1e-7);
                                    dx_data[i] = g * (p - tv) / (p * (1.0 - p)).max(1e-12);
                                }
                                vec![ndarray::ArrayD::from_shape_vec(x.shape(), dx_data)
                                    .unwrap_or_else(|e| {
                                        debug!("shape encoding failed (infallible): {e}");
                                        ArrayD::zeros(vec![0])
                                    })]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let x = &saved_gpu[0];
                                let t = &saved_gpu[1];
                                let shape = x.shape();
                                let eps = crate::gpu::GpuTensor::from_cpu(
                                    &ndarray::ArrayD::from_elem(shape.to_vec(), 1e-7),
                                )
                                .map_err(|e| format!("BCE backward: eps from_cpu failed: {e}"))?;
                                let one_minus_eps = crate::gpu::GpuTensor::from_cpu(
                                    &ndarray::ArrayD::from_elem(shape.to_vec(), 1.0 - 1e-7),
                                )
                                .map_err(|e| format!("BCE backward: 1-eps from_cpu failed: {e}"))?;
                                let x_minus_eps = ctx
                                    .sub(x, &eps)
                                    .map_err(|e| format!("BCE backward: sub failed: {e}"))?;
                                let r = ctx
                                    .relu(&x_minus_eps)
                                    .map_err(|e| format!("BCE backward: relu failed: {e}"))?;
                                let p = ctx
                                    .add(&r, &eps)
                                    .map_err(|e| format!("BCE backward: add failed: {e}"))?;
                                let diff = ctx
                                    .sub(&one_minus_eps, &p)
                                    .map_err(|e| format!("BCE backward: sub2 failed: {e}"))?;
                                let r2 = ctx
                                    .relu(&diff)
                                    .map_err(|e| format!("BCE backward: relu2 failed: {e}"))?;
                                let p = ctx
                                    .sub(&one_minus_eps, &r2)
                                    .map_err(|e| format!("BCE backward: sub3 failed: {e}"))?;
                                let p_minus_t = ctx
                                    .sub(&p, t)
                                    .map_err(|e| format!("BCE backward: sub4 failed: {e}"))?;
                                let num = ctx
                                    .mul(grad_gpu, &p_minus_t)
                                    .map_err(|e| format!("BCE backward: mul failed: {e}"))?;
                                let ones = crate::gpu::GpuTensor::ones(&shape)
                                    .map_err(|e| format!("BCE backward: ones failed: {e}"))?;
                                let one_minus_p = ctx
                                    .sub(&ones, &p)
                                    .map_err(|e| format!("BCE backward: sub5 failed: {e}"))?;
                                let denom = ctx
                                    .mul(&p, &one_minus_p)
                                    .map_err(|e| format!("BCE backward: mul2 failed: {e}"))?;
                                let eps2 = crate::gpu::GpuTensor::from_cpu(
                                    &ndarray::ArrayD::from_elem(shape.to_vec(), 1e-12),
                                )
                                .map_err(|e| format!("BCE backward: eps2 from_cpu failed: {e}"))?;
                                let denom_minus_eps2 = ctx
                                    .sub(&denom, &eps2)
                                    .map_err(|e| format!("BCE backward: sub6 failed: {e}"))?;
                                let r3 = ctx
                                    .relu(&denom_minus_eps2)
                                    .map_err(|e| format!("BCE backward: relu3 failed: {e}"))?;
                                let denom = ctx
                                    .add(&r3, &eps2)
                                    .map_err(|e| format!("BCE backward: add2 failed: {e}"))?;
                                let result = ctx
                                    .div(&num, &denom)
                                    .map_err(|e| format!("BCE backward: div failed: {e}"))?;
                                Ok(vec![result])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd nn backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let tgt = target.data();
    assert_eq!(data.shape(), tgt.shape(), "BCE: shape mismatch");

    let mut loss_data = vec![0.0f32; data.len()];
    for (i, (&x, &t)) in data.iter().zip(tgt.iter()).enumerate() {
        let p = x.clamp(1e-7, 1.0 - 1e-7);
        loss_data[i] = -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
    }
    let loss = ArrayD::from_shape_vec(data.shape(), loss_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    if !input.requires_grad() {
        return Tensor::new(loss);
    }

    let saved_data = data.clone();
    let saved_tgt = tgt.clone();
    Tensor::with_grad_fn(
        loss,
        vec![input.clone()],
        vec![saved_data, saved_tgt],
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let t = &saved[1];
            let mut dx_data = vec![0.0f32; x.len()];
            for (i, (&g, (&xv, &tv))) in grad.iter().zip(x.iter().zip(t.iter())).enumerate() {
                let p = xv.clamp(1e-7, 1.0 - 1e-7);
                dx_data[i] = g * (p - tv) / (p * (1.0 - p)).max(1e-12);
            }
            vec![
                ArrayD::from_shape_vec(x.shape(), dx_data).unwrap_or_else(|e| {
                    debug!("shape encoding failed (infallible): {e}");
                    ArrayD::zeros(vec![0])
                }),
            ]
        }),
    )
}

pub fn cross_entropy_loss(input: &Tensor, target: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let in_storage = input.storage();
        let t_storage = target.storage();
        #[cfg(feature = "device-cuda")]
        if let (Storage::Cuda(cu_in, _), Storage::Cuda(cu_t, _)) = (&in_storage, &t_storage) {
            if let Ok(ctx) = CudaRuntime::global() {
                match ctx.cross_entropy(cu_in, cu_t) {
                    Ok(cuda_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let cu_in_saved = cu_in.clone();
                        let cu_t_saved = cu_t.clone();
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![],
                            Box::new(move |grad, _saved| {
                                let data = cu_in_saved.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                let tgt = cu_t_saved.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                let batch = cu_in_saved.shape()[0];
                                let classes = cu_in_saved.shape()[1];
                                let mut lsm = vec![0.0f32; data.len()];
                                for b in 0..batch {
                                    let mut mx = f32::NEG_INFINITY;
                                    for c in 0..classes { mx = mx.max(data[b * classes + c]); }
                                    let mut sum_exp = 0.0;
                                    for c in 0..classes { sum_exp += (data[b * classes + c] - mx).exp(); }
                                    let log_sum = sum_exp.ln();
                                    for c in 0..classes { lsm[b * classes + c] = (data[b * classes + c] - mx) - log_sum; }
                                }
                                let mut dx = vec![0.0f32; batch * classes];
                                for b in 0..batch {
                                    let t = tgt[b] as usize;
                                    let g = grad[b];
                                    for c in 0..classes {
                                        let p = lsm[b * classes + c].exp();
                                        dx[b * classes + c] = g * if c == t { p - 1.0 } else { p };
                                    }
                                }
                                vec![ArrayD::from_shape_vec(vec![batch, classes], dx).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) })]
                            }),
                        );
                    }
                    Err(e) => warn!("CUDA cross_entropy failed: {e}"),
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let in_storage = input.storage();
        let t_storage = target.storage();
        if let (Storage::Gpu(gpu_in, _), Storage::Gpu(gpu_t, _)) = (&in_storage, &t_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.cross_entropy(gpu_in, gpu_t) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_in_for_cpu = gpu_in.clone();
                        let gpu_t_for_cpu = gpu_t.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_in.clone(), gpu_t.clone()],
                            Box::new(move |grad, _saved| {
                                let data = gpu_in_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!(
                                        "GPU CPU-backward readback failed in cross_entropy: {e}"
                                    );
                                    ndarray::ArrayD::zeros(gpu_in_for_cpu.shape())
                                });
                                let tgt = gpu_t_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!(
                                        "GPU CPU-backward readback failed in cross_entropy: {e}"
                                    );
                                    ndarray::ArrayD::zeros(gpu_t_for_cpu.shape())
                                });
                                let batch = data.shape()[0];
                                let classes = data.shape()[1];
                                let mut lsm = vec![0.0f32; data.len()];
                                for b in 0..batch {
                                    let mut mx = f32::NEG_INFINITY;
                                    for c in 0..classes {
                                        mx = mx.max(data[[b, c]]);
                                    }
                                    let mut sum_exp = 0.0;
                                    for c in 0..classes {
                                        sum_exp += (data[[b, c]] - mx).exp();
                                    }
                                    let log_sum = sum_exp.ln();
                                    for c in 0..classes {
                                        lsm[b * classes + c] = (data[[b, c]] - mx) - log_sum;
                                    }
                                }
                                let mut dx = vec![0.0f32; batch * classes];
                                for b in 0..batch {
                                    let t = tgt[b] as usize;
                                    let g = grad[b];
                                    for c in 0..classes {
                                        let p = lsm[b * classes + c].exp();
                                        dx[b * classes + c] = g * if c == t { p - 1.0 } else { p };
                                    }
                                }
                                vec![ArrayD::from_shape_vec(vec![batch, classes], dx)
                                    .unwrap_or_else(|e| {
                                        debug!("shape encoding failed (infallible): {e}");
                                        ArrayD::zeros(vec![0])
                                    })]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                use crate::gpu_backward::cross_entropy_backward_gpu;
                                let logits = &saved_gpu[0];
                                let targets = &saved_gpu[1];
                                cross_entropy_backward_gpu(ctx, logits, targets, grad_gpu)
                                    .map_err(|e| format!("cross entropy gpu backward failed: {e}"))
                                    .map(|d| vec![d])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd nn backward failed: {e}"),
                }
            }
        }
    }

    let data = input.data();
    let tgt = target.data();
    assert_eq!(
        data.ndim(),
        2,
        "CrossEntropy: input must be [batch, classes]"
    );
    assert_eq!(
        tgt.ndim(),
        1,
        "CrossEntropy: target must be 1D (class indices)"
    );
    assert_eq!(data.shape()[0], tgt.len(), "CrossEntropy: batch mismatch");

    let batch = data.shape()[0];
    let classes = data.shape()[1];

    // Numerically stable log_softmax
    let mut loss_data = vec![0.0f32; batch];
    let mut lsm_data = vec![0.0f32; data.len()];

    for b in 0..batch {
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..classes {
            max_val = max_val.max(data[[b, c]]);
        }
        let mut sum_exp = 0.0f32;
        for c in 0..classes {
            sum_exp += (data[[b, c]] - max_val).exp();
        }
        let log_sum = sum_exp.ln();
        for c in 0..classes {
            lsm_data[b * classes + c] = (data[[b, c]] - max_val) - log_sum;
        }
        let t = tgt[b] as usize;
        if t >= classes {
            tracing::warn!(
                "cross_entropy: target index {} out of range [0, {})",
                t,
                classes
            );
            loss_data[b] = 0.0;
            continue;
        }
        loss_data[b] = -lsm_data[b * classes + t];
    }
    let loss = ArrayD::from_shape_vec(vec![batch], loss_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    if !input.requires_grad() {
        return Tensor::new(loss);
    }

    let saved_lsm = ArrayD::from_shape_vec(vec![batch, classes], lsm_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });
    let saved_tgt = tgt.clone();

    Tensor::with_grad_fn(
        loss,
        vec![input.clone()],
        vec![saved_lsm, saved_tgt],
        Box::new(move |grad, saved| {
            let lsm = &saved[0];
            let tgt_saved = &saved[1];
            let batch = lsm.shape()[0];
            let classes = lsm.shape()[1];
            let mut dx_data = vec![0.0f32; batch * classes];
            for b in 0..batch {
                let t = tgt_saved[b] as usize;
                if t >= classes {
                    tracing::warn!(
                        "cross_entropy backward: target index {} out of range [0, {})",
                        t,
                        classes
                    );
                    continue;
                }
                let g = grad[b]; // upstream gradient per sample
                for c in 0..classes {
                    let p = lsm[[b, c]].exp();
                    let d = if c == t { p - 1.0 } else { p };
                    dx_data[b * classes + c] = g * d;
                }
            }
            vec![
                ArrayD::from_shape_vec(vec![batch, classes], dx_data).unwrap_or_else(|e| {
                    debug!("shape encoding failed (infallible): {e}");
                    ArrayD::zeros(vec![0])
                }),
            ]
        }),
    )
}

pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = math::sub(input, target);
    let sq = math::mul(&diff, &diff);
    crate::ops::reduce::mean(&sq)
}

pub fn embedding(input_ids: &Tensor, weight: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let ids_storage = input_ids.storage();
        let w_storage = weight.storage();
        #[cfg(feature = "device-cuda")]
        if let (Storage::Cuda(cu_ids, _), Storage::Cuda(cu_w, _)) = (&ids_storage, &w_storage) {
            if let Ok(ctx) = CudaRuntime::global() {
                match ctx.embedding(cu_ids, cu_w) {
                    Ok(cuda_result) => {
                        if !weight.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let w_shape = weight.shape().to_vec();
                        let cu_ids_saved = cu_ids.clone();
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input_ids.clone(), weight.clone()],
                            vec![],
                            Box::new(move |grad, _saved| {
                                let ids_arr = cu_ids_saved.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                let d = grad.shape()[1];
                                let vocab_size = w_shape[0];
                                let mut d_weight = ArrayD::<f32>::zeros(vec![vocab_size, d]);
                                for i in 0..ids_arr.len() {
                                    let idx = ids_arr[i] as usize;
                                    if idx >= vocab_size { continue; }
                                    for j in 0..d { d_weight[[idx, j]] += grad[[i, j]]; }
                                }
                                vec![ArrayD::zeros(grad.shape()), d_weight.into_dyn()]
                            }),
                        );
                    }
                    Err(e) => warn!("CUDA embedding failed: {e}"),
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let ids_storage = input_ids.storage();
        let w_storage = weight.storage();
        if let (Storage::Gpu(gpu_ids, _), Storage::Gpu(gpu_w, _)) = (&ids_storage, &w_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.embedding(gpu_ids, gpu_w) {
                    Ok(gpu_result) => {
                        if !weight.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_ids_for_cpu = gpu_ids.clone();
                        let w_shape = weight.shape().to_vec();
                        let w_shape_gpu = w_shape.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input_ids.clone(), weight.clone()],
                            vec![],
                            vec![gpu_ids.clone()],
                            Box::new(move |grad, _saved| {
                                let ids_arr = gpu_ids_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!(
                                        "GPU CPU-backward readback failed in embedding: {e}"
                                    );
                                    ndarray::ArrayD::zeros(gpu_ids_for_cpu.shape())
                                });
                                let d = grad.shape()[1];
                                let vocab_size = w_shape[0];
                                let mut d_weight = ArrayD::<f32>::zeros(vec![vocab_size, d]);
                                for i in 0..ids_arr.len() {
                                    let idx = ids_arr[i] as usize;
                                    if idx >= vocab_size {
                                        tracing::warn!(
                                            "embedding backward: index {} out of range [0, {})",
                                            idx,
                                            vocab_size
                                        );
                                        continue;
                                    }
                                    for j in 0..d {
                                        d_weight[[idx, j]] += grad[[i, j]];
                                    }
                                }
                                vec![ArrayD::zeros(grad.shape()), d_weight.into_dyn()]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                use crate::gpu_backward::embedding_backward_gpu;
                                let ids_gpu = &saved_gpu[0];
                                let (d_input, d_weight) =
                                    embedding_backward_gpu(ctx, ids_gpu, grad_gpu, w_shape_gpu[0])
                                        .map_err(|e| {
                                            format!("embedding backward gpu failed: {e}")
                                        })?;
                                Ok(vec![d_input, d_weight])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd nn backward failed: {e}"),
                }
            }
        }
    }

    let ids = input_ids.data();
    let w = weight.data();
    let w_shape = w.shape().to_vec();
    assert_eq!(w_shape.len(), 2, "Embedding weight must be 2D");
    assert!(ids.ndim() == 1, "Embedding: input_ids must be 1D");

    let vocab = w_shape[0];
    let dim = w_shape[1];
    let seq_len = ids.len();

    let mut result_data = Vec::with_capacity(seq_len * dim);
    for i in 0..seq_len {
        let idx = ids[i] as usize;
        assert!(idx < vocab, "embedding index out of bounds");
        for d in 0..dim {
            result_data.push(w[[idx, d]]);
        }
    }
    let result = ArrayD::from_shape_vec(vec![seq_len, dim], result_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    if !weight.requires_grad() {
        return Tensor::new(result);
    }

    let ids_saved = ids.clone();

    Tensor::with_grad_fn(
        result,
        vec![input_ids.clone(), weight.clone()],
        vec![ids_saved],
        Box::new(move |grad, saved| {
            let ids_arr = &saved[0];
            let d = grad.shape()[1];
            let vocab_size = w_shape[0];
            let mut d_weight = ArrayD::<f32>::zeros(vec![vocab_size, d]);

            for i in 0..ids_arr.len() {
                let idx = ids_arr[i] as usize;
                if idx >= vocab_size {
                    tracing::warn!(
                        "embedding backward: index {} out of range [0, {})",
                        idx,
                        vocab_size
                    );
                    continue;
                }
                for j in 0..d {
                    d_weight[[idx, j]] += grad[[i, j]];
                }
            }
            vec![ArrayD::zeros(grad.shape()), d_weight.into_dyn()]
        }),
    )
}

pub fn rms_norm_2d(input: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let in_storage = input.storage();
        let w_storage = weight.storage();
        #[cfg(feature = "device-cuda")]
        if let (Storage::Cuda(cu_in, _), Storage::Cuda(cu_w, _)) = (&in_storage, &w_storage) {
            if let Ok(ctx) = CudaRuntime::global() {
                match ctx.rms_norm(cu_in, cu_w, eps) {
                    Ok(cuda_result) => {
                        let requires_grad = input.requires_grad() || weight.requires_grad();
                        if !requires_grad {
                            let id = next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let cpu_x = input.data();
                        let cpu_w = weight.data();
                        let _gpu_in_saved = cu_in.clone();
                        let _gpu_w_saved = cu_w.clone();
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone(), weight.clone()],
                            vec![cpu_x.clone(), cpu_w.clone()],
                            Box::new(move |grad, saved| {
                                let x = &saved[0];
                                let w = &saved[1];
                                let hidden = x.shape()[1] as f32;
                                let dim = x.shape()[1];
                                let batch = x.shape()[0];
                                let mut dx_data = vec![0.0f32; x.len()];
                                let mut dw_data = vec![0.0f32; dim];
                                for b in 0..batch {
                                    let mut ssq = 0.0f32;
                                    for j in 0..dim { ssq += x[[b, j]] * x[[b, j]]; }
                                    let rms = (ssq / hidden + eps).sqrt();
                                    let inv_rms = 1.0 / rms;
                                    let mut sum_x_g = 0.0f32;
                                    for k in 0..dim { sum_x_g += x[[b, k]] * grad[[b, k]]; }
                                    let rms_grad_factor = -inv_rms.powi(3) * (1.0 / hidden);
                                    for j in 0..dim {
                                        let xv = x[[b, j]]; let wv = w[[j]]; let g = grad[[b, j]];
                                        dx_data[b * dim + j] = g * wv * inv_rms + wv * xv * rms_grad_factor * sum_x_g;
                                        dw_data[j] += g * xv * inv_rms;
                                    }
                                }
                                vec![
                                    ArrayD::from_shape_vec(x.shape(), dx_data).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) }),
                                    ArrayD::from_shape_vec(vec![dim], dw_data).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) }),
                                ]
                            }),
                        );
                    }
                    Err(e) => warn!("CUDA rms_norm failed: {e}"),
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let in_storage = input.storage();
        let w_storage = weight.storage();
        if let (Storage::Gpu(gpu_in, _), Storage::Gpu(gpu_w, _)) = (&in_storage, &w_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.rms_norm(gpu_in, gpu_w, eps) {
                    Ok(gpu_result) => {
                        let requires_grad = input.requires_grad() || weight.requires_grad();
                        if !requires_grad {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let cpu_x = input.data();
                        let cpu_w = weight.data();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone(), weight.clone()],
                            vec![cpu_x.clone(), cpu_w.clone()],
                            vec![gpu_in.clone(), gpu_w.clone()],
                            Box::new(move |grad, saved| {
                                let x = &saved[0];
                                let w = &saved[1];
                                let hidden = x.shape()[1] as f32;
                                let dim = x.shape()[1];
                                let batch = x.shape()[0];
                                let mut dx_data = vec![0.0f32; x.len()];
                                let mut dw_data = vec![0.0f32; dim];
                                for b in 0..batch {
                                    let mut ssq = 0.0f32;
                                    for j in 0..dim {
                                        ssq += x[[b, j]] * x[[b, j]];
                                    }
                                    let rms = (ssq / hidden + eps).sqrt();
                                    let inv_rms = 1.0 / rms;
                                    let mut sum_x_g = 0.0f32;
                                    for k in 0..dim {
                                        sum_x_g += x[[b, k]] * grad[[b, k]];
                                    }
                                    let rms_grad_factor = -inv_rms.powi(3) * (1.0 / hidden);
                                    for j in 0..dim {
                                        let xv = x[[b, j]];
                                        let wv = w[[j]];
                                        let g = grad[[b, j]];
                                        dx_data[b * dim + j] =
                                            g * wv * inv_rms + wv * xv * rms_grad_factor * sum_x_g;
                                        dw_data[j] += g * xv * inv_rms;
                                    }
                                }
                                vec![
                                    ArrayD::from_shape_vec(x.shape(), dx_data)
                                        .unwrap_or_else(|e| {
                                            debug!("shape encoding failed (infallible): {e}");
                                            ArrayD::zeros(vec![0])
                                        }),
                                    ArrayD::from_shape_vec(vec![dim], dw_data).unwrap_or_else(
                                        |e| {
                                            debug!("shape encoding failed (infallible): {e}");
                                            ArrayD::zeros(vec![0])
                                        },
                                    ),
                                ]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                use crate::gpu_backward::rms_norm_backward_gpu;
                                let input_gpu = &saved_gpu[0];
                                let weight_gpu = &saved_gpu[1];
                                let (dx, dw) = rms_norm_backward_gpu(
                                    ctx, input_gpu, weight_gpu, grad_gpu, eps,
                                )
                                .map_err(|e| format!("rms_norm gpu backward failed: {e}"))?;
                                Ok(vec![dx, dw])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd nn backward failed: {e}"),
                }
            }
        }
    }

    let data = input.data();
    let shape = data.shape().to_vec();
    assert_eq!(shape.len(), 2, "RMSNorm: input must be [batch, features]");

    let hidden = shape[1] as f32;
    let batch = shape[0];

    let mut result_data = vec![0.0f32; data.len()];
    for i in 0..batch {
        let mut ssq = 0.0f32;
        for j in 0..shape[1] {
            ssq += data[[i, j]] * data[[i, j]];
        }
        let rms = (ssq / hidden + eps).sqrt();
        let w = weight.data();
        for j in 0..shape[1] {
            result_data[i * shape[1] + j] = data[[i, j]] / rms * w[[j]];
        }
    }
    let result = ArrayD::from_shape_vec(shape.clone(), result_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    let requires_grad = input.requires_grad() || weight.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let orig_data = data;
    let w_data = weight.data();
    let saved = vec![
        orig_data,
        w_data,
        ArrayD::from_elem(vec![1], hidden),
        ArrayD::from_elem(vec![1], eps),
    ];

    let inputs = vec![input.clone(), weight.clone()];

    Tensor::with_grad_fn(
        result,
        inputs,
        saved,
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let w = &saved[1];
            let hidden = saved[2].iter().copied().next().unwrap_or(1.0);
            let eps = saved[3].iter().copied().next().unwrap_or(1e-6);
            let batch = x.shape()[0];
            let dim = x.shape()[1];

            let mut dx_data = vec![0.0f32; x.len()];
            let mut dw_data = vec![0.0f32; dim];

            for b in 0..batch {
                let mut ssq = 0.0f32;
                for j in 0..dim {
                    ssq += x[[b, j]] * x[[b, j]];
                }
                let rms = (ssq / hidden + eps).sqrt();
                let inv_rms = 1.0 / rms;
                let inv_hidden = 1.0 / hidden;
                let rms_grad_factor = -inv_rms.powi(3) * inv_hidden;

                for j in 0..dim {
                    let xv = x[[b, j]];
                    let wv = w[[j]];
                    let g = grad[[b, j]];
                    let mut sum_x_g = 0.0f32;
                    for k in 0..dim {
                        sum_x_g += x[[b, k]] * grad[[b, k]];
                    }
                    let dx = g * wv * inv_rms + wv * xv * rms_grad_factor * sum_x_g;
                    dx_data[b * dim + j] = dx;
                    dw_data[j] += g * xv * inv_rms;
                }
            }

            vec![
                ArrayD::from_shape_vec(x.shape(), dx_data).unwrap_or_else(|e| {
                    debug!("shape encoding failed (infallible): {e}");
                    ArrayD::zeros(vec![0])
                }),
                ArrayD::from_shape_vec(vec![dim], dw_data).unwrap_or_else(|e| {
                    debug!("shape encoding failed (infallible): {e}");
                    ArrayD::zeros(vec![0])
                }),
            ]
        }),
    )
}

fn attention_backward_cpu(
    grad: &ArrayD<f32>,
    q_arr: &ArrayD<f32>,
    k_arr: &ArrayD<f32>,
    v_arr: &ArrayD<f32>,
    scale: f32,
) -> Vec<ArrayD<f32>> {
    let batch = q_arr.shape()[0];
    let heads = q_arr.shape()[1];
    let seq = q_arr.shape()[2];
    let dim = q_arr.shape()[3];

    let mut dq = ArrayD::zeros(vec![batch, heads, seq, dim]);
    let mut dk = ArrayD::zeros(vec![batch, heads, seq, dim]);
    let mut dv = ArrayD::zeros(vec![batch, heads, seq, dim]);

    let grad_4d = match grad.to_shape(vec![batch, heads, seq, dim]) {
        Ok(v) => v,
        Err(e) => {
            debug!("attention backward grad reshape failed: {e}");
            return vec![ArrayD::zeros(vec![0]); 3];
        }
    };

    for b in 0..batch {
        for h in 0..heads {
            let q_h = q_arr.slice(s![b, h, .., ..]);
            let k_h = k_arr.slice(s![b, h, .., ..]);
            let v_h = v_arr.slice(s![b, h, .., ..]);
            let g_h = grad_4d.slice(s![b, h, .., ..]);

            // 1. Scores = Q @ K^T / scale: [S, S]
            let scores = q_h.dot(&k_h.t()) / scale;

            // 2. Causal softmax: P[i,j] = exp(S[i,j]) / sum_{k<=i} exp(S[i,k])
            let mut p = ndarray::Array2::<f32>::zeros((seq, seq));
            for i in 0..seq {
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..=i {
                    max_val = max_val.max(scores[[i, j]]);
                }
                let mut sum_exp = 0.0;
                for j in 0..=i {
                    let e = (scores[[i, j]] - max_val).exp();
                    p[[i, j]] = e;
                    sum_exp += e;
                }
                if sum_exp > 0.0 {
                    for j in 0..=i {
                        p[[i, j]] /= sum_exp;
                    }
                }
            }

            // 3. dV = P^T @ dO: [S, D]
            let p_t = p.t();
            let dv_h = p_t.dot(&g_h);
            dv.slice_mut(s![b, h, .., ..]).assign(&dv_h);

            // 4. dP = dO @ V^T: [S, S]
            let v_t = v_h.t();
            let dp_h = g_h.dot(&v_t);

            // 5. dS = P * (dP - sum(P*dP, axis=-1))
            let mut ds_h = ndarray::Array2::<f32>::zeros((seq, seq));
            for i in 0..seq {
                let mut sum_p_dp = 0.0;
                for j in 0..=i {
                    sum_p_dp += p[[i, j]] * dp_h[[i, j]];
                }
                for j in 0..=i {
                    ds_h[[i, j]] = p[[i, j]] * (dp_h[[i, j]] - sum_p_dp);
                }
            }

            // 6. dQ = dS @ K / scale: [S, D]
            let dq_h = ds_h.dot(&k_h) / scale;
            dq.slice_mut(s![b, h, .., ..]).assign(&dq_h);

            // 7. dK = dS^T @ Q / scale: [S, D]
            let ds_t = ds_h.t();
            let dk_h = ds_t.dot(&q_h) / scale;
            dk.slice_mut(s![b, h, .., ..]).assign(&dk_h);
        }
    }

    vec![dq.into_dyn(), dk.into_dyn(), dv.into_dyn()]
}

pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let q_storage = q.storage();
        let k_storage = k.storage();
        let v_storage = v.storage();
        #[cfg(feature = "device-cuda")]
        if let (Storage::Cuda(cq, _), Storage::Cuda(ck, _), Storage::Cuda(cv, _)) = (&q_storage, &k_storage, &v_storage) {
            if q.ndim() == 4 && k.ndim() == 4 && v.ndim() == 4 {
                if let Ok(ctx) = CudaRuntime::global() {
                    match ctx.fused_attention(cq, ck, cv, scale, true) {
                        Ok(cuda_result) => {
                            if !q.requires_grad() && !k.requires_grad() && !v.requires_grad() {
                                let id = next_tensor_id();
                                return Tensor::from_cuda(cuda_result, id, false);
                            }
                            let cq_saved = cq.clone();
                            let ck_saved = ck.clone();
                            let cv_saved = cv.clone();
                            return Tensor::from_cuda_with_grad_fn(
                                cuda_result,
                                vec![q.clone(), k.clone(), v.clone()],
                                vec![],
                                Box::new(move |grad, _saved| {
                                    let qs = cq_saved.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                    let ks = ck_saved.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                    let vs = cv_saved.to_cpu_vec(&ctx.stream).unwrap_or_default();
                                    let qs_arr = ArrayD::from_shape_vec(cq_saved.shape(), qs).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) });
                                    let ks_arr = ArrayD::from_shape_vec(ck_saved.shape(), ks).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) });
                                    let vs_arr = ArrayD::from_shape_vec(cv_saved.shape(), vs).unwrap_or_else(|e| { debug!("shape encoding failed: {e}"); ArrayD::zeros(vec![0]) });
                                    attention_backward_cpu(grad, &qs_arr, &ks_arr, &vs_arr, scale)
                                }),
                            );
                        }
                        Err(e) => warn!("CUDA causal_attention failed: {e}"),
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let q_storage = q.storage();
        let k_storage = k.storage();
        let v_storage = v.storage();
        if let (Storage::Gpu(gq, _), Storage::Gpu(gk, _), Storage::Gpu(gv, _)) =
            (&q_storage, &k_storage, &v_storage)
        {
            if q.ndim() == 4 && k.ndim() == 4 && v.ndim() == 4 {
                if let Ok(ctx) = crate::gpu::GpuContext::global() {
                    match ctx.fused_attention(gq, gk, gv, scale, true) {
                        Ok(gpu_result) => {
                            if !q.requires_grad() && !k.requires_grad() && !v.requires_grad() {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let gq_for_cpu = gq.clone();
                            let gk_for_cpu = gk.clone();
                            let gv_for_cpu = gv.clone();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![q.clone(), k.clone(), v.clone()],
                                vec![],
                                vec![gq.clone(), gk.clone(), gv.clone()],
                                Box::new(move |grad, _saved| {
                                    let qs = gq_for_cpu.to_cpu().unwrap_or_else(|e| { tracing::warn!("GPU CPU-backward readback failed in causal_attention q: {e}"); ndarray::ArrayD::zeros(gq_for_cpu.shape()) });
                                    let ks = gk_for_cpu.to_cpu().unwrap_or_else(|e| { tracing::warn!("GPU CPU-backward readback failed in causal_attention k: {e}"); ndarray::ArrayD::zeros(gk_for_cpu.shape()) });
                                    let vs = gv_for_cpu.to_cpu().unwrap_or_else(|e| { tracing::warn!("GPU CPU-backward readback failed in causal_attention v: {e}"); ndarray::ArrayD::zeros(gv_for_cpu.shape()) });
                                    attention_backward_cpu(grad, &qs, &ks, &vs, scale)
                                }),
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    use crate::gpu_backward::attention_backward_gpu;
                                    let gq = &saved_gpu[0];
                                    let gk = &saved_gpu[1];
                                    let gv = &saved_gpu[2];
                                    let (dq_gpu, dk_gpu, dv_gpu) =
                                        attention_backward_gpu(ctx, gq, gk, gv, grad_gpu, scale)
                                            .map_err(|e| {
                                                format!("fused attention backward gpu failed: {e}")
                                            })?;
                                    Ok(vec![dq_gpu, dk_gpu, dv_gpu])
                                })),
                            );
                        }
                        Err(e) => warn!("autograd nn backward failed: {e}"),
                    }
                }
            }
        }
    }

    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    assert!(
        q_data.ndim() == 4,
        "causal_attention: q must be [batch, heads, seq, dim], got {:?}",
        q_data.shape()
    );
    let batch = q_data.shape()[0];
    let heads = q_data.shape()[1];
    let seq = q_data.shape()[2];
    let dim = q_data.shape()[3];

    let requires_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();

    let mut output_data = ArrayD::<f32>::zeros(vec![batch, heads, seq, dim]);
    for b in 0..batch {
        for h in 0..heads {
            let q_slice = q_data.slice(s![b, h, .., ..]);
            let k_slice = k_data.slice(s![b, h, .., ..]);
            let v_slice = v_data.slice(s![b, h, .., ..]);

            let scores = q_slice.dot(&k_slice.t()) / scale;

            let mut p = ndarray::Array2::<f32>::zeros((seq, seq));
            for i in 0..seq {
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..=i {
                    max_val = max_val.max(scores[[i, j]]);
                }
                let mut sum_exp = 0.0;
                for j in 0..=i {
                    let e = (scores[[i, j]] - max_val).exp();
                    p[[i, j]] = e;
                    sum_exp += e;
                }
                if sum_exp > 0.0 {
                    for j in 0..=i {
                        p[[i, j]] /= sum_exp;
                    }
                }
            }

            let out = p.dot(&v_slice);
            output_data.slice_mut(s![b, h, .., ..]).assign(&out);
        }
    }

    if !requires_grad {
        return Tensor::new(output_data);
    }

    let saved = vec![
        q_data.clone(),
        k_data.clone(),
        v_data.clone(),
        ArrayD::from_elem(vec![1], scale),
    ];

    Tensor::with_grad_fn(
        output_data,
        vec![q.clone(), k.clone(), v.clone()],
        saved,
        Box::new(move |grad, saved| {
            let q_saved = &saved[0];
            let k_saved = &saved[1];
            let v_saved = &saved[2];
            let s = saved[3].iter().copied().next().unwrap_or(1.0);

            attention_backward_cpu(grad, q_saved, k_saved, v_saved, s)
        }),
    )
}

pub fn causal_softmax(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let Storage::Cuda(cu_input, _) = &storage {
            if let Ok(ctx) = CudaRuntime::global() {
                match ctx.causal_softmax(cu_input) {
                    Ok(cuda_out) => {
                        let requires_grad = input.requires_grad();
                        if !requires_grad {
                            return Tensor::from_cuda(cuda_out, next_tensor_id(), false);
                        }
                        let soft_shape = cuda_out.shape().clone();
                        let soft_cpu = cuda_out.to_cpu_vec(&ctx.stream).unwrap_or_default();
                        let soft_arr = ArrayD::from_shape_vec(soft_shape, soft_cpu).unwrap_or_else(|e| {
                            debug!("causal_softmax shape mismatch (infallible): {e}");
                            ArrayD::zeros(vec![0])
                        });
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_out,
                            vec![input.clone()],
                            vec![soft_arr],
                            Box::new(move |grad, saved| {
                                let soft = &saved[0];
                                let seq = soft.shape()[0];
                                let mut dx = vec![0.0f32; soft.len()];
                                for i in 0..seq {
                                    let mut sum_sg = 0.0f32;
                                    for j in 0..=i {
                                        sum_sg += soft[[i, j]] * grad[[i, j]];
                                    }
                                    for j in 0..=i {
                                        dx[i * seq + j] = soft[[i, j]] * (grad[[i, j]] - sum_sg);
                                    }
                                }
                                vec![ArrayD::from_shape_vec(soft.shape(), dx).unwrap_or_else(|e| {
                                    debug!("shape encoding failed (infallible): {e}");
                                    ArrayD::zeros(vec![0])
                                })]
                            }),
                        );
                    }
                    Err(e) => warn!("CUDA causal_softmax failed: {e}"),
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.causal_softmax(gpu_input) {
                    Ok(gpu_out) => {
                        let requires_grad = input.requires_grad();
                        if !requires_grad {
                            return Tensor::from_gpu(
                                gpu_out,
                                crate::tensor::next_tensor_id(),
                                false,
                            );
                        }
                        // For grad tracking: readback to CPU for backward
                        let soft_cpu = gpu_out.to_cpu().unwrap_or_else(|e| {
                            warn!("causal_softmax GPU readback failed: {e}");
                            ArrayD::zeros(vec![0])
                        });
                        let inputs = vec![input.clone()];
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_out,
                            inputs,
                            vec![soft_cpu],
                            vec![],
                            Box::new(move |grad, _saved| {
                                let soft = &_saved[0];
                                let seq = soft.shape()[0];
                                let mut dx = vec![0.0f32; soft.len()];
                                for i in 0..seq {
                                    let mut sum_sg = 0.0f32;
                                    for j in 0..=i {
                                        sum_sg += soft[[i, j]] * grad[[i, j]];
                                    }
                                    for j in 0..=i {
                                        dx[i * seq + j] = soft[[i, j]] * (grad[[i, j]] - sum_sg);
                                    }
                                }
                                vec![ArrayD::from_shape_vec(soft.shape(), dx)
                                    .unwrap_or_else(|e| {
                                        debug!("shape encoding failed (infallible): {e}");
                                        ArrayD::zeros(vec![0])
                                    })]
                            }),
                            None,
                        );
                    }
                    Err(e) => warn!("autograd nn backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let shape = data.shape().to_vec();
    assert_eq!(
        shape.len(),
        2,
        "causal_softmax: input must be [seq_len, seq_len]"
    );
    let seq_len = shape[0];

    let mut result_data = vec![0.0f32; data.len()];
    for i in 0..seq_len {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..=i {
            max_val = max_val.max(data[[i, j]]);
        }
        let mut sum_exp = 0.0f32;
        for j in 0..=i {
            sum_exp += (data[[i, j]] - max_val).exp();
        }
        for j in 0..=i {
            result_data[i * seq_len + j] = (data[[i, j]] - max_val).exp() / sum_exp;
        }
        for j in (i + 1)..seq_len {
            result_data[i * seq_len + j] = 0.0;
        }
    }
    let result = ArrayD::from_shape_vec(shape.clone(), result_data).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let saved = result.clone();

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(move |grad, saved| {
            let soft = &saved[0];
            let seq = soft.shape()[0];
            let mut dx_data = vec![0.0f32; soft.len()];
            for i in 0..seq {
                let mut sum_sg = 0.0f32;
                for j in 0..=i {
                    sum_sg += soft[[i, j]] * grad[[i, j]];
                }
                for j in 0..=i {
                    dx_data[i * seq + j] = soft[[i, j]] * (grad[[i, j]] - sum_sg);
                }
            }
            vec![
                ArrayD::from_shape_vec(soft.shape(), dx_data).unwrap_or_else(|e| {
                    debug!("shape encoding failed (infallible): {e}");
                    ArrayD::zeros(vec![0])
                }),
            ]
        }),
    )
}
