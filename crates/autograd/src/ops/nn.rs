use ndarray::{s, ArrayD};
use tracing::debug;

use super::super::tensor::Tensor;
use super::math;
#[cfg(feature = "gpu")]
use crate::{tensor::next_tensor_id, Storage};

pub fn softmax(input: &Tensor, axis: usize) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if axis == input.ndim() - 1 && input.ndim() == 2 {
                if let Ok(ctx) = crate::gpu::GpuContext::global() {
                    match ctx.softmax(gpu_input) {
                        Ok(gpu_result) => {
                            if !input.requires_grad() {
                                let id = next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let soft_cpu = gpu_result.to_cpu();
                            let soft_shape = soft_cpu.shape().to_vec();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![soft_cpu],
                                vec![],
                                Box::new(move |grad, saved| {
                                    let soft = &saved[0];
                                    let g = grad.clone();
                                    let n = soft.len();
                                    let batch = soft_shape[0];
                                    let dim = soft_shape[1];
                                    let mut dx = vec![0.0f32; n];
                                    for b in 0..batch {
                                        let base = b * dim;
                                        let mut sum_sg = 0.0;
                                        for j in 0..dim {
                                            sum_sg += soft[base + j] * g[base + j];
                                        }
                                        for j in 0..dim {
                                            let idx = base + j;
                                            dx[idx] = soft[idx] * (g[idx] - sum_sg);
                                        }
                                    }
                                    vec![ArrayD::from_shape_vec(soft_shape.clone(), dx).unwrap()]
                                }),
                                None,
                            );
                        }
                        Err(e) => debug!("autograd nn backward failed: {e}")
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

    let result =
        ArrayD::from_shape_vec(soft_shape.clone(), result_data).expect("data length matches shape");

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
            let dx = ArrayD::from_shape_vec(soft_shape.clone(), dx_data)
                .expect("data length matches shape");
            vec![dx]
        }),
    )
}

pub fn log_softmax(input: &Tensor, axis: usize) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if axis == input.ndim() - 1 && input.ndim() == 2 {
                if let Ok(ctx) = crate::gpu::GpuContext::global() {
                    // GPU: softmax(input) then ln = log_softmax
                    match ctx.softmax(gpu_input) {
                        Ok(soft) => match ctx.elementwise_unary(&soft, crate::gpu::ElemOp::Ln) {
                            Ok(gpu_result) => {
                                let requires_grad = input.requires_grad();
                                if !requires_grad {
                                    let id = crate::tensor::next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let soft_cpu = soft.to_cpu();
                                let shape = soft_cpu.shape().to_vec();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![input.clone()],
                                    vec![soft_cpu],
                                    vec![],
                                    Box::new(move |grad, saved| {
                                        let soft = &saved[0];
                                        let sum_g = grad.sum_axis(ndarray::Axis(axis)).into_dyn();
                                        let dx = grad - soft * &sum_g;
                                        vec![dx]
                                    }),
                                    None,
                                );
                            }
                            Err(e) => debug!("autograd nn backward failed: {e}")
                        },
                        Err(e) => debug!("autograd nn backward failed: {e}")
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

    let saved = data.clone();

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
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

    let data = input.data();
    let scale = 1.0 / (1.0 - rate);
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
    let mask_arr =
        ArrayD::from_shape_vec(data.shape().to_vec(), mask).expect("data length matches shape");
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
    #[cfg(feature = "gpu")]
    {
        let in_storage = input.storage();
        if let Storage::Gpu(gpu_in) = &in_storage {
            let has_gpu_weight = weight.map_or(true, |w| matches!(w.storage(), Storage::Gpu(_)));
            let has_gpu_bias = bias.map_or(true, |b| matches!(b.storage(), Storage::Gpu(_)));
            if has_gpu_weight && has_gpu_bias {
                if let (Some(w), Some(b)) = (weight, bias) {
                    if let (Storage::Gpu(gpu_w), Storage::Gpu(gpu_b)) = (&w.storage(), &b.storage())
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
                                            .unwrap(),
                                            ArrayD::from_shape_vec(vec![input.shape()[0]], std_arr)
                                                .unwrap(),
                                            ArrayD::from_elem(vec![1], n),
                                        ],
                                        vec![],
                                        Box::new(move |grad, saved| {
                                            let x = &saved[0];
                                            let mean = &saved[1];
                                            let std = &saved[2];
                                            let n_val =
                                                saved[3].iter().copied().next().unwrap_or(1.0);
                                            let batch = x.shape()[0];
                                            let dim = x.shape()[1];
                                            let mut dx = grad.clone();
                                            let gs = grad.as_slice().unwrap_or_else(|| {
                                                let v: Vec<f32> = grad.iter().copied().collect();
                                                // leak is acceptable — backward runs once per grad_fn
                                                Box::leak(v.into_boxed_slice())
                                            });
                                            let xs = x.as_slice().unwrap_or_else(|| {
                                                let v: Vec<f32> = x.iter().copied().collect();
                                                Box::leak(v.into_boxed_slice())
                                            });
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
                                                for j in 0..dim {
                                                    let idx = b * dim + j;
                                                    let gv = gs[idx];
                                                    let xv = xs[idx];
                                                    let xhat = (xv - m) / s;
                                                    let dx_val = inv_s
                                                        * (gv
                                                            - sum_dy / n_val
                                                            - xhat * sum_dy_xhat / n_val);
                                                    let mut inner = dx.clone();
                                                    if let Some(slice) = inner.as_slice_mut() {
                                                        slice[idx] = dx_val;
                                                    } else {
                                                        let mut v: Vec<f32> = inner.iter().copied().collect();
                                                        v[idx] = dx_val;
                                                        let shape = inner.shape().to_vec();
                                                        inner = ArrayD::from_shape_vec(shape, v).expect("shape matches");
                                                    }
                                                    dx = inner;
                                                }
                                            }
                                            vec![dx]
                                        }),
                                        None,
                                    );
                                }
                                Err(e) => debug!("autograd nn backward failed: {e}")
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
    let mut normalized =
        ArrayD::from_shape_vec(shape.clone(), result_data).expect("data length matches shape");

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
        || weight.map_or(false, |w| w.requires_grad())
        || bias.map_or(false, |b| b.requires_grad());

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

    let mean_arr =
        ArrayD::from_shape_vec(vec![shape[0]], mean.clone()).expect("data length matches shape");
    let std_arr =
        ArrayD::from_shape_vec(vec![shape[0]], std.clone()).expect("data length matches shape");
    let orig_data = data.clone();

    Tensor::with_grad_fn(
        result,
        inputs,
        vec![orig_data, mean_arr, std_arr, ArrayD::from_elem(vec![1], n)],
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let mean = &saved[1];
            let std = &saved[2];
            let n = saved[3].iter().copied().next().unwrap_or(1.0);
            let batch = x.shape()[0];
            let dim = x.shape()[1];

            // dL/dx_i = (1/N * sigma) * (N * dL/dy_i - sum(dL/dy) - x_hat_i * sum(dL/dy * x_hat))
            let mut dx = grad.clone();
            let g_slice = match grad.as_slice() {
                Some(s) => s,
                None => {
                    tracing::error!("layernorm backward: grad not contiguous");
                    return vec![ArrayD::zeros(x.shape().to_vec())];
                }
            };
            let x_slice = match x.as_slice() {
                Some(s) => s,
                None => {
                    tracing::error!("layernorm backward: x not contiguous");
                    return vec![ArrayD::zeros(x.shape().to_vec())];
                }
            };
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
                        return vec![ArrayD::zeros(x.shape().to_vec())];
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
    #[cfg(feature = "gpu")]
    {
        let in_storage = input.storage();
        let t_storage = target.storage();
        if let (Storage::Gpu(gpu_in), Storage::Gpu(gpu_t)) = (&in_storage, &t_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.elementwise_binary(gpu_in, gpu_t, crate::gpu::ElemOp::BinaryCrossEntropy) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = crate::tensor::next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let data_cpu = gpu_in.to_cpu();
                        let tgt_cpu = gpu_t.to_cpu();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![data_cpu, tgt_cpu],
                            vec![gpu_in.clone(), gpu_t.clone()],
                            Box::new(move |grad, saved| {
                                let x = &saved[0];
                                let t = &saved[1];
                                let mut dx_data = vec![0.0f32; x.len()];
                                for (i, (&g, (&xv, &tv))) in grad.iter().zip(x.iter().zip(t.iter())).enumerate() {
                                    let p = xv.clamp(1e-7, 1.0 - 1e-7);
                                    dx_data[i] = g * (p - tv) / (p * (1.0 - p)).max(1e-12);
                                }
                                vec![ndarray::ArrayD::from_shape_vec(x.shape().to_vec(), dx_data)
                                    .expect("data length matches shape")]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                // d(bce)/dx = g * (p - t) / (p * (1-p))  where p = clamp(x, 1e-7, 1-1e-7)
                                // clamp via relu trick: min(max(x, lo), hi) = hi - relu(hi - x - relu(x - lo))
                                let x = &saved_gpu[0];
                                let t = &saved_gpu[1];
                                let shape = x.shape();
                                let eps = crate::gpu::GpuTensor::from_cpu(&ndarray::ArrayD::from_elem(shape.to_vec(), 1e-7)).unwrap();
                                let one_minus_eps = crate::gpu::GpuTensor::from_cpu(&ndarray::ArrayD::from_elem(shape.to_vec(), 1.0 - 1e-7)).unwrap();
                                // max(x, 1e-7) = relu(x - 1e-7) + 1e-7
                                let x_minus_eps = ctx.sub(x, &eps).unwrap();
                                let r = ctx.relu(&x_minus_eps).unwrap();
                                let p = ctx.add(&r, &eps).unwrap();
                                // min(p, 1-1e-7) = (1-1e-7) - relu((1-1e-7) - p)
                                let diff = ctx.sub(&one_minus_eps, &p).unwrap();
                                let r2 = ctx.relu(&diff).unwrap();
                                let p = ctx.sub(&one_minus_eps, &r2).unwrap();
                                // numerator = grad * (p - t)
                                let p_minus_t = ctx.sub(&p, t).unwrap();
                                let num = ctx.mul(grad_gpu, &p_minus_t).unwrap();
                                // denominator = p * (1-p), clamped to 1e-12
                                let ones = crate::gpu::GpuTensor::from_cpu(&ndarray::ArrayD::from_elem(shape.to_vec(), 1.0)).unwrap();
                                let one_minus_p = ctx.sub(&ones, &p).unwrap();
                                let denom = ctx.mul(&p, &one_minus_p).unwrap();
                                let eps2 = crate::gpu::GpuTensor::from_cpu(&ndarray::ArrayD::from_elem(shape.to_vec(), 1e-12)).unwrap();
                                // max(denom, 1e-12) = relu(denom - 1e-12) + 1e-12
                                let denom_minus_eps2 = ctx.sub(&denom, &eps2).unwrap();
                                let r3 = ctx.relu(&denom_minus_eps2).unwrap();
                                let denom = ctx.add(&r3, &eps2).unwrap();
                                // result = num / denom
                                vec![ctx.div(&num, &denom).unwrap()]
                            })),
                        );
                    }
                    Err(e) => debug!("autograd nn backward failed: {e}")
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
    let loss = ArrayD::from_shape_vec(data.shape().to_vec(), loss_data)
        .expect("data length matches shape");

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
            vec![ArrayD::from_shape_vec(x.shape().to_vec(), dx_data)
                .expect("data length matches shape")]
        }),
    )
}

pub fn cross_entropy_loss(input: &Tensor, target: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let in_storage = input.storage();
        let t_storage = target.storage();
        if let (Storage::Gpu(gpu_in), Storage::Gpu(gpu_t)) = (&in_storage, &t_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.cross_entropy(gpu_in, gpu_t) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let lsm_data = {
                            let data = input.data();
                            let tgt = target.data();
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
                            ArrayD::from_shape_vec(vec![batch, classes], lsm).expect("lsm")
                        };
                        let saved_tgt = target.data();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![lsm_data, saved_tgt],
                            vec![],
                            Box::new(move |grad, saved| {
                                let lsm = &saved[0];
                                let tgt_saved = &saved[1];
                                let batch = lsm.shape()[0];
                                let classes = lsm.shape()[1];
                                let mut dx = vec![0.0f32; batch * classes];
                                for b in 0..batch {
                                    let t = tgt_saved[b] as usize;
                                    let g = grad[b];
                                    for c in 0..classes {
                                        let p = lsm[[b, c]].exp();
                                        dx[b * classes + c] = g * if c == t { p - 1.0 } else { p };
                                    }
                                }
                                vec![ArrayD::from_shape_vec(vec![batch, classes], dx).expect("dx")]
                            }),
                            None,
                        );
                    }
                    Err(e) => debug!("autograd nn backward failed: {e}")
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
        loss_data[b] = -lsm_data[b * classes + t];
    }
    let loss = ArrayD::from_shape_vec(vec![batch], loss_data).expect("data length matches shape");

    if !input.requires_grad() {
        return Tensor::new(loss);
    }

    let saved_lsm =
        ArrayD::from_shape_vec(vec![batch, classes], lsm_data).expect("data length matches shape");
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
                let g = grad[b]; // upstream gradient per sample
                for c in 0..classes {
                    let p = lsm[[b, c]].exp();
                    let d = if c == t { p - 1.0 } else { p };
                    dx_data[b * classes + c] = g * d;
                }
            }
            vec![ArrayD::from_shape_vec(vec![batch, classes], dx_data)
                .expect("data length matches shape")]
        }),
    )
}

pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = math::sub(input, target);
    let sq = math::mul(&diff, &diff);
    crate::ops::reduce::mean(&sq)
}

pub fn embedding(input_ids: &Tensor, weight: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let ids_storage = input_ids.storage();
        let w_storage = weight.storage();
        if let (Storage::Gpu(gpu_ids), Storage::Gpu(gpu_w)) = (&ids_storage, &w_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                match ctx.embedding(gpu_ids, gpu_w) {
                    Ok(gpu_result) => {
                        if !weight.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let ids_saved = input_ids.data();
                        let w_shape = weight.shape().to_vec();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input_ids.clone(), weight.clone()],
                            vec![ids_saved],
                            vec![],
                            Box::new(move |grad, saved| {
                                let ids_arr = &saved[0];
                                let d = grad.shape()[1];
                                let vocab_size = w_shape[0];
                                let mut d_weight = ArrayD::<f32>::zeros(vec![vocab_size, d]);
                                for i in 0..ids_arr.len() {
                                    let idx = ids_arr[i] as usize;
                                    for j in 0..d {
                                        d_weight[[idx, j]] += grad[[i, j]];
                                    }
                                }
                                vec![ArrayD::zeros(grad.shape().to_vec()), d_weight.into_dyn()]
                            }),
                            None,
                        );
                    }
                    Err(e) => debug!("autograd nn backward failed: {e}")
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
    let result =
        ArrayD::from_shape_vec(vec![seq_len, dim], result_data).expect("data length matches shape");

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
                for j in 0..d {
                    d_weight[[idx, j]] += grad[[i, j]];
                }
            }
            vec![ArrayD::zeros(grad.shape().to_vec()), d_weight.into_dyn()]
        }),
    )
}

pub fn rms_norm_2d(input: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let in_storage = input.storage();
        let w_storage = weight.storage();
        if let (Storage::Gpu(gpu_in), Storage::Gpu(gpu_w)) = (&in_storage, &w_storage) {
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
                            vec![], // no gpu_saved
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
                                        dx_data[b * dim + j] =
                                            g * wv * inv_rms + wv * xv * rms_grad_factor * sum_x_g;
                                        dw_data[j] += g * xv * inv_rms;
                                    }
                                }
                                vec![
                                    ArrayD::from_shape_vec(x.shape().to_vec(), dx_data)
                                        .expect("dx"),
                                    ArrayD::from_shape_vec(vec![dim], dw_data).expect("dw"),
                                ]
                            }),
                            None,
                        );
                    }
                    Err(e) => debug!("autograd nn backward failed: {e}")
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
    let result =
        ArrayD::from_shape_vec(shape.clone(), result_data).expect("data length matches shape");

    let requires_grad = input.requires_grad() || weight.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let orig_data = data.clone();
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
                ArrayD::from_shape_vec(x.shape().to_vec(), dx_data)
                    .expect("data length matches shape"),
                ArrayD::from_shape_vec(vec![dim], dw_data).expect("data length matches shape"),
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

    let grad_4d = grad.to_shape(vec![batch, heads, seq, dim]).unwrap();

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
    #[cfg(feature = "gpu")]
    {
        let q_storage = q.storage();
        let k_storage = k.storage();
        let v_storage = v.storage();
        if let (Storage::Gpu(gq), Storage::Gpu(gk), Storage::Gpu(gv)) =
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
                            let q_data = q.data();
                            let k_data = k.data();
                            let v_data = v.data();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![q.clone(), k.clone(), v.clone()],
                                vec![
                                    q_data.clone(),
                                    k_data.clone(),
                                    v_data.clone(),
                                    ArrayD::from_elem(vec![1], scale),
                                ],
                                vec![gq.clone(), gk.clone(), gv.clone()],
                                Box::new(move |grad, saved| {
                                    let qs = &saved[0];
                                    let ks = &saved[1];
                                    let vs = &saved[2];
                                    let s = saved[3].iter().copied().next().unwrap_or(1.0);
                                    attention_backward_cpu(grad, qs, ks, vs, s)
                                }),
                                None,
                            );
                        }
                        Err(e) => debug!("autograd nn backward failed: {e}")
                    }
                }
            }
        }
    }

    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    assert_eq!(
        q_data.ndim(),
        3,
        "causal_attention: q must be [batch, heads, dim]"
    );
    let batch = q_data.shape()[0];
    let heads = q_data.shape()[1];
    let dim = q_data.shape()[2];
    let seq = heads;

    let output_data = v_data.clone();

    let requires_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();

    let result = output_data.clone();

    if !requires_grad {
        return Tensor::new(result);
    }

    let saved = vec![
        q_data.clone(),
        k_data.clone(),
        v_data.clone(),
        ArrayD::from_elem(vec![1], scale),
    ];

    Tensor::with_grad_fn(
        result,
        vec![q.clone(), k.clone(), v.clone()],
        saved,
        Box::new(move |grad, saved| {
            let q_saved = &saved[0];
            let k_saved = &saved[1];
            let v_saved = &saved[2];
            let s = saved[3].iter().copied().next().unwrap_or(1.0);

            let grad_4d = grad.to_shape(vec![batch, heads, seq, dim]).unwrap().to_owned().into_dyn();
            let q_4d = q_saved.to_shape(vec![batch, heads, seq, dim]).unwrap().to_owned().into_dyn();
            let k_4d = k_saved.to_shape(vec![batch, heads, seq, dim]).unwrap().to_owned().into_dyn();
            let v_4d = v_saved.to_shape(vec![batch, heads, seq, dim]).unwrap().to_owned().into_dyn();

            attention_backward_cpu(&grad_4d, &q_4d, &k_4d, &v_4d, s)
        }),
    )
}

pub fn causal_softmax(input: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
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
                        // For grad tracking: use elementwise backward on GPU
                        let saved = gpu_out.clone();
                        let inputs = vec![input.clone()];
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_out,
                            inputs,
                            vec![], // CPU saved — not needed for 1D causal
                            vec![saved],
                            Box::new(move |grad, _saved| {
                                let soft = _saved[0].clone();
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
                                vec![ArrayD::from_shape_vec(soft.shape().to_vec(), dx).unwrap()]
                            }),
                            None,
                        );
                    }
                    Err(e) => debug!("autograd nn backward failed: {e}")
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
    let result =
        ArrayD::from_shape_vec(shape.clone(), result_data).expect("data length matches shape");

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
            vec![ArrayD::from_shape_vec(soft.shape().to_vec(), dx_data)
                .expect("data length matches shape")]
        }),
    )
}
