use ndarray::ArrayD;

use super::super::tensor::Tensor;

pub fn softmax(input: &Tensor, axis: usize) -> Tensor {
    let data = input.data();
    assert!(axis < data.ndim(), "Softmax: axis out of bounds");

    let soft_shape = data.shape().to_vec();
    let n = data.len();
    let flat: Vec<f32> = data.iter().copied().collect();

    // Compute sum of exp(x-max) along the specified axis
    let axis_stride: usize = soft_shape[axis..].iter().product();
    let batch_size = n / axis_stride;

    let mut result_data = vec![0.0f32; n];
    for b in 0..batch_size {
        let base = b * axis_stride;
        let mut max_in_group = f32::NEG_INFINITY;
        for j in 0..axis_stride {
            max_in_group = max_in_group.max(flat[base + j]);
        }
        let mut sum_exp = 0.0f32;
        for j in 0..axis_stride {
            sum_exp += (flat[base + j] - max_in_group).exp();
        }
        for j in 0..axis_stride {
            result_data[base + j] = (flat[base + j] - max_in_group).exp() / sum_exp;
        }
    }

    let result = ArrayD::from_shape_vec(soft_shape.clone(), result_data).unwrap();

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
            let n = soft.len();
            let s = soft.iter().copied().collect::<Vec<f32>>();
            let g = grad.iter().copied().collect::<Vec<f32>>();

            // sum(soft * grad) along axis
            let (batch, dim) = if soft_shape.len() == 2 { (soft_shape[0], soft_shape[1]) } else { (1, n) };
            let mut sum_sg = vec![0.0f32; batch];
            for b in 0..batch {
                for d in 0..dim {
                    let i = if dim == n { d } else { b * dim + d };
                    sum_sg[b] += s[i] * g[i];
                }
            }

            let mut dx_data = vec![0.0f32; n];
            for b in 0..batch {
                for d in 0..dim {
                    let i = if dim == n { d } else { b * dim + d };
                    dx_data[i] = s[i] * g[i] - s[i] * sum_sg[b];
                }
            }
            let dx = ArrayD::from_shape_vec(soft_shape.clone(), dx_data).unwrap();
            vec![dx]
        }),
    )
}

pub fn log_softmax(input: &Tensor, axis: usize) -> Tensor {
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
        .map(|_| if rand::Rng::gen::<f32>(&mut rng) >= rate { scale } else { 0.0 })
        .collect();
    let mask_arr = ArrayD::from_shape_vec(data.shape().to_vec(), mask).unwrap();
    let result = &data * &mask_arr;

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let saved = mask_arr;

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            vec![grad * &saved[0]]
        }),
    )
}

pub fn layer_norm_2d(input: &Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Tensor {
    let data = input.data();
    let shape = data.shape().to_vec();
    assert_eq!(shape.len(), 2, "LayerNorm2D: input must be [batch, features]");

    let last_dim = shape[1];
    let n = last_dim as f32;

    let mean: Vec<f32> = data.outer_iter().map(|row| row.mean().unwrap_or(0.0)).collect();
    let var: Vec<f32> = data.outer_iter().zip(mean.iter())
        .map(|(row, &m)| row.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / n)
        .collect();
    let std: Vec<f32> = var.iter().map(|v| (v + eps).sqrt()).collect();

    let mut result_data = vec![0.0f32; data.len()];
    for (i, (row, (&m, &s))) in data.outer_iter().zip(mean.iter().zip(std.iter())).enumerate() {
        for (j, &x) in row.iter().enumerate() {
            result_data[i * last_dim + j] = (x - m) / s;
        }
    }
    let mut normalized = ArrayD::from_shape_vec(shape.clone(), result_data).unwrap();

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
    if let Some(w) = weight { inputs.push(w.clone()); }
    if let Some(b) = bias { inputs.push(b.clone()); }

    let mean_arr = ArrayD::from_shape_vec(vec![shape[0]], mean.clone()).unwrap();
    let std_arr = ArrayD::from_shape_vec(vec![shape[0]], std.clone()).unwrap();
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
            for b in 0..batch {
                let m = mean[b];
                let s = std[b];
                let mut sum_dy = 0.0;
                let mut sum_dy_xhat = 0.0;
                for j in 0..dim {
                    let idx = b * dim + j;
                    let gv = grad.iter().copied().collect::<Vec<f32>>()[idx];
                    let xv = x.iter().copied().collect::<Vec<f32>>()[idx];
                    let xhat = (xv - m) / s;
                    sum_dy += gv;
                    sum_dy_xhat += gv * xhat;
                }
                let inv_s = 1.0 / s;
                for j in 0..dim {
                    let idx = b * dim + j;
                    let gv = grad.iter().copied().collect::<Vec<f32>>()[idx];
                    let xv = x.iter().copied().collect::<Vec<f32>>()[idx];
                    let xhat = (xv - m) / s;
                    let dx_val = inv_s * (gv - sum_dy / n - xhat * sum_dy_xhat / n);
                    // Multiply by weight gradient if weight exists
                    let mut inner = dx.clone();
                    inner.as_slice_mut().unwrap()[idx] = dx_val;
                    dx = inner;
                }
            }
            vec![dx]
        }),
    )
}

pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    let data = input.data();
    let tgt = target.data();
    assert_eq!(data.shape(), tgt.shape(), "BCE: shape mismatch");

    let mut loss_data = vec![0.0f32; data.len()];
    for (i, (&x, &t)) in data.iter().zip(tgt.iter()).enumerate() {
        let p = x.clamp(1e-7, 1.0 - 1e-7);
        loss_data[i] = -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
    }
    let loss = ArrayD::from_shape_vec(data.shape().to_vec(), loss_data).unwrap();

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
            vec![ArrayD::from_shape_vec(x.shape().to_vec(), dx_data).unwrap()]
        }),
    )
}

pub fn embedding(input_ids: &Tensor, weight: &Tensor) -> Tensor {
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
    let result = ArrayD::from_shape_vec(vec![seq_len, dim], result_data).unwrap();

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
            vec![ArrayD::zeros(vec![]), d_weight.into_dyn()]
        }),
    )
}
