use ndarray::ArrayD;

use super::super::broadcast;
use super::super::tensor::Tensor;
#[cfg(feature = "gpu")]
use crate::{Storage, tensor::next_tensor_id};
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuTensor, ElemOp};

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(_))
            && matches!(&b_storage, Storage::Gpu(_));
        if on_gpu {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga), Storage::Gpu(gb)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.add(ga, gb) {
                            Ok(gpu_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let a_shape_saved = ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                let b_shape_saved = ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    vec![],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, db]
                                    }),
                                    None,
                                );
                            }
                            Err(_) => {}
                        }
                    }
                }
                _ => {}
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
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
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
    #[cfg(feature = "gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(_))
            && matches!(&b_storage, Storage::Gpu(_));
        if on_gpu {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga), Storage::Gpu(gb)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.sub(ga, gb) {
                            Ok(gpu_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let a_shape_saved = ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                let b_shape_saved = ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    vec![],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, -db]
                                    }),
                                    None,
                                );
                            }
                            Err(_) => {}
                        }
                    }
                }
                _ => {}
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
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
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
    #[cfg(feature = "gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(_))
            && matches!(&b_storage, Storage::Gpu(_));
        if on_gpu {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga), Storage::Gpu(gb)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.mul(ga, gb) {
                            Ok(gpu_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let ga_clone = ga.clone();
                                let gb_clone = gb.clone();
                                let a_shape_saved = ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                let b_shape_saved = ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    vec![ga_clone, gb_clone],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, db]
                                    }),
                                    None,
                                );
                            }
                            Err(_) => {}
                        }
                    }
                }
                _ => {}
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
    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();
    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            a_bc.clone(),
            b_bc.clone(),
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
        ],
        Box::new(|grad, saved| {
            let a_val = &saved[0];
            let b_val = &saved[1];
            let a_shape: Vec<usize> = saved[2].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[3].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(&(grad.clone() * b_val), &a_shape);
            let db = broadcast::reduce_grad_for_shape(&(grad.clone() * a_val), &b_shape);
            vec![da, db]
        }),
    )
}

pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(_))
            && matches!(&b_storage, Storage::Gpu(_));
        if on_gpu {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga), Storage::Gpu(gb)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.div(ga, gb) {
                            Ok(gpu_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let a_shape = a.shape();
                                let b_shape = b.shape();
                                let ga_clone = ga.clone();
                                let gb_clone = gb.clone();
                                let a_shape_saved = ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                let b_shape_saved = ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a_shape_saved, b_shape_saved],
                                    vec![ga_clone, gb_clone],
                                    Box::new(|grad, saved| {
                                        let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
                                        let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
                                        let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
                                        let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
                                        vec![da, db]
                                    }),
                                    None,
                                );
                            }
                            Err(_) => {}
                        }
                    }
                }
                _ => {}
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
    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();
    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            a_bc.clone(),
            b_bc.clone(),
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).expect("data length matches shape"),
        ],
        Box::new(|grad, saved| {
            let a_val = &saved[0];
            let b_val = &saved[1];
            let a_shape: Vec<usize> = saved[2].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[3].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(&(grad.clone() / b_val), &a_shape);
            let db = broadcast::reduce_grad_for_shape(&(grad.clone() * (-a_val) / (b_val * b_val)), &b_shape);
            vec![da, db]
        }),
    )
}

pub fn exp(input: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.exp(gpu_input) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let result_cpu = gpu_result.to_cpu();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![result_cpu],
                            vec![],
                            Box::new(|grad, saved| {
                                let e = &saved[0];
                                vec![grad.clone() * e]
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
    let result = data.mapv(|x| x.exp());
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let saved = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let e = &saved[0];
            vec![grad.clone() * e]
        }),
    )
}

pub fn ln(input: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Ln) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let input_cpu = gpu_input.to_cpu();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![input_cpu],
                            vec![gpu_input.clone()],
                            Box::new(|grad, saved| {
                                let x = &saved[0];
                                let dx = x.mapv(|v| 1.0 / v.max(1e-38));
                                vec![grad.clone() * dx]
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
    let result = data.mapv(|x| x.max(1e-38).ln());
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let saved = data.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let x = &saved[0];
            let dx = x.mapv(|v| 1.0 / v.max(1e-38));
            vec![grad.clone() * dx]
        }),
    )
}

pub fn powf(input: &Tensor, exponent: f32) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                let exp_tensor = GpuTensor::from_cpu(&ArrayD::from_elem(vec![1], exponent)).unwrap();
                match ctx.elementwise_binary(gpu_input, &exp_tensor, ElemOp::Powf) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let input_cpu = gpu_input.to_cpu();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![input_cpu],
                            vec![gpu_input.clone()],
                            Box::new(move |grad, saved| {
                                let x = &saved[0];
                                let dx = x.mapv(|v| {
                                    if v == 0.0 && exponent < 1.0 { 0.0 }
                                    else { exponent * v.powf(exponent - 1.0) }
                                });
                                vec![grad.clone() * dx]
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
    let result = data.mapv(|x| x.powf(exponent));
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let saved_x = data.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved_x],
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let dx = x.mapv(|v| {
                if v == 0.0 && exponent < 1.0 {
                    0.0
                } else {
                    exponent * v.powf(exponent - 1.0)
                }
            });
            vec![grad.clone() * dx]
        }),
    )
}

pub fn sqrt(input: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Sqrt) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let result_cpu = gpu_result.to_cpu();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![result_cpu],
                            vec![],
                            Box::new(|grad, saved| {
                                let s = &saved[0];
                                let dx = s.mapv(|v| 0.5 / v.max(1e-38));
                                vec![grad.clone() * dx]
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
    let result = data.mapv(|x| x.max(0.0).sqrt());
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let saved = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let s = &saved[0];
            let dx = s.mapv(|v| 0.5 / v.max(1e-38));
            vec![grad.clone() * dx]
        }),
    )
}

pub fn neg(a: &Tensor) -> Tensor {
    #[cfg(feature = "gpu")]
    {
        let storage = a.storage();
        if let Storage::Gpu(gpu_input) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Neg) {
                    Ok(gpu_result) => {
                        if !a.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![a.clone()],
                            vec![],
                            vec![],
                            Box::new(|grad, _| vec![-grad.clone()]),
                            None,
                        );
                    }
                    Err(_) => {}
                }
            }
        }
    }
    let a_data = a.data();
    let result = -&a_data;
    if !a.requires_grad() {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![a.clone()],
        vec![],
        Box::new(|grad, _| vec![-grad.clone()]),
    )
}
