use tracing::warn;

use super::super::tensor::Tensor;
#[cfg(feature = "device-gpu")]
use crate::gpu::{ElemOp, GpuContext, GpuTensor};
#[cfg(feature = "device-gpu")]
use crate::{tensor::next_tensor_id, Storage};

pub fn relu(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Relu) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_for_cpu = gpu_input.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_input.clone()],
                            Box::new(move |grad, _saved| {
                                let x = gpu_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!("GPU CPU-backward readback failed in relu: {e}");
                                    ndarray::ArrayD::zeros(gpu_for_cpu.shape())
                                });
                                let mask = x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                                vec![grad * mask]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let x = &saved_gpu[0];
                                let relu_out = ctx
                                    .elementwise_unary(x, ElemOp::Relu)
                                    .map_err(|e| format!("GPU backward relu failed: {e}"))?;
                                let eps = GpuTensor::ones(&x.shape())
                                    .map_err(|e| format!("GPU backward relu ones failed: {e}"))?;
                                ctx.fill_constant(&eps, 1e-12)
                                    .map_err(|e| format!("GPU backward relu fill_constant failed: {e}"))?;
                                let denom = ctx
                                    .add(&relu_out, &eps)
                                    .map_err(|e| format!("GPU backward add failed: {e}"))?;
                                let mask = ctx
                                    .div(&relu_out, &denom)
                                    .map_err(|e| format!("GPU backward div failed: {e}"))?;
                                let result = ctx
                                    .mul(grad_gpu, &mask)
                                    .map_err(|e| format!("GPU backward mul failed: {e}"))?;
                                Ok(vec![result])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd activation backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| if x > 0.0 { x } else { 0.0 });
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![data],
        Box::new(|grad, saved| {
            let input_data = &saved[0];
            let mask = input_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            vec![grad * mask]
        }),
    )
}

pub fn gelu(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Gelu) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_for_cpu = gpu_input.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_input.clone()],
                            Box::new(move |grad, _saved| {
                                let x = gpu_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!("GPU CPU-backward readback failed in gelu: {e}");
                                    ndarray::ArrayD::zeros(gpu_for_cpu.shape())
                                });
                                let sqrt_2_over_pi = (2.0 / std::f64::consts::PI) as f32;
                                let gelu_grad = x.mapv(|xv| {
                                    let x3 = xv * xv * xv;
                                    let tanh_arg = sqrt_2_over_pi * (xv + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    let sech2 = 1.0 - t * t;
                                    0.5 * (1.0 + t)
                                        + 0.5
                                            * xv
                                            * sech2
                                            * sqrt_2_over_pi
                                            * (1.0 + 0.134145 * xv * xv)
                                });
                                vec![grad * gelu_grad]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let da = crate::gpu_backward::gelu_backward(
                                    ctx,
                                    &saved_gpu[0],
                                    grad_gpu,
                                )
                                .map_err(|e| format!("gelu_backward: {e}"))?;
                                Ok(vec![da])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd activation backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI) as f32;
    let result = data.mapv(|x| {
        let x3 = x * x * x;
        let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x3);
        0.5 * x * (1.0 + tanh_arg.tanh())
    });
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![data],
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let sqrt_2_over_pi = (2.0 / std::f64::consts::PI) as f32;
            let gelu_grad = x.mapv(|xv| {
                let x3 = xv * xv * xv;
                let tanh_arg = sqrt_2_over_pi * (xv + 0.044715 * x3);
                let t = tanh_arg.tanh();
                let sech2 = 1.0 - t * t;
                0.5 * (1.0 + t) + 0.5 * xv * sech2 * sqrt_2_over_pi * (1.0 + 0.134145 * xv * xv)
            });
            vec![grad * gelu_grad]
        }),
    )
}

pub fn tanh(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Tanh) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_for_cpu = gpu_result.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_for_cpu.clone()],
                            Box::new(move |grad, _saved| {
                                let t = gpu_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!("GPU CPU-backward readback failed in tanh: {e}");
                                    ndarray::ArrayD::zeros(gpu_for_cpu.shape())
                                });
                                let grad_t = t.mapv(|v| 1.0 - v * v);
                                vec![grad * grad_t]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let y = &saved_gpu[0];
                                let y2 = ctx
                                    .mul(y, y)
                                    .map_err(|e| format!("GPU backward tanh mul failed: {e}"))?;
                                let ones = GpuTensor::ones(&y.shape())
                                    .map_err(|e| format!("GPU backward tanh ones failed: {e}"))?;
                                let local = ctx
                                    .sub(&ones, &y2)
                                    .map_err(|e| format!("GPU backward tanh sub failed: {e}"))?;
                                let result = ctx
                                    .mul(grad_gpu, &local)
                                    .map_err(|e| format!("GPU backward tanh mul failed: {e}"))?;
                                Ok(vec![result])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd activation backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| x.tanh());
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let saved = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let t = &saved[0];
            let grad_t = t.mapv(|v| 1.0 - v * v);
            vec![grad * grad_t]
        }),
    )
}

pub fn leaky_relu(input: &Tensor, negative_slope: f32) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                let mut gpu_result = gpu_input.clone();
                match ctx.leaky_relu_inplace(&mut gpu_result, negative_slope) {
                    Ok(()) => {
                        if !input.requires_grad() {
                            let id = crate::tensor::next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_for_cpu = gpu_input.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_input.clone()],
                            Box::new(move |grad, _saved| {
                                let x = gpu_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!(
                                        "GPU CPU-backward readback failed in leaky_relu: {e}"
                                    );
                                    ndarray::ArrayD::zeros(gpu_for_cpu.shape())
                                });
                                let grad_x = x.mapv(|v| if v > 0.0 { 1.0 } else { negative_slope });
                                vec![grad * grad_x]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let da = crate::gpu_backward::leaky_relu_backward(
                                    ctx,
                                    &saved_gpu[0],
                                    grad_gpu,
                                    negative_slope,
                                )
                                .map_err(|e| format!("leaky_relu_backward: {e}"))?;
                                Ok(vec![da])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd activation backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| if x > 0.0 { x } else { x * negative_slope });
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![data],
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let grad_x = x.mapv(|v| if v > 0.0 { 1.0 } else { negative_slope });
            vec![grad * grad_x]
        }),
    )
}

pub fn sigmoid(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Sigmoid) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_for_cpu = gpu_result.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_for_cpu.clone()],
                            Box::new(move |grad, _saved| {
                                let sig = gpu_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!(
                                        "GPU CPU-backward readback failed in sigmoid: {e}"
                                    );
                                    ndarray::ArrayD::zeros(gpu_for_cpu.shape())
                                });
                                let sig_grad = sig.mapv(|s| s * (1.0 - s));
                                vec![grad * sig_grad]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let y = &saved_gpu[0];
                                let ones = GpuTensor::ones(&y.shape())
                                    .map_err(|e| format!("GPU backward sigmoid ones failed: {e}"))?;
                                let one_minus_y = ctx
                                    .sub(&ones, y)
                                    .map_err(|e| format!("GPU backward sigmoid sub failed: {e}"))?;
                                let dy = ctx
                                    .mul(y, &one_minus_y)
                                    .map_err(|e| format!("GPU backward sigmoid mul failed: {e}"))?;
                                let result = ctx
                                    .mul(grad_gpu, &dy)
                                    .map_err(|e| format!("GPU backward sigmoid mul failed: {e}"))?;
                                Ok(vec![result])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd activation backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            x.exp() / (1.0 + x.exp())
        }
    });
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    let saved = result.clone();
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let sig = &saved[0];
            let sig_grad = sig.mapv(|s| s * (1.0 - s));
            vec![grad * sig_grad]
        }),
    )
}

pub fn silu(input: &Tensor) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.elementwise_unary(gpu_input, ElemOp::Silu) {
                    Ok(gpu_result) => {
                        if !input.requires_grad() {
                            let id = next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let gpu_for_cpu = gpu_input.clone();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![],
                            vec![gpu_input.clone()],
                            Box::new(move |grad, _saved| {
                                let x = gpu_for_cpu.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!("GPU CPU-backward readback failed in silu: {e}");
                                    ndarray::ArrayD::zeros(gpu_for_cpu.shape())
                                });
                                let sig = x.mapv(|v| {
                                    if v >= 0.0 {
                                        1.0 / (1.0 + (-v).exp())
                                    } else {
                                        v.exp() / (1.0 + v.exp())
                                    }
                                });
                                let silu_grad = &sig + x * &sig * (1.0 - &sig);
                                vec![grad * silu_grad]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let x = &saved_gpu[0];
                                let sig =
                                    ctx.elementwise_unary(x, ElemOp::Sigmoid).map_err(|e| {
                                        format!("GPU backward silu sigmoid failed: {e}")
                                    })?;
                                let ones = GpuTensor::ones(&x.shape())
                                    .map_err(|e| format!("GPU backward silu ones failed: {e}"))?;
                                let one_minus_sig = ctx
                                    .sub(&ones, &sig)
                                    .map_err(|e| format!("GPU backward silu sub failed: {e}"))?;
                                let x_sig = ctx
                                    .mul(x, &sig)
                                    .map_err(|e| format!("GPU backward silu mul failed: {e}"))?;
                                let term2 = ctx
                                    .mul(&x_sig, &one_minus_sig)
                                    .map_err(|e| format!("GPU backward silu mul2 failed: {e}"))?;
                                let local_grad = ctx
                                    .add(&sig, &term2)
                                    .map_err(|e| format!("GPU backward silu add failed: {e}"))?;
                                let result = ctx
                                    .mul(grad_gpu, &local_grad)
                                    .map_err(|e| format!("GPU backward silu mul3 failed: {e}"))?;
                                Ok(vec![result])
                            })),
                        );
                    }
                    Err(e) => warn!("autograd activation backward failed: {e}"),
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| {
        x * if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            x.exp() / (1.0 + x.exp())
        }
    });
    if !input.requires_grad() {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![data],
        Box::new(|grad, saved| {
            let x = &saved[0];
            let sig = x.mapv(|v| {
                if v >= 0.0 {
                    1.0 / (1.0 + (-v).exp())
                } else {
                    v.exp() / (1.0 + v.exp())
                }
            });
            let silu_grad = &sig + x * &sig * (1.0 - &sig);
            vec![grad * silu_grad]
        }),
    )
}

pub fn swiglu(gate: &Tensor, x: &Tensor) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let g_storage = gate.storage();
        let x_storage = x.storage();
        if let (Storage::Gpu(gpu_gate, _), Storage::Gpu(gpu_x, _)) = (&g_storage, &x_storage) {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                // Use existing GPU ops: silu(gate) * x
                match ctx.silu(gpu_gate) {
                    Ok(silu_gate) => {
                        match ctx.mul(&silu_gate, gpu_x) {
                            Ok(gpu_result) => {
                                let requires_grad = gate.requires_grad() || x.requires_grad();
                                if !requires_grad {
                                    let id = crate::tensor::next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                let gpu_gate_for_cpu = gpu_gate.clone();
                                let gpu_x_for_cpu = gpu_x.clone();
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![gate.clone(), x.clone()],
                                    vec![],
                                    vec![gpu_gate.clone(), gpu_x.clone()],
                                    Box::new(move |grad, _saved| {
                                        let g = gpu_gate_for_cpu.to_cpu().unwrap_or_else(|e| { tracing::warn!("GPU CPU-backward readback failed in swiglu gate: {e}"); ndarray::ArrayD::zeros(gpu_gate_for_cpu.shape()) });
                                        let x_val = gpu_x_for_cpu.to_cpu().unwrap_or_else(|e| {
                                            tracing::warn!(
                                                "GPU CPU-backward readback failed in swiglu x: {e}"
                                            );
                                            ndarray::ArrayD::zeros(gpu_x_for_cpu.shape())
                                        });
                                        let sig = g.mapv(|v| {
                                            if v >= 0.0 {
                                                1.0 / (1.0 + (-v).exp())
                                            } else {
                                                v.exp() / (1.0 + v.exp())
                                            }
                                        });
                                        let dsilu = &sig + g * &sig * (1.0 - &sig);
                                        let d_gate = grad * x_val * dsilu;
                                        let d_x = grad * &sig;
                                        vec![d_gate, d_x]
                                    }),
                                    Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                        use crate::gpu_backward::swiglu_backward_gpu;
                                        let gate_gpu = &saved_gpu[0];
                                        let x_gpu = &saved_gpu[1];
                                        let (d_gate, d_x) =
                                            swiglu_backward_gpu(ctx, gate_gpu, x_gpu, grad_gpu)
                                                .map_err(|e| {
                                                    format!("swiglu gpu backward failed: {e}")
                                                })?;
                                        Ok(vec![d_gate, d_x])
                                    })),
                                );
                            }
                            Err(e) => warn!("autograd activation backward failed: {e}"),
                        }
                    }
                    Err(e) => warn!("autograd activation backward failed: {e}"),
                }
            }
        }
    }
    let gate_data = gate.data();
    let x_data = x.data();
    let sig = gate_data.mapv(|g| {
        if g >= 0.0 {
            1.0 / (1.0 + (-g).exp())
        } else {
            g.exp() / (1.0 + g.exp())
        }
    });
    let silu_gate = &gate_data * &sig;
    let result = &silu_gate * &x_data;
    let requires_grad = gate.requires_grad() || x.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }
    let saved = vec![gate_data, x_data];
    Tensor::with_grad_fn(
        result,
        vec![gate.clone(), x.clone()],
        saved,
        Box::new(|grad, saved| {
            let g = &saved[0];
            let x_val = &saved[1];
            let sig = g.mapv(|v| {
                if v >= 0.0 {
                    1.0 / (1.0 + (-v).exp())
                } else {
                    v.exp() / (1.0 + v.exp())
                }
            });
            let dsilu = &sig + g * &sig * (1.0 - &sig);
            let d_gate = grad * x_val * dsilu;
            let d_x = grad * &sig;
            vec![d_gate, d_x]
        }),
    )
}
