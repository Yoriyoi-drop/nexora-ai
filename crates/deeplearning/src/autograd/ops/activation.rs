use ndarray::ArrayD;

use crate::autograd::tensor::Tensor;

use crate::gen_activation_op;

#[cfg(feature = "device-gpu")]
use crate::autograd::gpu::GpuContext;
#[cfg(feature = "device-gpu")]
use crate::autograd::gpu::GpuTensor;

// ── relu ─────────────────────────────────────────────────────────────────────────

gen_activation_op!(relu,
    |x| if x > 0.0 { x } else { 0.0 },
    |data: &ArrayD<f32>, _result: &ArrayD<f32>| vec![data.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let x = &saved[0];
        let mask = x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        vec![grad * mask]
    },
    Relu,
    |gi: &GpuTensor, _gr: &GpuTensor| vec![gi.clone()],
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let x = &sg[0];
        let relu_out = ctx
            .elementwise_unary(x, crate::autograd::gpu::ElemOp::Relu)
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
            .mul(gg, &mask)
            .map_err(|e| format!("GPU backward mul failed: {e}"))?;
        Ok(vec![result])
    },
);

// ── gelu ─────────────────────────────────────────────────────────────────────────

fn gelu_f32(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI) as f32;
    let x3 = x * x * x;
    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x3);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

fn gelu_grad(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI) as f32;
    let x3 = x * x * x;
    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x3);
    let t = tanh_arg.tanh();
    let sech2 = 1.0 - t * t;
    0.5 * (1.0 + t) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 0.134145 * x * x)
}

gen_activation_op!(gelu,
    gelu_f32,
    |data: &ArrayD<f32>, _result: &ArrayD<f32>| vec![data.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let x = &saved[0];
        let grad_out = x.mapv(gelu_grad);
        vec![grad * grad_out]
    },
    Gelu,
    |gi: &GpuTensor, _gr: &GpuTensor| vec![gi.clone()],
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let da = crate::autograd::gpu_backward::gelu_backward(ctx, &sg[0], gg)
            .map_err(|e| format!("gelu_backward: {e}"))?;
        Ok(vec![da])
    },
);

// ── tanh ─────────────────────────────────────────────────────────────────────────

gen_activation_op!(tanh,
    |x| x.tanh(),
    |_data: &ArrayD<f32>, result: &ArrayD<f32>| vec![result.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let t = &saved[0];
        let grad_t = t.mapv(|v| 1.0 - v * v);
        vec![grad * grad_t]
    },
    Tanh,
    |_gi: &GpuTensor, gr: &GpuTensor| vec![gr.clone()],
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let y = &sg[0];
        let y2 = ctx
            .mul(y, y)
            .map_err(|e| format!("GPU backward tanh mul failed: {e}"))?;
        let ones = GpuTensor::ones(&y.shape())
            .map_err(|e| format!("GPU backward tanh ones failed: {e}"))?;
        let local = ctx
            .sub(&ones, &y2)
            .map_err(|e| format!("GPU backward tanh sub failed: {e}"))?;
        let result = ctx
            .mul(gg, &local)
            .map_err(|e| format!("GPU backward tanh mul failed: {e}"))?;
        Ok(vec![result])
    },
);

// ── sigmoid ──────────────────────────────────────────────────────────────────────

fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

gen_activation_op!(sigmoid,
    sigmoid_f32,
    |_data: &ArrayD<f32>, result: &ArrayD<f32>| vec![result.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let sig = &saved[0];
        let sig_grad = sig.mapv(|s| s * (1.0 - s));
        vec![grad * sig_grad]
    },
    Sigmoid,
    |_gi: &GpuTensor, gr: &GpuTensor| vec![gr.clone()],
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let y = &sg[0];
        let ones = GpuTensor::ones(&y.shape())
            .map_err(|e| format!("GPU backward sigmoid ones failed: {e}"))?;
        let one_minus_y = ctx
            .sub(&ones, y)
            .map_err(|e| format!("GPU backward sigmoid sub failed: {e}"))?;
        let dy = ctx
            .mul(y, &one_minus_y)
            .map_err(|e| format!("GPU backward sigmoid mul failed: {e}"))?;
        let result = ctx
            .mul(gg, &dy)
            .map_err(|e| format!("GPU backward sigmoid mul failed: {e}"))?;
        Ok(vec![result])
    },
);

// ── silu ─────────────────────────────────────────────────────────────────────────

fn silu_f32(x: f32) -> f32 {
    let sig = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    };
    x * sig
}

fn silu_grad(x: f32) -> f32 {
    let sig = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    };
    sig + x * sig * (1.0 - sig)
}

gen_activation_op!(silu,
    silu_f32,
    |data: &ArrayD<f32>, _result: &ArrayD<f32>| vec![data.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let x = &saved[0];
        let grad_out = x.mapv(silu_grad);
        vec![grad * grad_out]
    },
    Silu,
    |gi: &GpuTensor, _gr: &GpuTensor| vec![gi.clone()],
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let x = &sg[0];
        let sig = ctx
            .elementwise_unary(x, crate::autograd::gpu::ElemOp::Sigmoid)
            .map_err(|e| format!("GPU backward silu sigmoid failed: {e}"))?;
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
            .mul(gg, &local_grad)
            .map_err(|e| format!("GPU backward silu mul3 failed: {e}"))?;
        Ok(vec![result])
    },
);

// ── leaky_relu (manual — extra param, different wgpu path) ───────────────────────

pub fn leaky_relu(input: &Tensor, negative_slope: f32) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let storage = input.storage();
        if let crate::autograd::device::Storage::Gpu(gpu_input, _) = &storage {
            if let Ok(ctx) = crate::autograd::gpu::GpuContext::global() {
                let mut gpu_result = gpu_input.clone();
                match ctx.leaky_relu_inplace(&mut gpu_result, negative_slope) {
                    Ok(()) => {
                        if !input.requires_grad() {
                            let id = crate::autograd::tensor::next_tensor_id();
                            return Tensor::from_gpu(gpu_result, id, false);
                        }
                        let cpu_data = input.data();
                        return Tensor::from_gpu_with_grad_fn(
                            gpu_result,
                            vec![input.clone()],
                            vec![cpu_data.clone()],
                            vec![gpu_input.clone()],
                            Box::new(move |grad, saved| {
                                let x = &saved[0];
                                let grad_x =
                                    x.mapv(|v| if v > 0.0 { 1.0 } else { negative_slope });
                                vec![grad * grad_x]
                            }),
                            Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                let da = crate::autograd::gpu_backward::leaky_relu_backward(
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
                    Err(e) => tracing::warn!("autograd leaky_relu GPU failed: {e}"),
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

// ── swiglu (manual — 2 inputs, compound ops) ────────────────────────────────────

pub fn swiglu(gate: &Tensor, x: &Tensor) -> Tensor {
    #[cfg(feature = "device-gpu")]
    {
        let g_storage = gate.storage();
        let x_storage = x.storage();
        if let (crate::autograd::device::Storage::Gpu(gpu_gate, _), crate::autograd::device::Storage::Gpu(gpu_x, _)) =
            (&g_storage, &x_storage)
        {
            if let Ok(ctx) = GpuContext::global() {
                match ctx.silu(gpu_gate) {
                    Ok(silu_gate) => match ctx.mul(&silu_gate, gpu_x) {
                        Ok(gpu_result) => {
                            let requires_grad = gate.requires_grad() || x.requires_grad();
                            if !requires_grad {
                                let id = crate::autograd::tensor::next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let gate_data = gate.data();
                            let x_data = x.data();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![gate.clone(), x.clone()],
                                vec![gate_data.clone(), x_data.clone()],
                                vec![gpu_gate.clone(), gpu_x.clone()],
                                Box::new(move |grad, saved| {
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
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    let (d_gate, d_x) =
                                        crate::autograd::gpu_backward::swiglu_backward_gpu(
                                            ctx,
                                            &saved_gpu[0],
                                            &saved_gpu[1],
                                            grad_gpu,
                                        )
                                        .map_err(|e| {
                                            format!("swiglu gpu backward failed: {e}")
                                        })?;
                                    Ok(vec![d_gate, d_x])
                                })),
                            );
                        }
                        Err(e) => tracing::warn!("autograd swiglu GPU mul failed: {e}"),
                    },
                    Err(e) => tracing::warn!("autograd swiglu GPU silu failed: {e}"),
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
