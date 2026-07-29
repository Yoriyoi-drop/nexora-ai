use ndarray::{ArrayD, IxDyn};
use tracing::debug;

use crate::autograd::broadcast;
use crate::autograd::tensor::Tensor;

use crate::gen_unary_op;
use crate::gen_binary_op;

#[cfg(feature = "device-gpu")]
use crate::autograd::gpu::GpuTensor;

// ── Helpers for shape encoding ──────────────────────────────────────────────────

fn encode_shape(s: &[usize]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(vec![s.len()], s.iter().map(|&x| x as f32).collect())
        .unwrap_or_else(|e| {
            debug!("shape encoding failed (infallible): {e}");
            ArrayD::zeros(vec![0])
        })
}

// ── Unary ops (gen_unary_op!) ────────────────────────────────────────────────────

gen_unary_op!(exp,
    |x| x.exp(),
    |_data: &ArrayD<f32>, result: &ArrayD<f32>| vec![result.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| { let r = &saved[0]; vec![grad * r] },
    Exp, exp,
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let da = crate::autograd::gpu_backward::exp_backward(ctx, &sg[0], gg)
            .map_err(|e| format!("exp_backward: {e}"))?;
        Ok(vec![da])
    },
);

gen_unary_op!(ln,
    |x| x.ln(),
    |data: &ArrayD<f32>, _result: &ArrayD<f32>| vec![data.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| { let x = &saved[0]; vec![grad / x] },
    Ln, ln,
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let da = crate::autograd::gpu_backward::ln_backward(ctx, &sg[0], gg)
            .map_err(|e| format!("ln_backward: {e}"))?;
        Ok(vec![da])
    },
);

gen_unary_op!(sqrt,
    |x| x.sqrt(),
    |_data: &ArrayD<f32>, result: &ArrayD<f32>| vec![result.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| { let r = &saved[0]; vec![grad / (2.0 * r)] },
    Sqrt, sqrt,
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let da = crate::autograd::gpu_backward::sqrt_backward(ctx, &sg[0], gg)
            .map_err(|e| format!("sqrt_backward: {e}"))?;
        Ok(vec![da])
    },
);

gen_unary_op!(neg,
    |x| -x,
    |_data: &ArrayD<f32>, _result: &ArrayD<f32>| vec![],
    |grad: &ArrayD<f32>, _saved: &[ArrayD<f32>]| vec![-grad],
    Neg, neg,
    move |_sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let da = crate::autograd::gpu_backward::neg_backward(ctx, gg)
            .map_err(|e| format!("neg_backward: {e}"))?;
        Ok(vec![da])
    },
);

// ── Binary ops (gen_binary_op!) ─────────────────────────────────────────────────

gen_binary_op!(add,
    |a: &ArrayD<f32>, b: &ArrayD<f32>| a + b,
    |a: &ArrayD<f32>, b: &ArrayD<f32>, _a_bc: &ArrayD<f32>, _b_bc: &ArrayD<f32>, _result: &ArrayD<f32>|
        vec![encode_shape(a.shape()), encode_shape(b.shape())],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let sa: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
        let sb: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
        let da = broadcast::reduce_grad_for_shape(grad, &sa);
        let db = broadcast::reduce_grad_for_shape(grad, &sb);
        vec![da, db]
    },
    add, add,
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let a_shape = sg[0].shape();
        let b_shape = sg[1].shape();
        let (da, db) = crate::autograd::gpu_backward::add_backward(ctx, &a_shape, &b_shape, gg)
            .map_err(|e| format!("add_backward: {e}"))?;
        Ok(vec![da, db])
    },
    |_: &GpuTensor, _: &GpuTensor, _: &GpuTensor| vec![],
);

gen_binary_op!(sub,
    |a: &ArrayD<f32>, b: &ArrayD<f32>| a - b,
    |a: &ArrayD<f32>, b: &ArrayD<f32>, _a_bc: &ArrayD<f32>, _b_bc: &ArrayD<f32>, _result: &ArrayD<f32>|
        vec![encode_shape(a.shape()), encode_shape(b.shape())],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let sa: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
        let sb: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
        let da = broadcast::reduce_grad_for_shape(grad, &sa);
        let db = broadcast::reduce_grad_for_shape(grad, &sb);
        vec![da, -db]
    },
    sub, sub,
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let a_shape = sg[0].shape();
        let b_shape = sg[1].shape();
        let (da, db) = crate::autograd::gpu_backward::sub_backward(ctx, &a_shape, &b_shape, gg)
            .map_err(|e| format!("sub_backward: {e}"))?;
        Ok(vec![da, db])
    },
    |_: &GpuTensor, _: &GpuTensor, _: &GpuTensor| vec![],
);

gen_binary_op!(mul,
    |a: &ArrayD<f32>, b: &ArrayD<f32>| a * b,
    |_a: &ArrayD<f32>, _b: &ArrayD<f32>, a_bc: &ArrayD<f32>, b_bc: &ArrayD<f32>, _result: &ArrayD<f32>|
        vec![a_bc.clone(), b_bc.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let a_bc = &saved[0];
        let b_bc = &saved[1];
        let da = grad * b_bc;
        let db = grad * a_bc;
        vec![da, db]
    },
    mul, mul,
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let (da, db) = crate::autograd::gpu_backward::mul_backward(ctx, &sg[0], &sg[1], gg)
            .map_err(|e| format!("mul_backward: {e}"))?;
        Ok(vec![da, db])
    },
    |_: &GpuTensor, _: &GpuTensor, _: &GpuTensor| vec![],
);

gen_binary_op!(div,
    |a: &ArrayD<f32>, b: &ArrayD<f32>| a / b,
    |_a: &ArrayD<f32>, _b: &ArrayD<f32>, a_bc: &ArrayD<f32>, b_bc: &ArrayD<f32>, result: &ArrayD<f32>|
        vec![a_bc.clone(), b_bc.clone(), result.clone()],
    |grad: &ArrayD<f32>, saved: &[ArrayD<f32>]| {
        let _a_bc = &saved[0];
        let b_bc = &saved[1];
        let result_val = &saved[2];
        let da = grad / b_bc;
        let db = -grad * result_val / b_bc;
        vec![da, db]
    },
    div, div,
    move |sg: &[GpuTensor], gg: &GpuTensor, ctx: &GpuContext| {
        let (da, db) = crate::autograd::gpu_backward::div_backward(ctx, &sg[1], &sg[2], gg)
            .map_err(|e| format!("div_backward: {e}"))?;
        Ok(vec![da, db])
    },
    |_: &GpuTensor, _: &GpuTensor, result: &GpuTensor| vec![result.clone()],
);

// ── powf — special (extra exponent param, no native wgpu) ───────────────────────

pub fn powf(input: &Tensor, exponent: f32) -> Tensor {
    #[cfg(feature = "device-cuda")]
    {
        let input_storage = input.storage();
        #[cfg(feature = "device-cuda")]
        if let crate::autograd::device::Storage::Cuda(cu_input, _) = &input_storage {
            if let Ok(ctx) = crate::autograd::gpu::CudaRuntime::global() {
                match ctx.powf(cu_input, exponent) {
                    Ok(cuda_result) => {
                        let requires_grad = input.requires_grad();
                        if !requires_grad {
                            let id = crate::autograd::tensor::next_tensor_id();
                            return Tensor::from_cuda(cuda_result, id, false);
                        }
                        let exponent_saved = ArrayD::from_elem(IxDyn(&[]), exponent);
                        let input_cpu = input.data();
                        return Tensor::from_cuda_with_grad_fn(
                            cuda_result,
                            vec![input.clone()],
                            vec![exponent_saved, input_cpu],
                            Box::new(|grad, saved| {
                                let exp = saved[0][IxDyn(&[])];
                                let input_val = &saved[1];
                                let da = grad * exp * input_val.mapv(|x| x.powf(exp - 1.0));
                                vec![da]
                            }),
                        );
                    }
                    Err(e) => {
                        crate::autograd::ops::GPU_MATH_FALLBACKS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::warn!(error = %e, "CUDA powf failed, falling back");
                    }
                }
            }
        }
    }
    #[cfg(feature = "device-gpu")]
    {
        use crate::autograd::gpu::gpu_recovery::RECOVERY_MANAGER;
        use crate::autograd::gpu::{GpuContext, GpuError, GpuTensor};
        let input_storage = input.storage();
        if !RECOVERY_MANAGER.is_circuit_open() {
            if let crate::autograd::device::Storage::Gpu(gpu_input, _) = &input_storage {
                if let Ok(_ctx) = GpuContext::global() {
                    let result_arr = input.data().mapv(|x| x.powf(exponent));
                    match GpuTensor::from_cpu(&result_arr) {
                        Ok(gpu_result) => {
                            RECOVERY_MANAGER.record_recovery();
                            let requires_grad = input.requires_grad();
                            if !requires_grad {
                                let id = crate::autograd::tensor::next_tensor_id();
                                return Tensor::from_gpu(gpu_result, id, false);
                            }
                            let exponent_saved = ArrayD::from_elem(IxDyn(&[]), exponent);
                            let input_cpu = input.data();
                            let exponent_val = exponent;
                            let gpu_result_for_saved = gpu_result.clone();
                            return Tensor::from_gpu_with_grad_fn(
                                gpu_result,
                                vec![input.clone()],
                                vec![exponent_saved, input_cpu],
                                vec![gpu_input.clone(), gpu_result_for_saved],
                                Box::new(|grad, saved| {
                                    let exp = saved[0][IxDyn(&[])];
                                    let input_val = &saved[1];
                                    let da = grad * exp * input_val.mapv(|x| x.powf(exp - 1.0));
                                    vec![da]
                                }),
                                Some(Box::new(move |saved_gpu, grad_gpu, ctx| {
                                    let exponent = exponent_val;
                                    let da = crate::autograd::gpu_backward::powf_backward(
                                        ctx,
                                        &saved_gpu[0],
                                        &saved_gpu[1],
                                        exponent,
                                        grad_gpu,
                                    )
                                    .map_err(|e| format!("powf_backward: {e}"))?;
                                    Ok(vec![da])
                                })),
                            );
                        }
                        Err(e) => {
                            crate::autograd::ops::GPU_MATH_FALLBACKS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            RECOVERY_MANAGER.record_failure(&GpuError::Device(format!(
                                "GPU powf failed: {e}"
                            )));
                            tracing::warn!(
                                error = %e,
                                "GPU powf failed, falling back to CPU (fallback #{})",
                                crate::autograd::ops::GPU_MATH_FALLBACKS
                                    .load(std::sync::atomic::Ordering::Relaxed)
                            );
                        }
                    }
                }
            }
        }
    }
    let data = input.data();
    let result = data.mapv(|x| x.powf(exponent));
    let requires_grad = input.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }
    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![ArrayD::from_elem(IxDyn(&[]), exponent), data],
        Box::new(|grad, saved| {
            let exp = saved[0][IxDyn(&[])];
            let input_val = &saved[1];
            let da = grad * exp * input_val.mapv(|x| x.powf(exp - 1.0));
            vec![da]
        }),
    )
}
