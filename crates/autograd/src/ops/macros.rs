/// Generate a unary op with CUDA → wgpu → CPU fallback.
///
/// Parameters:
/// - `$op_name`: function name
/// - `$cpu_forward`: `|x: f32| -> f32`
/// - `$cpu_saved`: `|data: &ArrayD, result: &ArrayD| -> Vec<ArrayD>` saved for backward
/// - `$backward`: `|grad: &ArrayD, saved: &[ArrayD]| -> Vec<ArrayD>`
/// - `$elem_op`: `ElemOp` variant for wgpu
/// - `$cuda_fn`: method name on `CudaRuntime`
/// - `$gpu_backward`: closure `|saved_gpu: &[GpuTensor], grad_gpu: &GpuTensor, ctx: &GpuContext| -> Result<Vec<GpuTensor>, String>`
#[macro_export]
macro_rules! gen_unary_op {
    (
        $op_name:ident,
        $cpu_forward:expr,
        $cpu_saved:expr,
        $backward:expr,
        $elem_op:ident,
        $cuda_fn:ident,
        $gpu_backward:expr,
    ) => {
        pub fn $op_name(input: &$crate::tensor::Tensor) -> $crate::tensor::Tensor {
            #[cfg(feature = "device-cuda")]
            {
                let input_storage = input.storage();
                if let $crate::device::Storage::Cuda(cu_input, _) = &input_storage {
                    if let Ok(ctx) = $crate::gpu::CudaRuntime::global() {
                        match ctx.$cuda_fn(cu_input) {
                            Ok(cuda_result) => {
                                let requires_grad = input.requires_grad();
                                if !requires_grad {
                                    let id = $crate::tensor::next_tensor_id();
                                    return $crate::tensor::Tensor::from_cuda(cuda_result, id, false);
                                }
                                let input_data = input.data();
                                let result_data = input_data.mapv($cpu_forward);
                                let saved = $cpu_saved(&input_data, &result_data);
                                return $crate::tensor::Tensor::from_cuda_with_grad_fn(
                                    cuda_result,
                                    vec![input.clone()],
                                    saved,
                                    Box::new($backward),
                                );
                            }
                            Err(e) => {
                                $crate::ops::GPU_MATH_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(error = %e, concat!("CUDA ", stringify!($op_name), " failed, falling back"));
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "device-gpu")]
            {
                use $crate::gpu::{ElemOp, GpuContext, GpuError};
                use $crate::gpu::gpu_recovery::RECOVERY_MANAGER;
                let input_storage = input.storage();
                if !RECOVERY_MANAGER.is_circuit_open() {
                    if let $crate::device::Storage::Gpu(gpu_input, _) = &input_storage {
                        if let Ok(ctx) = GpuContext::global() {
                            match ctx.elementwise_unary(gpu_input, ElemOp::$elem_op) {
                                Ok(gpu_result) => {
                                    RECOVERY_MANAGER.record_recovery();
                                    let requires_grad = input.requires_grad();
                                    if !requires_grad {
                                        let id = $crate::tensor::next_tensor_id();
                                        return $crate::tensor::Tensor::from_gpu(gpu_result, id, false);
                                    }
                                    let input_data = input.data();
                                    let result_data = input_data.mapv($cpu_forward);
                                    let saved = $cpu_saved(&input_data, &result_data);
                                    let gpu_bwd = $gpu_backward;
                                    return $crate::tensor::Tensor::from_gpu_with_grad_fn(
                                        gpu_result,
                                        vec![input.clone()],
                                        saved,
                                        vec![gpu_input.clone()],
                                        Box::new($backward),
                                        Some(Box::new(move |sg, gg, ctx| (gpu_bwd)(sg, gg, ctx))),
                                    );
                                }
                                Err(e) => {
                                    $crate::ops::GPU_MATH_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    RECOVERY_MANAGER.record_failure(&GpuError::Device(
                                        format!(concat!("GPU ", stringify!($op_name), " failed: {}"), e)
                                    ));
                                    tracing::warn!(
                                        error = %e,
                                        concat!("GPU ", stringify!($op_name), " failed, falling back to CPU (fallback #{})"),
                                        $crate::ops::GPU_MATH_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
                                    );
                                }
                            }
                        }
                    }
                }
            }
            let data = input.data();
            let result = data.mapv($cpu_forward);
            if !input.requires_grad() {
                return $crate::tensor::Tensor::new(result);
            }
            let saved = $cpu_saved(&data, &result);
            $crate::tensor::Tensor::with_grad_fn(
                result,
                vec![input.clone()],
                saved,
                Box::new($backward),
            )
        }
    };
}

/// Generate a binary op with CUDA → wgpu → CPU fallback.
///
/// The macro defines `a_shape` and `b_shape` (both `Vec<usize>`) which the
/// `$gpu_backward` closure may reference.
///
/// `$extra_saved_gpu` is `|ga: &GpuTensor, gb: &GpuTensor, gpu_result: &GpuTensor| -> Vec<GpuTensor>`
/// for tensors beyond `[ga, gb]` (e.g. div needs `[gpu_result]`).
#[macro_export]
macro_rules! gen_binary_op {
    (
        $op_name:ident,
        $cpu_forward:expr,
        $cpu_saved:expr,
        $backward:expr,
        $wgpu_fn:ident,
        $cuda_fn:ident,
        $gpu_backward:expr,
        $extra_saved_gpu:expr,
    ) => {
        pub fn $op_name(a: &$crate::tensor::Tensor, b: &$crate::tensor::Tensor) -> $crate::tensor::Tensor {
            let _a_shape: Vec<usize> = a.shape();
            let _b_shape: Vec<usize> = b.shape();
            #[cfg(feature = "device-cuda")]
            {
                let a_storage = a.storage();
                let b_storage = b.storage();
                if let ($crate::device::Storage::Cuda(ca, _), $crate::device::Storage::Cuda(cb, _)) = (&a_storage, &b_storage) {
                    if let Ok(ctx) = $crate::gpu::CudaRuntime::global() {
                        match ctx.$cuda_fn(ca, cb) {
                            Ok(cuda_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    let id = $crate::tensor::next_tensor_id();
                                    return $crate::tensor::Tensor::from_cuda(cuda_result, id, false);
                                }
                                let a_data = a.data();
                                let b_data = b.data();
                                let (a_bc, b_bc, _) = $crate::broadcast::broadcast_arrays(&a_data, &b_data);
                                let result = $cpu_forward(&a_bc, &b_bc);
                                let saved = $cpu_saved(&a_data, &b_data, &a_bc, &b_bc, &result);
                                return $crate::tensor::Tensor::from_cuda_with_grad_fn(
                                    cuda_result,
                                    vec![a.clone(), b.clone()],
                                    saved,
                                    Box::new($backward),
                                );
                            }
                            Err(e) => {
                                $crate::ops::GPU_MATH_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(error = %e, concat!("CUDA ", stringify!($op_name), " failed, falling back"));
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "device-gpu")]
            {
                use $crate::gpu::{GpuContext, GpuError};
                use $crate::gpu::gpu_recovery::RECOVERY_MANAGER;
                let a_storage = a.storage();
                let b_storage = b.storage();
                let on_gpu = matches!(&a_storage, $crate::device::Storage::Gpu(..))
                    && matches!(&b_storage, $crate::device::Storage::Gpu(..));
                if on_gpu && !RECOVERY_MANAGER.is_circuit_open() {
                    if let ($crate::device::Storage::Gpu(ga, _), $crate::device::Storage::Gpu(gb, _)) = (&a_storage, &b_storage) {
                        if let Ok(ctx) = GpuContext::global() {
                            match ctx.$wgpu_fn(ga, gb) {
                                Ok(gpu_result) => {
                                    RECOVERY_MANAGER.record_recovery();
                                    let requires_grad = a.requires_grad() || b.requires_grad();
                                    if !requires_grad {
                                        let id = $crate::tensor::next_tensor_id();
                                        return $crate::tensor::Tensor::from_gpu(gpu_result, id, false);
                                    }
                                    let a_data = a.data();
                                    let b_data = b.data();
                                    let (a_bc, b_bc, _) = $crate::broadcast::broadcast_arrays(&a_data, &b_data);
                                    let result = $cpu_forward(&a_bc, &b_bc);
                                    let saved = $cpu_saved(&a_data, &b_data, &a_bc, &b_bc, &result);
                                    let mut saved_gpu = vec![ga.clone(), gb.clone()];
                                    saved_gpu.extend($extra_saved_gpu(ga, gb, &gpu_result));
                                    let gpu_bwd = $gpu_backward;
                                    return $crate::tensor::Tensor::from_gpu_with_grad_fn(
                                        gpu_result,
                                        vec![a.clone(), b.clone()],
                                        saved,
                                        saved_gpu,
                                        Box::new($backward),
                                        Some(Box::new(move |sg, gg, ctx| (gpu_bwd)(sg, gg, ctx))),
                                    );
                                }
                                Err(e) => {
                                    $crate::ops::GPU_MATH_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    RECOVERY_MANAGER.record_failure(&GpuError::Device(
                                        format!(concat!("GPU ", stringify!($op_name), " failed: {}"), e)
                                    ));
                                    tracing::warn!(
                                        error = %e,
                                        concat!("GPU ", stringify!($op_name), " failed, falling back to CPU (fallback #{})"),
                                        $crate::ops::GPU_MATH_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
                                    );
                                }
                            }
                        }
                    }
                }
            }
            let a_data = a.data();
            let b_data = b.data();
            let (a_bc, b_bc, _) = $crate::broadcast::broadcast_arrays(&a_data, &b_data);
            let result = $cpu_forward(&a_bc, &b_bc);
            let requires_grad = a.requires_grad() || b.requires_grad();
            if !requires_grad {
                return $crate::tensor::Tensor::new(result);
            }
            let saved = $cpu_saved(&a_data, &b_data, &a_bc, &b_bc, &result);
            $crate::tensor::Tensor::with_grad_fn(
                result,
                vec![a.clone(), b.clone()],
                saved,
                Box::new($backward),
            )
        }
    };
}

/// Generate an activation op with wgpu → CPU fallback (no CUDA path).
///
/// `$gpu_saved` is `|gpu_input: &GpuTensor, gpu_result: &GpuTensor| -> Vec<GpuTensor>`.
#[macro_export]
macro_rules! gen_activation_op {
    (
        $op_name:ident,
        $cpu_forward:expr,
        $cpu_saved:expr,
        $backward:expr,
        $elem_op:ident,
        $gpu_saved:expr,
        $gpu_backward:expr,
    ) => {
        pub fn $op_name(input: &$crate::tensor::Tensor) -> $crate::tensor::Tensor {
            #[cfg(feature = "device-gpu")]
            {
                use $crate::gpu::{ElemOp, GpuContext};
                let storage = input.storage();
                if let $crate::device::Storage::Gpu(gpu_input, _) = &storage {
                    if let Ok(ctx) = GpuContext::global() {
                        match ctx.elementwise_unary(gpu_input, ElemOp::$elem_op) {
                            Ok(gpu_result) => {
                                if !input.requires_grad() {
                                    let id = $crate::tensor::next_tensor_id();
                                    return $crate::tensor::Tensor::from_gpu(gpu_result, id, false);
                                }
                                let cpu_data = input.data();
                                let cpu_result = cpu_data.mapv($cpu_forward);
                                let saved = $cpu_saved(&cpu_data, &cpu_result);
                                let saved_gpu = $gpu_saved(gpu_input, &gpu_result);
                                let gpu_bwd = $gpu_backward;
                                return $crate::tensor::Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![input.clone()],
                                    saved,
                                    saved_gpu,
                                    Box::new($backward),
                                    Some(Box::new(move |sg, gg, ctx| (gpu_bwd)(sg, gg, ctx))),
                                );
                            }
                            Err(e) => tracing::warn!(
                                error = %e,
                                concat!("autograd ", stringify!($op_name), " GPU forward failed"),
                            ),
                        }
                    }
                }
            }
            let data = input.data();
            let result = data.mapv($cpu_forward);
            if !input.requires_grad() {
                return $crate::tensor::Tensor::new(result);
            }
            let saved = $cpu_saved(&data, &result);
            $crate::tensor::Tensor::with_grad_fn(
                result,
                vec![input.clone()],
                saved,
                Box::new($backward),
            )
        }
    };
}
