
use super::super::tensor::Tensor;
#[cfg(feature = "gpu")]
use crate::Storage;
#[cfg(feature = "gpu")]
use ndarray::ArrayD;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(a_shape.len(), 2, "MatMul: a must be 2D");
    assert_eq!(b_shape.len(), 2, "MatMul: b must be 2D");
    assert_eq!(a_shape[1], b_shape[0], "MatMul: inner dims must match");

    #[cfg(feature = "gpu")]
    {
        let a_storage = a.storage();
        let b_storage = b.storage();
        let on_gpu = matches!(&a_storage, Storage::Gpu(_))
            && matches!(&b_storage, Storage::Gpu(_));
        if on_gpu {
            match (&a_storage, &b_storage) {
                (Storage::Gpu(ga), Storage::Gpu(gb)) => {
                    if let Ok(ctx) = crate::gpu::GpuContext::global() {
                        match ctx.matmul(ga, gb) {
                            Ok(gpu_result) => {
                                let requires_grad = a.requires_grad() || b.requires_grad();
                                if !requires_grad {
                                    // Full GPU residency: return tensor with Storage::Gpu
                                    let id = crate::tensor::next_tensor_id();
                                    return Tensor::from_gpu(gpu_result, id, false);
                                }
                                // Training path: GPU matmul with GPU-native backward
                                let a_gpu = ga.clone();
                                let b_gpu = gb.clone();
                                // Saved GPU tensors needed for backward
                                let saved_gpu = vec![a_gpu.clone(), b_gpu.clone()];
                                let gpu_backward: crate::tape::GpuBackwardFn = Box::new(
                                    move |saved_gpu, grad_gpu, ctx| {
                                        let ga = &saved_gpu[0];
                                        let gb = &saved_gpu[1];
                                        match ctx.matmul_backward(ga, gb, grad_gpu) {
                                            Ok((da, db)) => vec![da, db],
                                            Err(_) => {
                                                vec![
                                                    crate::gpu::GpuTensor::from_cpu(
                                                        &ArrayD::zeros(ga.shape()),
                                                    ).unwrap(),
                                                    crate::gpu::GpuTensor::from_cpu(
                                                        &ArrayD::zeros(gb.shape()),
                                                    ).unwrap(),
                                                ]
                                            }
                                        }
                                    },
                                );
                                // CPU backward fallback
                                let cpu_backward: Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>> =
                                    Box::new(move |grad, saved| {
                                    let a_val = &saved[0];
                                    let b_val = &saved[1];
                                    let grad_arr = grad.clone();
                                    let grad_mat = grad_arr.view()
                                        .into_dimensionality::<ndarray::Ix2>().expect("grad 2D");
                                    let a_mat = a_val.view()
                                        .into_dimensionality::<ndarray::Ix2>().expect("a 2D");
                                    let b_mat = b_val.view()
                                        .into_dimensionality::<ndarray::Ix2>().expect("b 2D");
                                    let da = grad_mat.dot(&b_mat.t()).into_dyn();
                                    let db = a_mat.t().dot(&grad_mat).into_dyn();
                                    vec![da, db]
                                });
                                return Tensor::from_gpu_with_grad_fn(
                                    gpu_result,
                                    vec![a.clone(), b.clone()],
                                    vec![a.data(), b.data()],
                                    saved_gpu,
                                    cpu_backward,
                                    Some(gpu_backward),
                                );
                            }
                            Err(_) => { /* fall through to CPU */ }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let a_data = a.data();
    let b_data = b.data();

    let a_view = a_data.view();
    let b_view = b_data.view();

    let a_mat = a_view.into_dimensionality::<ndarray::Ix2>()
        .expect("MatMul: a must be 2D");
    let b_mat = b_view.into_dimensionality::<ndarray::Ix2>()
        .expect("MatMul: b must be 2D");

    let result = a_mat.dot(&b_mat).into_dyn();

    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let saved = vec![a_data.clone(), b_data.clone()];

    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        saved,
        Box::new(|grad, saved| {
            let a_val = &saved[0];
            let b_val = &saved[1];
            let grad_arr = grad.clone();

            let grad_mat = grad_arr.view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("grad must be 2D");

            let a_mat = a_val.view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("a must be 2D");

            let b_mat = b_val.view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("b must be 2D");

            let da = grad_mat.dot(&b_mat.t()).into_dyn();
            let db = a_mat.t().dot(&grad_mat).into_dyn();

            vec![da, db]
        }),
    )
}
