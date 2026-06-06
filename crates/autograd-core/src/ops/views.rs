use ndarray::{ArrayD, Axis};
use tracing::debug;

use crate::device::Storage;
use super::super::tensor::{next_tensor_id, Tensor};

/// Concatenate tensors along an axis.
/// Backward: each input gets its slice of the gradient.
pub fn cat(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "cat: at least one tensor required");

    // GPU-H1: GPU path — keep result on GPU
    #[cfg(feature = "gpu")]
    if tensors.iter().all(|t| matches!(t.storage(), Storage::Gpu(_))) {
        return cat_gpu(tensors, axis);
    }
    #[cfg(feature = "cuda")]
    if tensors.iter().all(|t| matches!(t.storage(), Storage::Cuda(_))) {
        return cat_cuda(tensors, axis);
    }

    let arrays: Vec<ArrayD<f32>> = tensors.iter().map(|t| t.data()).collect();
    let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e| {
        debug!("cat concatenate failed: {e}");
        ArrayD::zeros(vec![0])
    });

    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::new(result);
    }

    let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let saved_sizes = ArrayD::from_shape_vec(
        vec![dim_sizes.len()],
        dim_sizes.iter().map(|&x| x as f32).collect(),
    )
    .unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    Tensor::with_grad_fn(
        result,
        input_tensors,
        vec![saved_sizes],
        Box::new(move |grad, saved| {
            let sizes: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let mut offset = 0usize;
            let mut grads = Vec::with_capacity(sizes.len());
            for &len in &sizes {
                let indices: Vec<usize> = (offset..offset + len).collect();
                let g = grad.select(Axis(axis), &indices).into_dyn();
                grads.push(g);
                offset += len;
            }
            grads
        }),
    )
}

/// Stack tensors along a new axis.
pub fn stack(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "stack: at least one tensor required");

    // GPU-H1: GPU path — keep result on GPU
    #[cfg(feature = "gpu")]
    if tensors.iter().all(|t| matches!(t.storage(), Storage::Gpu(_))) {
        return stack_gpu(tensors, axis);
    }
    #[cfg(feature = "cuda")]
    if tensors.iter().all(|t| matches!(t.storage(), Storage::Cuda(_))) {
        return stack_cuda(tensors, axis);
    }

    let mut expanded: Vec<ArrayD<f32>> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let mut shape = t.shape().to_vec();
        shape.insert(axis.min(shape.len()), 1);
        let arr = t.data().into_shape(shape).unwrap_or_else(|e| {
            debug!("stack reshape failed (infallible): {e}");
            t.data()
        });
        expanded.push(arr);
    }

    let views: Vec<ndarray::ArrayViewD<f32>> = expanded.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e| {
        debug!("stack concatenate failed: {e}");
        ArrayD::zeros(vec![0])
    });

    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::new(result);
    }

    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let n = tensors.len();
    let saved_n = ArrayD::from_shape_vec(vec![1], vec![n as f32]).unwrap_or_else(|e| {
        debug!("shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![1])
    });

    Tensor::with_grad_fn(
        result,
        input_tensors,
        vec![saved_n],
        Box::new(move |grad, saved| {
            let count = saved[0][0] as usize;
            let mut grads = Vec::with_capacity(count);
            for i in 0..count {
                let g = grad.index_axis(Axis(axis), i).to_owned().into_dyn();
                grads.push(g);
            }
            grads
        }),
    )
}

// ─── GPU (wgpu) helpers ────────────────────────────────────────────────────

/// GPU cat: download all GPU tensors to CPU, concat on CPU, upload result to GPU.
/// Result stays GPU-resident — no immediate readback for downstream ops.
#[cfg(feature = "gpu")]
fn cat_gpu(tensors: &[&Tensor], axis: usize) -> Tensor {
    use crate::gpu::{GpuContext, GpuTensor};
    let arrays: Vec<ArrayD<f32>> = tensors.iter().map(|t| t.data()).collect();
    let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e| {
        debug!("cat_gpu concatenate failed: {e}");
        ArrayD::zeros(vec![0])
    });
    let _ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(_) => return Tensor::new(result),
    };
    let gpu_result = match GpuTensor::from_cpu(&result) {
        Ok(g) => g,
        Err(_) => return Tensor::new(result),
    };
    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::from_gpu(gpu_result, next_tensor_id(), false);
    }
    let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let saved_sizes = ArrayD::from_shape_vec(
        vec![dim_sizes.len()],
        dim_sizes.iter().map(|&x| x as f32).collect(),
    )
    .unwrap_or_else(|e| {
        debug!("cat_gpu shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });
    // GPU result with CPU backward fallback
    Tensor::from_gpu_with_grad_fn(
        gpu_result,
        input_tensors,
        vec![saved_sizes],
        vec![],
        Box::new(move |grad, saved| {
            let sizes: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let mut offset = 0usize;
            let mut grads = Vec::with_capacity(sizes.len());
            for &len in &sizes {
                let indices: Vec<usize> = (offset..offset + len).collect();
                let g = grad.select(Axis(axis), &indices).into_dyn();
                grads.push(g);
                offset += len;
            }
            grads
        }),
        None,
    )
}

#[cfg(feature = "gpu")]
fn stack_gpu(tensors: &[&Tensor], axis: usize) -> Tensor {
    use crate::gpu::{GpuContext, GpuTensor};
    let mut expanded: Vec<ArrayD<f32>> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let mut shape = t.shape().to_vec();
        shape.insert(axis.min(shape.len()), 1);
        let arr = t.data().into_shape(shape).unwrap_or_else(|e| {
            debug!("stack_gpu reshape failed (infallible): {e}");
            t.data()
        });
        expanded.push(arr);
    }
    let views: Vec<ndarray::ArrayViewD<f32>> = expanded.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e| {
        debug!("stack_gpu concatenate failed: {e}");
        ArrayD::zeros(vec![0])
    });
    let _ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(_) => return Tensor::new(result),
    };
    let gpu_result = match GpuTensor::from_cpu(&result) {
        Ok(g) => g,
        Err(_) => return Tensor::new(result),
    };
    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::from_gpu(gpu_result, next_tensor_id(), false);
    }
    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let n = tensors.len();
    let saved_n = ArrayD::from_shape_vec(vec![1], vec![n as f32]).unwrap_or_else(|e| {
        debug!("stack_gpu shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![1])
    });
    Tensor::from_gpu_with_grad_fn(
        gpu_result,
        input_tensors,
        vec![saved_n],
        vec![],
        Box::new(move |grad, saved| {
            let count = saved[0][0] as usize;
            let mut grads = Vec::with_capacity(count);
            for i in 0..count {
                let g = grad.index_axis(Axis(axis), i).to_owned().into_dyn();
                grads.push(g);
            }
            grads
        }),
        None,
    )
}

// ─── CUDA helpers ──────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn cat_cuda(tensors: &[&Tensor], axis: usize) -> Tensor {
    use crate::gpu::cuda::{CudaRuntime, CudaTensor};
    let arrays: Vec<ArrayD<f32>> = tensors.iter().map(|t| t.data()).collect();
    let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e| {
        debug!("cat_cuda concatenate failed: {e}");
        ArrayD::zeros(vec![0])
    });
    let cuda = match CudaRuntime::global() {
        Ok(c) => c,
        Err(_) => return Tensor::new(result),
    };
    let shape = result.shape().to_vec();
    let flat: Vec<f32> = result.iter().copied().collect();
    let cuda_result = match CudaTensor::from_cpu(&cuda.device, shape, &flat) {
        Ok(g) => g,
        Err(_) => return Tensor::new(result),
    };
    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::from_cuda(cuda_result, next_tensor_id(), false);
    }
    let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let saved_sizes = ArrayD::from_shape_vec(
        vec![dim_sizes.len()],
        dim_sizes.iter().map(|&x| x as f32).collect(),
    )
    .unwrap_or_else(|e| {
        debug!("cat_cuda shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });
    Tensor::from_cuda_with_grad_fn(
        cuda_result,
        input_tensors,
        vec![saved_sizes],
        Box::new(move |grad, saved| {
            let sizes: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let mut offset = 0usize;
            let mut grads = Vec::with_capacity(sizes.len());
            for &len in &sizes {
                let indices: Vec<usize> = (offset..offset + len).collect();
                let g = grad.select(Axis(axis), &indices).into_dyn();
                grads.push(g);
                offset += len;
            }
            grads
        }),
    )
}

#[cfg(feature = "cuda")]
fn stack_cuda(tensors: &[&Tensor], axis: usize) -> Tensor {
    use crate::gpu::cuda::{CudaRuntime, CudaTensor};
    let mut expanded: Vec<ArrayD<f32>> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let mut shape = t.shape().to_vec();
        shape.insert(axis.min(shape.len()), 1);
        let arr = t.data().into_shape(shape).unwrap_or_else(|e| {
            debug!("stack_cuda reshape failed (infallible): {e}");
            t.data()
        });
        expanded.push(arr);
    }
    let views: Vec<ndarray::ArrayViewD<f32>> = expanded.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e| {
        debug!("stack_cuda concatenate failed: {e}");
        ArrayD::zeros(vec![0])
    });
    let cuda = match CudaRuntime::global() {
        Ok(c) => c,
        Err(_) => return Tensor::new(result),
    };
    let shape = result.shape().to_vec();
    let flat: Vec<f32> = result.iter().copied().collect();
    let cuda_result = match CudaTensor::from_cpu(&cuda.device, shape, &flat) {
        Ok(g) => g,
        Err(_) => return Tensor::new(result),
    };
    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::from_cuda(cuda_result, next_tensor_id(), false);
    }
    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let n = tensors.len();
    let saved_n = ArrayD::from_shape_vec(vec![1], vec![n as f32]).unwrap_or_else(|e| {
        debug!("stack_cuda shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![1])
    });
    Tensor::from_cuda_with_grad_fn(
        cuda_result,
        input_tensors,
        vec![saved_n],
        Box::new(move |grad, saved| {
            let count = saved[0][0] as usize;
            let mut grads = Vec::with_capacity(count);
            for i in 0..count {
                let g = grad.index_axis(Axis(axis), i).to_owned().into_dyn();
                grads.push(g);
            }
            grads
        }),
    )
}
