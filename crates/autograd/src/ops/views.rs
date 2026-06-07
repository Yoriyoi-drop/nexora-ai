use ndarray::{ArrayD, Axis};
use tracing::debug;

use super::super::device::Storage;
use super::super::tensor::{next_tensor_id, Tensor};

/// Concatenate tensors along an axis.
/// Backward: each input gets its slice of the gradient.
pub fn cat(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "cat: at least one tensor required");

    // GPU-H1: GPU path — keep result on GPU
    #[cfg(feature = "gpu")]
    if tensors.iter().all(|t| t.is_gpu()) {
        return cat_gpu(tensors, axis);
    }
    #[cfg(feature = "device-cuda")]
    if tensors.iter().all(|t| matches!(t.storage(), Storage::Cuda(..))) {
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
    if tensors.iter().all(|t| t.is_gpu()) {
        return stack_gpu(tensors, axis);
    }
    #[cfg(feature = "device-cuda")]
    if tensors.iter().all(|t| matches!(t.storage(), Storage::Cuda(..))) {
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

// ─── GPU (wgpu) helpers — native GPU concat/stack ────────────────────────

/// Collect copy operations for GPU cat/stack.
#[cfg(feature = "gpu")]
struct GpuCopyOp {
    src_buffer: wgpu::Buffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}

#[cfg(feature = "gpu")]
fn extract_gpu_buffer(t: &Tensor) -> Option<wgpu::Buffer> {
    t.gpu_buffer()
}

/// Native GPU cat: uses wgpu copy_buffer_to_buffer to concatenate on-device.
/// No CPU round-trip. Works for any axis.
#[cfg(feature = "gpu")]
fn cat_gpu(tensors: &[&Tensor], axis: usize) -> Tensor {
    use crate::gpu::GpuContext;
    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(e) => {
            debug!("cat_gpu: GpuContext unavailable: {e}, falling back");
            let arrays: Vec<ArrayD<f32>> = tensors.iter().map(|t| t.data()).collect();
            let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|a| a.view()).collect();
            let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e2| {
                debug!("cat concatenate failed: {e2}"); ArrayD::zeros(vec![0])
            });
            return Tensor::new(result);
        }
    };

    let mut out_shape = tensors[0].shape().to_vec();
    let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
    out_shape[axis] = dim_sizes.iter().sum();
    let out_numel: usize = out_shape.iter().product();
    let out_buffer = ctx.alloc_or_create_buffer(
        (out_numel * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );

    let stride_axis: usize = tensors[0].shape()[axis + 1..].iter().product();
    let outer_batches: usize = tensors[0].shape()[..axis].iter().product();
    let elem_size = stride_axis * 4;

    // Collect all copy ops before entering encoder
    let mut copies: Vec<GpuCopyOp> = Vec::new();
    let mut axis_offset: usize = 0;
    for t in tensors {
        let buf = match extract_gpu_buffer(t) {
            Some(b) => b,
            None => continue,
        };
        let input_axis_dim = t.shape()[axis];
        let batch_bytes = (input_axis_dim * stride_axis * 4) as u64;

        if axis == 0 {
            copies.push(GpuCopyOp {
                src_buffer: buf,
                src_offset: 0,
                dst_offset: (axis_offset * stride_axis * 4) as u64,
                size: batch_bytes,
            });
        } else {
            for b in 0..outer_batches {
                let src_off = (b * input_axis_dim * stride_axis * 4) as u64;
                let dst_off = (b * out_shape[axis] * stride_axis * 4 + axis_offset * elem_size) as u64;
                copies.push(GpuCopyOp {
                    src_buffer: buf.clone(),
                    src_offset: src_off,
                    dst_offset: dst_off,
                    size: elem_size as u64,
                });
            }
        }
        axis_offset += input_axis_dim;
    }

    ctx.with_encoder(|enc| {
        for op in &copies {
            enc.copy_buffer_to_buffer(&op.src_buffer, op.src_offset, &out_buffer, op.dst_offset, op.size);
        }
    });

    let gpu_result = crate::gpu::GpuTensor {
        shape: out_shape.clone(),
        buffer: out_buffer,
        dtype: crate::gpu::GpuDtype::F32,
        device_id: 0,
    };

    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::from_gpu(gpu_result, next_tensor_id(), false);
    }

    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let saved_sizes = ArrayD::from_shape_vec(
        vec![dim_sizes.len()],
        dim_sizes.iter().map(|&x| x as f32).collect(),
    ).unwrap_or_else(|e| {
        debug!("cat_gpu shape encoding failed (infallible): {e}");
        ArrayD::zeros(vec![0])
    });

    Tensor::from_gpu_with_grad_fn(
        gpu_result, input_tensors, vec![saved_sizes], vec![],
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

/// Native GPU stack: uses wgpu copy_buffer_to_buffer to stack on-device.
/// No CPU round-trip. Works for any axis.
#[cfg(feature = "gpu")]
fn stack_gpu(tensors: &[&Tensor], axis: usize) -> Tensor {
    use crate::gpu::GpuContext;
    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(e) => {
            debug!("stack_gpu: GpuContext unavailable: {e}, falling back");
            let mut expanded: Vec<ArrayD<f32>> = Vec::with_capacity(tensors.len());
            for t in tensors {
                let mut shape = t.shape().to_vec();
                shape.insert(axis.min(shape.len()), 1);
                let arr = t.data().into_shape(shape).unwrap_or_else(|e2| {
                    debug!("stack reshape failed: {e2}"); t.data()
                });
                expanded.push(arr);
            }
            let views: Vec<ndarray::ArrayViewD<f32>> = expanded.iter().map(|a| a.view()).collect();
            let result = ndarray::concatenate(Axis(axis), &views).unwrap_or_else(|e2| {
                debug!("stack concatenate failed: {e2}"); ArrayD::zeros(vec![0])
            });
            return Tensor::new(result);
        }
    };

    let mut out_shape = tensors[0].shape().to_vec();
    let insert_axis = axis.min(out_shape.len());
    out_shape.insert(insert_axis, tensors.len());
    let out_numel: usize = out_shape.iter().product();
    let out_buffer = ctx.alloc_or_create_buffer(
        (out_numel * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );

    let element_stride: usize = tensors[0].shape()[insert_axis..].iter().product();
    let outer_batches: usize = tensors[0].shape()[..insert_axis].iter().product();
    let batch_bytes = (element_stride * 4) as u64;
    let total_stride = tensors.len() * element_stride;

    let mut copies: Vec<GpuCopyOp> = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let buf = match extract_gpu_buffer(t) {
            Some(b) => b,
            None => continue,
        };
        for b in 0..outer_batches {
            let src_off = (b * element_stride * 4) as u64;
            let dst_off = (b * total_stride * 4 + i * element_stride * 4) as u64;
            copies.push(GpuCopyOp {
                src_buffer: buf.clone(),
                src_offset: src_off,
                dst_offset: dst_off,
                size: batch_bytes,
            });
        }
    }

    ctx.with_encoder(|enc| {
        for op in &copies {
            enc.copy_buffer_to_buffer(&op.src_buffer, op.src_offset, &out_buffer, op.dst_offset, op.size);
        }
    });

    let gpu_result = crate::gpu::GpuTensor {
        shape: out_shape.clone(),
        buffer: out_buffer,
        dtype: crate::gpu::GpuDtype::F32,
        device_id: 0,
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
        gpu_result, input_tensors, vec![saved_n], vec![],
        Box::new(move |grad, saved| {
            let count = saved[0][0] as usize;
            let mut grads = Vec::with_capacity(count);
            for i in 0..count {
                let g = grad.index_axis(Axis(insert_axis), i).to_owned().into_dyn();
                grads.push(g);
            }
            grads
        }),
        None,
    )
}

// ─── CUDA helpers ──────────────────────────────────────────────────────────

#[cfg(feature = "device-cuda")]
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
    let cuda_result = match CudaTensor::from_cpu(&cuda.stream, shape, &flat, cuda.device_id) {
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

#[cfg(feature = "device-cuda")]
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
    let cuda_result = match CudaTensor::from_cpu(&cuda.stream, shape, &flat, cuda.device_id) {
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
