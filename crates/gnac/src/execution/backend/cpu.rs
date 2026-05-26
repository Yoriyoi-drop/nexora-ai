use std::collections::HashMap;

use nexora_autograd::ops;
use nexora_autograd::Tensor;

use crate::execution::{GraphIR, IROpType};
use crate::DLResult;

pub struct CpuBackend;

impl CpuBackend {
    pub fn execute(
        ir: &GraphIR,
        inputs: HashMap<String, Tensor>,
    ) -> DLResult<HashMap<String, Tensor>> {
        let mut tensors: HashMap<String, Tensor> = inputs;

        for op in &ir.operations {
            let output = dispatch_op(op, &tensors)?;
            for (i, output_tensor) in output.into_iter().enumerate() {
                let name = if i < op.outputs.len() {
                    op.outputs[i].name.clone()
                } else {
                    format!("{}_{}", op.op_type.name(), i)
                };
                tensors.insert(name, output_tensor);
            }
        }

        Ok(tensors)
    }
}

fn dispatch_op(
    op: &crate::execution::IROperation,
    tensors: &HashMap<String, Tensor>,
) -> DLResult<Vec<Tensor>> {
    let get_input = |idx: usize| -> DLResult<Tensor> {
        let name = &op.inputs[idx].name;
        tensors
            .get(name)
            .cloned()
            .ok_or_else(|| crate::DeepLearningError::Computation {
                reason: format!("Input '{}' not found for op {:?}", name, op.op_type),
            })
    };

    match op.op_type {
        IROpType::MatMul => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            Ok(vec![ops::matmul(&a, &b)])
        }
        IROpType::Add => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            Ok(vec![ops::add(&a, &b)])
        }
        IROpType::Mul => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            Ok(vec![ops::mul(&a, &b)])
        }
        IROpType::Relu => {
            let a = get_input(0)?;
            Ok(vec![ops::relu(&a)])
        }
        IROpType::Gelu => {
            let a = get_input(0)?;
            Ok(vec![ops::gelu(&a)])
        }
        IROpType::Softmax => {
            let a = get_input(0)?;
            let axis = op.attributes.get("axis").map(|&v| v as usize).unwrap_or(0);
            Ok(vec![ops::softmax(&a, axis)])
        }
        IROpType::LayerNorm => {
            let a = get_input(0)?;
            let normalized = ops::layer_norm(&a, None, None, 1e-5);
            Ok(vec![normalized])
        }
        IROpType::Reshape => {
            let a = get_input(0)?;
            let shape: Vec<usize> = op
                .attributes
                .get("shape")
                .map(|&v| vec![v as usize])
                .unwrap_or_default();
            if shape.is_empty() {
                return Err(crate::DeepLearningError::Computation {
                    reason: "Reshape op requires 'shape' attribute".to_string(),
                });
            }
            Ok(vec![ops::reshape(&a, &shape)])
        }
        IROpType::Transpose => {
            let a = get_input(0)?;
            Ok(vec![ops::transpose(&a)])
        }
        IROpType::Concat => {
            let input_refs: Vec<&Tensor> = op
                .inputs
                .iter()
                .filter_map(|input| tensors.get(&input.name))
                .collect();
            if input_refs.is_empty() {
                return Err(crate::DeepLearningError::Computation {
                    reason: "Concat requires at least one input".to_string(),
                });
            }
            let axis = op.attributes.get("axis").map(|&v| v as usize).unwrap_or(0);
            Ok(vec![ops::cat(&input_refs, axis)])
        }
        IROpType::Dropout => {
            let a = get_input(0)?;
            let rate = op.attributes.get("rate").copied().unwrap_or(0.0) as f32;
            Ok(vec![ops::dropout(&a, rate, true)])
        }
        IROpType::Input | IROpType::Output => {
            let a = get_input(0)?;
            Ok(vec![a])
        }
        IROpType::Conv2D => {
            let input = get_input(0)?;
            let kernel = get_input(1)?;
            let stride = op.attributes.get("stride").copied().unwrap_or(1.0) as usize;
            let padding = op.attributes.get("padding").copied().unwrap_or(0.0) as usize;
            let data = input.data();
            let kernel_data = kernel.data();
            let in_shape = data.shape();
            if in_shape.len() != 4 {
                return Err(crate::DeepLearningError::Computation {
                    reason: format!("Conv2D requires 4D input [N, C, H, W], got {:?}", in_shape),
                });
            }
            let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let k_shape = kernel_data.shape();
            if k_shape.len() != 4 {
                return Err(crate::DeepLearningError::Computation {
                    reason: format!(
                        "Conv2D kernel requires 4D [OC, IC, KH, KW], got {:?}",
                        k_shape
                    ),
                });
            }
            let (oc, _ic, kh, kw) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
            let oh = ((h + 2 * padding).saturating_sub(kh)) / stride + 1;
            let ow = ((w + 2 * padding).saturating_sub(kw)) / stride + 1;

            let mut out = ndarray::Array4::<f32>::zeros((n, oc, oh, ow));
            let k_slice = kernel_data.as_slice().unwrap_or(&[]);
            let i_slice = data.as_slice().unwrap_or(&[]);

            for bn in 0..n {
                for co in 0..oc {
                    for ci in 0..c {
                        for oh_idx in 0..oh {
                            for ow_idx in 0..ow {
                                let mut sum = 0.0;
                                for kh_idx in 0..kh {
                                    for kw_idx in 0..kw {
                                        let ih = oh_idx * stride + kh_idx;
                                        let iw = ow_idx * stride + kw_idx;
                                        if ih < padding
                                            || ih >= h + padding
                                            || iw < padding
                                            || iw >= w + padding
                                        {
                                            continue;
                                        }
                                        let ih_clamped = ih.wrapping_sub(padding);
                                        let iw_clamped = iw.wrapping_sub(padding);
                                        let i_idx = bn * c * h * w
                                            + ci * h * w
                                            + ih_clamped * w
                                            + iw_clamped;
                                        let k_idx =
                                            co * c * kh * kw + ci * kh * kw + kh_idx * kw + kw_idx;
                                        if i_idx < i_slice.len() && k_idx < k_slice.len() {
                                            sum += i_slice[i_idx] * k_slice[k_idx];
                                        }
                                    }
                                }
                                out[[bn, co, oh_idx, ow_idx]] += sum;
                            }
                        }
                    }
                }
            }
            Ok(vec![Tensor::new(out.into_dyn())])
        }
        IROpType::Attention => {
            let q = get_input(0)?;
            let k = get_input(1)?;
            let v = get_input(2)?;
            let scale = op.attributes.get("scale").copied().unwrap_or(1.0) as f32;
            let scale_sqrt = scale.max(1.0).sqrt().recip();
            let scores = ops::matmul(&q, &ops::transpose(&k));
            let scaled = ops::mul(&scores, &Tensor::from_slice(&[scale_sqrt], &[1]));
            let attn = ops::softmax(&scaled, scaled.ndim().saturating_sub(1));
            Ok(vec![ops::matmul(&attn, &v)])
        }
        IROpType::Split => {
            let a = get_input(0)?;
            let axis = op.attributes.get("axis").map(|&v| v as usize).unwrap_or(0);
            let chunks = op
                .attributes
                .get("chunks")
                .map(|&v| v as usize)
                .unwrap_or(2);
            let data = a.data();
            let shape = data.shape().to_vec();
            if axis >= shape.len() {
                return Err(crate::DeepLearningError::Computation {
                    reason: format!("Split axis {} out of range for shape {:?}", axis, shape),
                });
            }
            let dim_size = shape[axis];
            let chunk_size = dim_size / chunks;
            if chunk_size == 0 {
                return Err(crate::DeepLearningError::Computation {
                    reason: format!(
                        "Split chunk size 0 for axis {} dim {} chunks {}",
                        axis, dim_size, chunks
                    ),
                });
            }
            let flat = data.as_slice().unwrap_or(&[]);
            let inner = shape[axis + 1..].iter().product::<usize>().max(1);
            let outer_before = shape[..axis].iter().product::<usize>().max(1);
            let mut results = Vec::with_capacity(chunks);
            for c in 0..chunks {
                let start = c * chunk_size;
                let end = if c == chunks - 1 {
                    dim_size
                } else {
                    (c + 1) * chunk_size
                };
                let count = end - start;
                let chunk_size_total = count * inner;
                let mut chunk_data = vec![0.0f32; outer_before * chunk_size_total];
                for ob in 0..outer_before {
                    let src_base = ob * dim_size * inner + start * inner;
                    let dst_base = ob * chunk_size_total;
                    for j in 0..chunk_size_total {
                        if src_base + j < flat.len() {
                            chunk_data[dst_base + j] = flat[src_base + j];
                        }
                    }
                }
                let mut chunk_shape = shape.clone();
                chunk_shape[axis] = count;
                let arr =
                    ndarray::ArrayD::from_shape_vec(chunk_shape, chunk_data).map_err(|e| {
                        crate::DeepLearningError::Computation {
                            reason: format!("Split shape mismatch: {}", e),
                        }
                    })?;
                results.push(Tensor::new(arr));
            }
            Ok(results)
        }
        IROpType::Custom(_) => Err(crate::DeepLearningError::Computation {
            reason: format!("Custom op '{:?}' not available in CPU backend", op.op_type),
        }),
    }
}

impl IROpType {
    fn name(&self) -> &str {
        match self {
            IROpType::Conv2D => "conv2d",
            IROpType::MatMul => "matmul",
            IROpType::Add => "add",
            IROpType::Mul => "mul",
            IROpType::Relu => "relu",
            IROpType::Gelu => "gelu",
            IROpType::Softmax => "softmax",
            IROpType::LayerNorm => "layer_norm",
            IROpType::Attention => "attention",
            IROpType::Reshape => "reshape",
            IROpType::Transpose => "transpose",
            IROpType::Concat => "concat",
            IROpType::Split => "split",
            IROpType::Dropout => "dropout",
            IROpType::Input => "input",
            IROpType::Output => "output",
            IROpType::Custom(s) => s,
        }
    }
}
