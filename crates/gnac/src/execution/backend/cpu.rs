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
        tensors.get(name).cloned().ok_or_else(|| {
            crate::DeepLearningError::Computation {
                reason: format!("Input '{}' not found for op {:?}", name, op.op_type),
            }
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
            let input_refs: Vec<&Tensor> = op.inputs.iter().filter_map(|input| {
                tensors.get(&input.name)
            }).collect();
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
        IROpType::Conv2D => Err(crate::DeepLearningError::Computation {
            reason: "Conv2D not yet implemented in CPU backend".to_string(),
        }),
        IROpType::Attention => Err(crate::DeepLearningError::Computation {
            reason: "Attention not yet implemented in CPU backend".to_string(),
        }),
        IROpType::Split => Err(crate::DeepLearningError::Computation {
            reason: "Split not yet implemented in CPU backend".to_string(),
        }),
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
