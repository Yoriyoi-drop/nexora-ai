use std::collections::HashMap;

use nexora_autograd::gpu::{GpuContext, GpuTensor};
use nexora_autograd::Tensor;

use crate::canvas::NeuralGraph;
use crate::execution::backend::CpuBackend;
use crate::execution::optimizer::GraphOptimizer;
use crate::execution::{ExecutionBackend, GraphIR, IROpType};
use crate::DLResult;
use tracing;

/// Compiled Graph Execution — untuk training final
/// Graf di-optimize dan di-compile ke backend target
pub struct CompiledExecutor {
    pub backend: ExecutionBackend,
    pub optimizer: GraphOptimizer,
}

impl CompiledExecutor {
    pub fn new(backend: ExecutionBackend) -> Self {
        CompiledExecutor {
            backend,
            optimizer: GraphOptimizer::new(),
        }
    }

    /// Compile graf ke IR yang sudah dioptimasi
    pub fn compile(&self, graph: &NeuralGraph) -> DLResult<GraphIR> {
        let mut ir = GraphIR::from_graph(graph, self.backend);
        ir = self.optimizer.optimize(&ir);
        Ok(ir)
    }

    /// Jalankan compiled graph dengan input tensors
    pub fn execute(
        &self,
        ir: &GraphIR,
        inputs: HashMap<String, Tensor>,
    ) -> DLResult<HashMap<String, Tensor>> {
        let optimized_ops = &ir.operations;
        tracing::info!(
            "Executing compiled graph '{}' on {} with {} ops",
            ir.name,
            ir.backend.name(),
            optimized_ops.len()
        );

        match self.backend {
            ExecutionBackend::CPU => CpuBackend::execute(ir, inputs),
            ExecutionBackend::WGPU | ExecutionBackend::CUDA => self.execute_gpu(ir, inputs),
        }
    }

    /// Execute compiled graph on GPU via GpuContext (WGPU or CUDA backend).
    fn execute_gpu(
        &self,
        ir: &GraphIR,
        inputs: HashMap<String, Tensor>,
    ) -> DLResult<HashMap<String, Tensor>> {
        let ctx = GpuContext::global().map_err(|e| crate::DeepLearningError::Computation {
            reason: format!("GPU backend requires GPU context: {}", e),
        })?;

        let backend_name = if self.backend == ExecutionBackend::CUDA {
            "CUDA"
        } else {
            "WGPU"
        };
        tracing::info!(
            "Executing compiled graph '{}' on {} with {} ops",
            ir.name,
            backend_name,
            ir.operations.len()
        );

        // Upload all input tensors to GPU
        let mut gpu_tensors: HashMap<String, GpuTensor> = HashMap::new();
        for (name, tensor) in &inputs {
            let data = tensor.data();
            let gpu_tensor =
                GpuTensor::from_cpu(&data).map_err(|e| crate::DeepLearningError::Computation {
                    reason: format!("Failed to upload tensor '{}' to GPU: {}", name, e),
                })?;
            gpu_tensors.insert(name.clone(), gpu_tensor);
        }

        // Batch unsupported ops for efficient CPU fallback
        let mut fallback_ops: Vec<(usize, _)> = Vec::new();

        for (op_idx, op) in ir.operations.iter().enumerate() {
            let get_gpu = |name: &str| -> DLResult<&GpuTensor> {
                gpu_tensors
                    .get(name)
                    .ok_or_else(|| crate::DeepLearningError::Computation {
                        reason: format!("GPU tensor '{}' not found for op {:?}", name, op.op_type),
                    })
            };

            match op.op_type {
                IROpType::MatMul => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let b = get_gpu(&op.inputs[1].name)?;
                    let b_t =
                        ctx.transpose(b)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU transpose failed: {}", e),
                            })?;
                    let result =
                        ctx.matmul(a, &b_t)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU matmul failed: {}", e),
                            })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                IROpType::Add => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let b = get_gpu(&op.inputs[1].name)?;
                    let result =
                        ctx.add(a, b)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU add failed: {}", e),
                            })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                IROpType::Mul => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let b = get_gpu(&op.inputs[1].name)?;
                    let result =
                        ctx.mul(a, b)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU mul failed: {}", e),
                            })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                IROpType::Relu => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let result =
                        ctx.relu(a)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU relu failed: {}", e),
                            })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                IROpType::Gelu => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let result =
                        ctx.gelu(a)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU gelu failed: {}", e),
                            })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                IROpType::Softmax => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let result =
                        ctx.softmax(a)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU softmax failed: {}", e),
                            })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                IROpType::Transpose => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let result =
                        ctx.transpose(a)
                            .map_err(|e| crate::DeepLearningError::Computation {
                                reason: format!("GPU transpose failed: {}", e),
                            })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                IROpType::LayerNorm => {
                    let a = get_gpu(&op.inputs[0].name)?;
                    let shape = a.shape();
                    let numel = if shape.len() >= 1 {
                        shape[shape.len() - 1]
                    } else {
                        1
                    };
                    let weight_data = ndarray::ArrayD::from_elem(vec![numel], 1.0f32);
                    let weight = GpuTensor::from_cpu(&weight_data).map_err(|e| {
                        crate::DeepLearningError::Computation {
                            reason: format!("Failed to create layer_norm weight: {}", e),
                        }
                    })?;
                    let result = ctx.rms_norm(a, &weight, 1e-5).map_err(|e| {
                        crate::DeepLearningError::Computation {
                            reason: format!("GPU layer_norm failed: {}", e),
                        }
                    })?;
                    if let Some(output) = op.outputs.first() {
                        gpu_tensors.insert(output.name.clone(), result);
                    }
                }
                // Collect unsupported ops for batched CPU fallback
                _ => {
                    fallback_ops.push((op_idx, op.clone()));
                }
            }
        }

        // Batched CPU fallback: download all needed tensors once, run all fallback ops on CPU, re-upload results
        if !fallback_ops.is_empty() {
            tracing::info!(
                "GNAC GPU fallback: running {} unsupported ops on CPU (batched)",
                fallback_ops.len()
            );

            // Download all GPU tensors to CPU
            let cpu_tensors: HashMap<String, Tensor> = gpu_tensors
                .iter()
                .map(|(k, gpu)| {
                    let arr = gpu.to_cpu().unwrap_or_else(|e| {
                        tracing::warn!("GNAC GPU fallback readback failed: {e}");
                        ndarray::ArrayD::zeros(gpu.shape())
                    });
                    (k.clone(), Tensor::new(arr))
                })
                .collect();

            // Run all fallback ops on CPU
            let mut fallback_ir =
                GraphIR::new(&format!("{}_gpu_fallback", ir.name), ExecutionBackend::CPU);
            fallback_ir.operations = fallback_ops.iter().map(|(_, op)| op.clone()).collect();
            let cpu_results = CpuBackend::execute(&fallback_ir, cpu_tensors)?;

            // Re-upload CPU results to GPU
            for (name, tensor) in cpu_results {
                let data = tensor.data();
                if let Ok(gpu_tensor) = GpuTensor::from_cpu(&data) {
                    gpu_tensors.insert(name, gpu_tensor);
                }
            }
        }

        // Sync GPU before readback
        ctx.flush();
        ctx.sync();

        // Read all GPU tensors back to CPU
        let mut outputs = HashMap::new();
        for (name, gpu_tensor) in &gpu_tensors {
            let cpu_data = gpu_tensor.to_cpu().unwrap_or_else(|e| {
                tracing::warn!("GNAC final readback failed for {name}: {e}");
                ndarray::ArrayD::zeros(gpu_tensor.shape())
            });
            outputs.insert(name.clone(), Tensor::new(cpu_data));
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::GraphNode;
    use crate::execution::{IROpType, IRValue};
    use crate::NodeType;

    #[test]
    fn test_compiled_executor_new() {
        let exec = CompiledExecutor::new(ExecutionBackend::WGPU);
        assert_eq!(exec.backend, ExecutionBackend::WGPU);
    }

    #[test]
    fn test_compile_empty() {
        let exec = CompiledExecutor::new(ExecutionBackend::CPU);
        let g = NeuralGraph::new("empty");
        let ir = exec.compile(&g).unwrap();
        assert_eq!(ir.operations.len(), 0);
    }

    #[test]
    fn test_compile_with_nodes() {
        let exec = CompiledExecutor::new(ExecutionBackend::CPU);
        let mut g = NeuralGraph::new("g");
        g.add_node(GraphNode::new(NodeType::Input, "in", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Output, "out", 0.0, 0.0));
        let ir = exec.compile(&g).unwrap();
        assert!(ir.operations.len() <= 2);
    }

    #[test]
    fn test_execute_empty() {
        let exec = CompiledExecutor::new(ExecutionBackend::CPU);
        let ir = GraphIR::new("test", ExecutionBackend::CPU);
        let result = exec.execute(&ir, HashMap::new()).unwrap();
        assert!(result.is_empty());
    }
}
