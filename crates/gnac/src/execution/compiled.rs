use std::collections::HashMap;

use nexora_autograd::Tensor;

use crate::canvas::NeuralGraph;
use crate::execution::backend::CpuBackend;
use crate::execution::optimizer::GraphOptimizer;
use crate::execution::{ExecutionBackend, GraphIR};
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
            _ => Err(crate::DeepLearningError::Computation {
                reason: format!(
                    "Backend '{:?}' not yet implemented — use CPU backend",
                    self.backend
                ),
            }),
        }
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
        let exec = CompiledExecutor::new(ExecutionBackend::CUDA);
        assert_eq!(exec.backend, ExecutionBackend::CUDA);
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

    #[test]
    fn test_cpu_backend_rejects_gpu_backend() {
        let exec = CompiledExecutor::new(ExecutionBackend::CUDA);
        let ir = GraphIR::new("test", ExecutionBackend::CUDA);
        let result = exec.execute(&ir, HashMap::new());
        assert!(result.is_err());
    }
}
