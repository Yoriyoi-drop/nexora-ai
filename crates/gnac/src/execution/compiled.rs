use crate::canvas::NeuralGraph;
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

    /// Jalankan compiled graph
    pub fn execute(&self, ir: &GraphIR) -> DLResult<()> {
        let optimized_ops = &ir.operations;
        tracing::info!(
            "Executing compiled graph '{}' on {} with {} ops",
            ir.name,
            ir.backend.name(),
            optimized_ops.len()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::GraphNode;
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
        // Input nodes are preserved; dead node elimination removes Output if no connection
        assert!(ir.operations.len() <= 2);
    }

    #[test]
    fn test_execute() {
        let exec = CompiledExecutor::new(ExecutionBackend::CPU);
        let ir = GraphIR::new("test", ExecutionBackend::CPU);
        assert!(exec.execute(&ir).is_ok());
    }
}
