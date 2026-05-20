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
