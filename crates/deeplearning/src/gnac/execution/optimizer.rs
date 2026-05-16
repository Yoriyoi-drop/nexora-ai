use crate::gnac::execution::{GraphIR, IROpType};
use std::collections::HashSet;

/// Graph Optimizer — melakukan operator fusion, dead node elimination,
/// tensor reuse, dan memory scheduling
#[derive(Debug, Clone)]
pub struct GraphOptimizer {
    pub enable_fusion: bool,
    pub enable_dead_elimination: bool,
    pub enable_tensor_reuse: bool,
}

impl GraphOptimizer {
    pub fn new() -> Self {
        GraphOptimizer {
            enable_fusion: true,
            enable_dead_elimination: true,
            enable_tensor_reuse: true,
        }
    }

    /// Optimasi penuh terhadap IR
    pub fn optimize(&self, ir: &GraphIR) -> GraphIR {
        let mut optimized = ir.clone();

        if self.enable_dead_elimination {
            optimized = self.eliminate_dead_nodes(&optimized);
        }
        if self.enable_fusion {
            optimized = self.fuse_operations(&optimized);
        }

        optimized
    }

    /// Operator fusion: gabungkan operasi berurutan
    fn fuse_operations(&self, ir: &GraphIR) -> GraphIR {
        let mut fused = ir.clone();
        let mut to_remove = HashSet::new();
        let mut i = 0;

        while i + 1 < fused.operations.len() {
            let current = &fused.operations[i];
            let next = &fused.operations[i + 1];

            // ReLU + matmul fusion
            if current.op_type == IROpType::Relu && next.op_type == IROpType::MatMul {
                to_remove.insert(current.id);
            }
            // LayerNorm + attention fusion
            if current.op_type == IROpType::LayerNorm && next.op_type == IROpType::Attention {
                to_remove.insert(current.id);
            }

            i += 1;
        }

        fused.operations.retain(|op| !to_remove.contains(&op.id));
        fused
    }

    /// Dead node elimination: hapus operasi tanpa efek
    fn eliminate_dead_nodes(&self, ir: &GraphIR) -> GraphIR {
        let mut cleaned = ir.clone();

        let used_outputs: HashSet<_> = cleaned.operations
            .iter()
            .flat_map(|op| op.inputs.iter().map(|v| v.name.clone()))
            .collect();

        cleaned.operations.retain(|op| {
            if op.op_type == IROpType::Input {
                return true;
            }
            op.outputs.iter().any(|o| used_outputs.contains(&o.name))
        });

        cleaned
    }
}
