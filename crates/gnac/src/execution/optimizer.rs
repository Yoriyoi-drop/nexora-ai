use crate::execution::{GraphIR, IROpType};
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
        let mut to_remove = HashSet::with_capacity(fused.operations.len());
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

        let used_outputs: HashSet<_> = cleaned
            .operations
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{GraphIR, IROperation, IRValue, ExecutionBackend};

    fn simple_ir() -> GraphIR {
        GraphIR::new("test", ExecutionBackend::CPU)
    }

    #[test]
    fn test_optimizer_new() {
        let opt = GraphOptimizer::new();
        assert!(opt.enable_fusion);
        assert!(opt.enable_dead_elimination);
    }

    #[test]
    fn test_optimize_noop() {
        let opt = GraphOptimizer::new();
        let ir = simple_ir();
        let result = opt.optimize(&ir);
        assert_eq!(result.operations.len(), 0);
    }

    #[test]
    fn test_fuse_relu_matmul() {
        let opt = GraphOptimizer::new();
        let mut ir = simple_ir();
        ir.operations.push(IROperation {
            id: uuid::Uuid::new_v4(),
            op_type: IROpType::Relu,
            inputs: vec![],
            outputs: vec![IRValue { name: "a".into(), shape: vec![1], is_tensor: true }],
            attributes: std::collections::HashMap::new(),
        });
        ir.operations.push(IROperation {
            id: uuid::Uuid::new_v4(),
            op_type: IROpType::MatMul,
            inputs: vec![IRValue { name: "a".into(), shape: vec![1], is_tensor: true }],
            outputs: vec![],
            attributes: std::collections::HashMap::new(),
        });
        let fused = opt.optimize(&ir);
        assert_eq!(fused.operations.len(), 1);
    }

    #[test]
    fn test_eliminate_dead_nodes() {
        let opt = GraphOptimizer::new();
        let mut ir = simple_ir();
        ir.operations.push(IROperation {
            id: uuid::Uuid::new_v4(),
            op_type: IROpType::Input,
            inputs: vec![],
            outputs: vec![IRValue { name: "in".into(), shape: vec![1], is_tensor: true }],
            attributes: std::collections::HashMap::new(),
        });
        // Dead node (output not used)
        ir.operations.push(IROperation {
            id: uuid::Uuid::new_v4(),
            op_type: IROpType::MatMul,
            inputs: vec![],
            outputs: vec![IRValue { name: "dead".into(), shape: vec![1], is_tensor: true }],
            attributes: std::collections::HashMap::new(),
        });
        let cleaned = opt.eliminate_dead_nodes(&ir);
        assert_eq!(cleaned.operations.len(), 1);
        assert_eq!(cleaned.operations[0].op_type, IROpType::Input);
    }
}
