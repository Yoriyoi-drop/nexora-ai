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
    /// Tags the surviving node with a "fused_<op>" attribute to track what was fused.
    fn fuse_operations(&self, ir: &GraphIR) -> GraphIR {
        let mut fused = ir.clone();
        let mut to_remove = HashSet::with_capacity(fused.operations.len());
        let mut i = 0;

        while i + 1 < fused.operations.len() {
            let relu_idx = i;
            let matmul_idx = i + 1;

            // ReLU + MatMul fusion
            if fused.operations[relu_idx].op_type == IROpType::Relu
                && fused.operations[matmul_idx].op_type == IROpType::MatMul
            {
                // Fix up the MatMul's inputs to skip the ReLU node
                let relu_out_names: Vec<String> = fused.operations[relu_idx]
                    .outputs
                    .iter()
                    .map(|v| v.name.clone())
                    .collect();
                let relu_in_names: Vec<String> = fused.operations[relu_idx]
                    .inputs
                    .iter()
                    .map(|v| v.name.clone())
                    .collect();
                for matmul_input in fused.operations[matmul_idx].inputs.iter_mut() {
                    for (r_out, r_in) in relu_out_names.iter().zip(relu_in_names.iter()) {
                        if matmul_input.name == *r_out {
                            matmul_input.name = r_in.clone();
                        }
                    }
                }
                to_remove.insert(fused.operations[relu_idx].id);
                fused.operations[matmul_idx]
                    .attributes
                    .insert("fused_relu".to_string(), 1.0);
            }

            // LayerNorm + Attention fusion
            if fused.operations[relu_idx].op_type == IROpType::LayerNorm
                && fused.operations[matmul_idx].op_type == IROpType::Attention
            {
                // Fix up the Attention's inputs to skip the LayerNorm node
                let ln_out_names: Vec<String> = fused.operations[relu_idx]
                    .outputs
                    .iter()
                    .map(|v| v.name.clone())
                    .collect();
                let ln_in_names: Vec<String> = fused.operations[relu_idx]
                    .inputs
                    .iter()
                    .map(|v| v.name.clone())
                    .collect();
                for attn_input in fused.operations[matmul_idx].inputs.iter_mut() {
                    for (l_out, l_in) in ln_out_names.iter().zip(ln_in_names.iter()) {
                        if attn_input.name == *l_out {
                            attn_input.name = l_in.clone();
                        }
                    }
                }
                to_remove.insert(fused.operations[relu_idx].id);
                fused.operations[matmul_idx]
                    .attributes
                    .insert("fused_layernorm".to_string(), 1.0);
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
    use crate::execution::{ExecutionBackend, GraphIR, IROperation, IRValue};

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
            outputs: vec![IRValue {
                name: "a".into(),
                shape: vec![1],
                is_tensor: true,
            }],
            attributes: std::collections::HashMap::new(),
        });
        ir.operations.push(IROperation {
            id: uuid::Uuid::new_v4(),
            op_type: IROpType::MatMul,
            inputs: vec![IRValue {
                name: "a".into(),
                shape: vec![1],
                is_tensor: true,
            }],
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
            outputs: vec![IRValue {
                name: "in".into(),
                shape: vec![1],
                is_tensor: true,
            }],
            attributes: std::collections::HashMap::new(),
        });
        // Dead node (output not used)
        ir.operations.push(IROperation {
            id: uuid::Uuid::new_v4(),
            op_type: IROpType::MatMul,
            inputs: vec![],
            outputs: vec![IRValue {
                name: "dead".into(),
                shape: vec![1],
                is_tensor: true,
            }],
            attributes: std::collections::HashMap::new(),
        });
        let cleaned = opt.eliminate_dead_nodes(&ir);
        assert_eq!(cleaned.operations.len(), 1);
        assert_eq!(cleaned.operations[0].op_type, IROpType::Input);
    }
}
