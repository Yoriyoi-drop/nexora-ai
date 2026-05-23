use crate::canvas::{GraphNode, NeuralGraph};
use crate::execution::ExecutionBackend;
use uuid::Uuid;

/// Intermediate Representation — jembatan antara graf visual dan backend compiler
#[derive(Debug, Clone)]
pub struct GraphIR {
    pub id: Uuid,
    pub name: String,
    pub operations: Vec<IROperation>,
    pub backend: ExecutionBackend,
}

#[derive(Debug, Clone)]
pub struct IROperation {
    pub id: Uuid,
    pub op_type: IROpType,
    pub inputs: Vec<IRValue>,
    pub outputs: Vec<IRValue>,
    pub attributes: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct IRValue {
    pub name: String,
    pub shape: Vec<usize>,
    pub is_tensor: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IROpType {
    Conv2D,
    MatMul,
    Add,
    Mul,
    Relu,
    Gelu,
    Softmax,
    LayerNorm,
    Attention,
    Reshape,
    Transpose,
    Concat,
    Split,
    Dropout,
    Input,
    Output,
    Custom(&'static str),
}

impl GraphIR {
    pub fn new(name: &str, backend: ExecutionBackend) -> Self {
        GraphIR {
            id: Uuid::new_v4(),
            name: name.to_string(),
            operations: Vec::new(),
            backend,
        }
    }

    /// Konversi dari NeuralGraph ke IR
    pub fn from_graph(graph: &NeuralGraph, backend: ExecutionBackend) -> Self {
        let mut ir = GraphIR::new(&graph.name, backend);

        for node in graph.nodes.values() {
            let op = IROperation {
                id: node.id,
                op_type: Self::map_node_type(node),
                inputs: node
                    .inputs
                    .iter()
                    .map(|p| IRValue {
                        name: format!("{}_{}", node.name, p.name),
                        shape: p.tensor.shape.clone(),
                        is_tensor: true,
                    })
                    .collect(),
                outputs: node
                    .outputs
                    .iter()
                    .map(|p| IRValue {
                        name: format!("{}_{}", node.name, p.name),
                        shape: p.tensor.shape.clone(),
                        is_tensor: true,
                    })
                    .collect(),
                attributes: node.params.hyperparameters.clone(),
            };
            ir.operations.push(op);
        }

        ir
    }

    fn map_node_type(node: &GraphNode) -> IROpType {
        use crate::NodeType;
        match node.node_type {
            NodeType::Conv2D => IROpType::Conv2D,
            NodeType::Linear | NodeType::MatMul => IROpType::MatMul,
            NodeType::ReLU => IROpType::Relu,
            NodeType::GELU => IROpType::Gelu,
            NodeType::Softmax => IROpType::Softmax,
            NodeType::LayerNorm | NodeType::RMSNorm => IROpType::LayerNorm,
            NodeType::SelfAttention | NodeType::MultiHeadAttention => IROpType::Attention,
            NodeType::Reshape => IROpType::Reshape,
            NodeType::Transpose => IROpType::Transpose,
            NodeType::Concat => IROpType::Concat,
            NodeType::Split => IROpType::Split,
            NodeType::Dropout => IROpType::Dropout,
            NodeType::Input => IROpType::Input,
            NodeType::Output => IROpType::Output,
            NodeType::Add => IROpType::Add,
            NodeType::Mul => IROpType::Mul,
            _ => IROpType::Custom("generic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::GraphNode;
    use crate::NodeType;

    #[test]
    fn test_ir_new() {
        let ir = GraphIR::new("test", ExecutionBackend::CPU);
        assert_eq!(ir.name, "test");
        assert!(ir.operations.is_empty());
        assert_eq!(ir.backend, ExecutionBackend::CPU);
    }

    #[test]
    fn test_ir_from_graph() {
        let mut g = NeuralGraph::new("g");
        g.add_node(GraphNode::new(NodeType::Input, "in", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::ReLU, "relu", 0.0, 0.0));
        let ir = GraphIR::from_graph(&g, ExecutionBackend::CPU);
        assert_eq!(ir.operations.len(), 2);
    }

    #[test]
    fn test_ir_op_value() {
        let v = IRValue {
            name: "x".to_string(),
            shape: vec![1, 64],
            is_tensor: true,
        };
        assert!(v.is_tensor);
        assert_eq!(v.shape, vec![1, 64]);
    }

    #[test]
    fn test_map_node_type_all() {
        let cases = [
            (NodeType::Conv2D, IROpType::Conv2D),
            (NodeType::Linear, IROpType::MatMul),
            (NodeType::MatMul, IROpType::MatMul),
            (NodeType::ReLU, IROpType::Relu),
            (NodeType::GELU, IROpType::Gelu),
            (NodeType::Softmax, IROpType::Softmax),
            (NodeType::LayerNorm, IROpType::LayerNorm),
            (NodeType::RMSNorm, IROpType::LayerNorm),
            (NodeType::SelfAttention, IROpType::Attention),
            (NodeType::Reshape, IROpType::Reshape),
            (NodeType::Transpose, IROpType::Transpose),
            (NodeType::Concat, IROpType::Concat),
            (NodeType::Split, IROpType::Split),
            (NodeType::Dropout, IROpType::Dropout),
            (NodeType::Input, IROpType::Input),
            (NodeType::Output, IROpType::Output),
            (NodeType::Add, IROpType::Add),
            (NodeType::Mul, IROpType::Mul),
            (NodeType::Embedding, IROpType::Custom("generic")),
        ];
        for (nt, expected) in cases.iter() {
            let node = GraphNode::new(nt.clone(), "x", 0.0, 0.0);
            let result = GraphIR::map_node_type(&node);
            assert_eq!(result, expected.clone(), "mismatch for {:?}", nt);
        }
    }
}
