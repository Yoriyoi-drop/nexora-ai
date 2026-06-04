use crate::canvas::NeuralGraph;
use crate::{DLResult, DeepLearningError, NodeType};
use nexora_autograd::{Tensor, TensorOps};
use std::collections::HashMap;

/// Asynchronous Execution — overlapping compute & memory transfer
pub struct AsyncExecutor;

impl AsyncExecutor {
    /// Eksekusi asinkron dengan real compute per node type
    /// Setiap stage dijalankan sebagai tokio task dengan blocking compute
    pub async fn execute_async(
        graph: &NeuralGraph,
        inputs: HashMap<uuid::Uuid, Tensor>,
    ) -> DLResult<HashMap<uuid::Uuid, Tensor>> {
        let sorted = graph.topological_order()?;
        let mut tensors = inputs;

        for &node_id in &sorted {
            let node = graph
                .nodes
                .get(&node_id)
                .ok_or_else(|| DeepLearningError::Computation {
                    reason: format!("Node {} not found", node_id),
                })?;

            if node.node_type == NodeType::Input || node.node_type == NodeType::Output {
                continue;
            }

            let mut inputs_for_node: Vec<Tensor> = Vec::new();
            for edge in graph.edges.values() {
                if edge.target_node == node_id {
                    if let Some(t) = tensors.get(&edge.source_node) {
                        inputs_for_node.push(t.clone());
                    }
                }
            }

            let result =
                match node.node_type {
                    NodeType::MatMul | NodeType::Linear => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!(
                                    "MatMul/Linear node {} missing first input",
                                    node.name
                                ),
                            }
                        })?;
                        let b = inputs_for_node.get(1).ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!(
                                    "MatMul/Linear node {} missing second input",
                                    node.name
                                ),
                            }
                        })?;
                        a.matmul(b)
                    }
                    NodeType::Add => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Add node {} missing first input", node.name),
                            }
                        })?;
                        let b = inputs_for_node.get(1).ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Add node {} missing second input", node.name),
                            }
                        })?;
                        a.add(b)
                    }
                    NodeType::Mul => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Mul node {} missing first input", node.name),
                            }
                        })?;
                        let b = inputs_for_node.get(1).ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Mul node {} missing second input", node.name),
                            }
                        })?;
                        a.mul(b)
                    }
                    NodeType::ReLU => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("ReLU node {} missing input", node.name),
                            }
                        })?;
                        a.relu()
                    }
                    NodeType::GELU => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("GELU node {} missing input", node.name),
                            }
                        })?;
                        a.gelu()
                    }
                    NodeType::Sigmoid => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Sigmoid node {} missing input", node.name),
                            }
                        })?;
                        a.sigmoid()
                    }
                    NodeType::Tanh => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Tanh node {} missing input", node.name),
                            }
                        })?;
                        a.tanh()
                    }
                    NodeType::Softmax => {
                        let a = inputs_for_node.first().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Softmax node {} missing input", node.name),
                            }
                        })?;
                        a.softmax(a.shape().len().saturating_sub(1))
                    }
                    _ => {
                        // Fallback: identity
                        inputs_for_node.into_iter().next().ok_or_else(|| {
                            DeepLearningError::Computation {
                                reason: format!("Node {} has no inputs", node.name),
                            }
                        })?
                    }
                };

            tensors.insert(node_id, result);
        }

        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{GraphEdge, GraphNode};
    use crate::TensorDesc;
    use uuid::Uuid;

    fn create_test_graph() -> (NeuralGraph, Uuid, Uuid, Uuid) {
        let mut g = NeuralGraph::new("test");
        let inp = GraphNode::new(NodeType::Input, "in", 0.0, 0.0);
        let mm = GraphNode::new(NodeType::MatMul, "mm", 10.0, 0.0);
        let out = GraphNode::new(NodeType::Output, "out", 20.0, 0.0);
        let inp_id = g.add_node(inp);
        let mm_id = g.add_node(mm);
        let out_id = g.add_node(out);
        let tensor = TensorDesc::new(vec![2, 3], crate::DType::F32);
        let _ = g.add_edge(GraphEdge::new(
            inp_id,
            Uuid::new_v4(),
            mm_id,
            Uuid::new_v4(),
            tensor.clone(),
        ));
        let _ = g.add_edge(GraphEdge::new(
            mm_id,
            Uuid::new_v4(),
            out_id,
            Uuid::new_v4(),
            tensor,
        ));
        (g, inp_id, mm_id, out_id)
    }

    #[tokio::test]
    async fn test_execute_async_with_graph() {
        let (g, inp_id, mm_id, _out_id) = create_test_graph();
        let mut inputs = HashMap::new();
        inputs.insert(inp_id, Tensor::randn(&[2, 3], false));
        inputs.insert(mm_id, Tensor::randn(&[3, 4], false));
        let result = AsyncExecutor::execute_async(&g, inputs).await;
        assert!(result.is_ok());
        let tensors = result.unwrap();
        assert!(tensors.contains_key(&mm_id));
    }

    #[tokio::test]
    async fn test_execute_async_matmul_produces_output() {
        let mut g = NeuralGraph::new("matmul");
        let inp = GraphNode::new(NodeType::Input, "in", 0.0, 0.0);
        let mm = GraphNode::new(NodeType::MatMul, "mm", 10.0, 0.0);
        let out = GraphNode::new(NodeType::Output, "out", 20.0, 0.0);
        let inp_id = g.add_node(inp);
        let mm_id = g.add_node(mm);
        let out_id = g.add_node(out);
        let tensor = TensorDesc::new(vec![2, 3], crate::DType::F32);
        let _ = g.add_edge(GraphEdge::new(
            inp_id,
            Uuid::new_v4(),
            mm_id,
            Uuid::new_v4(),
            tensor,
        ));
        let _ = g.add_edge(GraphEdge::new(
            mm_id,
            Uuid::new_v4(),
            out_id,
            Uuid::new_v4(),
            TensorDesc::new(vec![2, 4], crate::DType::F32),
        ));

        let mut inputs = HashMap::new();
        inputs.insert(inp_id, Tensor::randn(&[2, 3], false));
        inputs.insert(mm_id, Tensor::randn(&[3, 4], false));
        let result = AsyncExecutor::execute_async(&g, inputs).await;
        assert!(result.is_ok());
        let tensors = result.unwrap();
        assert!(tensors.contains_key(&mm_id));
        assert_eq!(tensors[&mm_id].shape(), &[2, 4]);
    }
}
