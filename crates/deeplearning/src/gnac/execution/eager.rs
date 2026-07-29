use crate::gnac::canvas::NeuralGraph;
use crate::gnac::DLResult;

/// Graph topology validator and shape propagator.
///
/// Validates topological order and infers shapes through the graph. Does NOT
/// perform tensor computation — use [`CompiledExecutor`](super::compiled::CompiledExecutor)
/// for actual tensor dispatch.
pub struct GraphValidator;

impl GraphValidator {
    /// Validate graph topology (connectivity, input availability).
    pub fn validate(&self, graph: &NeuralGraph) -> DLResult<()> {
        let order = graph.topological_order()?;

        for node_id in order {
            let node = graph.get_node(&node_id).ok_or_else(|| {
                crate::gnac::DeepLearningError::Configuration {
                    reason: format!("Node {} not found during execution", node_id),
                }
            })?;

            if !node.params.enabled {
                continue;
            }

            // Validasi semua input tersedia
            let (incoming, _outgoing) = graph.get_node_connections(&node_id);
            if incoming.len() < node.inputs.len() {
                return Err(crate::gnac::DeepLearningError::Computation {
                    reason: format!(
                        "Node {}: missing inputs (have {}, need {})",
                        node.name,
                        incoming.len(),
                        node.inputs.len()
                    ),
                });
            }
        }

        Ok(())
    }

    /// Eksekusi dengan input tertentu
    pub fn execute_with_input(
        graph: &NeuralGraph,
        input_shape: &[usize],
    ) -> DLResult<Vec<Vec<usize>>> {
        let order = graph.topological_order()?;
        let mut shapes: std::collections::HashMap<uuid::Uuid, Vec<Vec<usize>>> =
            std::collections::HashMap::new();

        // Set input shape
        for input_node in graph.get_input_nodes() {
            shapes.insert(input_node.id, vec![input_shape.to_vec()]);
        }

        for node_id in order {
            let node = match graph.get_node(&node_id) {
                Some(n) => n,
                None => {
                    tracing::warn!(
                        "node {:?} not found in graph during execution — skipping",
                        node_id
                    );
                    continue;
                }
            };
            if !node.params.enabled {
                continue;
            }

            // Kumpulkan input shapes
            let (incoming, _) = graph.get_node_connections(&node_id);
            let mut input_shapes: Vec<Vec<usize>> = Vec::new();
            for edge in &incoming {
                if let Some(source_shapes) = shapes.get(&edge.source_node) {
                    if let Some(shape) = source_shapes.last() {
                        input_shapes.push(shape.clone());
                    }
                }
            }

            // Di sini sebenarnya akan dijalankan operasi sesungguhnya
            // Untuk eager mode, kita simulasikan output shapes berdasarkan input
            if let Ok(output_shapes) = crate::gnac::smart_tensor::propagation::ShapePropagator::new()
                .propagate(node, &input_shapes)
            {
                shapes.insert(node_id, output_shapes);
            }
        }

        // Return output shapes dari output nodes
        let mut result = Vec::new();
        for output_node in graph.get_output_nodes() {
            if let Some(shapes) = shapes.get(&output_node.id) {
                result.extend(shapes.clone());
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnac::canvas::GraphNode;
    use crate::gnac::NodeType;

    #[test]
    fn test_validate_empty() {
        let g = NeuralGraph::new("empty");
        let validator = GraphValidator;
        assert!(validator.validate(&g).is_ok());
    }

    #[test]
    fn test_validate_with_edges() {
        let mut g = NeuralGraph::new("g");
        let inp = GraphNode::new(NodeType::Input, "in", 0.0, 0.0);
        let out = GraphNode::new(NodeType::Output, "out", 0.0, 0.0);
        let inp_id = g.add_node(inp);
        let out_id = g.add_node(out);
        let tensor = crate::gnac::TensorDesc::new(vec![1, 64], crate::gnac::DType::F32);
        let _ = g.add_edge(crate::gnac::GraphEdge::new(
            inp_id,
            uuid::Uuid::new_v4(),
            out_id,
            uuid::Uuid::new_v4(),
            tensor,
        ));
        assert!(GraphValidator.validate(&g).is_ok());
    }
}
