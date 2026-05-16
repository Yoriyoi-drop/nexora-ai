use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;
use crate::gnac::execution::{GraphIR, ExecutionBackend};

/// Eager Execution — untuk editing visual responsif
/// Setiap operasi dijalankan segera begitu ditambahkan ke graf
pub struct EagerExecutor;

impl EagerExecutor {
    /// Eksekusi eager: jalankan operasi sekuensial
    pub fn execute(&self, graph: &NeuralGraph) -> DLResult<()> {
        let order = graph.topological_order()?;

        for node_id in order {
            let node = graph.get_node(&node_id)
                .ok_or_else(|| crate::DeepLearningError::Configuration {
                    reason: format!("Node {} not found during execution", node_id),
                })?;

            if !node.params.enabled {
                continue;
            }

            // Validasi semua input tersedia
            let (incoming, _outgoing) = graph.get_node_connections(&node_id);
            if incoming.len() < node.inputs.len() {
                return Err(crate::DeepLearningError::Computation {
                    reason: format!("Node {}: missing inputs (have {}, need {})",
                        node.name, incoming.len(), node.inputs.len()),
                });
            }
        }

        Ok(())
    }

    /// Eksekusi dengan input tertentu
    pub fn execute_with_input(graph: &NeuralGraph, input_shape: &[usize]) -> DLResult<Vec<Vec<usize>>> {
        let order = graph.topological_order()?;
        let mut shapes: std::collections::HashMap<uuid::Uuid, Vec<Vec<usize>>> = std::collections::HashMap::new();

        // Set input shape
        for input_node in graph.get_input_nodes() {
            shapes.insert(input_node.id, vec![input_shape.to_vec()]);
        }

        for node_id in order {
            let node = graph.get_node(&node_id).unwrap();
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
