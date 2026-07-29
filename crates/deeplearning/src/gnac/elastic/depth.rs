use crate::gnac::canvas::NeuralGraph;

/// Depth adjustment — tambah/kurangi kedalaman model runtime
pub struct DepthController;

impl DepthController {
    /// Kurangi depth dengan menghapus layer terakhir
    pub fn reduce_depth(graph: &mut NeuralGraph, target_depth: usize) {
        let order = graph.topological_order().unwrap_or_default();
        let input_id = graph.get_input_nodes().first().map(|n| n.id);
        let output_id = graph.get_output_nodes().first().map(|n| n.id);

        // Hapus node non-IO jika melebihi target depth
        let to_remove: Vec<_> = order
            .iter()
            .filter(|id| {
                let is_io =
                    input_id.map_or(false, |i| i == **id) || output_id.map_or(false, |o| o == **id);
                !is_io
            })
            .skip(target_depth)
            .copied()
            .collect();

        for id in to_remove {
            graph.remove_node(&id);
        }
    }

    /// Tambah depth (untuk input kompleks)
    pub fn increase_depth(_graph: &mut NeuralGraph) {
        tracing::info!("Increasing graph depth for complex input");
        // Dalam implementasi nyata, clone & insert layer tambahan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnac::canvas::GraphNode;
    use crate::gnac::NodeType;

    #[test]
    fn test_reduce_depth_empty() {
        let mut g = NeuralGraph::new("g");
        DepthController::reduce_depth(&mut g, 0);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_reduce_depth() {
        let mut g = NeuralGraph::new("g");
        g.add_node(GraphNode::new(NodeType::Input, "in", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Linear, "h1", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Linear, "h2", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Output, "out", 0.0, 0.0));
        let before = g.node_count();
        DepthController::reduce_depth(&mut g, 1);
        assert!(g.node_count() < before);
    }
}
