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
        let to_remove: Vec<_> = order.iter()
            .filter(|id| {
                let is_io = input_id.map_or(false, |i| i == **id) ||
                           output_id.map_or(false, |o| o == **id);
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
    pub fn increase_depth(graph: &mut NeuralGraph) {
        log::info!("Increasing graph depth for complex input");
        // Dalam implementasi nyata, clone & insert layer tambahan
    }
}
