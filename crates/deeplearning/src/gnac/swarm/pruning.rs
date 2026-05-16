use crate::gnac::canvas::NeuralGraph;
use crate::gnac::swarm::SwarmConfig;
use std::collections::HashSet;
use uuid::Uuid;

/// Evolutionary Graph Pruning
/// Arsitektur dipangkas berdasarkan: redundancy, marginal contribution, entropy
pub struct GraphPruner;

impl GraphPruner {
    /// Prune graf berdasarkan redundancy score
    pub fn prune(graph: &mut NeuralGraph, config: &SwarmConfig) {
        let mut to_remove = HashSet::new();

        // Prune node dengan kontribusi marginal rendah
        for (id, node) in &graph.nodes {
            let edge_count = graph.edges.values()
                .filter(|e| e.source_node == *id || e.target_node == *id)
                .count();
            if edge_count == 0 && node.node_type != crate::NodeType::Input && node.node_type != crate::NodeType::Output {
                to_remove.insert(*id);
            }
        }

        // Prune berdasarkan entropy
        for (id, edge) in &graph.edges {
            if edge.entropy_score < 0.01 {
                // Edge dengan entropy hampir 0: aktivasi dead
                // Tandai node output-nya
                to_remove.insert(edge.target_node);
            }
        }

        for id in to_remove {
            graph.remove_node(&id);
        }
    }

    /// Hitung redundancy score untuk sebuah node
    pub fn redundancy_score(graph: &NeuralGraph, node_id: &Uuid) -> f32 {
        let node = match graph.nodes.get(node_id) {
            Some(n) => n,
            None => return 1.0,
        };

        let mut score = 0.0;

        // Semakin banyak edge keluar, semakin penting node tsb
        let outgoing = graph.edges.values().filter(|e| e.source_node == *node_id).count();
        let incoming = graph.edges.values().filter(|e| e.target_node == *node_id).count();

        score -= outgoing as f32 * 0.2;
        score -= incoming as f32 * 0.1;

        // Node dengan tipe berbeda mendapatkan skor lebih rendah (lebih penting)
        let unique_types: HashSet<_> = graph.nodes.values().map(|n| &n.node_type).collect();
        if unique_types.len() == graph.nodes.len() {
            // Semua unik, semua penting
            score -= 0.1;
        }

        // Flops: node mahal lebih mungkin redundan
        score += (node.metadata.flops as f32 / 1_000_000.0) * 0.01;

        score
    }
}
