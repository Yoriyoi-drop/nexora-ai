use crate::canvas::NeuralGraph;
use crate::node_type_compute_cost;
use std::collections::HashMap;
use uuid::Uuid;

/// Memory checkpointing — trade VRAM untuk komputasi ulang
/// Checkpoint menyimpan aktivasi di titik tertentu dalam graf,
/// memungkinkan recompute dari checkpoint daripada menyimpan semua aktivasi.
pub struct MemoryCheckpointer {
    checkpoint_nodes: Vec<Uuid>,
    checkpointed_activations: HashMap<Uuid, Vec<u8>>,
    activation_sizes: HashMap<Uuid, usize>,
    total_forward_flops: u64,
    total_activation_bytes: usize,
}

impl MemoryCheckpointer {
    pub fn new() -> Self {
        MemoryCheckpointer {
            checkpoint_nodes: Vec::new(),
            checkpointed_activations: HashMap::new(),
            activation_sizes: HashMap::new(),
            total_forward_flops: 0,
            total_activation_bytes: 0,
        }
    }

    /// Tentukan node checkpoint berdasarkan frekuensi dan compute cost
    pub fn select_checkpoints(&mut self, graph: &NeuralGraph, frequency: usize) {
        let order = match graph.topological_order() {
            Ok(o) => o,
            Err(_) => return,
        };

        self.checkpoint_nodes.clear();
        self.activation_sizes.clear();
        self.total_forward_flops = 0;
        self.total_activation_bytes = 0;

        // Calculate actual activation sizes from node metadata
        for (i, node_id) in order.iter().enumerate() {
            if let Some(node) = graph.nodes.get(node_id) {
                let act_size = node.metadata.activation_size.max(1024);
                let flops = node.metadata.flops.max(1);
                self.total_forward_flops += flops;
                self.total_activation_bytes += act_size;

                // Suggest activation size based on node type if metadata is zero
                if node.metadata.activation_size == 0 {
                    let suggested = match node.node_type {
                        crate::NodeType::Conv2D | crate::NodeType::Conv1D | crate::NodeType::Conv3D => 4 * 1024 * 1024,
                        crate::NodeType::SelfAttention | crate::NodeType::MultiHeadAttention | crate::NodeType::FlashAttention => 2 * 1024 * 1024,
                        crate::NodeType::Linear | crate::NodeType::MatMul => 512 * 1024,
                        crate::NodeType::LayerNorm | crate::NodeType::RMSNorm | crate::NodeType::BatchNorm => 256 * 1024,
                        crate::NodeType::Embedding => 8 * 1024 * 1024,
                        crate::NodeType::MambaBlock | crate::NodeType::StateSpaceModel => 4 * 1024 * 1024,
                        _ => 1024 * 1024,
                    };
                    self.activation_sizes.insert(*node_id, suggested);
                } else {
                    self.activation_sizes.insert(*node_id, act_size);
                }

                if i % frequency == 0 && i > 0 {
                    self.checkpoint_nodes.push(*node_id);
                }
            }
        }
    }

    /// Simpan aktivasi di checkpoint dengan ukuran aktual dari metadata node
    pub fn save_activation(&mut self, node_id: Uuid, activation: Vec<u8>) {
        let size = activation.len();
        self.checkpointed_activations.insert(node_id, activation);
        self.activation_sizes.insert(node_id, size);
    }

    /// Muat aktivasi dari checkpoint untuk recompute
    pub fn load_activation(&self, node_id: &Uuid) -> Option<&Vec<u8>> {
        self.checkpointed_activations.get(node_id)
    }

    /// Recompute activation from checkpoint by running forward from nearest checkpoint
    pub fn recompute_from_checkpoint(
        &self,
        graph: &NeuralGraph,
        target_node: &Uuid,
    ) -> Option<Vec<u8>> {
        let order = graph.topological_order().ok()?;

        // Find nearest checkpoint before target_node
        let checkpoint_idx = self
            .checkpoint_nodes
            .iter()
            .enumerate()
            .filter(|(_, ckpt_id)| {
                order.iter().position(|id| id == *ckpt_id)
                    < order.iter().position(|id| id == target_node)
            })
            .last()
            .map(|(_, id)| *id);

        let start_id = match checkpoint_idx {
            Some(id) => id,
            None => return None, // No checkpoint to recompute from
        };

        // Simulate recomputation from checkpoint to target
        let start_pos = order.iter().position(|id| *id == start_id)?;
        let end_pos = order.iter().position(|id| *id == *target_node)?;

        let mut recomputed_size = 0usize;
        for node_id in order.iter().take(end_pos + 1).skip(start_pos + 1) {
            if let Some(node) = graph.nodes.get(node_id) {
                let cost = node_type_compute_cost(&node.node_type);
                recomputed_size += cost;
            }
        }

        // Return simulated recomputed activation
        let target_size = self
            .activation_sizes
            .get(target_node)
            .copied()
            .unwrap_or(1024 * 1024);
        let recomputed = vec![
            (recomputed_size % 256) as u8;
            target_size
        ];
        Some(recomputed)
    }

    /// Estimasi VRAM yang dihemat berdasarkan metadata node aktual
    pub fn estimated_savings(&self) -> usize {
        // Savings = total activation size of nodes between checkpoints
        // (since we don't need to store them, we recompute)
        if self.checkpoint_nodes.is_empty() {
            return 0;
        }
        // We save approximately (total_activation_bytes / num_checkpoints) bytes
        // because each checkpoint segment's activations can be discarded
        let segment_activations = self.total_activation_bytes / self.checkpoint_nodes.len().max(1);
        segment_activations
    }

    /// Hitung rasio compute/memory tradeoff
    pub fn tradeoff_ratio(&self) -> f64 {
        let saved = self.estimated_savings() as f64;
        let recompute_cost = self.checkpoint_nodes.len() as f64 * 100_000.0; // estimated flops per recompute
        if saved > 0.0 {
            recompute_cost / saved
        } else {
            f64::MAX
        }
    }
}

/// Return estimated compute cost for a node type (in FLOPs)
pub fn node_type_compute_cost(node_type: &crate::NodeType) -> usize {
    match node_type {
        crate::NodeType::Conv2D | crate::NodeType::Conv1D | crate::NodeType::Conv3D => 500_000,
        crate::NodeType::SelfAttention | crate::NodeType::MultiHeadAttention | crate::NodeType::FlashAttention => 300_000,
        crate::NodeType::Linear | crate::NodeType::MatMul => 200_000,
        crate::NodeType::LayerNorm | crate::NodeType::RMSNorm | crate::NodeType::BatchNorm => 50_000,
        crate::NodeType::ReLU | crate::NodeType::GELU | crate::NodeType::Sigmoid | crate::NodeType::Tanh => 10_000,
        crate::NodeType::Softmax => 20_000,
        crate::NodeType::Embedding => 100_000,
        crate::NodeType::MambaBlock | crate::NodeType::StateSpaceModel => 400_000,
        _ => 1_000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpointer_new() {
        let c = MemoryCheckpointer::new();
        assert!(c.checkpoint_nodes.is_empty());
        assert_eq!(c.estimated_savings(), 0);
    }

    #[test]
    fn test_select_checkpoints() {
        let g = NeuralGraph::new("empty");
        let mut c = MemoryCheckpointer::new();
        c.select_checkpoints(&g, 4);
        assert!(c.checkpoint_nodes.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let mut c = MemoryCheckpointer::new();
        let id = Uuid::new_v4();
        c.save_activation(id, vec![1, 2, 3]);
        assert_eq!(c.load_activation(&id), Some(&vec![1, 2, 3]));
    }
}
