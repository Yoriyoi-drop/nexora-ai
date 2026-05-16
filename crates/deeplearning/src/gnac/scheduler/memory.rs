use crate::gnac::canvas::NeuralGraph;
use std::collections::HashMap;
use uuid::Uuid;

/// Memory checkpointing — trade VRAM untuk komputasi ulang
pub struct MemoryCheckpointer {
    checkpoint_nodes: Vec<Uuid>,
    checkpointed_activations: HashMap<Uuid, Vec<u8>>,
}

impl MemoryCheckpointer {
    pub fn new() -> Self {
        MemoryCheckpointer {
            checkpoint_nodes: Vec::new(),
            checkpointed_activations: HashMap::new(),
        }
    }

    /// Tentukan node checkpoint berdasarkan frekuensi
    pub fn select_checkpoints(&mut self, graph: &NeuralGraph, frequency: usize) {
        let order = match graph.topological_order() {
            Ok(o) => o,
            Err(_) => return,
        };

        self.checkpoint_nodes.clear();
        for (i, node_id) in order.iter().enumerate() {
            if i % frequency == 0 && i > 0 {
                self.checkpoint_nodes.push(*node_id);
            }
        }
    }

    /// Simpan aktivasi di checkpoint
    pub fn save_activation(&mut self, node_id: Uuid, activation: Vec<u8>) {
        self.checkpointed_activations.insert(node_id, activation);
    }

    /// Muat aktivasi dari checkpoint
    pub fn load_activation(&self, node_id: &Uuid) -> Option<&Vec<u8>> {
        self.checkpointed_activations.get(node_id)
    }

    /// Estimasi VRAM yang dihemat
    pub fn estimated_savings(&self) -> usize {
        self.checkpoint_nodes.len() * 1024 * 1024 // 1MB per checkpoint
    }
}
