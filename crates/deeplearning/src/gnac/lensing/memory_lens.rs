use crate::gnac::canvas::NeuralGraph;
use crate::gnac::lensing::{NeuralLens, LensType, LensObservation, ObservationSeverity};

/// Memory Lens — menyorot konsumsi memori
pub struct MemoryLens;

impl NeuralLens for MemoryLens {
    fn lens_type(&self) -> LensType {
        LensType::Memory
    }

    fn observe(&self, graph: &NeuralGraph) -> LensObservation {
        let mut peak_memory_nodes = Vec::new();
        let mut total_activation_memory: usize = 0;

        for node in graph.nodes.values() {
            total_activation_memory += node.metadata.activation_size;
        }

        let threshold = total_activation_memory / graph.nodes.len().max(1) * 2;
        for node in graph.nodes.values() {
            if node.metadata.activation_size > threshold {
                peak_memory_nodes.push(node.id);
            }
        }

        let severity = if !peak_memory_nodes.is_empty() {
            ObservationSeverity::Warning
        } else {
            ObservationSeverity::Info
        };

        let summary = if !peak_memory_nodes.is_empty() {
            format!(
                "{} node(s) with high memory footprint. Total activation memory: {} MB. Consider gradient checkpointing or memory-efficient attention.",
                peak_memory_nodes.len(),
                total_activation_memory / 1_000_000
            )
        } else {
            format!("Total activation memory: {} MB. No memory bottlenecks.", total_activation_memory / 1_000_000)
        };

        LensObservation {
            lens_type: LensType::Memory,
            highlighted_nodes: peak_memory_nodes,
            highlighted_edges: vec![],
            summary,
            severity,
        }
    }

    fn name(&self) -> &str {
        "Memory Lens"
    }
}
