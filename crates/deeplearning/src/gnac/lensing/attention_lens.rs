use crate::gnac::canvas::NeuralGraph;
use crate::gnac::lensing::{NeuralLens, LensType, LensObservation, ObservationSeverity};
use crate::NodeType;

/// Attention Flow Lens — menyorot distribusi attention
pub struct AttentionFlowLens;

impl NeuralLens for AttentionFlowLens {
    fn lens_type(&self) -> LensType {
        LensType::AttentionFlow
    }

    fn observe(&self, graph: &NeuralGraph) -> LensObservation {
        let attention_nodes: Vec<_> = graph.nodes.values()
            .filter(|n| matches!(n.node_type, NodeType::SelfAttention | NodeType::MultiHeadAttention | NodeType::CrossAttention))
            .map(|n| n.id)
            .collect();

        let highlighted_edges: Vec<_> = graph.edges.values()
            .filter(|e| attention_nodes.contains(&e.target_node) || attention_nodes.contains(&e.source_node))
            .map(|e| e.id)
            .collect();

        let severity = if attention_nodes.is_empty() {
            ObservationSeverity::Info
        } else {
            ObservationSeverity::Info
        };

        let summary = if attention_nodes.is_empty() {
            "No attention nodes in graph.".to_string()
        } else {
            format!(
                "Found {} attention mechanism(s). Attention entropy and head importance available for inspection.",
                attention_nodes.len()
            )
        };

        LensObservation {
            lens_type: LensType::AttentionFlow,
            highlighted_nodes: attention_nodes,
            highlighted_edges,
            summary,
            severity,
        }
    }

    fn name(&self) -> &str {
        "Attention Flow Lens"
    }
}
