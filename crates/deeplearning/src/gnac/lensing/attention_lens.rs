use crate::gnac::canvas::NeuralGraph;
use crate::gnac::lensing::{LensObservation, LensType, NeuralLens, ObservationSeverity};
use crate::gnac::NodeType;

/// Attention Flow Lens — menyorot distribusi attention
pub struct AttentionFlowLens;

impl NeuralLens for AttentionFlowLens {
    fn lens_type(&self) -> LensType {
        LensType::AttentionFlow
    }

    fn observe(&self, graph: &NeuralGraph) -> LensObservation {
        let attention_nodes: Vec<_> = graph
            .nodes
            .values()
            .filter(|n| {
                matches!(
                    n.node_type,
                    NodeType::SelfAttention
                        | NodeType::MultiHeadAttention
                        | NodeType::CrossAttention
                )
            })
            .map(|n| n.id)
            .collect();

        let highlighted_edges: Vec<_> = graph
            .edges
            .values()
            .filter(|e| {
                attention_nodes.contains(&e.target_node) || attention_nodes.contains(&e.source_node)
            })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnac::canvas::GraphNode;
    use crate::gnac::DType;
    use crate::gnac::GraphEdge;
    use crate::gnac::NodeType;
    use crate::gnac::TensorDesc;
    use uuid::Uuid;

    fn graph_with_attention() -> NeuralGraph {
        let mut g = NeuralGraph::new("test");
        let attn = crate::gnac::canvas::GraphNode::new(crate::gnac::NodeType::SelfAttention, "attn", 0.0, 0.0);
        g.add_node(attn);
        g
    }

    #[test]
    fn test_attention_lens_name() {
        let lens = AttentionFlowLens;
        assert_eq!(lens.name(), "Attention Flow Lens");
    }

    #[test]
    fn test_attention_lens_type() {
        let lens = AttentionFlowLens;
        assert_eq!(lens.lens_type(), LensType::AttentionFlow);
    }

    #[test]
    fn test_attention_lens_no_attention() {
        let g = NeuralGraph::new("empty");
        let obs = AttentionFlowLens.observe(&g);
        assert_eq!(obs.severity, ObservationSeverity::Info);
        assert!(obs.summary.contains("No attention"));
    }

    #[test]
    fn test_attention_lens_with_attention() {
        let mut g = NeuralGraph::new("g");
        let attn = GraphNode::new(NodeType::SelfAttention, "attn", 0.0, 0.0);
        g.add_node(attn);
        let obs = AttentionFlowLens.observe(&g);
        assert!(obs.summary.contains("Found"));
    }
}
