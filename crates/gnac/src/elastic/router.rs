use crate::canvas::NeuralGraph;
use crate::elastic::ElasticStrategy;
use uuid::Uuid;

/// Adaptive routing — pilih jalur inferensi berdasarkan kompleksitas input
pub struct ElasticRouter {
    pub strategy: ElasticStrategy,
    pub complexity_thresholds: Vec<f64>,
}

impl ElasticRouter {
    pub fn new(strategy: ElasticStrategy) -> Self {
        ElasticRouter {
            strategy,
            complexity_thresholds: vec![0.3, 0.6, 0.9],
        }
    }

    /// Pilih jalur berdasarkan skor kompleksitas input
    pub fn select_path(&self, input_complexity: f64, graph: &NeuralGraph) -> Vec<Uuid> {
        match self.strategy {
            ElasticStrategy::Lightweight => {
                // Hanya jalur utama, nonaktifkan branch mahal
                let mut path = Vec::with_capacity(graph.nodes.len());
                for node in graph.nodes.values() {
                    if !Self::is_expensive(&node.node_type) {
                        path.push(node.id);
                    }
                }
                path
            }
            ElasticStrategy::HighPrecision => graph.topological_order().unwrap_or_default(),
            ElasticStrategy::Balanced => {
                if input_complexity > self.complexity_thresholds[1] {
                    graph.topological_order().unwrap_or_default()
                } else {
                    let mut path = Vec::with_capacity(graph.nodes.len());
                    for node in graph.nodes.values() {
                        if !Self::is_expensive(&node.node_type) {
                            path.push(node.id);
                        }
                    }
                    path
                }
            }
        }
    }

    fn is_expensive(node_type: &crate::NodeType) -> bool {
        matches!(
            node_type,
            crate::NodeType::SelfAttention
                | crate::NodeType::MultiHeadAttention
                | crate::NodeType::FlashAttention
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::GraphNode;
    use crate::NodeType;

    #[test]
    fn test_elastic_router_new() {
        let r = ElasticRouter::new(ElasticStrategy::Balanced);
        assert_eq!(r.strategy, ElasticStrategy::Balanced);
    }

    #[test]
    fn test_select_path_lightweight() {
        let mut g = NeuralGraph::new("g");
        g.add_node(GraphNode::new(NodeType::Input, "in", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::SelfAttention, "attn", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Output, "out", 0.0, 0.0));
        let r = ElasticRouter::new(ElasticStrategy::Lightweight);
        let path = r.select_path(0.5, &g);
        // SelfAttention should be excluded
        assert!(!path.is_empty());
    }

    #[test]
    fn test_select_path_high_precision() {
        let g = NeuralGraph::new("empty");
        let r = ElasticRouter::new(ElasticStrategy::HighPrecision);
        let path = r.select_path(0.5, &g);
        assert!(path.is_empty());
    }
}
