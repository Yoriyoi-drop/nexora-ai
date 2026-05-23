use crate::canvas::{CanvasPosition, NeuralGraph};
use crate::HealthStatus;
use uuid::Uuid;

/// MetaNode — subgraph kompleks yang di-collapse menjadi satu node
/// Tetap mempertahankan metadata agregat: FLOPs, health score, latency, params
#[derive(Debug, Clone)]
pub struct MetaNode {
    pub id: Uuid,
    pub name: String,
    pub inner_graph: NeuralGraph,
    pub position: CanvasPosition,
    pub aggregated_flops: u64,
    pub aggregated_params: usize,
    pub health: HealthStatus,
    pub latency_estimate_ms: f64,
}

impl MetaNode {
    pub fn new(name: &str, graph: NeuralGraph, x: f64, y: f64) -> Self {
        let flops = graph.total_flops();
        let params = graph.total_params();

        MetaNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            position: CanvasPosition::new(x, y),
            aggregated_flops: flops,
            aggregated_params: params,
            health: HealthStatus::Healthy,
            latency_estimate_ms: 0.0,
            inner_graph: graph,
        }
    }

    /// Ekstrak isi MetaNode kembali ke graf penuh
    pub fn expand(self) -> NeuralGraph {
        self.inner_graph
    }

    /// Update aggregated stats dari inner graph
    pub fn refresh_stats(&mut self) {
        self.aggregated_flops = self.inner_graph.total_flops();
        self.aggregated_params = self.inner_graph.total_params();

        let dead_count = self
            .inner_graph
            .nodes
            .values()
            .filter(|n| n.health == HealthStatus::Dead)
            .count();

        if dead_count > 0 {
            self.health = HealthStatus::Critical {
                reason: format!("{} dead nodes in subgraph", dead_count),
            };
        } else {
            self.health = HealthStatus::Healthy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::GraphNode;

    #[test]
    fn test_metanode_new() {
        let inner = NeuralGraph::new("inner");
        let meta = MetaNode::new("meta", inner, 10.0, 20.0);
        assert_eq!(meta.name, "meta");
        assert_eq!(meta.position.x, 10.0);
        assert_eq!(meta.aggregated_flops, 0);
    }

    #[test]
    fn test_metanode_expand() {
        let inner = NeuralGraph::new("inner");
        let meta = MetaNode::new("meta", inner, 0.0, 0.0);
        let expanded = meta.expand();
        assert_eq!(expanded.name, "inner");
    }

    #[test]
    fn test_metanode_refresh_stats_healthy() {
        let inner = NeuralGraph::new("inner");
        let mut meta = MetaNode::new("meta", inner, 0.0, 0.0);
        meta.refresh_stats();
        assert_eq!(meta.health, HealthStatus::Healthy);
    }

    #[test]
    fn test_metanode_refresh_stats_dead() {
        let mut inner = NeuralGraph::new("inner");
        let mut node = GraphNode::new(crate::NodeType::Linear, "dead", 0.0, 0.0);
        node.set_health(HealthStatus::Dead);
        inner.add_node(node);
        let mut meta = MetaNode::new("meta", inner, 0.0, 0.0);
        meta.refresh_stats();
        assert!(matches!(meta.health, HealthStatus::Critical { .. }));
    }
}
