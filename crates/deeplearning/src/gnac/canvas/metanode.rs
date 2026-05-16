use crate::gnac::canvas::{NeuralGraph, GraphNode, CanvasPosition, ZoomLevel};
use crate::gnac::HealthStatus;
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

        let dead_count = self.inner_graph
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
