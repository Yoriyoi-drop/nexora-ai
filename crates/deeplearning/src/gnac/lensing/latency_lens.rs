use crate::gnac::canvas::NeuralGraph;
use crate::gnac::lensing::{LensObservation, LensType, NeuralLens, ObservationSeverity};

/// Latency Lens — menyorot bottleneck latensi
pub struct LatencyLens;

impl NeuralLens for LatencyLens {
    fn lens_type(&self) -> LensType {
        LensType::Latency
    }

    fn observe(&self, graph: &NeuralGraph) -> LensObservation {
        let mut high_latency_nodes = Vec::new();

        let total_flops = graph.total_flops();
        let threshold = total_flops as f64 / graph.nodes.len().max(1) as f64 * 2.0;

        for node in graph.nodes.values() {
            if node.metadata.flops as f64 > threshold {
                high_latency_nodes.push(node.id);
            }
        }

        let severity = if !high_latency_nodes.is_empty() {
            ObservationSeverity::Warning
        } else {
            ObservationSeverity::Info
        };

        let summary = if !high_latency_nodes.is_empty() {
            format!(
                "{} node(s) with above-average FLOPs detected. Consider optimization: operator fusion, pruning, or precision reduction.",
                high_latency_nodes.len()
            )
        } else {
            format!(
                "Total graph FLOPs: {}. No latency bottlenecks detected.",
                total_flops
            )
        };

        LensObservation {
            lens_type: LensType::Latency,
            highlighted_nodes: high_latency_nodes,
            highlighted_edges: vec![],
            summary,
            severity,
        }
    }

    fn name(&self) -> &str {
        "Latency Lens"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_lens_name() {
        assert_eq!(LatencyLens.name(), "Latency Lens");
    }

    #[test]
    fn test_latency_lens_empty() {
        let g = NeuralGraph::new("empty");
        let obs = LatencyLens.observe(&g);
        assert_eq!(obs.severity, ObservationSeverity::Info);
        assert!(obs.summary.contains("bottlenecks") || obs.summary.contains("Total graph FLOPs"));
    }
}
