use crate::gnac::canvas::{NeuralGraph, GradientStatus};
use crate::gnac::lensing::{NeuralLens, LensType, LensObservation, ObservationSeverity};

/// Gradient Failure Lens — menyorot node dengan gradien tidak stabil
pub struct GradientFailureLens;

impl NeuralLens for GradientFailureLens {
    fn lens_type(&self) -> LensType {
        LensType::GradientFailure
    }

    fn observe(&self, graph: &NeuralGraph) -> LensObservation {
        let mut critical_nodes = Vec::new();
        let mut warning_nodes = Vec::new();
        let mut critical_edges = Vec::new();

        for edge in graph.edges.values() {
            match edge.gradient {
                GradientStatus::Exploding(norm) => {
                    critical_nodes.push(edge.source_node);
                    critical_nodes.push(edge.target_node);
                    critical_edges.push(edge.id);
                }
                GradientStatus::Vanishing(norm) => {
                    warning_nodes.push(edge.source_node);
                    warning_nodes.push(edge.target_node);
                }
                GradientStatus::Saturated => {
                    warning_nodes.push(edge.target_node);
                }
                _ => {}
            }
        }

        let mut highlighted = Vec::new();
        highlighted.extend(critical_nodes.clone());
        highlighted.extend(warning_nodes.clone());
        highlighted.sort();
        highlighted.dedup();

        let severity = if !critical_nodes.is_empty() {
            ObservationSeverity::Critical
        } else if !warning_nodes.is_empty() {
            ObservationSeverity::Warning
        } else {
            ObservationSeverity::Info
        };

        let summary = if !critical_nodes.is_empty() {
            format!(
                "Detected {} node(s) with exploding gradients and {} edge(s) affected. Consider gradient clipping or reducing learning rate.",
                critical_nodes.len(),
                critical_edges.len()
            )
        } else if !warning_nodes.is_empty() {
            format!(
                "Detected {} node(s) with vanishing/saturated gradients. Monitor training stability.",
                warning_nodes.len()
            )
        } else {
            "All gradients stable.".to_string()
        };

        LensObservation {
            lens_type: LensType::GradientFailure,
            highlighted_nodes: highlighted,
            highlighted_edges: critical_edges,
            summary,
            severity,
        }
    }

    fn name(&self) -> &str {
        "Gradient Failure Lens"
    }
}
