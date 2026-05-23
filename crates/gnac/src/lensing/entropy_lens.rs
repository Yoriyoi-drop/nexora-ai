use crate::canvas::NeuralGraph;
use crate::lensing::{LensObservation, LensType, NeuralLens, ObservationSeverity};

/// Activation Entropy Lens — menyorot entropi aktivasi
pub struct ActivationEntropyLens;

impl NeuralLens for ActivationEntropyLens {
    fn lens_type(&self) -> LensType {
        LensType::ActivationEntropy
    }

    fn observe(&self, graph: &NeuralGraph) -> LensObservation {
        let mut low_entropy_nodes = Vec::new(); // aktivasi mati/saturasi
        let mut high_entropy_nodes = Vec::new(); // aktivasi terlalu acak

        for (id, edge) in &graph.edges {
            let entropy = edge.entropy_score;
            if entropy < 0.1 {
                low_entropy_nodes.push(*id);
            } else if entropy > 4.0 {
                high_entropy_nodes.push(*id);
            }
        }

        let severity = if !low_entropy_nodes.is_empty() {
            ObservationSeverity::Warning
        } else if !high_entropy_nodes.is_empty() {
            ObservationSeverity::Warning
        } else {
            ObservationSeverity::Info
        };

        let mut summaries = Vec::new();
        if !low_entropy_nodes.is_empty() {
            summaries.push(format!(
                "{} edge(s) with low entropy (dead/saturated activations)",
                low_entropy_nodes.len()
            ));
        }
        if !high_entropy_nodes.is_empty() {
            summaries.push(format!(
                "{} edge(s) with high entropy (uncertain predictions)",
                high_entropy_nodes.len()
            ));
        }
        if summaries.is_empty() {
            summaries.push("Activation entropy within normal range.".to_string());
        }

        LensObservation {
            lens_type: LensType::ActivationEntropy,
            highlighted_nodes: vec![],
            highlighted_edges: low_entropy_nodes
                .into_iter()
                .chain(high_entropy_nodes)
                .collect(),
            summary: summaries.join(". "),
            severity,
        }
    }

    fn name(&self) -> &str {
        "Activation Entropy Lens"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_lens_name() {
        assert_eq!(ActivationEntropyLens.name(), "Activation Entropy Lens");
    }

    #[test]
    fn test_entropy_lens_empty() {
        let g = NeuralGraph::new("empty");
        let obs = ActivationEntropyLens.observe(&g);
        assert!(obs.summary.contains("normal range"));
        assert_eq!(obs.severity, ObservationSeverity::Info);
    }
}
