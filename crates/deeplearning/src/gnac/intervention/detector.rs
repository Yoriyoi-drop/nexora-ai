use crate::gnac::canvas::{NeuralGraph, GradientStatus, ActivationStats};
use crate::gnac::intervention::assistant::InterventionAdvice;

/// Deteksi anomali pada graf neural
pub struct AnomalyDetector;

#[derive(Debug, Clone)]
pub struct DetectedAnomaly {
    pub anomaly_type: AnomalyType,
    pub node_id: uuid::Uuid,
    pub severity: f32,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    ExplodingGradient,
    VanishingGradient,
    DeadActivation,
    UnstableAttention,
    ModeCollapse,
    SaturatedActivation,
}

impl AnomalyDetector {
    /// Scan seluruh graf untuk anomali
    pub fn scan(graph: &NeuralGraph) -> Vec<DetectedAnomaly> {
        let mut anomalies = Vec::new();

        for edge in graph.edges.values() {
            match edge.gradient {
                GradientStatus::Exploding(norm) => {
                    anomalies.push(DetectedAnomaly {
                        anomaly_type: AnomalyType::ExplodingGradient,
                        node_id: edge.target_node,
                        severity: (norm / 100.0).min(1.0),
                        description: format!("Exploding gradient (norm={:.2}) detected at edge {}", norm, edge.id),
                    });
                }
                GradientStatus::Vanishing(_) => {
                    anomalies.push(DetectedAnomaly {
                        anomaly_type: AnomalyType::VanishingGradient,
                        node_id: edge.target_node,
                        severity: 0.7,
                        description: format!("Vanishing gradient detected at edge {}", edge.id),
                    });
                }
                _ => {}
            }

            // Dead activation
            if edge.activation_distribution.sparsity > 0.95 {
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::DeadActivation,
                    node_id: edge.target_node,
                    severity: 0.8,
                    description: format!("Dead activation (sparsity={:.2}) at edge {}", edge.activation_distribution.sparsity, edge.id),
                });
            }

            // Saturated activation
            if edge.activation_distribution.std < 0.01 && edge.activation_distribution.mean.abs() > 2.0 {
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::SaturatedActivation,
                    node_id: edge.target_node,
                    severity: 0.6,
                    description: format!("Saturated activation (mean={:.2}, std={:.4})", edge.activation_distribution.mean, edge.activation_distribution.std),
                });
            }
        }

        anomalies
    }
}
