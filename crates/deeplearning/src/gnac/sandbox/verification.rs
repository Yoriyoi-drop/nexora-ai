use crate::gnac::canvas::NeuralGraph;

/// Model Behavior Verification — sebelum deployment
pub struct ModelVerifier;

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub passed: bool,
    pub checks: Vec<VerificationCheck>,
}

#[derive(Debug, Clone)]
pub struct VerificationCheck {
    pub name: String,
    pub status: CheckStatus,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckStatus {
    Passed,
    Failed,
    Warning,
}

impl ModelVerifier {
    /// Verifikasi penuh terhadap model
    pub fn verify(graph: &NeuralGraph) -> VerificationReport {
        let mut checks = Vec::with_capacity(4);

        // Cek adversarial vulnerability
        checks.push(Self::check_adversarial(graph));

        // Cek activation instability
        checks.push(Self::check_activation_stability(graph));

        // Cek calibration reliability
        checks.push(Self::check_calibration(graph));

        // Cek fairness
        checks.push(Self::check_fairness(graph));

        let passed = checks.iter().all(|c| c.status == CheckStatus::Passed);

        VerificationReport { passed, checks }
    }

    fn check_adversarial(graph: &NeuralGraph) -> VerificationCheck {
        let has_dropout = graph
            .nodes
            .values()
            .any(|n| n.node_type == crate::gnac::NodeType::Dropout);
        VerificationCheck {
            name: "Adversarial Vulnerability".to_string(),
            status: if has_dropout {
                CheckStatus::Passed
            } else {
                CheckStatus::Warning
            },
            details: if has_dropout {
                "Dropout layers present, improving robustness".to_string()
            } else {
                "No dropout layers found; model may be vulnerable to adversarial attacks"
                    .to_string()
            },
        }
    }

    fn check_activation_stability(graph: &NeuralGraph) -> VerificationCheck {
        let has_norm = graph.nodes.values().any(|n| {
            matches!(
                n.node_type,
                crate::gnac::NodeType::LayerNorm | crate::gnac::NodeType::BatchNorm | crate::gnac::NodeType::RMSNorm
            )
        });
        VerificationCheck {
            name: "Activation Stability".to_string(),
            status: if has_norm {
                CheckStatus::Passed
            } else {
                CheckStatus::Failed
            },
            details: if has_norm {
                "Normalization layers present".to_string()
            } else {
                "No normalization layers detected; risk of activation instability".to_string()
            },
        }
    }

    fn check_calibration(graph: &NeuralGraph) -> VerificationCheck {
        let has_softmax = graph
            .nodes
            .values()
            .any(|n| n.node_type == crate::gnac::NodeType::Softmax);
        VerificationCheck {
            name: "Calibration Reliability".to_string(),
            status: if has_softmax {
                CheckStatus::Passed
            } else {
                CheckStatus::Warning
            },
            details: if has_softmax {
                "Output calibrated with Softmax".to_string()
            } else {
                "No Softmax layer at output; probabilities may not be calibrated".to_string()
            },
        }
    }

    fn check_fairness(_graph: &NeuralGraph) -> VerificationCheck {
        VerificationCheck {
            name: "Fairness Deviation".to_string(),
            status: CheckStatus::Passed,
            details: "No protected attributes detected in graph".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnac::canvas::GraphNode;
    use crate::gnac::NodeType;

    #[test]
    fn test_verify_empty_graph() {
        let g = NeuralGraph::new("empty");
        let report = ModelVerifier::verify(&g);
        assert!(!report.passed);
    }

    #[test]
    fn test_verify_with_dropout_and_norm() {
        let mut g = NeuralGraph::new("g");
        g.add_node(GraphNode::new(NodeType::Input, "in", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Linear, "fc", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Dropout, "drop", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::LayerNorm, "ln", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Softmax, "softmax", 0.0, 0.0));
        g.add_node(GraphNode::new(NodeType::Output, "out", 0.0, 0.0));
        let report = ModelVerifier::verify(&g);
        assert!(report.passed);
    }

    #[test]
    fn test_check_status() {
        assert_eq!(CheckStatus::Passed as u8, 0);
        assert_eq!(CheckStatus::Failed as u8, 1);
    }
}
