use crate::gnac::intervention::detector::{DetectedAnomaly, AnomalyType};

/// Saran intervensi dari Diagnostic Assistant
#[derive(Debug, Clone)]
pub struct InterventionAdvice {
    pub anomaly: DetectedAnomaly,
    pub explanation: String,
    pub auto_fix: Option<String>,
    pub guided_tuning: Vec<TuningSuggestion>,
}

#[derive(Debug, Clone)]
pub struct TuningSuggestion {
    pub parameter: String,
    pub current_value: f64,
    pub suggested_value: f64,
    pub description: String,
}

/// Diagnostic Assistant — memberikan saran berbasis anomali
pub struct DiagnosticAssistant;

impl DiagnosticAssistant {
    pub fn analyze(anomaly: &DetectedAnomaly) -> InterventionAdvice {
        let explanation = Self::generate_explanation(anomaly);
        let auto_fix = Self::suggest_auto_fix(anomaly);
        let guided_tuning = Self::suggest_tuning(anomaly);

        InterventionAdvice {
            anomaly: anomaly.clone(),
            explanation,
            auto_fix,
            guided_tuning,
        }
    }

    fn generate_explanation(anomaly: &DetectedAnomaly) -> String {
        match anomaly.anomaly_type {
            AnomalyType::ExplodingGradient => {
                "Gradients have grown very large, causing numerical instability. \
                This often happens when the learning rate is too high or the network \
                depth is too large without proper normalization.".to_string()
            }
            AnomalyType::VanishingGradient => {
                "Gradients are becoming very small, preventing effective learning in \
                earlier layers. Consider using residual connections, different activation \
                functions (e.g., ReLU instead of sigmoid), or layer normalization.".to_string()
            }
            AnomalyType::DeadActivation => {
                "Neurons are producing near-zero outputs (dead activations). This is \
                common with ReLU activations when too many units become permanently inactive. \
                Consider using LeakyReLU, PReLU, or ELU activations.".to_string()
            }
            AnomalyType::UnstableAttention => {
                "Attention distributions are chaotic, indicating the model cannot \
                focus on relevant features. Try adjusting temperature, adding dropout, \
                or using attention normalization.".to_string()
            }
            AnomalyType::ModeCollapse => {
                "The model is collapsing to a single output mode, common in GANs and \
                certain generative architectures. Consider diversity-promoting loss terms \
                or architectural changes.".to_string()
            }
            AnomalyType::SaturatedActivation => {
                "Activations are saturated at extreme values, preventing gradient flow. \
                Consider normalization layers or reducing weight initialization scale.".to_string()
            }
        }
    }

    fn suggest_auto_fix(anomaly: &DetectedAnomaly) -> Option<String> {
        match anomaly.anomaly_type {
            AnomalyType::ExplodingGradient => Some("Apply gradient clipping (max_norm=1.0)".to_string()),
            AnomalyType::DeadActivation => Some("Replace ReLU with LeakyReLU (negative_slope=0.01)".to_string()),
            AnomalyType::SaturatedActivation => Some("Add BatchNormalization or LayerNormalization".to_string()),
            _ => None,
        }
    }

    fn suggest_tuning(anomaly: &DetectedAnomaly) -> Vec<TuningSuggestion> {
        match anomaly.anomaly_type {
            AnomalyType::ExplodingGradient => vec![
                TuningSuggestion {
                    parameter: "learning_rate".to_string(),
                    current_value: 0.001,
                    suggested_value: 0.0003,
                    description: "Reduce learning rate to stabilize gradient updates".to_string(),
                },
                TuningSuggestion {
                    parameter: "gradient_clip_norm".to_string(),
                    current_value: 0.0,
                    suggested_value: 1.0,
                    description: "Enable gradient clipping".to_string(),
                },
            ],
            AnomalyType::VanishingGradient => vec![
                TuningSuggestion {
                    parameter: "init_scale".to_string(),
                    current_value: 0.01,
                    suggested_value: 0.1,
                    description: "Increase weight initialization scale".to_string(),
                },
                TuningSuggestion {
                    parameter: "use_residual".to_string(),
                    current_value: 0.0,
                    suggested_value: 1.0,
                    description: "Add residual connections".to_string(),
                },
            ],
            _ => vec![
                TuningSuggestion {
                    parameter: "learning_rate".to_string(),
                    current_value: 0.001,
                    suggested_value: 0.0005,
                    description: "Slightly reduce learning rate".to_string(),
                },
            ],
        }
    }
}
