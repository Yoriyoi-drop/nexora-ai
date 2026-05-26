//! Guided Intervention System
//!
//! Ketika SmartTensor mendeteksi anomali (exploding gradient, dead activation,
//! unstable attention, mode collapse), Diagnostic Assistant aktif otomatis.
//! Menjelaskan masalah dalam bahasa natural dan menawarkan auto-fix, guided tuning,
//! atau kontrol manual penuh.

pub mod assistant;
pub mod detector;

pub use assistant::*;
pub use detector::*;

#[cfg(test)]
mod tests {
    use super::assistant::*;
    use super::detector::*;

    #[test]
    fn test_anomaly_type_debug() {
        let t = AnomalyType::ExplodingGradient;
        assert!(!format!("{:?}", t).is_empty());
    }

    #[test]
    fn test_diagnostic_assistant_exploding() {
        let anomaly = DetectedAnomaly {
            anomaly_type: AnomalyType::ExplodingGradient,
            node_id: uuid::Uuid::new_v4(),
            severity: 0.9,
            description: "test".to_string(),
        };
        let advice = DiagnosticAssistant::analyze(&anomaly);
        assert!(advice.explanation.contains("Gradients have grown"));
        assert_eq!(
            advice.auto_fix,
            Some("Apply gradient clipping (max_norm=1.0)".to_string())
        );
        assert_eq!(advice.guided_tuning.len(), 2);
    }

    #[test]
    fn test_diagnostic_assistant_vanishing() {
        let anomaly = DetectedAnomaly {
            anomaly_type: AnomalyType::VanishingGradient,
            node_id: uuid::Uuid::new_v4(),
            severity: 0.7,
            description: "test".to_string(),
        };
        let advice = DiagnosticAssistant::analyze(&anomaly);
        assert!(advice.explanation.contains("Gradients are becoming"));
        assert_eq!(advice.auto_fix, None);
    }
}
