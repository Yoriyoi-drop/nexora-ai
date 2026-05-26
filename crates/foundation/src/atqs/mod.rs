//! Re-export the ATQS crate plus foundation-specific integrations.

pub use nexora_atqs::*;

use crate::shared::base_model::{InputData, NxrInput, NxrModel, NxrOutput, OutputData};
pub use nexora_models::swift::NxrSwiftModel;

/// Enhanced ATQS with NXR-SWIFT integration
pub struct AtqsSwiftIntegration {
    pub atqs_compression: nexora_atqs::compression::AtqsCompression,
    pub swift_model: NxrSwiftModel,
    pub integration_config: AtqsSwiftConfig,
}

#[derive(Debug, Clone)]
pub struct AtqsSwiftConfig {
    pub enable_edge_optimization: bool,
    pub target_latency_ms: u32,
    pub edge_compression_level: u8,
}

impl Default for AtqsSwiftConfig {
    fn default() -> Self {
        Self {
            enable_edge_optimization: true,
            target_latency_ms: 1,
            edge_compression_level: 4,
        }
    }
}

impl AtqsSwiftIntegration {
    pub fn new() -> Self {
        Self {
            atqs_compression: nexora_atqs::compression::AtqsCompression::new(),
            swift_model: NxrSwiftModel::new(),
            integration_config: AtqsSwiftConfig::default(),
        }
    }

    pub async fn edge_optimized_compression(
        &self,
        data: &[u8],
    ) -> Result<EdgeOptimizedCompression, Box<dyn std::error::Error>> {
        let mut result = EdgeOptimizedCompression::new();

        if let Ok(atqs_result) = self.atqs_compression.compress(data).await {
            result.atqs_compression = Some(atqs_result);
        }

        if self.integration_config.enable_edge_optimization {
            let safe_len = data.len().to_string();
            let sanitized_input = format!("Optimize {} bytes for edge deployment", safe_len);
            let swift_input = NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: InputData::Text(sanitized_input),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };

            if let Ok(swift_result) = self.swift_model.infer(&swift_input).await {
                result.swift_optimization = Some(swift_result);
            }
        }

        result.combine_results(&self.integration_config);

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct EdgeOptimizedCompression {
    pub atqs_compression: Option<nexora_atqs::compression::CompressionResult>,
    pub swift_optimization: Option<NxrOutput>,
    pub combined_insights: Vec<String>,
    pub edge_recommendations: Vec<String>,
}

impl EdgeOptimizedCompression {
    pub fn new() -> Self {
        Self {
            atqs_compression: None,
            swift_optimization: None,
            combined_insights: Vec::new(),
            edge_recommendations: Vec::new(),
        }
    }

    fn combine_results(&mut self, config: &AtqsSwiftConfig) {
        if let Some(atqs) = &self.atqs_compression {
            self.combined_insights.push(format!(
                "ATQS Compression: {:.2}% reduction",
                atqs.compression_ratio
            ));
        }

        if let Some(swift) = &self.swift_optimization {
            if let OutputData::Text(text) = &swift.data {
                self.combined_insights
                    .push(format!("Edge Optimization: {}", text));

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                    if let Some(is_edge) = json.get("edge_optimized").and_then(|v| v.as_bool()) {
                        if is_edge {
                            self.edge_recommendations.push(
                                "Deploy with 4-bit quantization for maximum efficiency".to_string(),
                            );
                        }
                    }
                    if let Some(latency) = json.get("latency_ms").and_then(|v| v.as_f64()) {
                        if latency <= 1.0 {
                            self.edge_recommendations.push(
                                "Sub-millisecond latency achieved for real-time processing"
                                    .to_string(),
                            );
                        }
                    }
                } else {
                    if text.to_lowercase().contains("edge")
                        && text.to_lowercase().contains("optimized")
                    {
                        self.edge_recommendations.push(
                            "Deploy with 4-bit quantization for maximum efficiency".to_string(),
                        );
                    }
                    if text.to_lowercase().contains("latency") {
                        self.edge_recommendations.push(
                            "Latency optimization recommended for edge deployment".to_string(),
                        );
                    }
                }
            }
        }

        if config.target_latency_ms <= 1 {
            self.edge_recommendations
                .push("Ultra-low latency configuration optimized for edge devices".to_string());
        }
        if config.edge_compression_level <= 4 {
            self.edge_recommendations
                .push("High compression ratio maintained for edge deployment".to_string());
        }
    }

    pub fn summary(&self) -> String {
        let mut summary = String::new();

        if !self.combined_insights.is_empty() {
            summary.push_str("Combined Edge Insights:\n");
            for insight in &self.combined_insights {
                summary.push_str(&format!("- {}\n", insight));
            }
        }

        if !self.edge_recommendations.is_empty() {
            summary.push_str("\nEdge Deployment Recommendations:\n");
            for rec in &self.edge_recommendations {
                summary.push_str(&format!("- {}\n", rec));
            }
        }

        summary
    }
}

impl Default for AtqsSwiftIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atqs_swift_config_default() {
        let config = AtqsSwiftConfig::default();
        assert!(config.enable_edge_optimization);
        assert_eq!(config.target_latency_ms, 1);
        assert_eq!(config.edge_compression_level, 4);
    }

    #[test]
    fn test_edge_optimized_compression_new() {
        let result = EdgeOptimizedCompression::new();
        assert!(result.atqs_compression.is_none());
        assert!(result.swift_optimization.is_none());
        assert!(result.combined_insights.is_empty());
        assert!(result.edge_recommendations.is_empty());
    }

    #[test]
    fn test_edge_optimized_compression_summary_empty() {
        let result = EdgeOptimizedCompression::new();
        assert_eq!(result.summary(), "");
    }

    #[test]
    fn test_edge_optimized_compression_summary_with_recommendations() {
        let mut result = EdgeOptimizedCompression::new();
        result.edge_recommendations.push("test rec".to_string());
        let summary = result.summary();
        assert!(summary.contains("test rec"));
        assert!(summary.contains("Edge Deployment Recommendations"));
    }

    #[test]
    fn test_combine_results_with_config_only() {
        let mut result = EdgeOptimizedCompression::new();
        let config = AtqsSwiftConfig::default();
        result.combine_results(&config);
        assert!(result.edge_recommendations.len() >= 2);
        assert!(result.edge_recommendations[0].contains("4-bit quantization"));
    }

    #[test]
    fn test_combine_results_with_custom_config() {
        let mut result = EdgeOptimizedCompression::new();
        let config = AtqsSwiftConfig {
            enable_edge_optimization: false,
            target_latency_ms: 10,
            edge_compression_level: 6,
        };
        result.combine_results(&config);
        // With target_latency > 1 and compression_level > 4, no recommendations added
        assert!(result.edge_recommendations.is_empty());
    }

    #[test]
    fn test_atqs_swift_config_debug() {
        let config = AtqsSwiftConfig::default();
        let _debug = format!("{:?}", config);
    }

    #[test]
    fn test_edge_optimized_compression_debug() {
        let result = EdgeOptimizedCompression::new();
        let _debug = format!("{:?}", result);
    }

    #[test]
    fn test_atqs_swift_config_custom_values() {
        let config = AtqsSwiftConfig {
            enable_edge_optimization: false,
            target_latency_ms: 100,
            edge_compression_level: 8,
        };
        assert!(!config.enable_edge_optimization);
        assert_eq!(config.target_latency_ms, 100);
    }
}
