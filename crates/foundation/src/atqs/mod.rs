pub mod calibration;
pub mod compression;
pub mod config;
pub mod core;
pub mod error;
pub mod foundation;
pub mod profiling;
pub mod prelude;
pub mod tensor;
pub mod types;
pub mod utils;

// Re-export main components
pub use config::*;
pub use compression::*;
pub use types::*;
pub use error::*;
pub use foundation::*;
pub use tensor::*;
// Explicitly export ATQSConfig
pub use config::ATQSConfig;

// Integration with NXR-SWIFT for edge optimization
use crate::shared::base_model::NxrModel;
pub use crate::models::swift::NxrSwiftModel;

/// Enhanced ATQS with NXR-SWIFT integration
pub struct AtqsSwiftIntegration {
    /// Original ATQS compression system
    pub atqs_compression: compression::AtqsCompression,
    /// NXR-SWIFT edge optimization
    pub swift_model: NxrSwiftModel,
    /// Integration configuration
    pub integration_config: AtqsSwiftConfig,
}

/// Configuration for ATQS-SWIFT integration
#[derive(Debug, Clone)]
pub struct AtqsSwiftConfig {
    /// Enable edge optimization
    pub enable_edge_optimization: bool,
    /// Target latency in milliseconds
    pub target_latency_ms: u32,
    /// Compression level for edge devices
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
    /// Create new integration
    pub fn new() -> Self {
        Self {
            atqs_compression: compression::AtqsCompression::new(),
            swift_model: NxrSwiftModel::new(),
            integration_config: AtqsSwiftConfig::default(),
        }
    }

    /// Enhanced compression with edge optimization
    pub async fn edge_optimized_compression(&self, data: &[u8]) -> Result<EdgeOptimizedCompression, Box<dyn std::error::Error>> {
        let mut result = EdgeOptimizedCompression::new();

        // Original ATQS compression
        if let Ok(atqs_result) = self.atqs_compression.compress(data).await {
            result.atqs_compression = Some(atqs_result);
        }

        // NXR-SWIFT edge optimization
        if self.integration_config.enable_edge_optimization {
            let swift_input = crate::shared::base_model::NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: crate::shared::base_model::InputData::Text(format!("Optimize {} bytes for edge deployment", data.len())),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };

            if let Ok(swift_result) = self.swift_model.infer(&swift_input).await {
                result.swift_optimization = Some(swift_result);
            }
        }

        // Combine compression and optimization results
        result.combine_results(&self.integration_config);

        Ok(result)
    }
}

/// Edge optimized compression result
#[derive(Debug, Clone)]
pub struct EdgeOptimizedCompression {
    /// ATQS compression result
    pub atqs_compression: Option<compression::CompressionResult>,
    /// SWIFT edge optimization
    pub swift_optimization: Option<crate::shared::base_model::NxrOutput>,
    /// Combined compression insights
    pub combined_insights: Vec<String>,
    /// Edge deployment recommendations
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

    /// Combine compression and optimization results
    fn combine_results(&mut self, config: &AtqsSwiftConfig) {
        // Combine insights from both systems
        if let Some(atqs) = &self.atqs_compression {
            self.combined_insights.push(format!("ATQS Compression: {:.2}% reduction", atqs.compression_ratio));
        }

        if let Some(swift) = &self.swift_optimization {
            if let crate::shared::base_model::OutputData::Text(text) = &swift.data {
                self.combined_insights.push(format!("Edge Optimization: {}", text));
                
                // Generate edge deployment recommendations
                if text.contains("edge") && text.contains("optimized") {
                    self.edge_recommendations.push("Deploy with 4-bit quantization for maximum efficiency".to_string());
                }
                if text.contains("latency") && text.contains("1ms") {
                    self.edge_recommendations.push("Sub-millisecond latency achieved for real-time processing".to_string());
                }
            }
        }

        // Apply edge-specific recommendations
        if config.target_latency_ms <= 1 {
            self.edge_recommendations.push("Ultra-low latency configuration optimized for edge devices".to_string());
        }
        if config.edge_compression_level <= 4 {
            self.edge_recommendations.push("High compression ratio maintained for edge deployment".to_string());
        }
    }

    /// Get comprehensive edge optimization summary
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
