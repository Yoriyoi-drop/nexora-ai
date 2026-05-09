/// ORACLE - Optimized Retrieval-Augmented Code Learning Engine
/// 
/// Arsitektur pelatihan LLM generasi berikutnya yang menyatukan 6 metode
/// menjadi satu pipeline terpadu untuk pemahaman dan generasi kode skala besar.
/// Now integrated with NXR-VORTEX for enhanced code analysis capabilities.

pub mod backbone;
pub mod rope;
pub mod pretraining;
pub mod alignment;
pub mod trainer;
pub mod code_utils;
pub mod verifiers;

// Re-export main components for easier access
pub use backbone::*;
pub use rope::*;
pub use pretraining::*;
pub use alignment::*;
pub use trainer::*;
pub use code_utils::*;
pub use verifiers::*;

// Integration with NXR-VORTEX
pub use crate::models::vortex::NxrVortexModel;

/// Enhanced ORACLE with NXR-VORTEX integration
pub struct OracleVortexIntegration {
    /// Original ORACLE components
    pub oracle_trainer: trainer::OracleTrainer,
    /// NXR-VORTEX code analysis
    pub vortex_model: NxrVortexModel,
    /// Integration configuration
    pub integration_config: OracleVortexConfig,
}

/// Configuration for ORACLE-VORTEX integration
#[derive(Debug, Clone)]
pub struct OracleVortexConfig {
    /// Enable Vortex code analysis
    pub enable_vortex_analysis: bool,
    /// Analysis depth
    pub analysis_depth: u8,
    /// Integration frequency
    pub integration_frequency_ms: u64,
}

impl Default for OracleVortexConfig {
    fn default() -> Self {
        Self {
            enable_vortex_analysis: true,
            analysis_depth: 8,
            integration_frequency_ms: 1000,
        }
    }
}

impl OracleVortexIntegration {
    /// Create new integration
    pub fn new() -> Self {
        Self {
            oracle_trainer: trainer::OracleTrainer::new(),
            vortex_model: NxrVortexModel::new(),
            integration_config: OracleVortexConfig::default(),
        }
    }

    /// Enhanced code analysis with Vortex
    pub async fn enhanced_code_analysis(&self, code: &str) -> Result<EnhancedCodeAnalysis, Box<dyn std::error::Error>> {
        let mut analysis = EnhancedCodeAnalysis::new();

        // Original ORACLE analysis
        if let Ok(oracle_result) = self.oracle_trainer.analyze_code(code).await {
            analysis.oracle_analysis = Some(oracle_result);
        }

        // NXR-VORTEX analysis
        if self.integration_config.enable_vortex_analysis {
            let vortex_input = crate::shared::base_model::NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: crate::shared::base_model::InputData::Text(code.to_string()),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };

            if let Ok(vortex_result) = self.vortex_model.infer(&vortex_input).await {
                analysis.vortex_analysis = Some(vortex_result);
            }
        }

        Ok(analysis)
    }
}

/// Enhanced code analysis result
#[derive(Debug, Clone)]
pub struct EnhancedCodeAnalysis {
    /// ORACLE analysis result
    pub oracle_analysis: Option<trainer::CodeAnalysisResult>,
    /// Vortex analysis result
    pub vortex_analysis: Option<crate::shared::base_model::NxrOutput>,
    /// Combined insights
    pub combined_insights: Vec<String>,
}

impl EnhancedCodeAnalysis {
    pub fn new() -> Self {
        Self {
            oracle_analysis: None,
            vortex_analysis: None,
            combined_insights: Vec::new(),
        }
    }

    /// Get combined analysis summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        if let Some(oracle) = &self.oracle_analysis {
            summary.push_str(&format!("ORACLE Analysis: {}\n", oracle.quality_score));
        }
        
        if let Some(vortex) = &self.vortex_analysis {
            if let crate::shared::base_model::OutputData::Text(text) = &vortex.data {
                summary.push_str(&format!("VORTEX Analysis: {}\n", text));
            }
        }
        
        if !self.combined_insights.is_empty() {
            summary.push_str("Combined Insights:\n");
            for insight in &self.combined_insights {
                summary.push_str(&format!("- {}\n", insight));
            }
        }
        
        summary
    }
}

impl Default for OracleVortexIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Prelude module untuk import umum
pub mod prelude {
    pub use super::backbone::*;
    pub use super::rope::*;
    pub use super::pretraining::*;
    pub use super::alignment::*;
    pub use super::trainer::*;
    pub use super::code_utils::*;
    pub use super::verifiers::*;
}
