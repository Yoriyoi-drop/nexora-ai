// ORACLE module
//
// Re-exports from `nexora-oracle` crate, plus OracleVortexIntegration
// that depends on NXR-VORTEX (model series still in foundation).

pub use nexora_oracle::*;

use nexora_shared::base_model::NxrModel;
use crate::models::vortex::NxrVortexModel;

/// Enhanced ORACLE with NXR-VORTEX integration
pub struct OracleVortexIntegration {
    pub oracle_trainer: nexora_oracle::trainer::OracleTrainer,
    pub vortex_model: NxrVortexModel,
    pub integration_config: OracleVortexConfig,
}

#[derive(Debug, Clone)]
pub struct OracleVortexConfig {
    pub enable_vortex_analysis: bool,
    pub analysis_depth: u8,
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
    pub fn new() -> Self {
        Self {
            oracle_trainer: nexora_oracle::trainer::OracleTrainer::new(
                nexora_oracle::trainer::OracleConfig::default(), 32_000
            ).expect("failed to initialize ORACLE trainer"),
            vortex_model: NxrVortexModel::new(),
            integration_config: OracleVortexConfig::default(),
        }
    }

    pub async fn enhanced_code_analysis(&self, code: &str) -> Result<EnhancedCodeAnalysis, Box<dyn std::error::Error + Send + Sync>> {
        let mut analysis = EnhancedCodeAnalysis::new();

        if let Ok(oracle_result) = self.oracle_trainer.analyze_code(code).await {
            analysis.oracle_analysis = Some(oracle_result);
        }

        if self.integration_config.enable_vortex_analysis {
            let vortex_input = nexora_shared::base_model::NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: nexora_shared::base_model::InputData::Text(code.to_string()),
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

#[derive(Debug, Clone)]
pub struct EnhancedCodeAnalysis {
    pub oracle_analysis: Option<nexora_oracle::trainer::CodeAnalysisResult>,
    pub vortex_analysis: Option<nexora_shared::base_model::NxrOutput>,
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

    pub fn summary(&self) -> String {
        self.combined_insights.join("; ")
    }
}

impl Default for OracleVortexIntegration {
    fn default() -> Self {
        Self::new()
    }
}

pub mod prelude {
    pub use nexora_oracle::prelude::*;
}
