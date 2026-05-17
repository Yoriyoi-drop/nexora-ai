//! Foundation Reasoning Framework (SACA)
//! 
//! Unified coding intelligence and reasoning system
//! Now integrated with NXR-ÆTHER for enhanced emotional reasoning capabilities.

/// SACA Reasoning Framework
/// 
/// Symbolic Algorithmic Coding Architecture (SACA) framework
/// untuk unified coding intelligence dan advanced reasoning capabilities.
pub mod saca;

// Re-export main components
pub use saca::*;

// Integration with NXR-ÆTHER
use crate::shared::base_model::NxrModel;
pub use crate::models::aether::NxrAetherModel;

/// Enhanced SACA with NXR-ÆTHER integration
pub struct SacaAetherIntegration {
    /// Original SACA reasoning engine
    pub saca_engine: saca::SacaEngine,
    /// NXR-ÆTHER emotional reasoning
    pub aether_model: NxrAetherModel,
    /// Integration configuration
    pub integration_config: SacaAetherConfig,
}

/// Configuration for SACA-ÆTHER integration
#[derive(Debug, Clone)]
pub struct SacaAetherConfig {
    /// Enable emotional context in reasoning
    pub enable_emotional_context: bool,
    /// Empathy weight in reasoning decisions
    pub empathy_weight: f32,
    /// Emotional analysis depth
    pub emotional_depth: u8,
}

impl Default for SacaAetherConfig {
    fn default() -> Self {
        Self {
            enable_emotional_context: true,
            empathy_weight: 0.3,
            emotional_depth: 6,
        }
    }
}

impl SacaAetherIntegration {
    /// Create new integration
    pub fn new() -> Self {
        Self {
            saca_engine: saca::SacaEngine::new(),
            aether_model: NxrAetherModel::new(),
            integration_config: SacaAetherConfig::default(),
        }
    }

    /// Enhanced reasoning with emotional context
    pub async fn enhanced_reasoning(&self, problem: &str, context: &str) -> Result<EnhancedReasoningResult, Box<dyn std::error::Error>> {
        let mut result = EnhancedReasoningResult::new();

        // Original SACA reasoning
        if let Ok(saca_result) = self.saca_engine.reason(problem, context).await {
            result.logical_reasoning = Some(saca_result);
        }

        // NXR-ÆTHER emotional analysis
        if self.integration_config.enable_emotional_context {
            let aether_input = crate::shared::base_model::NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: crate::shared::base_model::InputData::Text(format!("{} {}", problem, context)),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };

            if let Ok(aether_result) = self.aether_model.infer(&aether_input).await {
                result.emotional_analysis = Some(aether_result);
            }
        }

        // Combine reasoning results
        result.combine_results(&self.integration_config);

        Ok(result)
    }
}

/// Enhanced reasoning result with emotional context
#[derive(Debug, Clone)]
pub struct EnhancedReasoningResult {
    /// Logical reasoning from SACA
    pub logical_reasoning: Option<saca::ReasoningResult>,
    /// Emotional analysis from ÆTHER
    pub emotional_analysis: Option<crate::shared::base_model::NxrOutput>,
    /// Combined reasoning insights
    pub combined_insights: Vec<String>,
    /// Emotional-aware recommendations
    pub emotional_recommendations: Vec<String>,
}

impl EnhancedReasoningResult {
    pub fn new() -> Self {
        Self {
            logical_reasoning: None,
            emotional_analysis: None,
            combined_insights: Vec::new(),
            emotional_recommendations: Vec::new(),
        }
    }

    /// Combine logical and emotional reasoning
    fn combine_results(&mut self, config: &SacaAetherConfig) {
        // Combine insights from both reasoning engines
        if let Some(logical) = &self.logical_reasoning {
            self.combined_insights.push(format!("Logical: {}", logical.conclusion));
        }

        if let Some(emotional) = &self.emotional_analysis {
            if let crate::shared::base_model::OutputData::Text(text) = &emotional.data {
                self.combined_insights.push(format!("Emotional: {}", text));
                
                // Generate emotional-aware recommendations
                if text.contains("empathy") || text.contains("understanding") {
                    self.emotional_recommendations.push("Consider emotional impact in decision making".to_string());
                }
                if text.contains("support") || text.contains("care") {
                    self.emotional_recommendations.push("Provide supportive and empathetic response".to_string());
                }
            }
        }

        // Apply empathy weight to recommendations
        if config.empathy_weight > 0.5 {
            self.emotional_recommendations.push("Prioritize emotional well-being in solution".to_string());
        }
    }

    /// Get comprehensive reasoning summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        if !self.combined_insights.is_empty() {
            summary.push_str("Combined Insights:\n");
            for insight in &self.combined_insights {
                summary.push_str("- ");
                summary.push_str(insight);
                summary.push('\n');
            }
        }
        
        if !self.emotional_recommendations.is_empty() {
            summary.push_str("\nEmotional Recommendations:\n");
            for rec in &self.emotional_recommendations {
                summary.push_str("- ");
                summary.push_str(rec);
                summary.push('\n');
            }
        }
        
        summary
    }
}

impl Default for SacaAetherIntegration {
    fn default() -> Self {
        Self::new()
    }
}
