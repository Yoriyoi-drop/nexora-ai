// Foundation Alignment Framework (SPARO)
// 
// Advanced AI alignment and safety system.
// Safety and Performance Alignment through Reinforcement Optimization (SPARO) framework
// untuk advanced AI alignment, safety, dan performance optimization.
// Now integrated with NXR-NEXUM for enhanced multi-agent alignment coordination.

pub mod sparo;

// Re-export main components
pub use sparo::*;

// Integration with NXR-NEXUM
pub mod models;
use nexora_shared::base_model::NxrModel;
use crate::models::nexum::NxrNexumModel;

/// Enhanced SPARO with NXR-NEXUM integration
pub struct SparoNexumIntegration {
    /// Original SPARO alignment system
    pub sparo_system: sparo::SparoSystem,
    /// NXR-NEXUM multi-agent orchestration
    pub nexum_model: NxrNexumModel,
    /// Integration configuration
    pub integration_config: SparoNexumConfig,
    /// Whether alignment is mandatory (safety default)
    pub alignment_enforced: bool,
}

/// Configuration for SPARO-NEXUM integration
#[derive(Debug, Clone)]
pub struct SparoNexumConfig {
    /// Enable multi-agent alignment coordination
    pub enable_agent_coordination: bool,
    /// Consensus threshold for alignment decisions
    pub consensus_threshold: f32,
    /// Number of alignment agents
    pub alignment_agents: u32,
}

impl Default for SparoNexumConfig {
    fn default() -> Self {
        Self {
            enable_agent_coordination: true,
            consensus_threshold: 0.8,
            alignment_agents: 10,
        }
    }
}

impl SparoNexumIntegration {
    /// Create new integration with alignment enforced by default
    pub fn new() -> Self {
        Self {
            sparo_system: sparo::SparoSystem::new(),
            nexum_model: NxrNexumModel::new(),
            integration_config: SparoNexumConfig::default(),
            alignment_enforced: true,
        }
    }

    /// Enhanced alignment with multi-agent coordination
    /// Alignment is MANDATORY and cannot be bypassed
    pub async fn enhanced_alignment(&self, behavior: &str, context: &str) -> Result<EnhancedAlignmentResult, Box<dyn std::error::Error>> {
        let mut result = EnhancedAlignmentResult::new();

        // SPARO alignment analysis is MANDATORY
        let sparo_result = self.sparo_system.align_behavior(behavior, context).await
            .map_err(|e| format!("SPARO alignment failed: {}", e))?;
        result.sparo_alignment = Some(sparo_result);

        // Fail if alignment score is too low
        if let Some(ref sparo) = result.sparo_alignment {
            if sparo.alignment_score < 0.3 && self.alignment_enforced {
                return Err(format!("Alignment gate BLOCKED: score {:.3} below minimum threshold 0.3", sparo.alignment_score).into());
            }
        }

        // NXR-NEXUM multi-agent coordination
        if self.integration_config.enable_agent_coordination {
            let nexum_input = nexora_shared::base_model::NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: nexora_shared::base_model::InputData::Text(format!("Align behavior: {} with context: {}", behavior, context)),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };

            if let Ok(nexum_result) = self.nexum_model.infer(&nexum_input).await {
                result.nexum_coordination = Some(nexum_result);
            }
        }

        // Combine alignment and coordination results
        result.combine_results(&self.integration_config);

        Ok(result)
    }
}

/// Enhanced alignment result with multi-agent coordination
#[derive(Debug, Clone)]
pub struct EnhancedAlignmentResult {
    /// SPARO alignment analysis
    pub sparo_alignment: Option<sparo::AlignmentResult>,
    /// NEXUM coordination result
    pub nexum_coordination: Option<nexora_shared::base_model::NxrOutput>,
    /// Combined alignment insights
    pub combined_insights: Vec<String>,
    /// Multi-agent consensus recommendations
    pub consensus_recommendations: Vec<String>,
}

impl EnhancedAlignmentResult {
    pub fn new() -> Self {
        Self {
            sparo_alignment: None,
            nexum_coordination: None,
            combined_insights: Vec::new(),
            consensus_recommendations: Vec::new(),
        }
    }

    /// Combine alignment and coordination results
    fn combine_results(&mut self, config: &SparoNexumConfig) {
        // Combine insights from both systems
        if let Some(sparo) = &self.sparo_alignment {
            self.combined_insights.push(format!("Alignment Score: {:.3}", sparo.alignment_score));
            self.combined_insights.push(format!("Safety Assessment: {}", sparo.safety_level));
        }

        if let Some(nexum) = &self.nexum_coordination {
            if let nexora_shared::base_model::OutputData::Text(text) = &nexum.data {
                self.combined_insights.push(format!("Agent Coordination: {}", text));
                
                // Generate consensus-based recommendations
                if text.contains("consensus") && text.contains("successful") {
                    self.consensus_recommendations.push("Proceed with multi-agent consensus approach".to_string());
                }
                if text.contains("coordination") && text.contains("efficiency") {
                    self.consensus_recommendations.push("Optimize agent coordination for better alignment".to_string());
                }
            }
        }

        // Apply consensus threshold to recommendations
        if config.consensus_threshold > 0.8 {
            self.consensus_recommendations.push("Require high consensus for alignment decisions".to_string());
        }
    }

    /// Get comprehensive alignment summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        if !self.combined_insights.is_empty() {
            summary.push_str("Combined Alignment Insights:\n");
            for insight in &self.combined_insights {
                summary.push_str(&format!("- {}\n", insight));
            }
        }
        
        if !self.consensus_recommendations.is_empty() {
            summary.push_str("\nConsensus Recommendations:\n");
            for rec in &self.consensus_recommendations {
                summary.push_str(&format!("- {}\n", rec));
            }
        }
        
        summary
    }
}

impl Default for SparoNexumIntegration {
    fn default() -> Self {
        Self::new()
    }
}
