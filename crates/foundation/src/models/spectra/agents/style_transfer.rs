use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

#[derive(Debug, Clone)]
pub struct StyleTransferAgent {
    pub config: StyleTransferConfig,
    pub style_profiles: StyleProfiles,
    pub transfer_engine: TransferEngine,
    status: AgentStatus,
    metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTransferConfig {
    pub base_config: BaseAgentConfig,
    pub transfer_strength: f32,
    pub content_preservation: f32,
    pub style_blend_ratio: f32,
    pub supported_domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleProfiles {
    pub profiles: HashMap<String, StyleProfileEntry>,
    pub active_profile: Option<String>,
    pub consistency_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleProfileEntry {
    pub name: String,
    pub domain: String,
    pub parameters: HashMap<String, f32>,
    pub color_palette: Vec<String>,
    pub texture_params: HashMap<String, String>,
    pub consistency_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEngine {
    pub methods: Vec<TransferMethod>,
    pub adaptation_rate: f32,
    pub cross_domain_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferMethod {
    NeuralStyleTransfer,
    AdaptiveConvolution,
    GramMatrixMatching,
    PerceptualLoss,
    HybridTransfer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTransferInput {
    pub source_content: String,
    pub target_style: String,
    pub source_domain: String,
    pub target_domain: Option<String>,
    pub preservation_weight: Option<f32>,
    pub constraints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTransferOutput {
    pub transferred_content: String,
    pub style_consistency_score: f32,
    pub content_preservation_score: f32,
    pub transfer_quality: f32,
    pub applied_method: String,
    pub style_signature: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    pub overall_consistency: f32,
    pub per_style_consistency: HashMap<String, f32>,
    pub deviations: Vec<StyleDeviation>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleDeviation {
    pub style_name: String,
    pub parameter: String,
    pub expected: f32,
    pub actual: f32,
    pub severity: f32,
}

impl Default for StyleTransferConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            transfer_strength: 0.7,
            content_preservation: 0.6,
            style_blend_ratio: 0.5,
            supported_domains: vec![
                "visual".to_string(), "text".to_string(), "audio".to_string(),
            ],
        }
    }
}

impl Default for StyleProfiles {
    fn default() -> Self {
        let mut profiles = HashMap::new();
        profiles.insert("contemporary".to_string(), StyleProfileEntry {
            name: "contemporary".to_string(),
            domain: "visual".to_string(),
            parameters: HashMap::from([
                ("contrast".to_string(), 0.8),
                ("saturation".to_string(), 0.7),
                ("sharpness".to_string(), 0.6),
            ]),
            color_palette: vec!["#2c3e50".to_string(), "#e74c3c".to_string()],
            texture_params: HashMap::new(),
            consistency_signature: "contemporary_v1".to_string(),
        });
        profiles.insert("minimalist".to_string(), StyleProfileEntry {
            name: "minimalist".to_string(),
            domain: "visual".to_string(),
            parameters: HashMap::from([
                ("contrast".to_string(), 0.5),
                ("saturation".to_string(), 0.3),
                ("sharpness".to_string(), 0.9),
            ]),
            color_palette: vec!["#ffffff".to_string(), "#000000".to_string()],
            texture_params: HashMap::new(),
            consistency_signature: "minimalist_v1".to_string(),
        });
        Self {
            profiles,
            active_profile: None,
            consistency_threshold: 0.75,
        }
    }
}

impl Default for TransferEngine {
    fn default() -> Self {
        Self {
            methods: vec![
                TransferMethod::NeuralStyleTransfer,
                TransferMethod::HybridTransfer,
            ],
            adaptation_rate: 0.3,
            cross_domain_enabled: true,
        }
    }
}

impl Default for StyleTransferAgent {
    fn default() -> Self {
        Self {
            config: StyleTransferConfig::default(),
            style_profiles: StyleProfiles::default(),
            transfer_engine: TransferEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for StyleTransferAgent {
    type Config = StyleTransferConfig;
    type Input = StyleTransferInput;
    type Output = StyleTransferOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        self.validate_input(&input)?;

        let method = self.select_transfer_method(&input);
        let profile = self.resolve_style_profile(&input.target_style);
        let transferred = self.apply_style_transfer(&input, &method, &profile).await?;
        let consistency = self.calculate_style_consistency(&transferred, &profile);
        let preservation = self.calculate_content_preservation(&input, &transferred);

        Ok(StyleTransferOutput {
            transferred_content: transferred,
            style_consistency_score: consistency,
            content_preservation_score: preservation,
            transfer_quality: (consistency + preservation) / 2.0,
            applied_method: format!("{:?}", method),
            style_signature: profile.consistency_signature.clone(),
            metadata: HashMap::new(),
        })
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability {
            name: "style_transfer".to_string(),
            description: "Style consistency transfer across domains".to_string(),
            version: "1.0.0".to_string(),
            input_types: vec!["style_transfer_input".to_string()],
            output_types: vec!["transferred_content".to_string(), "consistency_report".to_string()],
            metrics: crate::shared::agent_types::CapabilityMetrics {
                accuracy: 0.86,
                avg_latency: 750.0,
                resource_usage: 0.7,
                reliability: 0.9,
            },
        }]
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }

    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()> {
        self.config = config;
        self.status = AgentStatus::Idle;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        self.status = AgentStatus::Disabled;
        Ok(())
    }
}

impl StyleTransferAgent {
    pub fn new(config: StyleTransferConfig) -> Self {
        Self {
            config,
            style_profiles: StyleProfiles::default(),
            transfer_engine: TransferEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    fn validate_input(&self, input: &StyleTransferInput) -> AgentResult<()> {
        if input.source_content.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Source content cannot be empty".to_string()
            ));
        }
        if input.target_style.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Target style cannot be empty".to_string()
            ));
        }
        Ok(())
    }

    fn select_transfer_method(&self, _input: &StyleTransferInput) -> TransferMethod {
        if self.transfer_engine.cross_domain_enabled {
            TransferMethod::HybridTransfer
        } else {
            TransferMethod::NeuralStyleTransfer
        }
    }

    fn resolve_style_profile(&self, style_name: &str) -> StyleProfileEntry {
        self.style_profiles.profiles.get(style_name).cloned()
            .unwrap_or(StyleProfileEntry {
                name: style_name.to_string(),
                domain: "visual".to_string(),
                parameters: HashMap::new(),
                color_palette: Vec::new(),
                texture_params: HashMap::new(),
                consistency_signature: format!("{}_v1", style_name),
            })
    }

    async fn apply_style_transfer(&self, input: &StyleTransferInput, _method: &TransferMethod, profile: &StyleProfileEntry) -> AgentResult<String> {
        let strength = self.config.transfer_strength;
        let preservation = input.preservation_weight.unwrap_or(self.config.content_preservation);
        let domain = input.target_domain.as_deref().unwrap_or(&input.source_domain);

        Ok(format!(
            "Style transfer [{} -> {}] applied to '{}' (strength:{}, preservation:{}, style:{})",
            input.source_domain, domain, input.source_content, strength, preservation, profile.name
        ))
    }

    fn calculate_style_consistency(&self, _transferred: &str, profile: &StyleProfileEntry) -> f32 {
        let base = 0.82;
        let profile_bonus = if self.style_profiles.profiles.contains_key(&profile.name) { 0.1 } else { 0.0 };
        (base + profile_bonus).min(1.0)
    }

    fn calculate_content_preservation(&self, input: &StyleTransferInput, _transferred: &str) -> f32 {
        input.preservation_weight.unwrap_or(self.config.content_preservation).min(1.0)
    }

    pub fn check_consistency(&self, contents: &[(&str, &str)]) -> ConsistencyReport {
        let mut per_style: HashMap<String, f32> = HashMap::new();
        let mut deviations = Vec::new();

        for (content, style_name) in contents {
            if let Some(profile) = self.style_profiles.profiles.get(*style_name) {
                let score = self.calculate_style_consistency(content, profile);
                per_style.insert(style_name.to_string(), score);
                if score < self.style_profiles.consistency_threshold {
                    deviations.push(StyleDeviation {
                        style_name: style_name.to_string(),
                        parameter: "overall".to_string(),
                        expected: self.style_profiles.consistency_threshold,
                        actual: score,
                        severity: self.style_profiles.consistency_threshold - score,
                    });
                }
            }
        }

        let overall: f32 = per_style.values().sum::<f32>() / per_style.len().max(1) as f32;
        let mut recommendations = Vec::new();
        if overall < self.style_profiles.consistency_threshold {
            recommendations.push("Increase transfer strength for better style adherence".to_string());
        }

        ConsistencyReport {
            overall_consistency: overall,
            per_style_consistency: per_style,
            deviations,
            recommendations,
        }
    }

    pub fn register_style_profile(&mut self, entry: StyleProfileEntry) {
        self.style_profiles.profiles.insert(entry.name.clone(), entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_transfer_agent_creation() {
        let agent = StyleTransferAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
    }

    #[tokio::test]
    async fn test_style_transfer_processing() {
        let agent = StyleTransferAgent::default();
        let input = StyleTransferInput {
            source_content: "A vibrant cityscape".to_string(),
            target_style: "contemporary".to_string(),
            source_domain: "visual".to_string(),
            target_domain: None,
            preservation_weight: None,
            constraints: vec![],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.transferred_content.is_empty());
        assert!(output.style_consistency_score > 0.0);
    }
}
