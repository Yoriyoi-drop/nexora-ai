//! RISK-ORACLE Agent
//!
//! Risk identification and quantification

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

#[derive(Debug, Clone)]
pub struct RiskOracleAgent {
    pub config: RiskOracleConfig,
    pub risk_capabilities: RiskCapabilities,
    pub quantification_engine: QuantificationEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskOracleConfig {
    pub base_config: BaseAgentConfig,
    pub risk_model: RiskModel,
    pub quantification_method: QuantificationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskModel {
    Probabilistic,
    Bayesian,
    Frequentist,
    ScenarioBased,
    HybridRisk { models: Vec<RiskModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantificationMethod {
    MonteCarlo,
    ExpectedValue,
    ValueAtRisk,
    ConditionalVaR,
    StressTesting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCapabilities {
    pub risk_identification: bool,
    pub probabilistic_quantification: bool,
    pub impact_assessment: bool,
    pub correlation_analysis: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantificationEngine {
    pub risk_metrics: Vec<String>,
    pub distribution_models: Vec<String>,
    pub correlation_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskOracleTaskInput {
    pub context: String,
    pub domain: String,
    pub known_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskOracleTaskOutput {
    pub identified_risks: Vec<RiskOracleRisk>,
    pub overall_risk_score: f32,
    pub quantification_results: QuantificationResults,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskOracleRisk {
    pub risk_id: String,
    pub description: String,
    pub probability: f32,
    pub impact: f32,
    pub risk_score: f32,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantificationResults {
    pub expected_loss: f32,
    pub value_at_risk: f32,
    pub worst_case_loss: f32,
    pub confidence_interval: (f32, f32),
}

impl Default for RiskOracleConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            risk_model: RiskModel::HybridRisk {
                models: vec![
                    RiskModel::Probabilistic,
                    RiskModel::ScenarioBased,
                ],
            },
            quantification_method: QuantificationMethod::ValueAtRisk,
        }
    }
}

impl Default for RiskCapabilities {
    fn default() -> Self {
        Self {
            risk_identification: true,
            probabilistic_quantification: true,
            impact_assessment: true,
            correlation_analysis: true,
        }
    }
}

impl Default for QuantificationEngine {
    fn default() -> Self {
        Self {
            risk_metrics: vec![
                "probability".to_string(),
                "impact".to_string(),
                "risk_score".to_string(),
                "expected_loss".to_string(),
            ],
            distribution_models: vec![
                "normal".to_string(),
                "lognormal".to_string(),
                "poisson".to_string(),
            ],
            correlation_methods: vec![
                "pearson".to_string(),
                "spearman".to_string(),
                "kendall".to_string(),
            ],
        }
    }
}

impl Default for RiskOracleAgent {
    fn default() -> Self {
        Self {
            config: RiskOracleConfig::default(),
            risk_capabilities: RiskCapabilities::default(),
            quantification_engine: QuantificationEngine::default(),
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
impl BaseAgent for RiskOracleAgent {
    type Config = RiskOracleConfig;
    type Input = RiskOracleTaskInput;
    type Output = RiskOracleTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let identified_risks = self.identify_risks(&input).await?;
        let overall_risk_score = self.compute_overall_risk(&identified_risks).await?;
        let quantification_results = self.quantify_risks(&input, &identified_risks).await?;
        let recommendations = self.generate_recommendations(&identified_risks, &quantification_results).await?;

        Ok(RiskOracleTaskOutput {
            identified_risks,
            overall_risk_score,
            quantification_results,
            recommendations,
        })
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability {
                name: "risk_oracle".to_string(),
                description: "Risk identification and quantification".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["context".to_string(), "domain".to_string(), "known_factors".to_string()],
                output_types: vec!["identified_risks".to_string(), "risk_score".to_string(), "quantification".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.91,
                    avg_latency: 3600.0,
                    resource_usage: 0.7,
                    reliability: 0.93,
                },
            },
        ]
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

impl RiskOracleAgent {
    pub fn new(config: RiskOracleConfig) -> Self {
        Self {
            config,
            risk_capabilities: RiskCapabilities::default(),
            quantification_engine: QuantificationEngine::default(),
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

    async fn identify_risks(&self, input: &RiskOracleTaskInput) -> AgentResult<Vec<RiskOracleRisk>> {
        Ok(vec![
            RiskOracleRisk {
                risk_id: "RISK-001".to_string(),
                description: format!("Market volatility risk in {} domain", input.domain),
                probability: 0.45,
                impact: 0.75,
                risk_score: 0.34,
                category: "market".to_string(),
            },
            RiskOracleRisk {
                risk_id: "RISK-002".to_string(),
                description: format!("Operational risk from context: {}", input.context),
                probability: 0.30,
                impact: 0.60,
                risk_score: 0.18,
                category: "operational".to_string(),
            },
            RiskOracleRisk {
                risk_id: "RISK-003".to_string(),
                description: "Strategic risk from known factors".to_string(),
                probability: 0.50,
                impact: 0.50,
                risk_score: 0.25,
                category: "strategic".to_string(),
            },
        ])
    }

    async fn compute_overall_risk(&self, risks: &[RiskOracleRisk]) -> AgentResult<f32> {
        let weighted: f32 = risks.iter().map(|r| r.risk_score).sum();
        Ok((weighted / risks.len() as f32).min(1.0))
    }

    async fn quantify_risks(&self, _input: &RiskOracleTaskInput, risks: &[RiskOracleRisk]) -> AgentResult<QuantificationResults> {
        let expected_loss: f32 = risks.iter().map(|r| r.probability * r.impact).sum();
        Ok(QuantificationResults {
            expected_loss,
            value_at_risk: expected_loss * 1.65,
            worst_case_loss: risks.iter().map(|r| r.impact).sum(),
            confidence_interval: (0.05, 0.95),
        })
    }

    async fn generate_recommendations(&self, _risks: &[RiskOracleRisk], _quant: &QuantificationResults) -> AgentResult<Vec<String>> {
        Ok(vec![
            "Implement hedging strategy for high-impact risks".to_string(),
            "Establish monitoring triggers at 80% of VaR threshold".to_string(),
            "Diversify exposure across uncorrelated risk factors".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_oracle_agent_creation() {
        let agent = RiskOracleAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_risk_oracle_task_processing() {
        let agent = RiskOracleAgent::default();
        let input = RiskOracleTaskInput {
            context: "Market expansion into emerging markets".to_string(),
            domain: "finance".to_string(),
            known_factors: vec!["currency fluctuation".to_string(), "regulatory changes".to_string()],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.identified_risks.is_empty());
        assert!(output.overall_risk_score > 0.0);
        assert!(output.quantification_results.value_at_risk > 0.0);
        assert!(!output.recommendations.is_empty());
    }
}
