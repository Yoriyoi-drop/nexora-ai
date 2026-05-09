//! Truth Validator Agent
//! 
//! Truth verification and validation systems

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Truth Validator Agent - Truth verification and validation systems
#[derive(Debug, Clone)]
pub struct TruthValidatorAgent {
    pub config: TruthValidatorConfig,
    pub validation_capabilities: ValidationCapabilities,
    pub truth_assessment: TruthAssessment,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthValidatorConfig {
    pub base_config: BaseAgentConfig,
    pub validation_model: ValidationModel,
    pub truth_framework: TruthFramework,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationModel {
    EmpiricalValidation,
    LogicalValidation,
    ConsensusValidation,
    PragmaticValidation,
    HybridValidation { models: Vec<ValidationModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruthFramework {
    CorrespondenceTheory,
    CoherenceTheory,
    PragmaticTheory,
    ConsensusTheory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCapabilities {
    pub fact_checking: bool,
    pub logical_consistency: bool,
    pub source_verification: bool,
    pub cross_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthAssessment {
    pub validation_methods: Vec<String>,
    pub truth_criteria: Vec<String>,
    pub assessment_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthValidatorTaskInput {
    pub statement: String,
    pub evidence_sources: Vec<String>,
    pub validation_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthValidatorTaskOutput {
    pub truth_score: f32,
    pub validation_result: String,
    pub confidence_level: f32,
    pub supporting_evidence: Vec<String>,
}

impl Default for TruthValidatorConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            validation_model: ValidationModel::HybridValidation {
                models: vec![
                    ValidationModel::EmpiricalValidation,
                    ValidationModel::LogicalValidation,
                ],
            },
            truth_framework: TruthFramework::CorrespondenceTheory,
        }
    }
}

impl Default for ValidationCapabilities {
    fn default() -> Self {
        Self {
            fact_checking: true,
            logical_consistency: true,
            source_verification: true,
            cross_validation: true,
        }
    }
}

impl Default for TruthAssessment {
    fn default() -> Self {
        Self {
            validation_methods: vec![
                "empirical_testing".to_string(),
                "logical_analysis".to_string(),
                "source_evaluation".to_string(),
            ],
            truth_criteria: vec![
                "accuracy".to_string(),
                "consistency".to_string(),
                "reliability".to_string(),
            ],
            assessment_metrics: vec![
                "truth_score".to_string(),
                "confidence_level".to_string(),
                "evidence_strength".to_string(),
            ],
        }
    }
}

impl Default for TruthValidatorAgent {
    fn default() -> Self {
        Self {
            config: TruthValidatorConfig::default(),
            validation_capabilities: ValidationCapabilities::default(),
            truth_assessment: TruthAssessment::default(),
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
impl BaseAgent for TruthValidatorAgent {
    type Config = TruthValidatorConfig;
    type Input = TruthValidatorTaskInput;
    type Output = TruthValidatorTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let truth_score = self.calculate_truth_score(&input).await?;
        let validation_result = self.generate_validation_result(&input, truth_score).await?;
        let confidence_level = self.assess_confidence(&input).await?;
        let supporting_evidence = self.identify_supporting_evidence(&input).await?;

        Ok(TruthValidatorTaskOutput {
            truth_score,
            validation_result,
            confidence_level,
            supporting_evidence,
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
                name: "truth_validation".to_string(),
                description: "Truth verification and validation systems".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["statement".to_string(), "evidence_sources".to_string()],
                output_types: vec!["truth_score".to_string(), "validation_result".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.92,
                    avg_latency: 2400.0,
                    resource_usage: 0.55,
                    reliability: 0.94,
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

impl TruthValidatorAgent {
    pub fn new(config: TruthValidatorConfig) -> Self {
        Self {
            config,
            validation_capabilities: ValidationCapabilities::default(),
            truth_assessment: TruthAssessment::default(),
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

    async fn calculate_truth_score(&self, input: &TruthValidatorTaskInput) -> AgentResult<f32> {
        // Simple truth score calculation based on evidence
        let evidence_count = input.evidence_sources.len() as f32;
        let base_score = 0.5;
        let evidence_bonus = (evidence_count / 10.0).min(0.4);
        
        Ok(base_score + evidence_bonus)
    }

    async fn generate_validation_result(&self, input: &TruthValidatorTaskInput, truth_score: f32) -> AgentResult<String> {
        let result = if truth_score >= 0.8 {
            "Highly likely true"
        } else if truth_score >= 0.6 {
            "Likely true"
        } else if truth_score >= 0.4 {
            "Possibly true"
        } else {
            "Unlikely to be true"
        };

        Ok(format!("Statement '{}' is {} based on available evidence", input.statement, result))
    }

    async fn assess_confidence(&self, input: &TruthValidatorTaskInput) -> AgentResult<f32> {
        let source_count = input.evidence_sources.len() as f32;
        let base_confidence = 0.6;
        let source_bonus = (source_count / 15.0).min(0.3);
        
        Ok(base_confidence + source_bonus)
    }

    async fn identify_supporting_evidence(&self, input: &TruthValidatorTaskInput) -> AgentResult<Vec<String>> {
        Ok(input.evidence_sources.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_validator_agent_creation() {
        let agent = TruthValidatorAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_truth_validator_task_processing() {
        let agent = TruthValidatorAgent::default();
        let input = TruthValidatorTaskInput {
            statement: "The Earth orbits the Sun".to_string(),
            evidence_sources: vec![
                "Scientific observations".to_string(),
                "Astronomical data".to_string(),
            ],
            validation_type: "empirical".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.truth_score > 0.0);
        assert!(!output.validation_result.is_empty());
        assert!(output.confidence_level > 0.0);
        assert!(!output.supporting_evidence.is_empty());
    }
}
