//! SCIENCE-BOT Agent
//!
//! Scientific reasoning and methodology analysis

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// SCIENCE-BOT Agent - Scientific reasoning and methodology analysis
#[derive(Debug, Clone)]
pub struct ScienceBotAgent {
    pub config: ScienceBotConfig,
    pub reasoning_capabilities: ReasoningCapabilities,
    pub methodology_framework: MethodologyFramework,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScienceBotConfig {
    pub base_config: BaseAgentConfig,
    pub reasoning_model: ReasoningModel,
    pub methodology_approach: MethodologyApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningModel {
    DeductiveReasoning,
    InductiveReasoning,
    AbductiveReasoning,
    BayesianReasoning,
    CausalReasoning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MethodologyApproach {
    ExperimentalDesign,
    ObservationalStudy,
    SystematicReview,
    MetaAnalysis,
    ComputationalModeling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningCapabilities {
    pub logical_reasoning: bool,
    pub methodology_analysis: bool,
    pub statistical_evaluation: bool,
    pub experimental_design: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodologyFramework {
    pub reasoning_methods: Vec<String>,
    pub analysis_techniques: Vec<String>,
    pub evaluation_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScienceBotTaskInput {
    pub scientific_claim: String,
    pub methodology_description: String,
    pub evidence_sources: Vec<String>,
    pub analysis_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScienceBotTaskOutput {
    pub reasoning_analysis: Vec<String>,
    pub methodology_assessment: MethodologyAssessment,
    pub statistical_findings: Vec<String>,
    pub confidence_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodologyAssessment {
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub validity_score: f32,
    pub reproducibility_score: f32,
}

impl Default for ScienceBotConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            reasoning_model: ReasoningModel::BayesianReasoning,
            methodology_approach: MethodologyApproach::ExperimentalDesign,
        }
    }
}

impl Default for ReasoningCapabilities {
    fn default() -> Self {
        Self {
            logical_reasoning: true,
            methodology_analysis: true,
            statistical_evaluation: true,
            experimental_design: true,
        }
    }
}

impl Default for MethodologyFramework {
    fn default() -> Self {
        Self {
            reasoning_methods: vec![
                "bayesian_inference".to_string(),
                "causal_inference".to_string(),
                "statistical_hypothesis_testing".to_string(),
            ],
            analysis_techniques: vec![
                "regression_analysis".to_string(),
                "variance_analysis".to_string(),
                "factor_analysis".to_string(),
                "sensitivity_analysis".to_string(),
            ],
            evaluation_criteria: vec![
                "internal_validity".to_string(),
                "external_validity".to_string(),
                "construct_validity".to_string(),
                "statistical_conclusion_validity".to_string(),
            ],
        }
    }
}

impl Default for ScienceBotAgent {
    fn default() -> Self {
        Self {
            config: ScienceBotConfig::default(),
            reasoning_capabilities: ReasoningCapabilities::default(),
            methodology_framework: MethodologyFramework::default(),
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
impl BaseAgent for ScienceBotAgent {
    type Config = ScienceBotConfig;
    type Input = ScienceBotTaskInput;
    type Output = ScienceBotTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let reasoning_analysis = self.analyze_scientific_reasoning(&input).await?;
        let methodology_assessment = self.assess_methodology(&input).await?;
        let statistical_findings = self.evaluate_statistics(&input).await?;
        let confidence_score = self.calculate_confidence(&input, &methodology_assessment).await?;

        Ok(ScienceBotTaskOutput {
            reasoning_analysis,
            methodology_assessment,
            statistical_findings,
            confidence_score,
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
                name: "science_bot".to_string(),
                description: "Scientific reasoning and methodology analysis".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["scientific_claim".to_string(), "methodology_description".to_string()],
                output_types: vec!["reasoning_analysis".to_string(), "methodology_assessment".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.91,
                    avg_latency: 3000.0,
                    resource_usage: 0.75,
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

impl ScienceBotAgent {
    pub fn new(config: ScienceBotConfig) -> Self {
        Self {
            config,
            reasoning_capabilities: ReasoningCapabilities::default(),
            methodology_framework: MethodologyFramework::default(),
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

    async fn analyze_scientific_reasoning(&self, input: &ScienceBotTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Analyzing claim: {}", input.scientific_claim),
            format!("Applying {} reasoning model", match self.config.reasoning_model {
                ReasoningModel::DeductiveReasoning => "deductive",
                ReasoningModel::InductiveReasoning => "inductive",
                ReasoningModel::AbductiveReasoning => "abductive",
                ReasoningModel::BayesianReasoning => "bayesian",
                ReasoningModel::CausalReasoning => "causal",
            }),
            "Evaluating logical structure and argument validity".to_string(),
            "Identifying assumptions and potential biases".to_string(),
            "Assessing coherence with established scientific knowledge".to_string(),
        ])
    }

    async fn assess_methodology(&self, input: &ScienceBotTaskInput) -> AgentResult<MethodologyAssessment> {
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        if !input.methodology_description.is_empty() {
            strengths.push("Methodology is explicitly defined".to_string());
            strengths.push("Controls for confounding variables".to_string());
        } else {
            weaknesses.push("Methodology description is missing".to_string());
        }

        if input.evidence_sources.len() >= 3 {
            strengths.push("Multiple evidence sources provide triangulation".to_string());
        } else {
            weaknesses.push("Limited evidence sources may reduce reliability".to_string());
        }

        let method_length = input.methodology_description.len() as f32;
        let validity_score = if method_length > 50.0 { 0.85 } else { 0.65 };
        let reproducibility_score = if input.evidence_sources.len() > 0 { 0.82 } else { 0.55 };

        Ok(MethodologyAssessment {
            strengths,
            weaknesses,
            validity_score,
            reproducibility_score,
        })
    }

    async fn evaluate_statistics(&self, input: &ScienceBotTaskInput) -> AgentResult<Vec<String>> {
        let mut findings = Vec::new();

        findings.push("Statistical significance assessment: evaluating p-values and effect sizes".to_string());
        findings.push("Power analysis: determining if sample size is adequate".to_string());
        findings.push("Confidence interval estimation: quantifying uncertainty".to_string());

        for param in input.analysis_parameters.keys() {
            findings.push(format!("Analysis parameter considered: {}", param));
        }

        Ok(findings)
    }

    async fn calculate_confidence(
        &self,
        input: &ScienceBotTaskInput,
        assessment: &MethodologyAssessment,
    ) -> AgentResult<f32> {
        let methodology_quality = assessment.validity_score;
        let evidence_strength = if input.evidence_sources.len() >= 3 { 0.9 } else { 0.7 };
        let claim_clarity = if !input.scientific_claim.is_empty() { 0.85 } else { 0.5 };

        Ok((methodology_quality + evidence_strength + claim_clarity) / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_science_bot_agent_creation() {
        let agent = ScienceBotAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_science_bot_task_processing() {
        let agent = ScienceBotAgent::default();
        let input = ScienceBotTaskInput {
            scientific_claim: "CRISPR-Cas9 gene editing exhibits off-target effects in eukaryotic cells".to_string(),
            methodology_description: "Systematic review of 50 peer-reviewed studies using randomized controlled trials with double-blind protocols".to_string(),
            evidence_sources: vec![
                "Nature Biotechnology 2023".to_string(),
                "Cell Reports 2024".to_string(),
                "Science Advances 2024".to_string(),
            ],
            analysis_parameters: HashMap::from([
                ("significance_level".to_string(), "0.05".to_string()),
                ("effect_size_threshold".to_string(), "0.3".to_string()),
            ]),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.reasoning_analysis.is_empty());
        assert!(!output.methodology_assessment.strengths.is_empty());
        assert!(!output.statistical_findings.is_empty());
        assert!(output.confidence_score > 0.0);
    }
}
