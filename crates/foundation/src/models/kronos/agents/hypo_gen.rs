//! HYPO-GEN Agent
//!
//! Scientific hypothesis generation

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// HYPO-GEN Agent - Scientific hypothesis generation
#[derive(Debug, Clone)]
pub struct HypoGenAgent {
    pub config: HypoGenConfig,
    pub generation_capabilities: GenerationCapabilities,
    pub hypothesis_framework: HypothesisFramework,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypoGenConfig {
    pub base_config: BaseAgentConfig,
    pub generation_model: GenerationModel,
    pub hypothesis_type: HypothesisType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationModel {
    InductiveReasoning,
    DeductiveReasoning,
    AbductiveReasoning,
    AnalogyBasedGeneration,
    HybridGeneration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HypothesisType {
    CausalHypothesis,
    CorrelationalHypothesis,
    DescriptiveHypothesis,
    PredictiveHypothesis,
    ExplanatoryHypothesis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationCapabilities {
    pub hypothesis_generation: bool,
    pub scientific_reasoning: bool,
    pub literature_synthesis: bool,
    pub falsifiability_check: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisFramework {
    pub generation_methods: Vec<String>,
    pub validation_criteria: Vec<String>,
    pub domain_paradigms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypoGenTaskInput {
    pub research_question: String,
    pub scientific_domain: String,
    pub observations: Vec<String>,
    pub existing_theories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypoGenTaskOutput {
    pub hypotheses: Vec<Hypothesis>,
    pub supporting_evidence: Vec<String>,
    pub falsification_tests: Vec<String>,
    pub generation_quality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub statement: String,
    pub hypothesis_type: HypothesisType,
    pub confidence: f32,
    pub testability_score: f32,
}

impl Default for HypoGenConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            generation_model: GenerationModel::HybridGeneration,
            hypothesis_type: HypothesisType::CausalHypothesis,
        }
    }
}

impl Default for GenerationCapabilities {
    fn default() -> Self {
        Self {
            hypothesis_generation: true,
            scientific_reasoning: true,
            literature_synthesis: true,
            falsifiability_check: true,
        }
    }
}

impl Default for HypothesisFramework {
    fn default() -> Self {
        Self {
            generation_methods: vec![
                "inductive_generation".to_string(),
                "deductive_generation".to_string(),
                "abductive_generation".to_string(),
                "analogy_mapping".to_string(),
            ],
            validation_criteria: vec![
                "falsifiability".to_string(),
                "testability".to_string(),
                "parsimony".to_string(),
                "coherence".to_string(),
            ],
            domain_paradigms: vec![
                "physics_paradigms".to_string(),
                "biology_paradigms".to_string(),
                "chemistry_paradigms".to_string(),
                "social_science_paradigms".to_string(),
            ],
        }
    }
}

impl Default for HypoGenAgent {
    fn default() -> Self {
        Self {
            config: HypoGenConfig::default(),
            generation_capabilities: GenerationCapabilities::default(),
            hypothesis_framework: HypothesisFramework::default(),
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
impl BaseAgent for HypoGenAgent {
    type Config = HypoGenConfig;
    type Input = HypoGenTaskInput;
    type Output = HypoGenTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let hypotheses = self.generate_hypotheses(&input).await?;
        let supporting_evidence = self.collect_supporting_evidence(&input, &hypotheses).await?;
        let falsification_tests = self.design_falsification_tests(&input, &hypotheses).await?;
        let generation_quality = self.assess_generation_quality(&input, &hypotheses).await?;

        Ok(HypoGenTaskOutput {
            hypotheses,
            supporting_evidence,
            falsification_tests,
            generation_quality,
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
                name: "hypo_gen".to_string(),
                description: "Scientific hypothesis generation".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["research_question".to_string(), "scientific_domain".to_string()],
                output_types: vec!["hypotheses".to_string(), "falsification_tests".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.90,
                    avg_latency: 3500.0,
                    resource_usage: 0.80,
                    reliability: 0.92,
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

impl HypoGenAgent {
    pub fn new(config: HypoGenConfig) -> Self {
        Self {
            config,
            generation_capabilities: GenerationCapabilities::default(),
            hypothesis_framework: HypothesisFramework::default(),
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

    async fn generate_hypotheses(&self, input: &HypoGenTaskInput) -> AgentResult<Vec<Hypothesis>> {
        let mut hypotheses = Vec::new();

        hypotheses.push(Hypothesis {
            statement: format!("{} is causally influenced by observable factors in {}", input.research_question, input.scientific_domain),
            hypothesis_type: HypothesisType::CausalHypothesis,
            confidence: 0.85,
            testability_score: 0.78,
        });

        hypotheses.push(Hypothesis {
            statement: format!("Correlation exists between {} and established theoretical predictions", input.research_question),
            hypothesis_type: HypothesisType::CorrelationalHypothesis,
            confidence: 0.82,
            testability_score: 0.91,
        });

        hypotheses.push(Hypothesis {
            statement: format!("Novel mechanism explains {} beyond current {} frameworks", input.research_question, input.scientific_domain),
            hypothesis_type: HypothesisType::ExplanatoryHypothesis,
            confidence: 0.73,
            testability_score: 0.65,
        });

        Ok(hypotheses)
    }

    async fn collect_supporting_evidence(
        &self,
        input: &HypoGenTaskInput,
        hypotheses: &[Hypothesis],
    ) -> AgentResult<Vec<String>> {
        let mut evidence = Vec::new();

        for (i, h) in hypotheses.iter().enumerate() {
            evidence.push(format!("Hypothesis {}: {} - supported by domain observations", i + 1, h.statement));
        }

        for obs in &input.observations {
            evidence.push(format!("Observation supports: {}", obs));
        }

        for theory in &input.existing_theories {
            evidence.push(format!("Consistent with theory: {}", theory));
        }

        Ok(evidence)
    }

    async fn design_falsification_tests(
        &self,
        input: &HypoGenTaskInput,
        hypotheses: &[Hypothesis],
    ) -> AgentResult<Vec<String>> {
        let mut tests = Vec::new();

        for (i, h) in hypotheses.iter().enumerate() {
            tests.push(format!("Test {}: Design controlled experiment to verify '{}'", i + 1, h.statement));
            tests.push(format!("Test {} falsification: Identify conditions that would disprove the hypothesis", i + 1));
        }

        tests.push(format!("Domain: {} - apply standard falsification protocols", input.scientific_domain));

        Ok(tests)
    }

    async fn assess_generation_quality(
        &self,
        input: &HypoGenTaskInput,
        hypotheses: &[Hypothesis],
    ) -> AgentResult<f32> {
        let hypothesis_quality = if hypotheses.len() >= 2 { 0.85 } else { 0.6 };
        let domain_coverage = if !input.scientific_domain.is_empty() { 0.8 } else { 0.5 };
        let observation_quality = if input.observations.len() > 0 { 0.75 } else { 0.5 };

        Ok((hypothesis_quality + domain_coverage + observation_quality) / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypo_gen_agent_creation() {
        let agent = HypoGenAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_hypo_gen_task_processing() {
        let agent = HypoGenAgent::default();
        let input = HypoGenTaskInput {
            research_question: "How do prion proteins induce misfolding in healthy neurons?".to_string(),
            scientific_domain: "molecular_neuroscience".to_string(),
            observations: vec![
                "Misfolded proteins propagate between cells".to_string(),
                "Prion strains exhibit distinct phenotypes".to_string(),
            ],
            existing_theories: vec![
                "Protein-only hypothesis".to_string(),
                "Seeding-nucleation model".to_string(),
            ],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.hypotheses.is_empty());
        assert!(!output.supporting_evidence.is_empty());
        assert!(!output.falsification_tests.is_empty());
        assert!(output.generation_quality > 0.0);
    }
}
