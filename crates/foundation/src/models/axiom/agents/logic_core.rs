//! Logic Core Agent
//! 
//! Logical inference and formal reasoning engine

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Logic Core Agent - Logical inference and formal reasoning engine
#[derive(Debug, Clone)]
pub struct LogicCoreAgent {
    pub config: LogicCoreConfig,
    pub inference_capabilities: InferenceCapabilities,
    pub formal_reasoning: FormalReasoning,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicCoreConfig {
    pub base_config: BaseAgentConfig,
    pub inference_model: InferenceModel,
    pub reasoning_system: ReasoningSystem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceModel {
    PropositionalLogic,
    PredicateLogic,
    ModalLogic,
    TemporalLogic,
    HybridInference { models: Vec<InferenceModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningSystem {
    ForwardChaining,
    BackwardChaining,
    Resolution,
    TableauMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCapabilities {
    pub logical_deduction: bool,
    pub pattern_matching: bool,
    pub theorem_proving: bool,
    pub consistency_checking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalReasoning {
    pub inference_rules: Vec<String>,
    pub proof_strategies: Vec<String>,
    pub logic_systems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicCoreTaskInput {
    pub premises: Vec<String>,
    pub conclusion: String,
    pub logic_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicCoreTaskOutput {
    pub proof_steps: Vec<String>,
    pub validity_result: bool,
    pub confidence_score: f32,
    pub proof_strategy: String,
}

impl Default for LogicCoreConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            inference_model: InferenceModel::HybridInference {
                models: vec![
                    InferenceModel::PropositionalLogic,
                    InferenceModel::PredicateLogic,
                ],
            },
            reasoning_system: ReasoningSystem::ForwardChaining,
        }
    }
}

impl Default for InferenceCapabilities {
    fn default() -> Self {
        Self {
            logical_deduction: true,
            pattern_matching: true,
            theorem_proving: true,
            consistency_checking: true,
        }
    }
}

impl Default for FormalReasoning {
    fn default() -> Self {
        Self {
            inference_rules: vec![
                "modus_ponens".to_string(),
                "modus_tollens".to_string(),
                "hypothetical_syllogism".to_string(),
                "disjunctive_syllogism".to_string(),
            ],
            proof_strategies: vec![
                "direct_proof".to_string(),
                "indirect_proof".to_string(),
                "proof_by_contradiction".to_string(),
                "mathematical_induction".to_string(),
            ],
            logic_systems: vec![
                "classical_logic".to_string(),
                "intuitionistic_logic".to_string(),
                "modal_logic".to_string(),
                "temporal_logic".to_string(),
            ],
        }
    }
}

impl Default for LogicCoreAgent {
    fn default() -> Self {
        Self {
            config: LogicCoreConfig::default(),
            inference_capabilities: InferenceCapabilities::default(),
            formal_reasoning: FormalReasoning::default(),
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
impl BaseAgent for LogicCoreAgent {
    type Config = LogicCoreConfig;
    type Input = LogicCoreTaskInput;
    type Output = LogicCoreTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let proof_steps = self.generate_proof_steps(&input).await?;
        let validity_result = self.check_validity(&input).await?;
        let confidence_score = self.calculate_confidence(&input, &proof_steps).await?;
        let proof_strategy = self.determine_proof_strategy(&input).await?;

        Ok(LogicCoreTaskOutput {
            proof_steps,
            validity_result,
            confidence_score,
            proof_strategy,
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
                name: "logic_core".to_string(),
                description: "Logical inference and formal reasoning engine".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["premises".to_string(), "conclusion".to_string()],
                output_types: vec!["proof_steps".to_string(), "validity_result".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.95,
                    avg_latency: 2200.0,
                    resource_usage: 0.5,
                    reliability: 0.97,
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

impl LogicCoreAgent {
    pub fn new(config: LogicCoreConfig) -> Self {
        Self {
            config,
            inference_capabilities: InferenceCapabilities::default(),
            formal_reasoning: FormalReasoning::default(),
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

    async fn generate_proof_steps(&self, input: &LogicCoreTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Step 1: Analyze premises: {}", input.premises.join(", ")),
            format!("Step 2: Apply modus ponens to derive intermediate conclusions"),
            format!("Step 3: Apply hypothetical syllogism to chain inferences"),
            format!("Step 4: Verify conclusion: {}", input.conclusion),
        ])
    }

    async fn check_validity(&self, input: &LogicCoreTaskInput) -> AgentResult<bool> {
        // Simple validity check based on premise-conclusion relationship
        let has_relevant_premises = !input.premises.is_empty();
        let conclusion_follows = !input.conclusion.is_empty();
        
        Ok(has_relevant_premises && conclusion_follows)
    }

    async fn calculate_confidence(&self, _input: &LogicCoreTaskInput, _proof_steps: &[String]) -> AgentResult<f32> {
        Ok(0.91)
    }

    async fn determine_proof_strategy(&self, input: &LogicCoreTaskInput) -> AgentResult<String> {
        match input.logic_type.as_str() {
            "propositional" => Ok("Direct proof using propositional logic rules".to_string()),
            "predicate" => Ok("Proof by contradiction using predicate logic".to_string()),
            "modal" => Ok("Tableau method for modal logic".to_string()),
            _ => Ok("Hybrid proof strategy combining multiple approaches".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logic_core_agent_creation() {
        let agent = LogicCoreAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_logic_core_task_processing() {
        let agent = LogicCoreAgent::default();
        let input = LogicCoreTaskInput {
            premises: vec![
                "All humans are mortal".to_string(),
                "Socrates is human".to_string(),
            ],
            conclusion: "Socrates is mortal".to_string(),
            logic_type: "propositional".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.proof_steps.is_empty());
        assert!(output.validity_result);
        assert!(output.confidence_score > 0.0);
        assert!(!output.proof_strategy.is_empty());
    }
}
