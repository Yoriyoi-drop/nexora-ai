//! Creation Architect Agent
//! 
//! Creative design and innovative solution generation

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Creation Architect Agent - Creative design and innovative solution generation
#[derive(Debug, Clone)]
pub struct CreationArchitectAgent {
    pub config: CreationArchitectConfig,
    pub creativity_capabilities: CreativityCapabilities,
    pub innovation_engine: InnovationEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreationArchitectConfig {
    pub base_config: BaseAgentConfig,
    pub creativity_model: CreativityModel,
    pub innovation_approach: InnovationApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativityModel {
    DivergentThinking,
    ConvergentThinking,
    LateralThinking,
    SystemsThinking,
    HybridModel { models: Vec<CreativityModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InnovationApproach {
    IncrementalInnovation,
    RadicalInnovation,
    DisruptiveInnovation,
    ArchitecturalInnovation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativityCapabilities {
    pub idea_generation: bool,
    pub pattern_recognition: bool,
    pub concept_synthesis: bool,
    pub solution_prototyping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationEngine {
    pub ideation_methods: Vec<String>,
    pub evaluation_criteria: Vec<String>,
    pub prototyping_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreationArchitectTaskInput {
    pub problem_statement: String,
    pub constraints: Vec<String>,
    pub objectives: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreationArchitectTaskOutput {
    pub creative_solutions: Vec<String>,
    pub innovation_score: f32,
    pub feasibility_assessment: String,
    pub implementation_roadmap: Vec<String>,
}

impl Default for CreationArchitectConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            creativity_model: CreativityModel::HybridModel {
                models: vec![
                    CreativityModel::DivergentThinking,
                    CreativityModel::ConvergentThinking,
                ],
            },
            innovation_approach: InnovationApproach::IncrementalInnovation,
        }
    }
}

impl Default for CreativityCapabilities {
    fn default() -> Self {
        Self {
            idea_generation: true,
            pattern_recognition: true,
            concept_synthesis: true,
            solution_prototyping: true,
        }
    }
}

impl Default for InnovationEngine {
    fn default() -> Self {
        Self {
            ideation_methods: vec![
                "brainstorming".to_string(),
                "mind_mapping".to_string(),
                "scamper".to_string(),
            ],
            evaluation_criteria: vec![
                "novelty".to_string(),
                "feasibility".to_string(),
                "impact".to_string(),
            ],
            prototyping_tools: vec![
                "sketches".to_string(),
                "mockups".to_string(),
                "models".to_string(),
            ],
        }
    }
}

impl Default for CreationArchitectAgent {
    fn default() -> Self {
        Self {
            config: CreationArchitectConfig::default(),
            creativity_capabilities: CreativityCapabilities::default(),
            innovation_engine: InnovationEngine::default(),
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
impl BaseAgent for CreationArchitectAgent {
    type Config = CreationArchitectConfig;
    type Input = CreationArchitectTaskInput;
    type Output = CreationArchitectTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let creative_solutions = self.generate_creative_solutions(&input).await?;
        let innovation_score = self.calculate_innovation_score(&creative_solutions).await?;
        let feasibility_assessment = self.assess_feasibility(&input, &creative_solutions).await?;
        let implementation_roadmap = self.create_implementation_roadmap(&creative_solutions).await?;

        Ok(CreationArchitectTaskOutput {
            creative_solutions,
            innovation_score,
            feasibility_assessment,
            implementation_roadmap,
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
                name: "creation_architecture".to_string(),
                description: "Creative design and innovative solution generation".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["problem_statement".to_string(), "constraints".to_string()],
                output_types: vec!["creative_solutions".to_string(), "innovation_score".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.86,
                    avg_latency: 3500.0,
                    resource_usage: 0.7,
                    reliability: 0.88,
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

impl CreationArchitectAgent {
    pub fn new(config: CreationArchitectConfig) -> Self {
        Self {
            config,
            creativity_capabilities: CreativityCapabilities::default(),
            innovation_engine: InnovationEngine::default(),
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

    async fn generate_creative_solutions(&self, input: &CreationArchitectTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Innovative solution for: {}", input.problem_statement),
            "Alternative approach using design thinking".to_string(),
            "Systems-based solution addressing root causes".to_string(),
        ])
    }

    async fn calculate_innovation_score(&self, _solutions: &[String]) -> AgentResult<f32> {
        Ok(0.78)
    }

    async fn assess_feasibility(&self, _input: &CreationArchitectTaskInput, _solutions: &[String]) -> AgentResult<String> {
        Ok("High feasibility with moderate resource requirements".to_string())
    }

    async fn create_implementation_roadmap(&self, _solutions: &[String]) -> AgentResult<Vec<String>> {
        Ok(vec![
            "Phase 1: Research and analysis".to_string(),
            "Phase 2: Prototype development".to_string(),
            "Phase 3: Testing and refinement".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_architect_agent_creation() {
        let agent = CreationArchitectAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_creation_architect_task_processing() {
        let agent = CreationArchitectAgent::default();
        let input = CreationArchitectTaskInput {
            problem_statement: "How to improve team productivity?".to_string(),
            constraints: vec!["Limited budget".to_string()],
            objectives: vec!["Increase efficiency".to_string()],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.creative_solutions.is_empty());
        assert!(output.innovation_score > 0.0);
        assert!(!output.feasibility_assessment.is_empty());
        assert!(!output.implementation_roadmap.is_empty());
    }
}
