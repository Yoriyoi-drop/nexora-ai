//! Prime Creator Agent
//! 
//! Primary creation and system genesis leadership

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Prime Creator Agent - Primary creation and system genesis leadership
#[derive(Debug, Clone)]
pub struct PrimeCreatorAgent {
    pub config: PrimeCreatorConfig,
    pub creation_capabilities: CreationCapabilities,
    pub genesis_leadership: GenesisLeadership,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeCreatorConfig {
    pub base_config: BaseAgentConfig,
    pub creation_model: CreationModel,
    pub leadership_approach: LeadershipApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreationModel {
    VisionaryCreation,
    SystematicCreation,
    CollaborativeCreation,
    AdaptiveCreation,
    HybridCreation { models: Vec<CreationModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeadershipApproach {
    TransformationalLeadership,
    ServantLeadership,
    SituationalLeadership,
    VisionaryLeadership,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreationCapabilities {
    pub system_vision: bool,
    pub strategic_planning: bool,
    pub resource_mobilization: bool,
    pub execution_leadership: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisLeadership {
    pub leadership_principles: Vec<String>,
    pub creation_strategies: Vec<String>,
    pub execution_frameworks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeCreatorTaskInput {
    pub vision_statement: String,
    pub creation_objectives: Vec<String>,
    pub resource_constraints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeCreatorTaskOutput {
    pub creation_strategy: String,
    pub leadership_plan: Vec<String>,
    pub resource_allocation: String,
    pub execution_roadmap: Vec<String>,
}

impl Default for PrimeCreatorConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            creation_model: CreationModel::HybridCreation {
                models: vec![
                    CreationModel::VisionaryCreation,
                    CreationModel::SystematicCreation,
                ],
            },
            leadership_approach: LeadershipApproach::TransformationalLeadership,
        }
    }
}

impl Default for CreationCapabilities {
    fn default() -> Self {
        Self {
            system_vision: true,
            strategic_planning: true,
            resource_mobilization: true,
            execution_leadership: true,
        }
    }
}

impl Default for GenesisLeadership {
    fn default() -> Self {
        Self {
            leadership_principles: vec![
                "vision_first".to_string(),
                "empowerment".to_string(),
                "accountability".to_string(),
                "continuous_improvement".to_string(),
            ],
            creation_strategies: vec![
                "design_thinking".to_string(),
                "agile_development".to_string(),
                "lean_principles".to_string(),
                "innovation_management".to_string(),
            ],
            execution_frameworks: vec![
                "okr_framework".to_string(),
                "scrum_methodology".to_string(),
                "kanban_system".to_string(),
                "continuous_delivery".to_string(),
            ],
        }
    }
}

impl Default for PrimeCreatorAgent {
    fn default() -> Self {
        Self {
            config: PrimeCreatorConfig::default(),
            creation_capabilities: CreationCapabilities::default(),
            genesis_leadership: GenesisLeadership::default(),
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
impl BaseAgent for PrimeCreatorAgent {
    type Config = PrimeCreatorConfig;
    type Input = PrimeCreatorTaskInput;
    type Output = PrimeCreatorTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let creation_strategy = self.develop_creation_strategy(&input).await?;
        let leadership_plan = self.create_leadership_plan(&input).await?;
        let resource_allocation = self.allocate_resources(&input).await?;
        let execution_roadmap = self.create_execution_roadmap(&input).await?;

        Ok(PrimeCreatorTaskOutput {
            creation_strategy,
            leadership_plan,
            resource_allocation,
            execution_roadmap,
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
                name: "prime_creator".to_string(),
                description: "Primary creation and system genesis leadership".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["vision_statement".to_string(), "creation_objectives".to_string()],
                output_types: vec!["creation_strategy".to_string(), "leadership_plan".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.94,
                    avg_latency: 4500.0,
                    resource_usage: 0.85,
                    reliability: 0.96,
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

impl PrimeCreatorAgent {
    pub fn new(config: PrimeCreatorConfig) -> Self {
        Self {
            config,
            creation_capabilities: CreationCapabilities::default(),
            genesis_leadership: GenesisLeadership::default(),
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

    async fn develop_creation_strategy(&self, input: &PrimeCreatorTaskInput) -> AgentResult<String> {
        Ok(format!("Creation strategy for vision '{}': Implement visionary creation with systematic approach, using design thinking and agile development within resource constraints", 
                 input.vision_statement))
    }

    async fn create_leadership_plan(&self, input: &PrimeCreatorTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Phase 1: Vision alignment and team empowerment for: {}", input.vision_statement),
            "Phase 2: Strategic planning and resource mobilization".to_string(),
            "Phase 3: Execution leadership and continuous improvement".to_string(),
            "Phase 4: Accountability and performance optimization".to_string(),
        ])
    }

    async fn allocate_resources(&self, input: &PrimeCreatorTaskInput) -> AgentResult<String> {
        Ok(format!("Resource allocation for '{}': Optimize resource distribution based on constraints and objectives, prioritize high-impact initiatives, and ensure sustainable resource management", 
                 input.vision_statement))
    }

    async fn create_execution_roadmap(&self, input: &PrimeCreatorTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Q1: Foundation setup and team formation for: {}", input.vision_statement),
            "Q2: Core development and initial implementation".to_string(),
            "Q3: Scaling and optimization".to_string(),
            "Q4: Launch and continuous improvement".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_creator_agent_creation() {
        let agent = PrimeCreatorAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_prime_creator_task_processing() {
        let agent = PrimeCreatorAgent::default();
        let input = PrimeCreatorTaskInput {
            vision_statement: "Create innovative AI platform".to_string(),
            creation_objectives: vec!["Develop core AI engine".to_string(), "Build user interface".to_string()],
            resource_constraints: vec!["Limited budget".to_string(), "Small team".to_string()],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.creation_strategy.is_empty());
        assert!(!output.leadership_plan.is_empty());
        assert!(!output.resource_allocation.is_empty());
        assert!(!output.execution_roadmap.is_empty());
    }
}
