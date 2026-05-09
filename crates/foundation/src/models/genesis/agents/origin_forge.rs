//! Origin Forge Agent
//! 
//! System origin creation and foundational architecture

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Origin Forge Agent - System origin creation and foundational architecture
#[derive(Debug, Clone)]
pub struct OriginForgeAgent {
    pub config: OriginForgeConfig,
    pub origination_capabilities: OriginationCapabilities,
    pub foundational_architecture: FoundationalArchitecture,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginForgeConfig {
    pub base_config: BaseAgentConfig,
    pub origination_model: OriginationModel,
    pub architecture_style: ArchitectureStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OriginationModel {
    CleanArchitecture,
    DomainDrivenDesign,
    MicroservicesFirst,
    EventSourced,
    HybridOrigination { models: Vec<OriginationModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureStyle {
    Monolithic,
    Microservices,
    Serverless,
    Distributed,
    HybridArchitecture { styles: Vec<ArchitectureStyle> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginationCapabilities {
    pub system_design: bool,
    pub component_architecture: bool,
    pub interface_definition: bool,
    pub scalability_planning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundationalArchitecture {
    pub design_principles: Vec<String>,
    pub architectural_patterns: Vec<String>,
    pub scalability_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginForgeTaskInput {
    pub system_domain: String,
    pub business_requirements: Vec<String>,
    pub technical_constraints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginForgeTaskOutput {
    pub origin_architecture: String,
    pub foundational_components: Vec<String>,
    pub scalability_plan: String,
    pub implementation_phases: Vec<String>,
}

impl Default for OriginForgeConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            origination_model: OriginationModel::HybridOrigination {
                models: vec![
                    OriginationModel::CleanArchitecture,
                    OriginationModel::DomainDrivenDesign,
                ],
            },
            architecture_style: ArchitectureStyle::Microservices,
        }
    }
}

impl Default for OriginationCapabilities {
    fn default() -> Self {
        Self {
            system_design: true,
            component_architecture: true,
            interface_definition: true,
            scalability_planning: true,
        }
    }
}

impl Default for FoundationalArchitecture {
    fn default() -> Self {
        Self {
            design_principles: vec![
                "single_responsibility".to_string(),
                "open_closed".to_string(),
                "dependency_inversion".to_string(),
                "interface_segregation".to_string(),
            ],
            architectural_patterns: vec![
                "repository_pattern".to_string(),
                "unit_of_work".to_string(),
                "command_query_separation".to_string(),
                "event_sourcing".to_string(),
            ],
            scalability_strategies: vec![
                "horizontal_scaling".to_string(),
                "load_balancing".to_string(),
                "caching_strategies".to_string(),
                "database_sharding".to_string(),
            ],
        }
    }
}

impl Default for OriginForgeAgent {
    fn default() -> Self {
        Self {
            config: OriginForgeConfig::default(),
            origination_capabilities: OriginationCapabilities::default(),
            foundational_architecture: FoundationalArchitecture::default(),
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
impl BaseAgent for OriginForgeAgent {
    type Config = OriginForgeConfig;
    type Input = OriginForgeTaskInput;
    type Output = OriginForgeTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let origin_architecture = self.create_origin_architecture(&input).await?;
        let foundational_components = self.define_foundational_components(&input).await?;
        let scalability_plan = self.create_scalability_plan(&input).await?;
        let implementation_phases = self.plan_implementation_phases(&input).await?;

        Ok(OriginForgeTaskOutput {
            origin_architecture,
            foundational_components,
            scalability_plan,
            implementation_phases,
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
                name: "origin_forge".to_string(),
                description: "System origin creation and foundational architecture".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["system_domain".to_string(), "business_requirements".to_string()],
                output_types: vec!["origin_architecture".to_string(), "foundational_components".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.90,
                    avg_latency: 3800.0,
                    resource_usage: 0.78,
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

impl OriginForgeAgent {
    pub fn new(config: OriginForgeConfig) -> Self {
        Self {
            config,
            origination_capabilities: OriginationCapabilities::default(),
            foundational_architecture: FoundationalArchitecture::default(),
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

    async fn create_origin_architecture(&self, input: &OriginForgeTaskInput) -> AgentResult<String> {
        Ok(format!("Origin architecture for '{}': Clean architecture with domain-driven design, microservices style, and event-sourced persistence", input.system_domain))
    }

    async fn define_foundational_components(&self, input: &OriginForgeTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Domain Layer: Core business logic for {}", input.system_domain),
            "Application Layer: Use cases and application services".to_string(),
            "Infrastructure Layer: External integrations and data persistence".to_string(),
            "Presentation Layer: API endpoints and user interfaces".to_string(),
        ])
    }

    async fn create_scalability_plan(&self, input: &OriginForgeTaskInput) -> AgentResult<String> {
        Ok(format!("Scalability plan for '{}': Horizontal scaling with load balancing, caching strategies, and database sharding to handle growth", input.system_domain))
    }

    async fn plan_implementation_phases(&self, _input: &OriginForgeTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            "Phase 1: Core domain and infrastructure setup".to_string(),
            "Phase 2: Application layer and use case implementation".to_string(),
            "Phase 3: Presentation layer and API development".to_string(),
            "Phase 4: Testing, deployment, and optimization".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin_forge_agent_creation() {
        let agent = OriginForgeAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_origin_forge_task_processing() {
        let agent = OriginForgeAgent::default();
        let input = OriginForgeTaskInput {
            system_domain: "E-commerce platform".to_string(),
            business_requirements: vec!["Product catalog".to_string(), "Order processing".to_string()],
            technical_constraints: vec!["High availability".to_string()],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.origin_architecture.is_empty());
        assert!(!output.foundational_components.is_empty());
        assert!(!output.scalability_plan.is_empty());
        assert!(!output.implementation_phases.is_empty());
    }
}
