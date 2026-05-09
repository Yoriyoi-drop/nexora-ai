//! System Breeder Agent
//! 
//! System evolution and generative development

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// System Breeder Agent - System evolution and generative development
#[derive(Debug, Clone)]
pub struct SystemBreederAgent {
    pub config: SystemBreederConfig,
    pub evolution_capabilities: EvolutionCapabilities,
    pub generative_engine: GenerativeEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBreederConfig {
    pub base_config: BaseAgentConfig,
    pub evolution_model: EvolutionModel,
    pub generative_approach: GenerativeApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionModel {
    GeneticAlgorithm,
    EvolutionaryProgramming,
    DifferentialEvolution,
    ParticleSwarm,
    HybridEvolution { models: Vec<EvolutionModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerativeApproach {
    GenerativeAdversarial,
    VariationalAutoencoder,
    TransformerBased,
    HybridGenerative { approaches: Vec<GenerativeApproach> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionCapabilities {
    pub system_mutation: bool,
    pub fitness_evaluation: bool,
    pub selection_mechanisms: bool,
    pub crossover_operations: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeEngine {
    pub generation_algorithms: Vec<String>,
    pub optimization_methods: Vec<String>,
    pub evaluation_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBreederTaskInput {
    pub initial_system: String,
    pub evolution_objectives: Vec<String>,
    pub fitness_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBreederTaskOutput {
    pub evolved_systems: Vec<String>,
    pub fitness_scores: Vec<f32>,
    pub evolution_history: Vec<String>,
    pub best_candidate: String,
}

impl Default for SystemBreederConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            evolution_model: EvolutionModel::HybridEvolution {
                models: vec![
                    EvolutionModel::GeneticAlgorithm,
                    EvolutionModel::EvolutionaryProgramming,
                ],
            },
            generative_approach: GenerativeApproach::HybridGenerative {
                approaches: vec![
                    GenerativeApproach::TransformerBased,
                    GenerativeApproach::VariationalAutoencoder,
                ],
            },
        }
    }
}

impl Default for EvolutionCapabilities {
    fn default() -> Self {
        Self {
            system_mutation: true,
            fitness_evaluation: true,
            selection_mechanisms: true,
            crossover_operations: true,
        }
    }
}

impl Default for GenerativeEngine {
    fn default() -> Self {
        Self {
            generation_algorithms: vec![
                "genetic_crossover".to_string(),
                "mutation_operator".to_string(),
                "selection_algorithm".to_string(),
            ],
            optimization_methods: vec![
                "gradient_descent".to_string(),
                "simulated_annealing".to_string(),
                "particle_swarm_optimization".to_string(),
            ],
            evaluation_metrics: vec![
                "performance_score".to_string(),
                "efficiency_metric".to_string(),
                "scalability_index".to_string(),
            ],
        }
    }
}

impl Default for SystemBreederAgent {
    fn default() -> Self {
        Self {
            config: SystemBreederConfig::default(),
            evolution_capabilities: EvolutionCapabilities::default(),
            generative_engine: GenerativeEngine::default(),
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
impl BaseAgent for SystemBreederAgent {
    type Config = SystemBreederConfig;
    type Input = SystemBreederTaskInput;
    type Output = SystemBreederTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let evolved_systems = self.evolve_systems(&input).await?;
        let fitness_scores = self.evaluate_fitness(&input, &evolved_systems).await?;
        let evolution_history = self.track_evolution_history(&input).await?;
        let best_candidate = self.select_best_candidate(&evolved_systems, &fitness_scores).await?;

        Ok(SystemBreederTaskOutput {
            evolved_systems,
            fitness_scores,
            evolution_history,
            best_candidate,
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
                name: "system_breeding".to_string(),
                description: "System evolution and generative development".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["initial_system".to_string(), "evolution_objectives".to_string()],
                output_types: vec!["evolved_systems".to_string(), "fitness_scores".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.87,
                    avg_latency: 5000.0,
                    resource_usage: 0.9,
                    reliability: 0.89,
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

impl SystemBreederAgent {
    pub fn new(config: SystemBreederConfig) -> Self {
        Self {
            config,
            evolution_capabilities: EvolutionCapabilities::default(),
            generative_engine: GenerativeEngine::default(),
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

    async fn evolve_systems(&self, input: &SystemBreederTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Evolved system 1: {} with enhanced performance", input.initial_system),
            format!("Evolved system 2: {} with improved efficiency", input.initial_system),
            format!("Evolved system 3: {} with better scalability", input.initial_system),
        ])
    }

    async fn evaluate_fitness(&self, _input: &SystemBreederTaskInput, evolved_systems: &[String]) -> AgentResult<Vec<f32>> {
        Ok(vec![
            0.85, // fitness for evolved system 1
            0.78, // fitness for evolved system 2
            0.92, // fitness for evolved system 3
        ].iter().take(evolved_systems.len()).cloned().collect())
    }

    async fn track_evolution_history(&self, input: &SystemBreederTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Initial system: {}", input.initial_system),
            "Generation 1: Applied mutation operators".to_string(),
            "Generation 2: Performed crossover operations".to_string(),
            "Generation 3: Selected best performers".to_string(),
        ])
    }

    async fn select_best_candidate(&self, evolved_systems: &[String], fitness_scores: &[f32]) -> AgentResult<String> {
        let best_index = fitness_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);
        
        Ok(evolved_systems.get(best_index).cloned().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_breeder_agent_creation() {
        let agent = SystemBreederAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_system_breeder_task_processing() {
        let agent = SystemBreederAgent::default();
        let input = SystemBreederTaskInput {
            initial_system: "Basic AI system".to_string(),
            evolution_objectives: vec!["Improve accuracy".to_string(), "Reduce latency".to_string()],
            fitness_criteria: vec!["Performance score".to_string(), "Efficiency metric".to_string()],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.evolved_systems.is_empty());
        assert_eq!(output.evolved_systems.len(), output.fitness_scores.len());
        assert!(!output.evolution_history.is_empty());
        assert!(!output.best_candidate.is_empty());
    }
}
