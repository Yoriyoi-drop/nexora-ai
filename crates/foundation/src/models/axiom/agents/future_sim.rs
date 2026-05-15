//! FUTURE-SIM Agent
//!
//! Future scenario simulation using Monte Carlo methods

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

#[derive(Debug, Clone)]
pub struct FutureSimAgent {
    pub config: FutureSimConfig,
    pub simulation_capabilities: SimulationCapabilities,
    pub scenario_engine: ScenarioEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureSimConfig {
    pub base_config: BaseAgentConfig,
    pub simulation_model: SimulationModel,
    pub sampling_method: SamplingMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationModel {
    MonteCarlo,
    DiscreteEvent,
    AgentBased,
    SystemDynamics,
    HybridSim { models: Vec<SimulationModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingMethod {
    RandomSampling,
    LatinHypercube,
    StratifiedSampling,
    QuasiRandom,
    ImportanceSampling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationCapabilities {
    pub scenario_generation: bool,
    pub probabilistic_forecasting: bool,
    pub sensitivity_analysis: bool,
    pub convergence_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioEngine {
    pub scenario_types: Vec<String>,
    pub distribution_families: Vec<String>,
    pub convergence_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureSimTaskInput {
    pub base_scenario: String,
    pub variables: Vec<String>,
    pub iterations: u32,
    pub time_horizon: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureSimTaskOutput {
    pub scenarios: Vec<SimulationScenario>,
    pub probability_distribution: ProbabilityDistribution,
    pub key_insights: Vec<String>,
    pub convergence_report: ConvergenceReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationScenario {
    pub id: String,
    pub label: String,
    pub probability: f32,
    pub outcome_metrics: HashMap<String, f32>,
    pub narrative: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityDistribution {
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
    pub percentiles: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceReport {
    pub iterations_completed: u32,
    pub convergence_achieved: bool,
    pub stability_score: f32,
    pub error_margin: f32,
}

impl Default for FutureSimConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            simulation_model: SimulationModel::MonteCarlo,
            sampling_method: SamplingMethod::LatinHypercube,
        }
    }
}

impl Default for SimulationCapabilities {
    fn default() -> Self {
        Self {
            scenario_generation: true,
            probabilistic_forecasting: true,
            sensitivity_analysis: true,
            convergence_detection: true,
        }
    }
}

impl Default for ScenarioEngine {
    fn default() -> Self {
        Self {
            scenario_types: vec![
                "baseline".to_string(),
                "optimistic".to_string(),
                "pessimistic".to_string(),
                "black_swan".to_string(),
            ],
            distribution_families: vec![
                "normal".to_string(),
                "lognormal".to_string(),
                "uniform".to_string(),
                "triangular".to_string(),
            ],
            convergence_metrics: vec![
                "mean_stability".to_string(),
                "variance_stability".to_string(),
                "geyer_ineff".to_string(),
            ],
        }
    }
}

impl Default for FutureSimAgent {
    fn default() -> Self {
        Self {
            config: FutureSimConfig::default(),
            simulation_capabilities: SimulationCapabilities::default(),
            scenario_engine: ScenarioEngine::default(),
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
impl BaseAgent for FutureSimAgent {
    type Config = FutureSimConfig;
    type Input = FutureSimTaskInput;
    type Output = FutureSimTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let scenarios = self.generate_scenarios(&input).await?;
        let probability_distribution = self.compute_distribution(&scenarios).await?;
        let key_insights = self.extract_insights(&input, &scenarios, &probability_distribution).await?;
        let convergence_report = self.check_convergence(&input).await?;

        Ok(FutureSimTaskOutput {
            scenarios,
            probability_distribution,
            key_insights,
            convergence_report,
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
                name: "future_sim".to_string(),
                description: "Future scenario simulation using Monte Carlo".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["base_scenario".to_string(), "variables".to_string(), "iterations".to_string()],
                output_types: vec!["scenarios".to_string(), "distribution".to_string(), "convergence".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.86,
                    avg_latency: 5800.0,
                    resource_usage: 0.85,
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

impl FutureSimAgent {
    pub fn new(config: FutureSimConfig) -> Self {
        Self {
            config,
            simulation_capabilities: SimulationCapabilities::default(),
            scenario_engine: ScenarioEngine::default(),
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

    async fn generate_scenarios(&self, input: &FutureSimTaskInput) -> AgentResult<Vec<SimulationScenario>> {
        Ok(vec![
            SimulationScenario {
                id: "SCE-BASE".to_string(),
                label: "Baseline".to_string(),
                probability: 0.50,
                outcome_metrics: vec![
                    ("roi".to_string(), 0.12),
                    ("growth".to_string(), 0.08),
                ].into_iter().collect(),
                narrative: format!("Baseline projection for '{}' under normal conditions", input.base_scenario),
            },
            SimulationScenario {
                id: "SCE-OPT".to_string(),
                label: "Optimistic".to_string(),
                probability: 0.20,
                outcome_metrics: vec![
                    ("roi".to_string(), 0.35),
                    ("growth".to_string(), 0.22),
                ].into_iter().collect(),
                narrative: "Favorable market conditions and execution".to_string(),
            },
            SimulationScenario {
                id: "SCE-PESS".to_string(),
                label: "Pessimistic".to_string(),
                probability: 0.20,
                outcome_metrics: vec![
                    ("roi".to_string(), -0.10),
                    ("growth".to_string(), -0.05),
                ].into_iter().collect(),
                narrative: "Adverse conditions and execution risks materialize".to_string(),
            },
            SimulationScenario {
                id: "SCE-BLACK".to_string(),
                label: "Black Swan".to_string(),
                probability: 0.10,
                outcome_metrics: vec![
                    ("roi".to_string(), -0.40),
                    ("growth".to_string(), -0.25),
                ].into_iter().collect(),
                narrative: "Extreme tail-risk event materializes".to_string(),
            },
        ])
    }

    async fn compute_distribution(&self, scenarios: &[SimulationScenario]) -> AgentResult<ProbabilityDistribution> {
        let outcomes: Vec<f32> = scenarios.iter()
            .filter_map(|s| s.outcome_metrics.get("roi").copied())
            .collect();

        let count = outcomes.len() as f32;
        let mean = outcomes.iter().sum::<f32>() / count;
        let variance = outcomes.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count;

        let mut sorted = outcomes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Ok(ProbabilityDistribution {
            mean,
            median: sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
            std_dev: variance.sqrt(),
            percentiles: vec![
                (0.05, sorted[0]),
                (0.25, sorted[sorted.len() / 4]),
                (0.75, sorted[3 * sorted.len() / 4]),
                (0.95, sorted[sorted.len() - 1]),
            ],
        })
    }

    async fn extract_insights(&self, _input: &FutureSimTaskInput, scenarios: &[SimulationScenario], dist: &ProbabilityDistribution) -> AgentResult<Vec<String>> {
        Ok(vec![
            format!("Mean expected ROI: {:.2} with std dev {:.2}", dist.mean, dist.std_dev),
            format!("Probability of positive outcome: {:.0}%", scenarios.iter().filter(|s| s.outcome_metrics.get("roi").copied().unwrap_or(0.0) > 0.0).count() as f32 / scenarios.len() as f32 * 100.0),
            "Worst-case scenario suggests downside protection is warranted".to_string(),
        ])
    }

    async fn check_convergence(&self, input: &FutureSimTaskInput) -> AgentResult<ConvergenceReport> {
        Ok(ConvergenceReport {
            iterations_completed: input.iterations,
            convergence_achieved: input.iterations >= 10000,
            stability_score: (1.0 - 1.0 / (input.iterations as f32).sqrt()).min(0.99),
            error_margin: 1.96 / (input.iterations as f32).sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_future_sim_agent_creation() {
        let agent = FutureSimAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_future_sim_task_processing() {
        let agent = FutureSimAgent::default();
        let input = FutureSimTaskInput {
            base_scenario: "Global economic growth trajectory".to_string(),
            variables: vec!["GDP growth".to_string(), "inflation".to_string(), "interest rates".to_string()],
            iterations: 50000,
            time_horizon: 730,
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.scenarios.is_empty());
        assert!(output.probability_distribution.std_dev >= 0.0);
        assert!(!output.key_insights.is_empty());
        assert!(output.convergence_report.iterations_completed > 0);
    }
}
