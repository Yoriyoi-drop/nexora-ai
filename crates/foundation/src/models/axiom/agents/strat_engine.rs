//! STRAT-ENGINE Agent
//!
//! Long-term strategy generation and evaluation

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

#[derive(Debug, Clone)]
pub struct StratEngineAgent {
    pub config: StratEngineConfig,
    pub strategy_capabilities: StrategyCapabilities,
    pub evaluation_framework: EvaluationFramework,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratEngineConfig {
    pub base_config: BaseAgentConfig,
    pub strategy_model: StrategyModel,
    pub evaluation_method: EvaluationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyModel {
    CompetitiveStrategy,
    GrowthStrategy,
    InnovationStrategy,
    RiskAdjustedStrategy,
    HybridStrategy { models: Vec<StrategyModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationMethod {
    WeightedScoring,
    MultiCriteria,
    RealOptions,
    ScenarioAnalysis,
    ROIAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCapabilities {
    pub strategic_planning: bool,
    pub scenario_generation: bool,
    pub trade_off_analysis: bool,
    pub outcome_prediction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationFramework {
    pub evaluation_criteria: Vec<String>,
    pub scoring_methods: Vec<String>,
    pub optimization_goals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratEngineTaskInput {
    pub objective: String,
    pub constraints: Vec<String>,
    pub time_horizon: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratEngineTaskOutput {
    pub strategies: Vec<StrategyProposal>,
    pub recommended_strategy: StrategyProposal,
    pub evaluation_scores: HashMap<String, f32>,
    pub trade_off_analysis: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyProposal {
    pub id: String,
    pub name: String,
    pub description: String,
    pub expected_roi: f32,
    pub risk_level: f32,
    pub time_to_maturity: u32,
    pub confidence: f32,
}

impl Default for StratEngineConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            strategy_model: StrategyModel::HybridStrategy {
                models: vec![
                    StrategyModel::CompetitiveStrategy,
                    StrategyModel::GrowthStrategy,
                ],
            },
            evaluation_method: EvaluationMethod::MultiCriteria,
        }
    }
}

impl Default for StrategyCapabilities {
    fn default() -> Self {
        Self {
            strategic_planning: true,
            scenario_generation: true,
            trade_off_analysis: true,
            outcome_prediction: true,
        }
    }
}

impl Default for EvaluationFramework {
    fn default() -> Self {
        Self {
            evaluation_criteria: vec![
                "roi".to_string(),
                "risk".to_string(),
                "feasibility".to_string(),
                "time_horizon".to_string(),
            ],
            scoring_methods: vec![
                "weighted_sum".to_string(),
                "ahp".to_string(),
                "topsis".to_string(),
            ],
            optimization_goals: vec![
                "maximize_roi".to_string(),
                "minimize_risk".to_string(),
                "balanced".to_string(),
            ],
        }
    }
}

impl Default for StratEngineAgent {
    fn default() -> Self {
        Self {
            config: StratEngineConfig::default(),
            strategy_capabilities: StrategyCapabilities::default(),
            evaluation_framework: EvaluationFramework::default(),
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
impl BaseAgent for StratEngineAgent {
    type Config = StratEngineConfig;
    type Input = StratEngineTaskInput;
    type Output = StratEngineTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let strategies = self.generate_strategies(&input).await?;
        let evaluation_scores = self.evaluate_strategies(&input, &strategies).await?;
        let recommended_strategy = self.select_best_strategy(&strategies, &evaluation_scores).await?;
        let trade_off_analysis = self.analyze_trade_offs(&strategies, &evaluation_scores).await?;

        Ok(StratEngineTaskOutput {
            strategies,
            recommended_strategy,
            evaluation_scores,
            trade_off_analysis,
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
                name: "strat_engine".to_string(),
                description: "Long-term strategy generation and evaluation".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["objective".to_string(), "constraints".to_string(), "time_horizon".to_string()],
                output_types: vec!["strategies".to_string(), "evaluation_scores".to_string(), "recommendation".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.88,
                    avg_latency: 4200.0,
                    resource_usage: 0.75,
                    reliability: 0.90,
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

impl StratEngineAgent {
    pub fn new(config: StratEngineConfig) -> Self {
        Self {
            config,
            strategy_capabilities: StrategyCapabilities::default(),
            evaluation_framework: EvaluationFramework::default(),
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

    async fn generate_strategies(&self, input: &StratEngineTaskInput) -> AgentResult<Vec<StrategyProposal>> {
        Ok(vec![
            StrategyProposal {
                id: "STRAT-A".to_string(),
                name: "Aggressive Growth".to_string(),
                description: format!("Rapid expansion strategy for objective '{}'", input.objective),
                expected_roi: 0.25,
                risk_level: 0.70,
                time_to_maturity: input.time_horizon / 2,
                confidence: 0.72,
            },
            StrategyProposal {
                id: "STRAT-B".to_string(),
                name: "Balanced Approach".to_string(),
                description: "Moderate growth with controlled risk exposure".to_string(),
                expected_roi: 0.15,
                risk_level: 0.40,
                time_to_maturity: input.time_horizon,
                confidence: 0.85,
            },
            StrategyProposal {
                id: "STRAT-C".to_string(),
                name: "Conservative Defense".to_string(),
                description: "Capital preservation with minimal risk".to_string(),
                expected_roi: 0.05,
                risk_level: 0.15,
                time_to_maturity: input.time_horizon * 2,
                confidence: 0.93,
            },
        ])
    }

    async fn evaluate_strategies(&self, _input: &StratEngineTaskInput, strategies: &[StrategyProposal]) -> AgentResult<HashMap<String, f32>> {
        let mut scores = HashMap::new();
        for s in strategies {
            let score = s.expected_roi * 0.4 + (1.0 - s.risk_level) * 0.3 + s.confidence * 0.3;
            scores.insert(s.id.clone(), score);
        }
        Ok(scores)
    }

    async fn select_best_strategy(&self, strategies: &[StrategyProposal], scores: &HashMap<String, f32>) -> AgentResult<StrategyProposal> {
        strategies.iter()
            .max_by(|a, b| {
                let sa = scores.get(&a.id).copied().unwrap_or(0.0);
                let sb = scores.get(&b.id).copied().unwrap_or(0.0);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| crate::shared::agent_types::AgentError::ProcessingFailed("No strategies to evaluate".to_string()))
    }

    async fn analyze_trade_offs(&self, strategies: &[StrategyProposal], scores: &HashMap<String, f32>) -> AgentResult<String> {
        let analysis: Vec<String> = strategies.iter().map(|s| {
            let score = scores.get(&s.id).copied().unwrap_or(0.0);
            format!("{}: ROI={:.2}, Risk={:.2}, Score={:.2}", s.name, s.expected_roi, s.risk_level, score)
        }).collect();
        Ok(format!("Trade-off analysis:\n{}", analysis.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strat_engine_agent_creation() {
        let agent = StratEngineAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_strat_engine_task_processing() {
        let agent = StratEngineAgent::default();
        let input = StratEngineTaskInput {
            objective: "Increase market share by 20%".to_string(),
            constraints: vec!["Budget limit $10M".to_string(), "18-month timeline".to_string()],
            time_horizon: 365,
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.strategies.is_empty());
        assert!(output.evaluation_scores.len() == output.strategies.len());
        assert!(!output.trade_off_analysis.is_empty());
    }
}
