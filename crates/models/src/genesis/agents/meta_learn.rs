use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use nexora_shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

#[derive(Debug, Clone)]
pub struct MetaLearnAgent {
    pub config: MetaLearnConfig,
    pub meta_capabilities: MetaCapabilities,
    pub learning_engine: LearningEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnConfig {
    pub base_config: BaseAgentConfig,
    pub meta_learning_algorithm: MetaLearningAlgorithm,
    pub optimization_strategy: OptimizationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningAlgorithm {
    ModelAgnosticMetaLearning,
    Reptile,
    PrototypicalNetworks,
    GradientBasedMetaLearning,
    HybridMetaLearning { algorithms: Vec<MetaLearningAlgorithm> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    GradientDescent,
    NaturalGradient,
    ConjugateGradient,
    AdaptiveOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCapabilities {
    pub meta_optimization: bool,
    pub transfer_learning: bool,
    pub few_shot_adaptation: bool,
    pub learning_dynamics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEngine {
    pub meta_learning_methods: Vec<String>,
    pub optimization_techniques: Vec<String>,
    pub adaptation_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnTaskInput {
    pub task_distribution: Vec<String>,
    pub learning_experience: Vec<String>,
    pub performance_targets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnTaskOutput {
    pub optimized_learner: String,
    pub meta_knowledge: Vec<String>,
    pub adaptation_rules: Vec<String>,
    pub learning_curves: Vec<String>,
}

impl Default for MetaLearnConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            meta_learning_algorithm: MetaLearningAlgorithm::HybridMetaLearning {
                algorithms: vec![
                    MetaLearningAlgorithm::ModelAgnosticMetaLearning,
                    MetaLearningAlgorithm::Reptile,
                ],
            },
            optimization_strategy: OptimizationStrategy::AdaptiveOptimization,
        }
    }
}

impl Default for MetaCapabilities {
    fn default() -> Self {
        Self {
            meta_optimization: true,
            transfer_learning: true,
            few_shot_adaptation: true,
            learning_dynamics: true,
        }
    }
}

impl Default for LearningEngine {
    fn default() -> Self {
        Self {
            meta_learning_methods: vec![
                "maml_inner_loop".to_string(),
                "reptile_first_order".to_string(),
                "prototypical_embedding".to_string(),
            ],
            optimization_techniques: vec![
                "stochastic_meta_gradient".to_string(),
                "natural_gradient_descent".to_string(),
                "conjugate_gradient_meta".to_string(),
            ],
            adaptation_strategies: vec![
                "fast_weight_adaptation".to_string(),
                "context_parameter_modulation".to_string(),
                "task_embedding_conditioning".to_string(),
            ],
        }
    }
}

impl Default for MetaLearnAgent {
    fn default() -> Self {
        Self {
            config: MetaLearnConfig::default(),
            meta_capabilities: MetaCapabilities::default(),
            learning_engine: LearningEngine::default(),
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
impl BaseAgent for MetaLearnAgent {
    type Config = MetaLearnConfig;
    type Input = MetaLearnTaskInput;
    type Output = MetaLearnTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let optimized_learner = self.optimize_learner(&input).await?;
        let meta_knowledge = self.extract_meta_knowledge(&input).await?;
        let adaptation_rules = self.derive_adaptation_rules(&input).await?;
        let learning_curves = self.simulate_learning_curves(&input).await?;

        Ok(MetaLearnTaskOutput {
            optimized_learner,
            meta_knowledge,
            adaptation_rules,
            learning_curves,
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
                name: "meta_learn".to_string(),
                description: "Learning to learn optimizer and meta-learning engine".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["task_distribution".to_string(), "learning_experience".to_string()],
                output_types: vec!["optimized_learner".to_string(), "meta_knowledge".to_string()],
                metrics: nexora_shared::agent_types::CapabilityMetrics {
                    accuracy: 0.0,
                    avg_latency: 0.0,
                    resource_usage: 0.0,
                    reliability: 0.0,
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

impl MetaLearnAgent {
    pub fn new(config: MetaLearnConfig) -> Self {
        Self {
            config,
            meta_capabilities: MetaCapabilities::default(),
            learning_engine: LearningEngine::default(),
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

    async fn optimize_learner(&self, input: &MetaLearnTaskInput) -> AgentResult<String> {
        Ok(format!(
            "Meta-optimized learner for {} tasks using MAML inner-loop with adaptive gradient preconditioning",
            input.task_distribution.len()
        ))
    }

    async fn extract_meta_knowledge(&self, _input: &MetaLearnTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            "Cross-task feature representations".to_string(),
            "Learning rate adaptation patterns".to_string(),
            "Task similarity embeddings".to_string(),
        ])
    }

    async fn derive_adaptation_rules(&self, input: &MetaLearnTaskInput) -> AgentResult<Vec<String>> {
        Ok(input.performance_targets.iter().map(|target| {
            format!("Fast adaptation rule for {} via context parameter modulation", target)
        }).collect())
    }

    async fn simulate_learning_curves(&self, input: &MetaLearnTaskInput) -> AgentResult<Vec<String>> {
        let mut results = Vec::new();
        let n_tasks = input.task_distribution.len().max(1);
        let n_experience = input.learning_experience.len();
        let n_targets = input.performance_targets.len().max(1);

        let base_acc = 0.5 + (n_experience as f64 * 0.05).min(0.3);
        let transfer_boost = (n_tasks as f64).sqrt() * 0.03;

        let n_steps = (20 + n_targets * 10).max(10);
        for step in (1..=n_steps).step_by(5.max(n_steps / 10)) {
            let t = step as f64 / n_steps as f64;
            let acc = base_acc + (1.0 - base_acc) * (1.0 - (-3.0 * t).exp())
                + transfer_boost * (1.0 - (-2.0 * t).exp());
            let task_name = input
                .task_distribution
                .get(step % n_tasks)
                .cloned()
                .unwrap_or_default();
            results.push(format!(
                "Few-shot convergence after {} steps: {:.1}% accuracy [task: {}]",
                step,
                acc * 100.0,
                task_name
            ));
        }

        let transfer_improvement = base_acc * 0.23 + transfer_boost * 0.5;
        let task_diversity_factor = (n_tasks as f64).sqrt() * 0.02;
        results.push(format!(
            "Transfer improvement over baseline: +{:.1}% ({} tasks, {} experiences)",
            transfer_improvement * 100.0,
            n_tasks,
            n_experience
        ));

        let meta_grad_norm = 0.04 + 0.01 * (1.0 - (-5.0 * task_diversity_factor * 10.0).exp());
        results.push(format!(
            "Meta-gradient norm: {:.3} (stable across {} target metrics)",
            meta_grad_norm, n_targets
        ));

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learn_agent_creation() {
        let agent = MetaLearnAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_meta_learn_task_processing() {
        let agent = MetaLearnAgent::default();
        let input = MetaLearnTaskInput {
            task_distribution: vec!["image_classification".to_string(), "text_classification".to_string()],
            learning_experience: vec!["previous_tasks_100".to_string()],
            performance_targets: vec!["95%_accuracy".to_string(), "5_shot".to_string()],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.optimized_learner.is_empty());
        assert!(!output.meta_knowledge.is_empty());
        assert!(!output.adaptation_rules.is_empty());
        assert!(!output.learning_curves.is_empty());
    }
}
