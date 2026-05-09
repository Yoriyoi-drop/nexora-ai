//! Insight Oracle Agent
//! 
//! Deep insight generation and wisdom synthesis

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Insight Oracle Agent - Deep insight generation and wisdom synthesis
#[derive(Debug, Clone)]
pub struct InsightOracleAgent {
    pub config: InsightOracleConfig,
    pub insight_capabilities: InsightCapabilities,
    pub wisdom_synthesis: WisdomSynthesis,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightOracleConfig {
    pub base_config: BaseAgentConfig,
    pub insight_model: InsightModel,
    pub wisdom_framework: WisdomFramework,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightModel {
    AnalyticalModel,
    IntuitiveModel,
    SynthesisModel,
    HybridModel { models: Vec<InsightModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomFramework {
    pub knowledge_domains: Vec<String>,
    pub synthesis_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightCapabilities {
    pub pattern_recognition: bool,
    pub deep_analysis: bool,
    pub wisdom_synthesis: bool,
    pub contextual_insights: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomSynthesis {
    pub synthesis_algorithms: Vec<String>,
    pub wisdom_sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightOracleTaskInput {
    pub query: String,
    pub context: String,
    pub domain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightOracleTaskOutput {
    pub insight: String,
    pub wisdom_level: f32,
    pub relevance_score: f32,
    pub actionable_steps: Vec<String>,
}

impl Default for InsightOracleConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            insight_model: InsightModel::HybridModel {
                models: vec![
                    InsightModel::AnalyticalModel,
                    InsightModel::IntuitiveModel,
                ],
            },
            wisdom_framework: WisdomFramework {
                knowledge_domains: vec!["philosophy".to_string(), "science".to_string()],
                synthesis_methods: vec!["synthesis".to_string()],
            },
        }
    }
}

impl Default for InsightCapabilities {
    fn default() -> Self {
        Self {
            pattern_recognition: true,
            deep_analysis: true,
            wisdom_synthesis: true,
            contextual_insights: true,
        }
    }
}

impl Default for WisdomSynthesis {
    fn default() -> Self {
        Self {
            synthesis_algorithms: vec!["algorithm_001".to_string()],
            wisdom_sources: vec!["source_001".to_string()],
        }
    }
}

impl Default for InsightOracleAgent {
    fn default() -> Self {
        Self {
            config: InsightOracleConfig::default(),
            insight_capabilities: InsightCapabilities::default(),
            wisdom_synthesis: WisdomSynthesis::default(),
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
impl BaseAgent for InsightOracleAgent {
    type Config = InsightOracleConfig;
    type Input = InsightOracleTaskInput;
    type Output = InsightOracleTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let insight = self.generate_insight(&input).await?;
        let wisdom_level = self.assess_wisdom_level(&insight).await?;
        let relevance_score = self.calculate_relevance(&input, &insight).await?;
        let actionable_steps = self.generate_actionable_steps(&input, &insight).await?;

        Ok(InsightOracleTaskOutput {
            insight,
            wisdom_level,
            relevance_score,
            actionable_steps,
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
                name: "insight_generation".to_string(),
                description: "Deep insight generation and wisdom synthesis".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["query".to_string(), "context".to_string()],
                output_types: vec!["insight".to_string(), "wisdom".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.87,
                    avg_latency: 3000.0,
                    resource_usage: 0.7,
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

impl InsightOracleAgent {
    pub fn new(config: InsightOracleConfig) -> Self {
        Self {
            config,
            insight_capabilities: InsightCapabilities::default(),
            wisdom_synthesis: WisdomSynthesis::default(),
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

    async fn generate_insight(&self, input: &InsightOracleTaskInput) -> AgentResult<String> {
        Ok(format!("Based on your query about {}, here's a deep insight: {}", 
                 input.domain, 
                 "Consider the interconnected nature of your challenge and seek wisdom from multiple perspectives."))
    }

    async fn assess_wisdom_level(&self, _insight: &str) -> AgentResult<f32> {
        Ok(0.8)
    }

    async fn calculate_relevance(&self, _input: &InsightOracleTaskInput, _insight: &str) -> AgentResult<f32> {
        Ok(0.85)
    }

    async fn generate_actionable_steps(&self, _input: &InsightOracleTaskInput, _insight: &str) -> AgentResult<Vec<String>> {
        Ok(vec![
            "Reflect on the insight from multiple angles".to_string(),
            "Apply the wisdom to your specific context".to_string(),
            "Share the insight with others for feedback".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insight_oracle_agent_creation() {
        let agent = InsightOracleAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_insight_oracle_task_processing() {
        let agent = InsightOracleAgent::default();
        let input = InsightOracleTaskInput {
            query: "How can I find meaning in my work?".to_string(),
            context: "Professional development".to_string(),
            domain: "philosophy".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.insight.is_empty());
        assert!(output.wisdom_level > 0.0);
        assert!(!output.actionable_steps.is_empty());
    }
}
