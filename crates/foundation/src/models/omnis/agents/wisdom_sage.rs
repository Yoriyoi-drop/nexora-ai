//! Wisdom Sage Agent
//! 
//! Ancient wisdom integration and modern application

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Wisdom Sage Agent - Ancient wisdom integration and modern application
#[derive(Debug, Clone)]
pub struct WisdomSageAgent {
    pub config: WisdomSageConfig,
    pub wisdom_capabilities: WisdomCapabilities,
    pub ancient_wisdom: AncientWisdom,
    pub modern_application: ModernApplication,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomSageConfig {
    pub base_config: BaseAgentConfig,
    pub wisdom_traditions: Vec<WisdomTradition>,
    pub application_domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WisdomTradition {
    EasternPhilosophy,
    WesternPhilosophy,
    IndigenousWisdom,
    SpiritualTraditions,
    MysticalTraditions,
    IntegratedWisdom { traditions: Vec<WisdomTradition> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomCapabilities {
    pub ancient_knowledge: bool,
    pub modern_synthesis: bool,
    pub practical_application: bool,
    pub timeless_principles: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AncientWisdom {
    pub philosophical_principles: Vec<String>,
    pub timeless_truths: Vec<String>,
    pub wisdom_sayings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModernApplication {
    pub contemporary_contexts: Vec<String>,
    pub practical_solutions: Vec<String>,
    pub integration_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomSageTaskInput {
    pub question: String,
    pub context: String,
    pub urgency_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomSageTaskOutput {
    pub wisdom_response: String,
    pub ancient_principle: String,
    pub modern_application: String,
    pub practical_guidance: Vec<String>,
}

impl Default for WisdomSageConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            wisdom_traditions: vec![
                WisdomTradition::EasternPhilosophy,
                WisdomTradition::WesternPhilosophy,
            ],
            application_domains: vec![
                "personal_growth".to_string(),
                "leadership".to_string(),
            ],
        }
    }
}

impl Default for WisdomCapabilities {
    fn default() -> Self {
        Self {
            ancient_knowledge: true,
            modern_synthesis: true,
            practical_application: true,
            timeless_principles: true,
        }
    }
}

impl Default for AncientWisdom {
    fn default() -> Self {
        Self {
            philosophical_principles: vec![
                "The middle way".to_string(),
                "Balance and harmony".to_string(),
                "Know thyself".to_string(),
            ],
            timeless_truths: vec![
                "Change is constant".to_string(),
                "All things are connected".to_string(),
                "Wisdom comes from experience".to_string(),
            ],
            wisdom_sayings: vec![
                "The journey of a thousand miles begins with a single step".to_string(),
                "As you think, so shall you become".to_string(),
            ],
        }
    }
}

impl Default for ModernApplication {
    fn default() -> Self {
        Self {
            contemporary_contexts: vec![
                "digital age".to_string(),
                "global connectivity".to_string(),
                "rapid change".to_string(),
            ],
            practical_solutions: vec![
                "mindful leadership".to_string(),
                "sustainable living".to_string(),
                "ethical technology".to_string(),
            ],
            integration_methods: vec![
                "meditation practice".to_string(),
                "reflective journaling".to_string(),
                "community dialogue".to_string(),
            ],
        }
    }
}

impl Default for WisdomSageAgent {
    fn default() -> Self {
        Self {
            config: WisdomSageConfig::default(),
            wisdom_capabilities: WisdomCapabilities::default(),
            ancient_wisdom: AncientWisdom::default(),
            modern_application: ModernApplication::default(),
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
impl BaseAgent for WisdomSageAgent {
    type Config = WisdomSageConfig;
    type Input = WisdomSageTaskInput;
    type Output = WisdomSageTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let wisdom_response = self.generate_wisdom_response(&input).await?;
        let ancient_principle = self.extract_ancient_principle(&input).await?;
        let modern_application = self.create_modern_application(&input, &ancient_principle).await?;
        let practical_guidance = self.provide_practical_guidance(&input).await?;

        Ok(WisdomSageTaskOutput {
            wisdom_response,
            ancient_principle,
            modern_application,
            practical_guidance,
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
                name: "wisdom_sage".to_string(),
                description: "Ancient wisdom integration and modern application".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["question".to_string(), "context".to_string()],
                output_types: vec!["wisdom_response".to_string(), "practical_guidance".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.91,
                    avg_latency: 2500.0,
                    resource_usage: 0.5,
                    reliability: 0.93,
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

impl WisdomSageAgent {
    pub fn new(config: WisdomSageConfig) -> Self {
        Self {
            config,
            wisdom_capabilities: WisdomCapabilities::default(),
            ancient_wisdom: AncientWisdom::default(),
            modern_application: ModernApplication::default(),
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

    async fn generate_wisdom_response(&self, input: &WisdomSageTaskInput) -> AgentResult<String> {
        Ok(format!("In response to your question '{}', consider this timeless wisdom: {}", 
                 input.question, 
                 "The greatest wisdom often comes from understanding that we are part of something larger than ourselves."))
    }

    async fn extract_ancient_principle(&self, _input: &WisdomSageTaskInput) -> AgentResult<String> {
        Ok("The principle of interconnectedness - all things are related and influence each other".to_string())
    }

    async fn create_modern_application(&self, _input: &WisdomSageTaskInput, _principle: &str) -> AgentResult<String> {
        Ok("Apply this principle by considering how your actions affect others and the environment in our interconnected world".to_string())
    }

    async fn provide_practical_guidance(&self, _input: &WisdomSageTaskInput) -> AgentResult<Vec<String>> {
        Ok(vec![
            "Practice mindfulness to recognize connections".to_string(),
            "Consider long-term consequences of your actions".to_string(),
            "Seek harmony between personal and collective good".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wisdom_sage_agent_creation() {
        let agent = WisdomSageAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_wisdom_sage_task_processing() {
        let agent = WisdomSageAgent::default();
        let input = WisdomSageTaskInput {
            question: "How can I find balance in my busy life?".to_string(),
            context: "Work-life balance".to_string(),
            urgency_level: "medium".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.wisdom_response.is_empty());
        assert!(!output.ancient_principle.is_empty());
        assert!(!output.modern_application.is_empty());
        assert!(!output.practical_guidance.is_empty());
    }
}
