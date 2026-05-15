//! Base Agent Trait and Common Functionality
//! 
//! Provides the foundation for all NXR agents

use async_trait::async_trait;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::shared::base_model::NxrModelResult;
use super::agent_types::{AgentStatus, AgentCapability, TaskPriority, AgentMetrics, AgentResult, AgentError};

/// Base trait that all NXR agents must implement
#[async_trait]
pub trait BaseAgent: Send + Sync {
    /// Agent-specific configuration type
    type Config: Clone + Send + Sync;
    
    /// Input type for this agent
    type Input: Clone + Send + Sync;
    
    /// Output type for this agent
    type Output: Clone + Send + Sync;

    /// Process a single task
    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output>;
    
    /// Get agent identifier
    fn agent_id(&self) -> &str;
    
    /// Get current agent status
    fn get_status(&self) -> AgentStatus;
    
    /// Get agent capabilities
    fn get_capabilities(&self) -> Vec<AgentCapability>;
    
    /// Get agent metrics
    fn get_metrics(&self) -> AgentMetrics;
    
    /// Initialize the agent with configuration
    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()>;
    
    /// Shutdown the agent gracefully
    async fn shutdown(&mut self) -> AgentResult<()>;
    
    /// Health check
    async fn health_check(&self) -> AgentResult<bool> {
        Ok(matches!(self.get_status(), AgentStatus::Idle | AgentStatus::Processing))
    }
    
    /// Check if agent can handle specific input type
    fn can_handle_input(&self, input_type: &str) -> bool {
        self.get_capabilities()
            .iter()
            .any(|cap| cap.input_types.contains(&input_type.to_string()))
    }
    
    /// Get supported input types
    fn supported_input_types(&self) -> Vec<String> {
        self.get_capabilities()
            .iter()
            .flat_map(|cap| cap.input_types.clone())
            .collect()
    }
    
    /// Get supported output types
    fn supported_output_types(&self) -> Vec<String> {
        self.get_capabilities()
            .iter()
            .flat_map(|cap| cap.output_types.clone())
            .collect()
    }
}

/// Agent configuration base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseAgentConfig {
    /// Agent identifier
    pub agent_id: String,
    /// Agent name
    pub name: String,
    /// Agent version
    pub version: String,
    /// Enable/disable agent
    pub enabled: bool,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Custom configuration parameters
    pub custom_params: HashMap<String, String>,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30000,
        }
    }
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f64,
    /// Maximum disk usage in MB
    pub max_disk_mb: u64,
    /// Maximum network bandwidth in MB/s
    pub max_network_mbps: f64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_percent: 80.0,
            max_disk_mb: 5120,
            max_network_mbps: 100.0,
        }
    }
}

/// Default implementation for BaseAgentConfig
impl Default for BaseAgentConfig {
    fn default() -> Self {
        Self {
            agent_id: "default_agent".to_string(),
            name: "Default Agent".to_string(),
            version: "1.0.0".to_string(),
            enabled: true,
            max_concurrent_tasks: 1,
            timeout_seconds: 30,
            retry_config: RetryConfig::default(),
            resource_limits: ResourceLimits::default(),
            custom_params: HashMap::new(),
        }
    }
}

/// Helper trait for agent lifecycle management
#[async_trait]
pub trait AgentLifecycle: BaseAgent {
    /// Pre-processing hook
    async fn pre_process(&self, input: &Self::Input) -> AgentResult<Self::Input> {
        Ok(input.clone())
    }
    
    /// Post-processing hook
    async fn post_process(&self, output: &Self::Output) -> AgentResult<Self::Output> {
        Ok(output.clone())
    }
    
    /// Error handling hook
    async fn handle_error(&self, error: &AgentError) -> AgentResult<()> {
        tracing::error!("Agent {} encountered error: {}", self.agent_id(), error);
        Ok(())
    }
    
    /// Process with lifecycle hooks
    async fn process_with_lifecycle(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let processed_input = self.pre_process(&input).await?;
        
        match self.process(processed_input).await {
            Ok(output) => {
                let final_output = self.post_process(&output).await?;
                Ok(final_output)
            }
            Err(error) => {
                self.handle_error(&error).await?;
                Err(error)
            }
        }
    }
}

/// Macro to implement common agent functionality
#[macro_export]
macro_rules! impl_base_agent {
    ($agent_type:ty, $config_type:ty, $input_type:ty, $output_type:ty) => {
        impl $crate::shared::base_agent::BaseAgent for $agent_type {
            type Config = $config_type;
            type Input = $input_type;
            type Output = $output_type;
            
            fn agent_id(&self) -> &str {
                &self.config.agent_id
            }
            
            fn get_status(&self) -> $crate::shared::agent_types::AgentStatus {
                self.status.clone()
            }
            
            fn get_capabilities(&self) -> Vec<$crate::shared::agent_types::AgentCapability> {
                self.capabilities.clone()
            }
            
            fn get_metrics(&self) -> $crate::shared::agent_types::AgentMetrics {
                self.metrics.clone()
            }
        }
    };
}
