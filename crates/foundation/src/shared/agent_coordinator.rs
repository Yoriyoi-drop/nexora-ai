//! Agent Coordinator System
//! 
//! Provides coordination functionality for multi-agent systems

use std::collections::HashMap;
use std::marker::PhantomData;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::base_model::NxrModelResult;
use super::agent_types::{AgentStatus, TaskRoutingRule, CommunicationChannel, TaskPriority, AgentResult};

/// Generic Agent Coordinator
#[derive(Debug, Clone)]
pub struct AgentCoordinator<T> {
    /// Coordination strategy
    pub strategy: CoordinationStrategy,
    /// Agent weights for load balancing
    pub agent_weights: HashMap<String, f32>,
    /// Task routing rules
    pub routing_rules: Vec<TaskRoutingRule>,
    /// Agent communication channels
    pub communication_channels: HashMap<String, CommunicationChannel>,
    /// Coordinator metrics
    pub metrics: CoordinatorMetrics,
    /// Phantom data for generic type
    phantom: PhantomData<T>,
}

/// Coordination Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// Sequential coordination (one agent at a time)
    Sequential,
    /// Parallel coordination (multiple agents simultaneously)
    Parallel,
    /// Hierarchical coordination (agent hierarchy)
    Hierarchical { 
        /// Hierarchy order (top to bottom)
        hierarchy: Vec<String> 
    },
    /// Adaptive coordination (dynamic strategy selection)
    Adaptive,
    /// Consensus-based coordination
    ConsensusBased { 
        /// Consensus threshold (0.0 - 1.0)
        threshold: f32 
    },
    /// Load-balanced coordination
    LoadBalanced,
    /// Priority-based coordination
    PriorityBased,
    /// Model-specific coordination variants
    EmpathyDriven { 
        /// Empathy weight factor
        empathy_weight: f32 
    },
    CreativeDriven { 
        /// Creativity weight factor
        creativity_weight: f32 
    },
    Consensus { 
        /// Consensus threshold
        threshold: f32 
    },
}

/// Coordinator Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorMetrics {
    /// Total tasks coordinated
    pub total_tasks: u64,
    /// Average coordination time (ms)
    pub avg_coordination_time: f64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Agent utilization rates
    pub agent_utilization: HashMap<String, f32>,
    /// Last coordination timestamp
    pub last_coordination: chrono::DateTime<chrono::Utc>,
}

impl Default for CoordinatorMetrics {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            avg_coordination_time: 0.0,
            success_rate: 1.0,
            agent_utilization: HashMap::new(),
            last_coordination: chrono::Utc::now(),
        }
    }
}

impl<T> AgentCoordinator<T> {
    /// Create a new agent coordinator
    pub fn new(strategy: CoordinationStrategy) -> Self {
        Self {
            strategy,
            agent_weights: HashMap::new(),
            routing_rules: Vec::new(),
            communication_channels: HashMap::new(),
            metrics: CoordinatorMetrics::default(),
            phantom: PhantomData,
        }
    }
    
    /// Add agent weight
    pub fn add_agent_weight(&mut self, agent_id: String, weight: f32) {
        self.agent_weights.insert(agent_id, weight);
    }
    
    /// Add routing rule
    pub fn add_routing_rule(&mut self, rule: TaskRoutingRule) {
        self.routing_rules.push(rule);
        // Sort rules by priority (higher priority first)
        self.routing_rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    /// Add communication channel
    pub fn add_communication_channel(&mut self, channel: CommunicationChannel) {
        self.communication_channels.insert(channel.id.clone(), channel);
    }
    
    /// Route task to appropriate agent
    pub fn route_task(&self, task_description: &str) -> AgentResult<String> {
        for rule in &self.routing_rules {
            if task_description.contains(&rule.pattern) {
                return Ok(rule.target_agent.clone());
            }
        }
        
        // Fallback to load balancing
        self.select_agent_by_load()
    }
    
    /// Select agent based on load weights
    fn select_agent_by_load(&self) -> AgentResult<String> {
        if self.agent_weights.is_empty() {
            return Err(super::agent_types::AgentError::ProcessingFailed(
                "No agents available for routing".to_string()
            ));
        }
        
        // Simple weighted random selection
        let total_weight: f32 = self.agent_weights.values().sum();
        let mut random = rand::random::<f32>() * total_weight;
        
        for (agent_id, weight) in &self.agent_weights {
            random -= weight;
            if random <= 0.0 {
                return Ok(agent_id.clone());
            }
        }
        
        // Fallback to first agent (or default if empty)
        Ok(self.agent_weights.keys().next().cloned().unwrap_or_default())
    }
    
    /// Get coordination strategy
    pub fn get_strategy(&self) -> &CoordinationStrategy {
        &self.strategy
    }
    
    /// Update coordination strategy
    pub fn update_strategy(&mut self, strategy: CoordinationStrategy) {
        self.strategy = strategy;
    }
    
    /// Get agent utilization
    pub fn get_agent_utilization(&self, agent_id: &str) -> Option<f32> {
        self.metrics.agent_utilization.get(agent_id).copied()
    }
    
    /// Update agent utilization
    pub fn update_agent_utilization(&mut self, agent_id: String, utilization: f32) {
        self.metrics.agent_utilization.insert(agent_id, utilization);
    }
}

/// Trait for coordination strategies
#[async_trait]
pub trait CoordinationStrategyTrait: Send + Sync {
    /// Coordinate task execution
    async fn coordinate(&self, agents: Vec<String>, task: &str) -> AgentResult<Vec<String>>;
    
    /// Get strategy name
    fn strategy_name(&self) -> &str;
    
    /// Check if strategy is applicable for given task
    fn is_applicable(&self, task: &str, agent_count: usize) -> bool;
}

/// Sequential coordination implementation
pub struct SequentialCoordinator;

#[async_trait]
impl CoordinationStrategyTrait for SequentialCoordinator {
    async fn coordinate(&self, agents: Vec<String>, _task: &str) -> AgentResult<Vec<String>> {
        Ok(agents) // Execute agents in order
    }
    
    fn strategy_name(&self) -> &str {
        "Sequential"
    }
    
    fn is_applicable(&self, _task: &str, _agent_count: usize) -> bool {
        true // Always applicable
    }
}

/// Parallel coordination implementation
pub struct ParallelCoordinator;

#[async_trait]
impl CoordinationStrategyTrait for ParallelCoordinator {
    async fn coordinate(&self, agents: Vec<String>, _task: &str) -> AgentResult<Vec<String>> {
        Ok(agents) // Execute all agents in parallel
    }
    
    fn strategy_name(&self) -> &str {
        "Parallel"
    }
    
    fn is_applicable(&self, _task: &str, agent_count: usize) -> bool {
        agent_count > 1 // Only meaningful with multiple agents
    }
}

/// Hierarchical coordination implementation
pub struct HierarchicalCoordinator {
    hierarchy: Vec<String>,
}

impl HierarchicalCoordinator {
    pub fn new(hierarchy: Vec<String>) -> Self {
        Self { hierarchy }
    }
}

#[async_trait]
impl CoordinationStrategyTrait for HierarchicalCoordinator {
    async fn coordinate(&self, agents: Vec<String>, _task: &str) -> AgentResult<Vec<String>> {
        // Sort agents according to hierarchy
        let mut sorted_agents = agents;
        sorted_agents.sort_by(|a, b| {
            let a_pos = self.hierarchy.iter().position(|x| x == a).unwrap_or(usize::MAX);
            let b_pos = self.hierarchy.iter().position(|x| x == b).unwrap_or(usize::MAX);
            a_pos.cmp(&b_pos)
        });
        Ok(sorted_agents)
    }
    
    fn strategy_name(&self) -> &str {
        "Hierarchical"
    }
    
    fn is_applicable(&self, _task: &str, _agent_count: usize) -> bool {
        !self.hierarchy.is_empty()
    }
}

/// Factory for creating coordination strategies
pub struct CoordinationStrategyFactory;

impl CoordinationStrategyFactory {
    /// Create strategy from enum
    pub fn create_strategy(strategy: &CoordinationStrategy) -> Box<dyn CoordinationStrategyTrait> {
        match strategy {
            CoordinationStrategy::Sequential => Box::new(SequentialCoordinator),
            CoordinationStrategy::Parallel => Box::new(ParallelCoordinator),
            CoordinationStrategy::Hierarchical { hierarchy } => {
                Box::new(HierarchicalCoordinator::new(hierarchy.clone()))
            }
            CoordinationStrategy::Adaptive => Box::new(SequentialCoordinator), // Fallback
            CoordinationStrategy::ConsensusBased { .. } => Box::new(SequentialCoordinator), // Fallback
            CoordinationStrategy::LoadBalanced => Box::new(ParallelCoordinator),
            CoordinationStrategy::PriorityBased => Box::new(SequentialCoordinator),
            CoordinationStrategy::EmpathyDriven { .. } => Box::new(SequentialCoordinator),
            CoordinationStrategy::CreativeDriven { .. } => Box::new(ParallelCoordinator),
            CoordinationStrategy::Consensus { .. } => Box::new(SequentialCoordinator),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_coordinator_creation() {
        let coordinator = AgentCoordinator::<()>::new(CoordinationStrategy::Sequential);
        assert!(matches!(coordinator.strategy, CoordinationStrategy::Sequential));
    }

    #[test]
    fn test_routing_rules() {
        let mut coordinator = AgentCoordinator::<()>::new(CoordinationStrategy::Sequential);
        
        let rule = TaskRoutingRule {
            pattern: "test".to_string(),
            target_agent: "test_agent".to_string(),
            priority: 1,
            weight: Some(1.0),
        };
        
        coordinator.add_routing_rule(rule);
        assert_eq!(coordinator.routing_rules.len(), 1);
        
        let result = coordinator.route_task("this is a test task");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test_agent");
    }
}
