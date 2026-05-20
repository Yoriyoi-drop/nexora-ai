//! Orchestration Engine - Agent orchestration and coordination

use std::collections::HashMap;
use super::tasks::{TaskDistribution, ExecutionMonitoring, AgentResourceUsage};

/// Orchestration Engine
#[derive(Debug, Clone)]
pub struct OrchestrationEngine {
    /// Orchestration mode
    pub orchestration_mode: OrchestrationMode,
    /// Agent coordination strategy
    pub coordination_strategy: AgentCoordinationStrategy,
    /// Task distribution system
    pub task_distribution: super::tasks::TaskDistributionSystem,
    /// Agent registry
    pub agent_registry: super::agents::AgentRegistry,
    /// Orchestration metrics
    pub metrics: OrchestrationMetrics,
}

/// Orchestration Mode
#[derive(Debug, Clone)]
pub enum OrchestrationMode {
    /// Centralized orchestration
    Centralized,
    /// Distributed orchestration
    Distributed,
    /// Hierarchical orchestration
    Hierarchical,
    /// Hybrid orchestration
    Hybrid { centralized_weight: f32, distributed_weight: f32 },
    /// Adaptive orchestration
    Adaptive,
    /// Swarm orchestration
    Swarm,
}

/// Agent Coordination Strategy
#[derive(Debug, Clone)]
pub enum AgentCoordinationStrategy {
    /// Direct coordination
    Direct,
    /// Mediated coordination
    Mediated,
    /// Hierarchical coordination
    Hierarchical,
    /// Peer-to-peer coordination
    PeerToPeer,
    /// Event-driven coordination
    EventDriven,
    /// Consensus-based coordination
    ConsensusBased,
}

/// Orchestration Metrics
#[derive(Debug, Clone)]
pub struct OrchestrationMetrics {
    /// Coordination accuracy
    pub coordination_accuracy: f32,
    /// Task distribution efficiency
    pub task_distribution_efficiency: f32,
    /// Agent utilization
    pub agent_utilization: f32,
    /// Response time
    pub response_time_ms: f64,
    /// Throughput
    pub throughput: f32,
    /// Error rate
    pub error_rate: f32,
}

/// Orchestration Result
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    pub selected_agents: Vec<String>,
    pub task_distribution: TaskDistribution,
    pub execution_monitoring: ExecutionMonitoring,
    pub result_collection: ResultCollection,
    pub orchestration_time_ms: u64,
    pub success_rate: f32,
}

impl OrchestrationResult {
    pub fn new() -> Self {
        Self {
            selected_agents: Vec::new(),
            task_distribution: TaskDistribution::new(),
            execution_monitoring: ExecutionMonitoring::new(),
            result_collection: ResultCollection::new(),
            orchestration_time_ms: 0,
            success_rate: 0.0,
        }
    }
}

/// Result Collection
#[derive(Debug, Clone)]
pub struct ResultCollection {
    pub agent_results: HashMap<String, AgentResult>,
    pub collection_time_ms: u64,
    pub overall_success: bool,
}

impl ResultCollection {
    pub fn new() -> Self {
        Self {
            agent_results: HashMap::new(),
            collection_time_ms: 0,
            overall_success: false,
        }
    }
}

/// Agent Result
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub agent_id: String,
    pub task_id: String,
    pub success: bool,
    pub result_data: String,
    pub execution_time_ms: u64,
    pub resource_usage: AgentResourceUsage,
}

use super::super::config as nexum_config;

impl From<nexum_config::OrchestrationMode> for OrchestrationMode {
    fn from(c: nexum_config::OrchestrationMode) -> Self {
        match c {
            nexum_config::OrchestrationMode::Centralized => Self::Centralized,
            nexum_config::OrchestrationMode::Distributed => Self::Distributed,
            nexum_config::OrchestrationMode::Hierarchical => Self::Hierarchical,
            nexum_config::OrchestrationMode::Hybrid { centralized_weight, distributed_weight } => Self::Hybrid { centralized_weight, distributed_weight },
            nexum_config::OrchestrationMode::Adaptive => Self::Adaptive,
            nexum_config::OrchestrationMode::Swarm => Self::Swarm,
            nexum_config::OrchestrationMode::Synchronous => Self::Centralized,
        }
    }
}

impl From<nexum_config::AgentCoordinationStrategy> for AgentCoordinationStrategy {
    fn from(c: nexum_config::AgentCoordinationStrategy) -> Self {
        match c {
            nexum_config::AgentCoordinationStrategy::Direct => Self::Direct,
            nexum_config::AgentCoordinationStrategy::Mediated => Self::Mediated,
            nexum_config::AgentCoordinationStrategy::Hierarchical => Self::Hierarchical,
            nexum_config::AgentCoordinationStrategy::PeerToPeer => Self::PeerToPeer,
            nexum_config::AgentCoordinationStrategy::EventDriven => Self::EventDriven,
            nexum_config::AgentCoordinationStrategy::ConsensusBased => Self::ConsensusBased,
        }
    }
}
