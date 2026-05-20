//! Resource optimization and allocation

use std::collections::HashMap;

/// Resource Optimizer
#[derive(Debug, Clone)]
pub struct ResourceOptimizer {
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Optimization algorithm
    pub optimization_algorithm: OptimizationAlgorithm,
    /// Resource monitor
    pub resource_monitor: ResourceMonitor,
    /// Allocation history
    pub allocation_history: Vec<ResourceAllocationRecord>,
    /// Optimization metrics
    pub metrics: ResourceOptimizationMetrics,
}

/// Allocation Strategy
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// Equal allocation
    Equal,
    /// Priority-based allocation
    PriorityBased,
    /// Demand-based allocation
    DemandBased,
    /// Performance-based allocation
    PerformanceBased,
    /// Market-based allocation
    MarketBased,
    /// Optimal allocation
    Optimal,
}

/// Optimization Algorithm
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    /// Linear programming
    LinearProgramming,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Particle swarm optimization
    ParticleSwarmOptimization,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Multi-objective optimization
    MultiObjective,
}

/// Resource Monitor
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// Monitoring frequency
    pub monitoring_frequency_ms: u64,
    /// Resource pool
    pub resource_pool: ResourcePool,
    /// Usage history
    pub usage_history: Vec<ResourceUsage>,
    /// Monitor metrics
    pub metrics: ResourceMonitorMetrics,
}

/// Resource Pool
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// Total resources
    pub total_resources: HashMap<String, f64>,
    /// Available resources
    pub available_resources: HashMap<String, f64>,
    /// Allocated resources
    pub allocated_resources: HashMap<String, f64>,
    /// Resource constraints
    pub constraints: Vec<ResourceConstraint>,
}

/// Resource Usage
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Usage ID
    pub id: String,
    /// Resource type
    pub resource_type: String,
    /// Usage amount
    pub usage_amount: f64,
    /// Usage time
    pub usage_time: chrono::DateTime<chrono::Utc>,
    /// Agent ID
    pub agent_id: String,
}

/// Resource Constraint
#[derive(Debug, Clone)]
pub struct ResourceConstraint {
    /// Constraint name
    pub name: String,
    /// Resource type
    pub resource_type: String,
    /// Minimum value
    pub min_value: f64,
    /// Maximum value
    pub max_value: f64,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Priority
    pub priority: u8,
}

/// Constraint Type
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Hard constraint
    Hard,
    /// Soft constraint
    Soft,
    /// Elastic constraint
    Elastic,
    /// Dynamic constraint
    Dynamic,
}

/// Resource Monitor Metrics
#[derive(Debug, Clone)]
pub struct ResourceMonitorMetrics {
    /// Monitoring accuracy
    pub monitoring_accuracy: f32,
    /// Resource utilization
    pub resource_utilization: f32,
    /// Allocation efficiency
    pub allocation_efficiency: f32,
    /// Constraint compliance
    pub constraint_compliance: f32,
}

/// Resource Allocation Record
#[derive(Debug, Clone)]
pub struct ResourceAllocationRecord {
    /// Allocation ID
    pub id: String,
    /// Agent ID
    pub agent_id: String,
    /// Resource allocation
    pub allocation: HashMap<String, f64>,
    /// Allocation time
    pub allocated_at: chrono::DateTime<chrono::Utc>,
    /// Allocation duration
    pub duration_ms: u64,
}

/// Resource Optimization Metrics
#[derive(Debug, Clone)]
pub struct ResourceOptimizationMetrics {
    /// Optimization efficiency
    pub optimization_efficiency: f32,
    /// Resource utilization
    pub resource_utilization: f32,
    /// Allocation fairness
    pub allocation_fairness: f32,
    /// Optimization speed
    pub optimization_speed_ms: f64,
}

/// Resource Requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_requirement: f32,
    pub memory_requirement: f32,
    pub network_requirement: f32,
    pub storage_requirement: f64,
    pub gpu_requirement: f32,
}

/// Resource Allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub allocation_result: AllocationResult,
    pub allocation_time_ms: u64,
    pub allocation_efficiency: f32,
}

impl ResourceAllocation {
    pub fn new() -> Self {
        Self {
            allocation_result: AllocationResult::new(),
            allocation_time_ms: 0,
            allocation_efficiency: 0.0,
        }
    }
}

/// Allocation Result
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub agent_allocations: HashMap<String, AgentAllocation>,
}

impl AllocationResult {
    pub fn new() -> Self {
        Self {
            agent_allocations: HashMap::new(),
        }
    }
}

/// Agent Allocation
#[derive(Debug, Clone)]
pub struct AgentAllocation {
    pub agent_id: String,
    pub cpu_allocation: f32,
    pub memory_allocation: f32,
    pub network_allocation: f32,
    pub storage_allocation: f64,
    pub gpu_allocation: f32,
}

use super::super::config as nexum_config;

impl From<nexum_config::AllocationStrategy> for AllocationStrategy {
    fn from(c: nexum_config::AllocationStrategy) -> Self {
        match c {
            nexum_config::AllocationStrategy::Equal => Self::Equal,
            nexum_config::AllocationStrategy::PriorityBased => Self::PriorityBased,
            nexum_config::AllocationStrategy::DemandBased => Self::DemandBased,
            nexum_config::AllocationStrategy::PerformanceBased => Self::PerformanceBased,
            nexum_config::AllocationStrategy::MarketBased => Self::MarketBased,
            nexum_config::AllocationStrategy::Optimal => Self::Optimal,
        }
    }
}

impl From<nexum_config::OptimizationAlgorithm> for OptimizationAlgorithm {
    fn from(c: nexum_config::OptimizationAlgorithm) -> Self {
        match c {
            nexum_config::OptimizationAlgorithm::LinearProgramming => Self::LinearProgramming,
            nexum_config::OptimizationAlgorithm::GeneticAlgorithm => Self::GeneticAlgorithm,
            nexum_config::OptimizationAlgorithm::SimulatedAnnealing => Self::SimulatedAnnealing,
            nexum_config::OptimizationAlgorithm::ParticleSwarmOptimization => {
                Self::ParticleSwarmOptimization
            }
            nexum_config::OptimizationAlgorithm::ReinforcementLearning => {
                Self::ReinforcementLearning
            }
            nexum_config::OptimizationAlgorithm::MultiObjective => Self::MultiObjective,
        }
    }
}

impl From<nexum_config::ConstraintType> for ConstraintType {
    fn from(c: nexum_config::ConstraintType) -> Self {
        match c {
            nexum_config::ConstraintType::Hard => Self::Hard,
            nexum_config::ConstraintType::Soft => Self::Soft,
            nexum_config::ConstraintType::Elastic => Self::Elastic,
            nexum_config::ConstraintType::Dynamic => Self::Dynamic,
        }
    }
}

impl From<nexum_config::ResourceConstraint> for ResourceConstraint {
    fn from(c: nexum_config::ResourceConstraint) -> Self {
        let resource_type = match c.resource_type {
            nexum_config::ResourceType::CPU => "CPU".to_string(),
            nexum_config::ResourceType::Memory => "Memory".to_string(),
            nexum_config::ResourceType::Network => "Network".to_string(),
            nexum_config::ResourceType::Storage => "Storage".to_string(),
            nexum_config::ResourceType::GPU => "GPU".to_string(),
            nexum_config::ResourceType::Custom { name, .. } => name,
        };
        Self {
            name: c.name,
            resource_type,
            min_value: c.min_value,
            max_value: c.max_value,
            constraint_type: c.constraint_type.into(),
            priority: c.priority,
        }
    }
}
