//! Task distribution, scheduling, and execution monitoring

use std::collections::HashMap;

/// Task Distribution System
#[derive(Debug, Clone)]
pub struct TaskDistributionSystem {
    /// Distribution method
    pub distribution_method: TaskDistributionMethod,
    /// Load balancer
    pub load_balancer: LoadBalancer,
    /// Task scheduler
    pub task_scheduler: TaskScheduler,
    /// Distribution metrics
    pub metrics: DistributionMetrics,
}

/// Task Distribution Method
#[derive(Debug, Clone)]
pub enum TaskDistributionMethod {
    /// Round-robin distribution
    RoundRobin,
    /// Load-balanced distribution
    LoadBalanced,
    /// Skill-based distribution
    SkillBased,
    /// Priority-based distribution
    PriorityBased,
    /// Dynamic distribution
    Dynamic,
    /// Optimal distribution
    Optimal,
}

/// Load Balancer
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Agent load tracking
    pub agent_loads: HashMap<String, AgentLoad>,
    /// Balancing metrics
    pub metrics: LoadBalancingMetrics,
}

/// Load Balancing Algorithm
#[derive(Debug, Clone)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Response time
    ResponseTime,
    /// Resource-based
    ResourceBased,
    /// Predictive
    Predictive,
}

/// Agent Load
#[derive(Debug, Clone)]
pub struct AgentLoad {
    /// Agent ID
    pub agent_id: String,
    /// Current task count
    pub current_tasks: usize,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Memory utilization
    pub memory_utilization: f32,
    /// Response time
    pub response_time_ms: f64,
    /// Last update
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Load Balancing Metrics
#[derive(Debug, Clone)]
pub struct LoadBalancingMetrics {
    /// Balance score
    pub balance_score: f32,
    /// Distribution efficiency
    pub distribution_efficiency: f32,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Load variance
    pub load_variance: f32,
}

/// Task Scheduler
#[derive(Debug, Clone)]
pub struct TaskScheduler {
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    /// Task queue
    pub task_queue: TaskQueue,
    /// Scheduling metrics
    pub metrics: SchedulingMetrics,
}

/// Scheduling Algorithm
#[derive(Debug, Clone)]
pub enum SchedulingAlgorithm {
    /// First-come-first-served
    FCFS,
    /// Shortest job first
    SJF,
    /// Priority scheduling
    Priority,
    /// Round-robin scheduling
    RoundRobin,
    /// Fair scheduling
    Fair,
    /// Predictive scheduling
    Predictive,
}

/// Task Queue
#[derive(Debug, Clone)]
pub struct TaskQueue {
    /// Queue type
    pub queue_type: QueueType,
    /// Tasks
    pub tasks: Vec<Task>,
    /// Queue metrics
    pub metrics: QueueMetrics,
}

/// Queue Type
#[derive(Debug, Clone)]
pub enum QueueType {
    /// FIFO queue
    FIFO,
    /// Priority queue
    Priority,
    /// Fair queue
    Fair,
    /// Deadline queue
    Deadline,
    /// Multiple queue
    Multiple { queues: HashMap<String, TaskQueue> },
}

/// Task
#[derive(Debug, Clone)]
pub struct Task {
    /// Task ID
    pub id: String,
    /// Task type
    pub task_type: TaskType,
    /// Task priority
    pub priority: TaskPriority,
    /// Task requirements
    pub requirements: TaskRequirements,
    /// Task status
    pub status: TaskStatus,
    /// Creation time
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Deadline
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

/// Task Type
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Computation task
    Computation,
    /// Communication task
    Communication,
    /// Coordination task
    Coordination,
    /// Monitoring task
    Monitoring,
    /// Maintenance task
    Maintenance,
}

/// Task Priority
#[derive(Debug, Clone)]
pub enum TaskPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
    /// Emergency priority
    Emergency,
}

/// Task Requirements
#[derive(Debug, Clone)]
pub struct TaskRequirements {
    /// CPU requirement
    pub cpu_requirement: f32,
    /// Memory requirement
    pub memory_requirement: f32,
    /// Network requirement
    pub network_requirement: f32,
    /// Storage requirement
    pub storage_requirement: f32,
    /// GPU requirement
    pub gpu_requirement: f32,
    /// Skill requirements
    pub skill_requirements: Vec<String>,
    /// Estimated duration
    pub estimated_duration_ms: u64,
}

/// Task Status
#[derive(Debug, Clone)]
pub enum TaskStatus {
    /// Pending
    Pending,
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Queue Metrics
#[derive(Debug, Clone)]
pub struct QueueMetrics {
    /// Queue length
    pub queue_length: usize,
    /// Average wait time
    pub avg_wait_time_ms: f64,
    /// Throughput
    pub throughput: f32,
    /// Queue utilization
    pub utilization: f32,
}

/// Distribution Metrics
#[derive(Debug, Clone)]
pub struct DistributionMetrics {
    /// Distribution efficiency
    pub efficiency: f32,
    /// Load balance score
    pub load_balance_score: f32,
    /// Task completion rate
    pub completion_rate: f32,
    /// Average distribution time
    pub avg_distribution_time_ms: f64,
}

/// Task Distribution
#[derive(Debug, Clone)]
pub struct TaskDistribution {
    pub subtasks: HashMap<String, Task>,
    pub distribution_time_ms: u64,
    pub distribution_efficiency: f32,
}

impl TaskDistribution {
    pub fn new() -> Self {
        Self {
            subtasks: HashMap::new(),
            distribution_time_ms: 0,
            distribution_efficiency: 0.0,
        }
    }
}

/// Execution Monitoring
#[derive(Debug, Clone)]
pub struct ExecutionMonitoring {
    pub agent_monitoring: HashMap<String, AgentMonitoring>,
    pub monitoring_time_ms: u64,
    pub overall_progress: f32,
}

impl ExecutionMonitoring {
    pub fn new() -> Self {
        Self {
            agent_monitoring: HashMap::new(),
            monitoring_time_ms: 0,
            overall_progress: 0.0,
        }
    }
}

/// Agent Monitoring
#[derive(Debug, Clone)]
pub struct AgentMonitoring {
    pub agent_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub status: ExecutionStatus,
    pub progress: f32,
    pub resource_usage: AgentResourceUsage,
    pub errors: Vec<String>,
}

/// Execution Status
#[derive(Debug, Clone)]
pub enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Agent Resource Usage
#[derive(Debug, Clone)]
pub struct AgentResourceUsage {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub network_usage: f32,
    pub gpu_usage: f32,
}

/// Scheduling Metrics
#[derive(Debug, Clone)]
pub struct SchedulingMetrics {
    pub scheduling_efficiency: f32,
    pub avg_wait_time_ms: f64,
    pub throughput: f32,
    pub fairness_score: f32,
}

use super::super::config as nexum_config;

impl From<nexum_config::TaskDistributionMethod> for TaskDistributionMethod {
    fn from(c: nexum_config::TaskDistributionMethod) -> Self {
        match c {
            nexum_config::TaskDistributionMethod::RoundRobin => Self::RoundRobin,
            nexum_config::TaskDistributionMethod::LoadBalanced => Self::LoadBalanced,
            nexum_config::TaskDistributionMethod::SkillBased => Self::SkillBased,
            nexum_config::TaskDistributionMethod::PriorityBased => Self::PriorityBased,
            nexum_config::TaskDistributionMethod::Dynamic => Self::Dynamic,
            nexum_config::TaskDistributionMethod::Optimal => Self::Optimal,
        }
    }
}
