//! Agent registry and agent information

use std::collections::HashMap;

/// Agent Registry
#[derive(Debug, Clone)]
pub struct AgentRegistry {
    /// Registered agents
    pub agents: HashMap<String, AgentInfo>,
    /// Agent groups
    pub agent_groups: HashMap<String, Vec<String>>,
    /// Registry metrics
    pub metrics: RegistryMetrics,
}

/// Agent Info
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Agent ID
    pub agent_id: String,
    /// Agent type
    pub agent_type: AgentType,
    /// Agent capabilities
    pub capabilities: Vec<String>,
    /// Agent status
    pub status: AgentStatus,
    /// Agent resources
    pub resources: AgentResources,
    /// Agent performance
    pub performance: AgentPerformance,
    /// Registration time
    pub registered_at: chrono::DateTime<chrono::Utc>,
}

/// Agent Type
#[derive(Debug, Clone)]
pub enum AgentType {
    /// Worker agent
    Worker,
    /// Manager agent
    Manager,
    /// Specialist agent
    Specialist,
    /// Coordinator agent
    Coordinator,
    /// Monitor agent
    Monitor,
    /// Gateway agent
    Gateway,
}

/// Agent Status
#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    /// Active
    Active,
    /// Inactive
    Inactive,
    /// Busy
    Busy,
    /// Failed
    Failed,
    /// Maintenance
    Maintenance,
}

/// Agent Resources
#[derive(Debug, Clone)]
pub struct AgentResources {
    /// CPU capacity
    pub cpu_capacity: f32,
    /// Memory capacity
    pub memory_capacity: f32,
    /// Network bandwidth
    pub network_bandwidth: f32,
    /// Storage capacity
    pub storage_capacity: f64,
    /// GPU capacity
    pub gpu_capacity: f32,
}

/// Agent Performance
#[derive(Debug, Clone)]
pub struct AgentPerformance {
    /// Success rate
    pub success_rate: f32,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Task completion rate
    pub task_completion_rate: f32,
    /// Reliability score
    pub reliability_score: f32,
}

/// Registry Metrics
#[derive(Debug, Clone)]
pub struct RegistryMetrics {
    /// Total agents
    pub total_agents: usize,
    /// Active agents
    pub active_agents: usize,
    /// Agent types distribution
    pub agent_types_distribution: HashMap<String, usize>,
    /// Registration rate
    pub registration_rate: f32,
}
