use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Enable event bus (pub/sub backbone)
    pub enable_eventbus: bool,

    /// Enable unified memory pools
    pub enable_memory_pools: bool,

    /// Enable DAG scheduler
    pub enable_dag_scheduler: bool,

    /// Enable GPU scheduler
    pub enable_gpu_scheduler: bool,

    /// Enable agent auto-scaling (3-49 agents)
    pub enable_agent_scaling: bool,

    /// Enable cost optimizer cascade
    pub enable_cost_optimizer: bool,

    /// Enable observability collector
    pub enable_observability: bool,

    /// Number of DAG scheduler workers (default: 4)
    pub dag_workers: usize,

    /// Number of GPUs for scheduler (default: 1)
    pub gpu_count: usize,

    /// Observability collection interval in seconds (default: 10)
    pub observability_interval_secs: u64,

    /// Agent scaling min agents (default: 3)
    pub agent_min: usize,

    /// Agent scaling max agents (default: 49)
    pub agent_max: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            enable_eventbus: true,
            enable_memory_pools: true,
            enable_dag_scheduler: false,
            enable_gpu_scheduler: false,
            enable_agent_scaling: false,
            enable_cost_optimizer: true,
            enable_observability: true,
            dag_workers: 4,
            gpu_count: 1,
            observability_interval_secs: 10,
            agent_min: 3,
            agent_max: 49,
        }
    }
}
