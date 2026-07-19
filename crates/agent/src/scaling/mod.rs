/// Dynamic Agent Scaling — Kubernetes-like autoscaler untuk AI agents
///
/// Strategi:
/// - Target utilization: 70-80% CPU/memory
/// - Scale up saat utilization > 80% atau queue depth > threshold
/// - Scale down saat utilization < 40% selama N cycle
/// - Min/max agents: 3/49
/// - Cooldown period: 30s antara scale events
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

mod autoscaler;
mod manager_autoscaler;
mod metrics_collector;
mod scaling_policy;
mod pool_manager;

pub use autoscaler::*;
pub use manager_autoscaler::*;
pub use metrics_collector::*;
pub use scaling_policy::*;
pub use pool_manager::*;

/// Konfigurasi scaling
#[derive(Debug, Clone)]
pub struct AgentScalingConfig {
    pub min_agents: usize,
    pub max_agents: usize,
    pub scale_up_threshold: f64,        // utilization (%) untuk scale up
    pub scale_down_threshold: f64,      // utilization (%) untuk scale down
    pub cooldown_period_secs: u64,
    pub evaluation_interval_secs: u64,
    pub scale_up_by: usize,
    pub scale_down_by: usize,
    pub queue_depth_threshold: usize,
}

impl Default for AgentScalingConfig {
    fn default() -> Self {
        Self {
            min_agents: 3,
            max_agents: 49,
            scale_up_threshold: 0.75,
            scale_down_threshold: 0.35,
            cooldown_period_secs: 30,
            evaluation_interval_secs: 10,
            scale_up_by: 3,
            scale_down_by: 2,
            queue_depth_threshold: 50,
        }
    }
}
