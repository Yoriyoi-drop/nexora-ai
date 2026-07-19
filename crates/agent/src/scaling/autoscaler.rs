use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};

use super::pool_manager::AgentPoolManager;
use super::scaling_policy::ScalingDecision;
use super::AgentScalingConfig;

/// Autoscaler — background task yang evaluate scaling secara periodik
pub struct AgentAutoscaler {
    pool_manager: Arc<AgentPoolManager>,
    running: AtomicBool,
}

impl AgentAutoscaler {
    pub fn new(config: AgentScalingConfig) -> Self {
        Self {
            pool_manager: Arc::new(AgentPoolManager::new(config)),
            running: AtomicBool::new(false),
        }
    }

    pub fn pool_manager(&self) -> &Arc<AgentPoolManager> {
        &self.pool_manager
    }

    /// Start background scaling loop
    pub async fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("AgentAutoscaler started");

        let pool = self.pool_manager.clone();
        let interval = pool.config().evaluation_interval_secs;

        while self.running.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(interval)).await;

            // Collect metrics
            let metrics = pool.metrics();
            let cpu = *metrics.avg_cpu_util.read().unwrap();
            let mem = *metrics.avg_mem_util.read().unwrap();
            let queue = metrics.queue_depth.load(Ordering::Relaxed);

            // Evaluate and scale
            let decision = pool.evaluate_scaling(cpu, mem, queue as usize);
            metrics.calculate_rps();

            match decision {
                ScalingDecision::ScaleUp(n) => {
                    info!("autoscaler: scale up +{} agents", n);
                }
                ScalingDecision::ScaleDown(n) => {
                    info!("autoscaler: scale down -{} agents", n);
                }
                ScalingDecision::Hold => {}
            }
        }
    }

    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("AgentAutoscaler shutdown");
    }
}
