use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

use super::scaling_policy::{ScalingDecision, ScalingPolicy};
use super::AgentScalingConfig;

/// Autoscaler that bridges scaling logic with AgentManager via command channel
pub struct ManagerAutoscaler {
    config: AgentScalingConfig,
    policy: RwLock<ScalingPolicy>,
    running: AtomicBool,
    last_scale_time: RwLock<Instant>,
    cmd_tx: tokio::sync::mpsc::Sender<crate::agent_manager::ManagerCommand>,
}

impl ManagerAutoscaler {
    pub fn new(
        config: AgentScalingConfig,
        cmd_tx: tokio::sync::mpsc::Sender<crate::agent_manager::ManagerCommand>,
    ) -> Self {
        let cfg = config.clone();
        Self {
            config,
            policy: RwLock::new(ScalingPolicy::new(cfg)),
            running: AtomicBool::new(false),
            last_scale_time: RwLock::new(Instant::now()),
            cmd_tx,
        }
    }

    fn evaluate(&self) -> (ScalingDecision, bool) {
        let last_scale = *self.last_scale_time.read();
        if last_scale.elapsed() < Duration::from_secs(self.config.cooldown_period_secs) {
            return (ScalingDecision::Hold, false);
        }

        let mut policy = self.policy.write();
        let decision = policy.evaluate(0.0, 0.0, 0, 0);

        let acted = matches!(decision, ScalingDecision::ScaleUp(_) | ScalingDecision::ScaleDown(_));
        if acted {
            *self.last_scale_time.write() = Instant::now();
        }

        (decision, acted)
    }

    pub async fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("ManagerAutoscaler started (min={}, max={})", self.config.min_agents, self.config.max_agents);

        let interval = self.config.evaluation_interval_secs;

        while self.running.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(interval)).await;
            if !self.running.load(Ordering::Relaxed) {
                break;
            }

            let (decision, _acted) = self.evaluate();

            match decision {
                ScalingDecision::ScaleUp(n) => {
                    for _ in 0..n {
                        let (tx, _rx) = tokio::sync::oneshot::channel();
                        let cmd = crate::agent_manager::ManagerCommand::SpawnAgent {
                            agent_type: "worker".into(),
                            config: crate::AgentConfig::default(),
                            response_tx: tx,
                        };
                        if self.cmd_tx.send(cmd).await.is_err() {
                            warn!("ManagerAutoscaler: cmd channel closed");
                            break;
                        }
                    }
                    info!("ManagerAutoscaler: scale up +{} agents", n);
                }
                ScalingDecision::ScaleDown(n) => {
                    for _ in 0..n {
                        let (tx, _rx) = tokio::sync::oneshot::channel();
                        let cmd = crate::agent_manager::ManagerCommand::StopRandomAgent {
                            response_tx: tx,
                        };
                        if self.cmd_tx.send(cmd).await.is_err() {
                            warn!("ManagerAutoscaler: cmd channel closed");
                            break;
                        }
                    }
                    info!("ManagerAutoscaler: scale down -{} agents", n);
                }
                ScalingDecision::Hold => {}
            }
        }
    }

    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}
