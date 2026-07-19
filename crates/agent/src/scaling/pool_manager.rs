use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::metrics_collector::ScalingMetrics;
use super::scaling_policy::{ScalingDecision, ScalingPolicy};
use super::AgentScalingConfig;

/// Status agent individual
#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Active,
    Idle,
    Draining,
    Stopped,
}

/// Agent yang dikelola pool
struct PooledAgent {
    pub id: Uuid,
    pub status: AgentStatus,
    pub created_at: Instant,
    pub last_active: Instant,
}

/// Pool manager — kelola lifecycle agent pool
pub struct AgentPoolManager {
    config: AgentScalingConfig,
    agents: DashMap<Uuid, Arc<RwLock<PooledAgent>>>,
    metrics: Arc<ScalingMetrics>,
    policy: RwLock<ScalingPolicy>,
    running: AtomicBool,
    last_scale_time: RwLock<Instant>,
}

impl AgentPoolManager {
    pub fn new(config: AgentScalingConfig) -> Self {
        let min_agents = config.min_agents;
        let manager = Self {
            agents: DashMap::new(),
            metrics: Arc::new(ScalingMetrics::new()),
            policy: RwLock::new(ScalingPolicy::new(config.clone())),
            running: AtomicBool::new(false),
            last_scale_time: RwLock::new(Instant::now()),
            config,
        };

        for _ in 0..min_agents {
            manager.spawn_agent();
        }

        manager
    }

    pub fn spawn_agent(&self) -> Uuid {
        let id = Uuid::new_v4();
        self.agents.insert(
            id,
            Arc::new(RwLock::new(PooledAgent {
                id,
                status: AgentStatus::Active,
                created_at: Instant::now(),
                last_active: Instant::now(),
            })),
        );
        let count = self.agents.len();
        self.metrics.set_active_agents(count);
        info!("agent spawned: id={}, total={}", id, count);
        id
    }

    pub fn stop_agent(&self) -> Option<Uuid> {
        // Cari agent idle untuk di-stop (LIFO)
        let victim = self.agents.iter().find(|e| {
            let agent = e.value().read();
            agent.status == AgentStatus::Idle
        });

        if let Some(entry) = victim {
            let id = entry.key();
            if let Some(agent) = self.agents.get(id) {
                agent.write().status = AgentStatus::Draining;
            }
            let id = *id;
            self.agents.remove(&id);
            let count = self.agents.len();
            self.metrics.set_active_agents(count);
            info!("agent stopped: id={}, total={}", id, count);
            return Some(id);
        }
        None
    }

    pub fn mark_active(&self, id: &Uuid) {
        if let Some(agent) = self.agents.get(id) {
            let mut a = agent.write();
            a.status = AgentStatus::Active;
            a.last_active = Instant::now();
        }
    }

    pub fn mark_idle(&self, id: &Uuid) {
        if let Some(agent) = self.agents.get(id) {
            agent.write().status = AgentStatus::Idle;
        }
    }

    /// Evaluasi scaling berdasarkan metrics
    pub fn evaluate_scaling(&self, cpu_util: f64, mem_util: f64, queue_depth: usize) -> ScalingDecision {
        let active = self.agents.len();
        let mut policy = self.policy.write();
        let decision = policy.evaluate(cpu_util, mem_util, queue_depth, active);

        // Cooldown check
        let last_scale = *self.last_scale_time.read();
        if last_scale.elapsed() < Duration::from_secs(self.config.cooldown_period_secs) {
            return ScalingDecision::Hold;
        }

        match decision {
            ScalingDecision::ScaleUp(n) => {
                *self.last_scale_time.write() = Instant::now();
                for _ in 0..n {
                    self.spawn_agent();
                }
                info!("SCALE UP: +{} agents (total={})", n, self.agents.len());
            }
            ScalingDecision::ScaleDown(n) => {
                *self.last_scale_time.write() = Instant::now();
                for _ in 0..n {
                    self.stop_agent();
                }
                info!("SCALE DOWN: -{} agents (total={})", n, self.agents.len());
            }
            ScalingDecision::Hold => {}
        }

        decision
    }

    pub fn metrics(&self) -> &Arc<ScalingMetrics> {
        &self.metrics
    }

    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    pub fn active_count(&self) -> usize {
        self.agents
            .iter()
            .filter(|e| e.value().read().status == AgentStatus::Active)
            .count()
    }

    pub fn idle_count(&self) -> usize {
        self.agents
            .iter()
            .filter(|e| e.value().read().status == AgentStatus::Idle)
            .count()
    }

    pub fn config(&self) -> &AgentScalingConfig {
        &self.config
    }
}
