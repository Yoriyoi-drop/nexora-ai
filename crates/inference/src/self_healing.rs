use crate::engine::{EngineState, InferenceEngine};
use crate::degradation::{DegradationConfig, DegradationLevel, DegradationManager};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct SelfHealingConfig {
    pub health_check_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub max_consecutive_failures: u64,
    pub restart_delay_ms: u64,
    pub degradation_config: DegradationConfig,
}

impl Default for SelfHealingConfig {
    fn default() -> Self {
        Self {
            health_check_interval_ms: 5_000,
            heartbeat_timeout_ms: 30_000,
            max_consecutive_failures: 3,
            restart_delay_ms: 1_000,
            degradation_config: DegradationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    Healthy,
    Degraded,
    Stuck,
    Failed,
}

pub struct SelfHealingWorker {
    engine_handle: Arc<RwLock<InferenceEngine>>,
    config: SelfHealingConfig,
    degradation: Arc<DegradationManager>,
    shutdown_flag: Arc<AtomicBool>,
    health: Arc<RwLock<WorkerHealth>>,
    consecutive_failures: Arc<AtomicU64>,
    last_heartbeat_ms: Arc<AtomicU64>,
    restart_count: Arc<AtomicU64>,
    worker_id: String,
}

impl SelfHealingWorker {
    pub fn new(
        engine_handle: Arc<RwLock<InferenceEngine>>,
        config: SelfHealingConfig,
    ) -> Self {
        let degradation = Arc::new(DegradationManager::new(config.degradation_config.clone()));
        Self {
            engine_handle,
            config,
            degradation,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            health: Arc::new(RwLock::new(WorkerHealth::Healthy)),
            consecutive_failures: Arc::new(AtomicU64::new(0)),
            last_heartbeat_ms: Arc::new(AtomicU64::new(0)),
            restart_count: Arc::new(AtomicU64::new(0)),
            worker_id: format!("healer-{}", Uuid::new_v4().to_string().split('-').next().unwrap_or("0000")),
        }
    }

    pub fn degradation_manager(&self) -> Arc<DegradationManager> {
        self.degradation.clone()
    }

    pub fn record_heartbeat(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_heartbeat_ms.store(now, Ordering::Relaxed);
    }

    pub async fn current_health(&self) -> WorkerHealth {
        *self.health.read().await
    }

    pub fn restart_count(&self) -> u64 {
        self.restart_count.load(Ordering::Relaxed)
    }

    pub async fn start(self: Arc<Self>) {
        let worker = Arc::clone(&self);
        tokio::spawn(async move {
            info!("SelfHealingWorker '{}' started", worker.worker_id);
            loop {
                if worker.shutdown_flag.load(Ordering::Relaxed) {
                    info!("SelfHealingWorker '{}' shutting down", worker.worker_id);
                    break;
                }

                tokio::time::sleep(Duration::from_millis(worker.config.health_check_interval_ms)).await;

                if let Err(e) = worker.run_health_check().await {
                    warn!("Health check failed: {}", e);
                }
            }
        });
    }

    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }

    async fn run_health_check(&self) -> Result<(), String> {
        let engine = self.engine_handle.read().await;
        let state = engine.get_engine_stats().await.state;

        match state {
            EngineState::ShuttingDown | EngineState::Shutdown => {
                self.shutdown();
                return Ok(());
            }
            EngineState::Uninitialized | EngineState::Initializing => {
                let mut health = self.health.write().await;
                *health = WorkerHealth::Degraded;
                return Ok(());
            }
            EngineState::Ready => {}
        }

        let stats = engine.get_engine_stats().await;
        let active_count = stats.active_requests_count;

        let heartbeat_age_ms = {
            let last = self.last_heartbeat_ms.load(Ordering::Relaxed);
            if last == 0 {
                0
            } else {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                now.saturating_sub(last)
            }
        };

        let mut new_health = WorkerHealth::Healthy;

        if heartbeat_age_ms > self.config.heartbeat_timeout_ms && active_count > 0 {
            new_health = WorkerHealth::Stuck;
            self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
            warn!(
                "Heartbeat stale for {}ms (limit: {}ms), marking stuck",
                heartbeat_age_ms, self.config.heartbeat_timeout_ms
            );
        }

        if active_count == 0 && stats.scheduler_stats.queued_requests > 0 {
            new_health = WorkerHealth::Stuck;
            self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
            warn!(
                "Requests queued ({}) but none active — possible lock-up",
                stats.scheduler_stats.queued_requests
            );
        }

        let deg_level = self.degradation.evaluate(
            stats.scheduler_stats.avg_processing_time_ms,
        ).await;

        if deg_level > DegradationLevel::Reduced {
            new_health = WorkerHealth::Degraded;
        }

        {
            let mut health = self.health.write().await;
            *health = new_health;
        }

        let fails = self.consecutive_failures.load(Ordering::Relaxed);
        if fails >= self.config.max_consecutive_failures {
            error!(
                "SelfHealingWorker '{}': {} consecutive failures detected, initiating recovery",
                self.worker_id, fails
            );
            self.perform_recovery().await;
        }

        debug!(
            "Health check: state={:?}, health={:?}, active={}, queued={}, deg={}, heartbeat_age={}ms",
            state, new_health, active_count, stats.scheduler_stats.queued_requests, deg_level, heartbeat_age_ms
        );

        Ok(())
    }

    async fn perform_recovery(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.restart_count.fetch_add(1, Ordering::Relaxed);
        info!("Performing recovery for SelfHealingWorker '{}'", self.worker_id);
        tokio::time::sleep(Duration::from_millis(self.config.restart_delay_ms)).await;
    }

    pub fn stats(&self) -> SelfHealingStats {
        SelfHealingStats {
            worker_id: self.worker_id.clone(),
            health: WorkerHealth::Healthy,
            consecutive_failures: self.consecutive_failures.load(Ordering::Relaxed),
            restart_count: self.restart_count.load(Ordering::Relaxed),
            degradation_level: DegradationLevel::None,
        }
    }
}

pub struct SelfHealingStats {
    pub worker_id: String,
    pub health: WorkerHealth,
    pub consecutive_failures: u64,
    pub restart_count: u64,
    pub degradation_level: DegradationLevel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceConfig;

    #[tokio::test]
    async fn test_initial_health_is_healthy() {
        let config = SelfHealingConfig::default();
        let engine = Arc::new(RwLock::new(InferenceEngine::new(InferenceConfig::default())));
        let worker = SelfHealingWorker::new(engine, config);
        assert_eq!(worker.current_health().await, WorkerHealth::Healthy);
    }

    #[tokio::test]
    async fn test_heartbeat_recording() {
        let config = SelfHealingConfig::default();
        let engine = Arc::new(RwLock::new(InferenceEngine::new(InferenceConfig::default())));
        let worker = SelfHealingWorker::new(engine, config);
        assert_eq!(worker.last_heartbeat_ms.load(Ordering::Relaxed), 0);
        worker.record_heartbeat();
        assert_ne!(worker.last_heartbeat_ms.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_restart_count_starts_zero() {
        let config = SelfHealingConfig::default();
        let engine = Arc::new(RwLock::new(InferenceEngine::new(InferenceConfig::default())));
        let worker = SelfHealingWorker::new(engine, config);
        assert_eq!(worker.restart_count(), 0);
    }

    #[tokio::test]
    async fn test_degradation_manager_access() {
        let config = SelfHealingConfig::default();
        let engine = Arc::new(RwLock::new(InferenceEngine::new(InferenceConfig::default())));
        let worker = SelfHealingWorker::new(engine, config);
        let dm = worker.degradation_manager();
        assert_eq!(dm.current_level().await, DegradationLevel::None);
    }

    #[tokio::test]
    async fn test_shutdown_flag() {
        let config = SelfHealingConfig::default();
        let engine = Arc::new(RwLock::new(InferenceEngine::new(InferenceConfig::default())));
        let worker = SelfHealingWorker::new(engine, config);
        assert!(!worker.shutdown_flag.load(Ordering::Relaxed));
        worker.shutdown();
        assert!(worker.shutdown_flag.load(Ordering::Relaxed));
    }
}
