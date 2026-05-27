use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DegradationLevel {
    None,
    Reduced,
    Minimal,
    ReadOnly,
    Unavailable,
}

impl DegradationLevel {
    pub fn description(&self) -> &'static str {
        match self {
            DegradationLevel::None => "normal operation",
            DegradationLevel::Reduced => "reduced capacity — higher latency expected",
            DegradationLevel::Minimal => "minimal functionality — priority requests only",
            DegradationLevel::ReadOnly => "read-only mode — no generation allowed",
            DegradationLevel::Unavailable => "service unavailable",
        }
    }

    pub fn allows_generation(&self) -> bool {
        matches!(self, DegradationLevel::None | DegradationLevel::Reduced | DegradationLevel::Minimal)
    }

    pub fn allows_all_requests(&self) -> bool {
        matches!(self, DegradationLevel::None | DegradationLevel::Reduced)
    }

    pub fn allows_priority_only(&self) -> bool {
        matches!(self, DegradationLevel::Minimal)
    }
}

impl std::fmt::Display for DegradationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DegradationLevel::None => write!(f, "none"),
            DegradationLevel::Reduced => write!(f, "reduced"),
            DegradationLevel::Minimal => write!(f, "minimal"),
            DegradationLevel::ReadOnly => write!(f, "read_only"),
            DegradationLevel::Unavailable => write!(f, "unavailable"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DegradationConfig {
    pub error_rate_reduced: f64,
    pub error_rate_minimal: f64,
    pub error_rate_readonly: f64,
    pub latency_ms_reduced: f64,
    pub latency_ms_minimal: f64,
    pub eval_interval_ms: u64,
    pub auto_recover: bool,
    pub recovery_cooldown_ms: u64,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            error_rate_reduced: 0.05,
            error_rate_minimal: 0.15,
            error_rate_readonly: 0.30,
            latency_ms_reduced: 2_000.0,
            latency_ms_minimal: 5_000.0,
            eval_interval_ms: 10_000,
            auto_recover: true,
            recovery_cooldown_ms: 30_000,
        }
    }
}

pub struct DegradationManager {
    level: Arc<RwLock<DegradationLevel>>,
    config: DegradationConfig,
    error_count: Arc<AtomicU64>,
    total_requests: Arc<AtomicU64>,
    last_escalation: Arc<RwLock<Instant>>,
    last_recovery: Arc<RwLock<Instant>>,
    transition_count: Arc<AtomicU64>,
}

impl DegradationManager {
    pub fn new(config: DegradationConfig) -> Self {
        Self {
            level: Arc::new(RwLock::new(DegradationLevel::None)),
            config,
            error_count: Arc::new(AtomicU64::new(0)),
            total_requests: Arc::new(AtomicU64::new(0)),
            last_escalation: Arc::new(RwLock::new(Instant::now())),
            last_recovery: Arc::new(RwLock::new(Instant::now())),
            transition_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub async fn current_level(&self) -> DegradationLevel {
        *self.level.read().await
    }

    pub fn record_request(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn error_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let errors = self.error_count.load(Ordering::Relaxed);
        errors as f64 / total as f64
    }

    pub fn transition_count(&self) -> u64 {
        self.transition_count.load(Ordering::Relaxed)
    }

    fn compute_target_level(&self, error_rate: f64, avg_latency_ms: f64) -> DegradationLevel {
        if error_rate >= self.config.error_rate_readonly || avg_latency_ms >= 10_000.0 {
            DegradationLevel::ReadOnly
        } else if error_rate >= self.config.error_rate_minimal || avg_latency_ms >= self.config.latency_ms_minimal {
            DegradationLevel::Minimal
        } else if error_rate >= self.config.error_rate_reduced || avg_latency_ms >= self.config.latency_ms_reduced {
            DegradationLevel::Reduced
        } else {
            DegradationLevel::None
        }
    }

    pub async fn evaluate(&self, avg_latency_ms: f64) -> DegradationLevel {
        let error_rate = self.error_rate();
        let target = self.compute_target_level(error_rate, avg_latency_ms);
        let current = *self.level.read().await;

        if target > current {
            let mut last_esc = self.last_escalation.write().await;
            if last_esc.elapsed() >= Duration::from_millis(self.config.eval_interval_ms) {
                *last_esc = Instant::now();
                let mut level = self.level.write().await;
                *level = target;
                self.transition_count.fetch_add(1, Ordering::Relaxed);
                warn!(
                    "Degradation escalated to {:?} (error_rate={:.3}, latency={:.1}ms)",
                    target, error_rate, avg_latency_ms
                );
                return target;
            }
        } else if target < current && self.config.auto_recover {
            let mut last_rec = self.last_recovery.write().await;
            if last_rec.elapsed() >= Duration::from_millis(self.config.recovery_cooldown_ms) {
                *last_rec = Instant::now();
                let mut level = self.level.write().await;
                *level = target;
                self.transition_count.fetch_add(1, Ordering::Relaxed);
                info!(
                    "Degradation recovered to {:?} (error_rate={:.3}, latency={:.1}ms)",
                    target, error_rate, avg_latency_ms
                );
                return target;
            }
        }

        current
    }

    pub async fn reset(&self) {
        *self.level.write().await = DegradationLevel::None;
        self.error_count.store(0, Ordering::Relaxed);
        self.total_requests.store(0, Ordering::Relaxed);
        debug!("DegradationManager reset to None");
    }

    pub fn stats(&self) -> DegradationStats {
        DegradationStats {
            level: DegradationLevel::None,
            error_rate: self.error_rate(),
            transition_count: self.transition_count.load(Ordering::Relaxed),
        }
    }
}

pub struct DegradationStats {
    pub level: DegradationLevel,
    pub error_rate: f64,
    pub transition_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_initial_level_is_none() {
        let mgr = DegradationManager::new(DegradationConfig::default());
        assert_eq!(mgr.current_level().await, DegradationLevel::None);
    }

    #[tokio::test]
    async fn test_escalation_on_high_error_rate() {
        let mgr = DegradationManager::new(DegradationConfig {
            error_rate_reduced: 0.05,
            eval_interval_ms: 0,
            ..Default::default()
        });

        for _ in 0..10 {
            mgr.record_request();
        }
        // 1 error out of 10 = 10% error rate, between reduced(5%) and minimal(15%)
        mgr.record_error();

        let level = mgr.evaluate(100.0).await;
        assert_eq!(level, DegradationLevel::Reduced);
    }

    #[tokio::test]
    async fn test_escalation_on_high_latency() {
        let mgr = DegradationManager::new(DegradationConfig {
            latency_ms_reduced: 2000.0,
            eval_interval_ms: 0,
            ..Default::default()
        });

        let level = mgr.evaluate(3000.0).await;
        assert_eq!(level, DegradationLevel::Reduced);
    }

    #[tokio::test]
    async fn test_transition_tracking() {
        let mgr = DegradationManager::new(DegradationConfig {
            error_rate_reduced: 0.01,
            eval_interval_ms: 0,
            ..Default::default()
        });

        assert_eq!(mgr.transition_count(), 0);
        mgr.record_request();
        mgr.record_error();
        mgr.evaluate(100.0).await;
        assert_eq!(mgr.transition_count(), 1);
    }
}
