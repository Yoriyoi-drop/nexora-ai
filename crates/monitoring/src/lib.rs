pub mod health;
pub mod metrics;
pub mod profiling;
pub mod tracing;

pub use health::{HealthChecker, HealthReport, HealthStatus, SystemMetrics};
pub use metrics::MetricsCollector;
pub use profiling::{Profiler, ProfilingConfig};
pub use tracing::{init_tracing, TracingConfig, TracingFormat, TracingLevel};

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_health: bool,
    pub enable_profiling: bool,
    pub metrics_retention_hours: u64,
    pub health_check_interval_seconds: u64,
    pub tracing: TracingConfig,
    pub profiling: ProfilingConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_health: true,
            enable_profiling: false,
            metrics_retention_hours: 24,
            health_check_interval_seconds: 30,
            tracing: TracingConfig::default(),
            profiling: ProfilingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub metrics: String,
    pub health: String,
    pub uptime_seconds: u64,
}

pub struct MonitoringSystem {
    config: MonitoringConfig,
    metrics: Option<Arc<MetricsCollector>>,
    health: Option<Arc<RwLock<HealthChecker>>>,
    profiler: Option<Arc<RwLock<Profiler>>>,
    start_time: Instant,
}

impl MonitoringSystem {
    pub fn new(config: MonitoringConfig) -> Self {
        let metrics = if config.enable_metrics {
            Some(Arc::new(MetricsCollector::new()))
        } else {
            None
        };

        let health = if config.enable_health {
            Some(Arc::new(RwLock::new(HealthChecker::new())))
        } else {
            None
        };

        let profiler = if config.enable_profiling {
            let mut p = Profiler::new(config.profiling.clone());
            p.start();
            Some(Arc::new(RwLock::new(p)))
        } else {
            None
        };

        Self {
            config,
            metrics,
            health,
            profiler,
            start_time: Instant::now(),
        }
    }

    pub fn metrics(&self) -> Option<&Arc<MetricsCollector>> {
        self.metrics.as_ref()
    }

    pub fn health_checker(&self) -> Option<&Arc<RwLock<HealthChecker>>> {
        self.health.as_ref()
    }

    pub fn profiler(&self) -> Option<&Arc<RwLock<Profiler>>> {
        self.profiler.as_ref()
    }

    pub async fn get_system_status(&self) -> SystemStatus {
        let metrics_status = self.metrics.as_ref().map_or("disabled".to_string(), |_| "running".to_string());
        let health_status = self.health.as_ref().map_or("disabled".to_string(), |_| "running".to_string());

        SystemStatus {
            metrics: metrics_status,
            health: health_status,
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }

    pub fn elapsed_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

impl std::fmt::Debug for MonitoringSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MonitoringSystem")
            .field("config", &self.config)
            .field("start_time", &self.start_time)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_system_create() {
        let config = MonitoringConfig::default();
        let system = MonitoringSystem::new(config);
        let status = system.get_system_status().await;
        assert_eq!(status.metrics, "running");
        assert_eq!(status.health, "running");
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = MonitoringConfig::default();
        let system = MonitoringSystem::new(config);
        let health = system.health_checker().unwrap();
        let report = {
            let checker = health.read().await;
            checker.check_health()
        };
        assert_eq!(report.status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let config = MonitoringConfig::default();
        let system = MonitoringSystem::new(config);
        let metrics = system.metrics().unwrap();
        metrics.record_request(true, 0.05);
        metrics.record_request(false, 0.10);
        let prom = metrics.gather_prometheus();
        assert!(prom.contains("nexora_requests_total"));
        assert!(prom.contains("nexora_request_failures_total"));
    }

    #[tokio::test]
    async fn test_profiler() {
        let config = MonitoringConfig {
            enable_profiling: true,
            profiling: ProfilingConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let system = MonitoringSystem::new(config);
        let profiler = system.profiler().unwrap();
        {
            let mut p = profiler.write().await;
            p.record_cpu(0.5);
            p.record_memory(1024.0 * 1024.0 * 256.0);
        }
        let p = profiler.read().await;
        assert!(!p.cpu_samples().is_empty());
        assert!(!p.memory_samples().is_empty());
    }
}
