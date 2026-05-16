use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub status: HealthStatus,
    pub uptime_seconds: u64,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub healthy: bool,
    pub message: String,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_ratio: f64,
    pub memory_usage_bytes: f64,
    pub active_connections: usize,
    pub queue_depth: usize,
    pub error_rate: f64,
    pub average_latency_ms: f64,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_ratio: 0.0,
            memory_usage_bytes: 0.0,
            active_connections: 0,
            queue_depth: 0,
            error_rate: 0.0,
            average_latency_ms: 0.0,
        }
    }
}

pub struct HealthChecker {
    start_time: Instant,
    pub metrics: SystemMetrics,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: SystemMetrics::default(),
        }
    }

    pub fn update_metrics(&mut self, metrics: SystemMetrics) {
        self.metrics = metrics;
    }

    pub fn check_health(&self) -> HealthReport {
        let mut checks = Vec::new();
        let mut all_healthy = true;
        let mut degraded = false;

        // CPU check
        let cpu_check = if self.metrics.cpu_usage_ratio > 0.95 {
            all_healthy = false;
            HealthCheck {
                name: "cpu".into(),
                healthy: false,
                message: format!("CPU usage critical: {:.1}%", self.metrics.cpu_usage_ratio * 100.0),
                latency_ms: 0,
            }
        } else if self.metrics.cpu_usage_ratio > 0.80 {
            degraded = true;
            HealthCheck {
                name: "cpu".into(),
                healthy: true,
                message: format!("CPU usage elevated: {:.1}%", self.metrics.cpu_usage_ratio * 100.0),
                latency_ms: 0,
            }
        } else {
            HealthCheck {
                name: "cpu".into(),
                healthy: true,
                message: format!("CPU usage normal: {:.1}%", self.metrics.cpu_usage_ratio * 100.0),
                latency_ms: 0,
            }
        };
        checks.push(cpu_check);

        // Memory check
        let mem_check = if self.metrics.memory_usage_bytes > 80_000_000_000.0 {
            all_healthy = false;
            HealthCheck {
                name: "memory".into(),
                healthy: false,
                message: format!("Memory usage critical: {:.1} GB", self.metrics.memory_usage_bytes / 1e9),
                latency_ms: 0,
            }
        } else if self.metrics.memory_usage_bytes > 60_000_000_000.0 {
            degraded = true;
            HealthCheck {
                name: "memory".into(),
                healthy: true,
                message: format!("Memory usage elevated: {:.1} GB", self.metrics.memory_usage_bytes / 1e9),
                latency_ms: 0,
            }
        } else {
            HealthCheck {
                name: "memory".into(),
                healthy: true,
                message: format!("Memory usage normal: {:.1} GB", self.metrics.memory_usage_bytes / 1e9),
                latency_ms: 0,
            }
        };
        checks.push(mem_check);

        // Error rate check
        let err_check = if self.metrics.error_rate > 0.30 {
            all_healthy = false;
            HealthCheck {
                name: "error_rate".into(),
                healthy: false,
                message: format!("Error rate critical: {:.1}%", self.metrics.error_rate * 100.0),
                latency_ms: 0,
            }
        } else if self.metrics.error_rate > 0.10 {
            degraded = true;
            HealthCheck {
                name: "error_rate".into(),
                healthy: true,
                message: format!("Error rate elevated: {:.1}%", self.metrics.error_rate * 100.0),
                latency_ms: 0,
            }
        } else {
            HealthCheck {
                name: "error_rate".into(),
                healthy: true,
                message: format!("Error rate normal: {:.1}%", self.metrics.error_rate * 100.0),
                latency_ms: 0,
            }
        };
        checks.push(err_check);

        // Latency check
        let lat_check = if self.metrics.average_latency_ms > 10_000.0 {
            all_healthy = false;
            HealthCheck {
                name: "latency".into(),
                healthy: false,
                message: format!("Latency critical: {:.1} ms", self.metrics.average_latency_ms),
                latency_ms: self.metrics.average_latency_ms as u64,
            }
        } else if self.metrics.average_latency_ms > 5_000.0 {
            degraded = true;
            HealthCheck {
                name: "latency".into(),
                healthy: true,
                message: format!("Latency elevated: {:.1} ms", self.metrics.average_latency_ms),
                latency_ms: self.metrics.average_latency_ms as u64,
            }
        } else {
            HealthCheck {
                name: "latency".into(),
                healthy: true,
                message: format!("Latency normal: {:.1} ms", self.metrics.average_latency_ms),
                latency_ms: self.metrics.average_latency_ms as u64,
            }
        };
        checks.push(lat_check);

        let status = if !all_healthy {
            HealthStatus::Unhealthy
        } else if degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        HealthReport {
            status,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            checks,
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HealthChecker")
            .field("start_time", &self.start_time)
            .field("metrics", &self.metrics)
            .finish()
    }
}
