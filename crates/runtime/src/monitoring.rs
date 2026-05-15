//! Monitoring Module
//! 
//! Provides performance monitoring and health check capabilities

use crate::{Result, InferenceError};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub request_count: u64,
    pub average_latency_ms: f64,
    pub error_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            request_count: 0,
            average_latency_ms: 0.0,
            error_rate: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
        }
    }
}

/// Health checker for system components
pub struct HealthChecker {
    last_check: Instant,
    metrics: PerformanceMetrics,
}

impl HealthChecker {
    /// Create new health checker
    pub fn new() -> Self {
        Self {
            last_check: Instant::now(),
            metrics: PerformanceMetrics::default(),
        }
    }
    
    /// Check system health
    pub async fn check_health(&mut self) -> Result<bool> {
        self.last_check = Instant::now();

        let mut healthy = true;

        if self.metrics.cpu_usage_percent > 95.0 {
            tracing::warn!("CPU usage critical: {}%", self.metrics.cpu_usage_percent);
            healthy = false;
        }

        if self.metrics.memory_usage_mb > 80_000.0 {
            tracing::warn!("Memory usage critical: {} MB", self.metrics.memory_usage_mb);
            healthy = false;
        }

        if self.metrics.error_rate > 0.3 {
            tracing::warn!("Error rate too high: {}", self.metrics.error_rate);
            healthy = false;
        }

        if self.metrics.average_latency_ms > 10_000.0 {
            tracing::warn!("Average latency too high: {} ms", self.metrics.average_latency_ms);
            healthy = false;
        }

        Ok(healthy)
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
    
    /// Update metrics
    pub fn update_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics = metrics;
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}
