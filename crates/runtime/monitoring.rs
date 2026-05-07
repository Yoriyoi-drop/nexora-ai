//! Performance Monitoring
//! 
//! Shared monitoring utilities for AI frameworks

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};

/// Performance metrics collector
pub struct PerformanceMonitor {
    name: String,
    metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    counters: Arc<RwLock<HashMap<String, Counter>>>,
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    gauges: Arc<RwLock<HashMap<String, Gauge>>>,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(name: String) -> Self {
        Self {
            name,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Record counter increment
    pub async fn increment_counter(&self, name: &str, value: u64) -> Result<()> {
        let mut counters = self.counters.write().await;
        let counter = counters.entry(name.to_string()).or_insert_with(|| Counter::new(name.to_string()));
        counter.increment(value);
        
        debug!("Monitor {}: Incremented counter '{}' by {}", self.name, name, value);
        Ok(())
    }
    
    /// Record gauge value
    pub async fn set_gauge(&self, name: &str, value: f64) -> Result<()> {
        let mut gauges = self.gauges.write().await;
        let gauge = gauges.entry(name.to_string()).or_insert_with(|| Gauge::new(name.to_string()));
        gauge.set(value);
        
        debug!("Monitor {}: Set gauge '{}' to {}", self.name, name, value);
        Ok(())
    }
    
    /// Record histogram observation
    pub async fn observe_histogram(&self, name: &str, value: f64) -> Result<()> {
        let mut histograms = self.histograms.write().await;
        let histogram = histograms.entry(name.to_string()).or_insert_with(|| Histogram::new(name.to_string()));
        histogram.observe(value);
        
        debug!("Monitor {}: Observed histogram '{}' value {}", self.name, name, value);
        Ok(())
    }
    
    /// Record timer observation
    pub async fn record_timer(&self, name: &str, duration: Duration) -> Result<()> {
        self.observe_histogram(name, duration.as_secs_f64()).await
    }
    
    /// Time a function execution
    pub async fn time_function<F, Fut, T>(&self, name: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let start = Instant::now();
        let result = f().await;
        let duration = start.elapsed();
        
        self.record_timer(name, duration).await?;
        
        // Also record success/failure
        match &result {
            Ok(_) => {
                self.increment_counter(&format!("{}_success", name), 1).await?;
            }
            Err(_) => {
                self.increment_counter(&format!("{}_error", name), 1).await?;
            }
        }
        
        result
    }
    
    /// Get all metrics
    pub async fn get_all_metrics(&self) -> HashMap<String, MetricValue> {
        let mut all_metrics = HashMap::new();
        
        // Collect counters
        {
            let counters = self.counters.read().await;
            for (name, counter) in counters.iter() {
                all_metrics.insert(name.clone(), MetricValue::Counter(counter.value()));
            }
        }
        
        // Collect gauges
        {
            let gauges = self.gauges.read().await;
            for (name, gauge) in gauges.iter() {
                all_metrics.insert(name.clone(), MetricValue::Gauge(gauge.value()));
            }
        }
        
        // Collect histograms
        {
            let histograms = self.histograms.read().await;
            for (name, histogram) in histograms.iter() {
                let stats = histogram.statistics();
                all_metrics.insert(name.clone(), MetricValue::Histogram(stats));
            }
        }
        
        all_metrics
    }
    
    /// Get metric summary
    pub async fn get_summary(&self) -> MonitoringSummary {
        let metrics = self.get_all_metrics().await;
        
        MonitoringSummary {
            name: self.name.clone(),
            total_metrics: metrics.len(),
            counters: metrics.values()
                .filter(|m| matches!(m, MetricValue::Counter(_)))
                .count(),
            gauges: metrics.values()
                .filter(|m| matches!(m, MetricValue::Gauge(_)))
                .count(),
            histograms: metrics.values()
                .filter(|m| matches!(m, MetricValue::Histogram(_)))
                .count(),
            metrics,
        }
    }
    
    /// Reset all metrics
    pub async fn reset(&self) -> Result<()> {
        {
            let mut counters = self.counters.write().await;
            for counter in counters.values_mut() {
                counter.reset();
            }
        }
        
        {
            let mut gauges = self.gauges.write().await;
            for gauge in gauges.values_mut() {
                gauge.reset();
            }
        }
        
        {
            let mut histograms = self.histograms.write().await;
            for histogram in histograms.values_mut() {
                histogram.reset();
            }
        }
        
        info!("Monitor {}: All metrics reset", self.name);
        Ok(())
    }
}

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(HistogramStatistics),
}

/// Counter metric
#[derive(Debug, Clone)]
pub struct Counter {
    name: String,
    value: u64,
}

impl Counter {
    pub fn new(name: String) -> Self {
        Self { name, value: 0 }
    }
    
    pub fn increment(&mut self, value: u64) {
        self.value += value;
    }
    
    pub fn value(&self) -> u64 {
        self.value
    }
    
    pub fn reset(&mut self) {
        self.value = 0;
    }
}

/// Gauge metric
#[derive(Debug, Clone)]
pub struct Gauge {
    name: String,
    value: f64,
}

impl Gauge {
    pub fn new(name: String) -> Self {
        Self { name, value: 0.0 }
    }
    
    pub fn set(&mut self, value: f64) {
        self.value = value;
    }
    
    pub fn value(&self) -> f64 {
        self.value
    }
    
    pub fn reset(&mut self) {
        self.value = 0.0;
    }
}

/// Histogram metric
#[derive(Debug, Clone)]
pub struct Histogram {
    name: String,
    observations: Vec<f64>,
    max_observations: usize,
}

impl Histogram {
    pub fn new(name: String) -> Self {
        Self {
            name,
            observations: Vec::new(),
            max_observations: 1000,
        }
    }
    
    pub fn observe(&mut self, value: f64) {
        self.observations.push(value);
        
        // Limit observations to prevent memory bloat
        if self.observations.len() > self.max_observations {
            self.observations.remove(0);
        }
    }
    
    pub fn statistics(&self) -> HistogramStatistics {
        if self.observations.is_empty() {
            return HistogramStatistics {
                name: self.name.clone(),
                count: 0,
                sum: 0.0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }
        
        let mut sorted = self.observations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean = sum / count as f64;
        let min = sorted[0];
        let max = sorted[count - 1];
        
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        
        let p95_index = (0.95 * count as f64) as usize;
        let p99_index = (0.99 * count as f64) as usize;
        
        let p95 = if p95_index < count { sorted[p95_index] } else { max };
        let p99 = if p99_index < count { sorted[p99_index] } else { max };
        
        HistogramStatistics {
            name: self.name.clone(),
            count,
            sum,
            min,
            max,
            mean,
            median,
            p95,
            p99,
        }
    }
    
    pub fn reset(&mut self) {
        self.observations.clear();
    }
}

/// Histogram statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStatistics {
    pub name: String,
    pub count: usize,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Monitoring summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSummary {
    pub name: String,
    pub total_metrics: usize,
    pub counters: usize,
    pub gauges: usize,
    pub histograms: usize,
    pub metrics: HashMap<String, MetricValue>,
}

/// Health check system
pub struct HealthChecker {
    name: String,
    checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
}

impl HealthChecker {
    /// Create new health checker
    pub fn new(name: String) -> Self {
        Self {
            name,
            checks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Add health check
    pub async fn add_check<F>(&self, name: String, check_fn: F)
    where
        F: Fn() -> Result<bool> + Send + Sync + 'static,
    {
        let health_check = HealthCheck {
            name: name.clone(),
            check_fn: Arc::new(Box::new(check_fn)),
        };
        
        let mut checks = self.checks.write().await;
        checks.insert(name, health_check);
    }
    
    /// Run all health checks
    pub async fn run_checks(&self) -> HealthStatus {
        let checks = self.checks.read().await;
        let mut results = HashMap::new();
        let mut overall_healthy = true;
        
        for (name, check) in checks.iter() {
            let result = match (check.check_fn)() {
                Ok(healthy) => HealthCheckResult {
                    name: name.clone(),
                    healthy,
                    message: if healthy { "OK".to_string() } else { "Failed".to_string() },
                    timestamp: std::time::SystemTime::now(),
                },
                Err(e) => HealthCheckResult {
                    name: name.clone(),
                    healthy: false,
                    message: format!("Error: {}", e),
                    timestamp: std::time::SystemTime::now(),
                },
            };
            
            if !result.healthy {
                overall_healthy = false;
            }
            
            results.insert(name.clone(), result);
        }
        
        HealthStatus {
            name: self.name.clone(),
            healthy: overall_healthy,
            checks: results,
        }
    }
}

/// Health check function
pub struct HealthCheck {
    name: String,
    check_fn: Arc<Box<dyn Fn() -> Result<bool> + Send + Sync>>,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub name: String,
    pub healthy: bool,
    pub message: String,
    pub timestamp: std::time::SystemTime,
}

/// Overall health status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub name: String,
    pub healthy: bool,
    pub checks: HashMap<String, HealthCheckResult>,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}
