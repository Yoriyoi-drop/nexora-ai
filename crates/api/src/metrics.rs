//! API Metrics - Rust implementation
//! 
//! Metrics collection and monitoring for API server

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use crate::{MetricsData, RouteMetrics};

/// Metrics collector for API performance monitoring
#[derive(Debug)]
pub struct MetricsCollector {
    metrics: Arc<RwLock<MetricsStorage>>,
    start_time: Instant,
}

/// Internal metrics storage
#[derive(Debug)]
struct MetricsStorage {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    response_times: Vec<Duration>,
    route_metrics: HashMap<String, RouteMetricStorage>,
    error_counts: HashMap<String, u64>,
    active_connections: usize,
    last_reset: Instant,
}

impl Default for MetricsStorage {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            response_times: Vec::new(),
            route_metrics: HashMap::new(),
            error_counts: HashMap::new(),
            active_connections: 0,
            last_reset: Instant::now(),
        }
    }
}

#[derive(Debug, Default)]
struct RouteMetricStorage {
    requests: u64,
    response_times: Vec<Duration>,
    error_count: u64,
    last_request: Option<Instant>,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(MetricsStorage::default())),
            start_time: Instant::now(),
        }
    }
    
    /// Record a request
    pub async fn record_request(&self, method: &str, path: &str, response_time: Duration, status: axum::http::StatusCode) {
        let mut metrics = self.metrics.write().await;
        
        // Update global metrics
        metrics.total_requests += 1;
        metrics.response_times.push(response_time);
        
        if status.is_success() {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
            let error_key = format!("{} {}", status.as_u16(), status.canonical_reason().unwrap_or("Unknown"));
            *metrics.error_counts.entry(error_key).or_insert(0) += 1;
        }
        
        // Update route-specific metrics
        let route_key = format!("{} {}", method, path);
        let route_metrics = metrics.route_metrics.entry(route_key.clone()).or_insert_with(RouteMetricStorage::default);
        
        route_metrics.requests += 1;
        route_metrics.response_times.push(response_time);
        route_metrics.last_request = Some(Instant::now());
        
        if !status.is_success() {
            route_metrics.error_count += 1;
        }
        
        // Keep only recent response times (last 1000)
        let response_times_len = {
            let metrics = self.metrics.read().await;
            metrics.response_times.len()
        };
        if response_times_len > 1000 {
            let mut metrics = self.metrics.write().await;
            metrics.response_times.drain(0..response_times_len - 1000);
        }
        
        let route_response_times_len = route_metrics.response_times.len();
        if route_response_times_len > 1000 {
            route_metrics.response_times.drain(0..route_response_times_len - 1000);
        }
    }
    
    /// Increment active connections
    pub async fn increment_connections(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.active_connections += 1;
    }
    
    /// Decrement active connections
    pub async fn decrement_connections(&self) {
        let mut metrics = self.metrics.write().await;
        if metrics.active_connections > 0 {
            metrics.active_connections -= 1;
        }
    }
    
    /// Get current metrics
    pub async fn get_current_metrics(&self) -> MetricsData {
        let metrics = self.metrics.read().await;
        
        let average_response_time = if metrics.response_times.is_empty() {
            0.0
        } else {
            let total: Duration = metrics.response_times.iter().sum();
            total.as_millis() as f64 / metrics.response_times.len() as f64
        };
        
        let requests_per_second = {
            let elapsed = metrics.last_reset.elapsed();
            if elapsed.as_secs() > 0 {
                metrics.total_requests as f64 / elapsed.as_secs() as f64
            } else {
                0.0
            }
        };
        
        let error_rate = if metrics.total_requests > 0 {
            (metrics.failed_requests as f64 / metrics.total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        let top_routes = metrics.route_metrics
            .iter()
            .map(|(route, storage)| {
                let avg_response_time = if storage.response_times.is_empty() {
                    0.0
                } else {
                    let total: Duration = storage.response_times.iter().sum();
                    total.as_millis() as f64 / storage.response_times.len() as f64
                };
                
                let error_rate = if storage.requests > 0 {
                    (storage.error_count as f64 / storage.requests as f64) * 100.0
                } else {
                    0.0
                };
                
                RouteMetrics {
                    path: route.clone(),
                    method: route.split(' ').next().unwrap_or("UNKNOWN").to_string(),
                    requests: storage.requests,
                    average_response_time_ms: avg_response_time,
                    error_rate_percent: error_rate,
                }
            })
            .collect::<Vec<_>>();
        
        MetricsData {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_secs(),
            requests_total: metrics.total_requests,
            requests_per_second,
            average_response_time_ms: average_response_time,
            error_rate_percent: error_rate,
            active_connections: metrics.active_connections,
            memory_usage_mb: self.get_memory_usage(),
            cpu_usage_percent: self.get_cpu_usage(),
            top_routes,
        }
    }
    
    /// Get route-specific metrics
    pub async fn get_route_metrics(&self, route: &str) -> Option<RouteMetrics> {
        let metrics = self.metrics.read().await;
        
        if let Some(storage) = metrics.route_metrics.get(route) {
            let average_response_time = if storage.response_times.is_empty() {
                0.0
            } else {
                let total: Duration = storage.response_times.iter().sum();
                total.as_millis() as f64 / storage.response_times.len() as f64
            };
            
            let error_rate = if storage.requests > 0 {
                (storage.error_count as f64 / storage.requests as f64) * 100.0
            } else {
                0.0
            };
            
            Some(RouteMetrics {
                path: route.to_string(),
                method: route.split(' ').next().unwrap_or("UNKNOWN").to_string(),
                requests: storage.requests,
                average_response_time_ms: average_response_time,
                error_rate_percent: error_rate,
            })
        } else {
            None
        }
    }
    
    /// Get all route metrics
    pub async fn get_all_route_metrics(&self) -> Vec<RouteMetrics> {
        let metrics = self.metrics.read().await;
        
        metrics.route_metrics
            .iter()
            .map(|(route, storage)| {
                let average_response_time = if storage.response_times.is_empty() {
                    0.0
                } else {
                    let total: Duration = storage.response_times.iter().sum();
                    total.as_millis() as f64 / storage.response_times.len() as f64
                };
                
                let error_rate = if storage.requests > 0 {
                    (storage.error_count as f64 / storage.requests as f64) * 100.0
                } else {
                    0.0
                };
                
                RouteMetrics {
                    path: route.clone(),
                    method: route.split(' ').next().unwrap_or("UNKNOWN").to_string(),
                    requests: storage.requests,
                    average_response_time_ms: average_response_time,
                    error_rate_percent: error_rate,
                }
            })
            .collect()
    }
    
    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = MetricsStorage {
            last_reset: Instant::now(),
            ..Default::default()
        };
    }
    
    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get memory usage in MB
    fn get_memory_usage(&self) -> f64 {
        // Try to get memory usage from /proc/self/status on Linux
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
        
        // Fallback: use a simple estimation based on process info
        // This is a rough estimate, in a real implementation you'd use proper memory profiling
        50.0 // Default fallback: 50MB
    }
    
    /// Get CPU usage as percentage
    fn get_cpu_usage(&self) -> f64 {
        // Simple CPU usage estimation based on process activity
        // This is a simplified implementation
        if let Ok(usage) = self.get_process_cpu_usage() {
            usage
        } else {
            // Fallback: return a reasonable default
            0.1 // 10% CPU usage as fallback
        }
    }
    
    /// Get process CPU usage from /proc/self/stat on Linux
    fn get_process_cpu_usage(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let stat_content = std::fs::read_to_string("/proc/self/stat")?;
        let parts: Vec<&str> = stat_content.split_whitespace().collect();
        
        if parts.len() < 17 {
            return Err("Insufficient data in /proc/self/stat".into());
        }
        
        // Get utime (user time) and stime (system time) from fields 14 and 15
        // These values are in clock ticks (CLK_TCK), not seconds
        let utime: u64 = parts[13].parse()?;
        let stime: u64 = parts[14].parse()?;
        let total_time = utime + stime;
        
        // Get total CPU time from /proc/stat (also in clock ticks)
        let stat_content = std::fs::read_to_string("/proc/stat")?;
        let first_line = stat_content.lines().next().ok_or("No data in /proc/stat")?;
        let cpu_parts: Vec<u64> = first_line.split_whitespace()
            .skip(1) // Skip "cpu"
            .take(4) // user, nice, system, idle
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if cpu_parts.len() < 4 {
            return Err("Insufficient CPU data in /proc/stat".into());
        }
        
        let total_cpu_time: u64 = cpu_parts.iter().sum();
        
        if total_cpu_time == 0 {
            return Ok(0.0);
        }
        
        // CLK_TCK is typically 100 on Linux (sysconf(_SC_CLK_TCK))
        let clk_tck = 100.0;
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as f64;
        
        // Convert both process time and system time from clock ticks to seconds
        let process_secs = total_time as f64 / clk_tck;
        let system_secs = total_cpu_time as f64 / clk_tck;
        
        // Calculate per-CPU percentage
        Ok((process_secs / system_secs / num_cpus) * 100.0)
    }
}

/// Metrics exporter for external monitoring systems
#[derive(Debug)]
pub struct MetricsExporter {
    metrics_collector: Arc<MetricsCollector>,
}

impl MetricsExporter {
    pub fn new(metrics_collector: Arc<MetricsCollector>) -> Self {
        Self { metrics_collector }
    }
    
    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let metrics = self.metrics_collector.get_current_metrics().await;
        
        format!(
            "# HELP nexora_api_requests_total Total number of API requests\n\
             # TYPE nexora_api_requests_total counter\n\
             nexora_api_requests_total {}\n\
             # HELP nexora_api_requests_per_second Requests per second\n\
             # TYPE nexora_api_requests_per_second gauge\n\
             nexora_api_requests_per_second {}\n\
             # HELP nexora_api_response_time_ms Average response time in milliseconds\n\
             # TYPE nexora_api_response_time_ms gauge\n\
             nexora_api_response_time_ms {}\n\
             # HELP nexora_api_error_rate_percent Error rate percentage\n\
             # TYPE nexora_api_error_rate_percent gauge\n\
             nexora_api_error_rate_percent {}\n\
             # HELP nexora_api_active_connections Active connections\n\
             # TYPE nexora_api_active_connections gauge\n\
             nexora_api_active_connections {}\n",
            metrics.requests_total,
            metrics.requests_per_second,
            metrics.average_response_time_ms,
            metrics.error_rate_percent,
            metrics.active_connections
        )
    }
    
    /// Export metrics in JSON format
    pub async fn export_json(&self) -> Result<serde_json::Value> {
        let metrics = self.metrics_collector.get_current_metrics().await;
        Ok(serde_json::to_value(metrics)?)
    }
}
