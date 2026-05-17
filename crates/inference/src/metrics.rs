//! Metrics Collection
//! 
//! Metrics collection dan analysis untuk inference engine.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tracing::debug;
use chrono::{DateTime, Utc};

use crate::Result;

/// Metrics collector untuk inference
pub struct MetricsCollector {
    /// Configuration
    config: MetricsConfig,
    /// Current metrics
    metrics: Arc<RwLock<InferenceMetrics>>,
    /// Historical metrics
    history: Arc<RwLock<VecDeque<TimestampedMetrics>>>,
    /// Alert thresholds
    alert_thresholds: Arc<RwLock<AlertThresholds>>,
    /// Active alerts
    active_alerts: Arc<RwLock<Vec<MetricAlert>>>,
}

/// Configuration untuk metrics collection
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// History size limit
    pub max_history_size: usize,
    /// Collection interval (seconds)
    pub collection_interval_seconds: u64,
    /// Enable automatic aggregation
    pub enable_aggregation: bool,
    /// Aggregation window (seconds)
    pub aggregation_window_seconds: u64,
    /// Enable alerts
    pub enable_alerts: bool,
    /// Alert check interval (seconds)
    pub alert_check_interval_seconds: u64,
    /// Retention period (hours)
    pub retention_hours: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            collection_interval_seconds: 10,
            enable_aggregation: true,
            aggregation_window_seconds: 60,
            enable_alerts: true,
            alert_check_interval_seconds: 30,
            retention_hours: 24,
        }
    }
}

/// Inference metrics
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    /// Request metrics
    pub requests: RequestMetrics,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Resource metrics
    pub resources: ResourceMetrics,
    /// Model metrics
    pub models: ModelMetrics,
    /// Error metrics
    pub errors: ErrorMetrics,
    /// Custom metrics
    pub custom: HashMap<String, f64>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Request metrics
#[derive(Debug, Clone, Default)]
pub struct RequestMetrics {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Cancelled requests
    pub cancelled_requests: u64,
    /// Timeout requests
    pub timeout_requests: u64,
    /// Average request size (tokens)
    pub avg_request_size: f64,
    /// Average response size (tokens)
    pub avg_response_size: f64,
    /// Requests per second
    pub requests_per_second: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// P50 latency (ms)
    pub p50_latency_ms: f64,
    /// P95 latency (ms)
    pub p95_latency_ms: f64,
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
    /// Throughput (tokens/second)
    pub throughput_tokens_per_second: f64,
    /// Throughput (requests/second)
    pub throughput_requests_per_second: f64,
    /// Average queue time (ms)
    pub avg_queue_time_ms: f64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
}

/// Resource metrics
#[derive(Debug, Clone, Default)]
pub struct ResourceMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    /// GPU memory usage (bytes)
    pub gpu_memory_usage_bytes: Option<u64>,
    /// GPU memory usage percentage
    pub gpu_memory_usage_percent: Option<f64>,
    /// Active connections
    pub active_connections: usize,
    /// Queue depth
    pub queue_depth: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Model metrics
#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    /// Model load time (ms)
    pub model_load_time_ms: u64,
    /// Model memory usage (bytes)
    pub model_memory_bytes: u64,
    /// Active models
    pub active_models: usize,
    /// Model switches
    pub model_switches: u64,
    /// Average inference time per token (ms)
    pub avg_inference_time_per_token_ms: f64,
    /// Model utilization percentage
    pub model_utilization_percent: f64,
}

/// Error metrics
#[derive(Debug, Clone, Default)]
pub struct ErrorMetrics {
    /// Total errors
    pub total_errors: u64,
    /// Error rate
    pub error_rate: f64,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    /// Recent errors
    pub recent_errors: VecDeque<ErrorEntry>,
}

/// Error entry
#[derive(Debug, Clone)]
pub struct ErrorEntry {
    pub timestamp: DateTime<Utc>,
    pub error_type: String,
    pub message: String,
    pub request_id: Option<Uuid>,
}

/// Timestamped metrics
#[derive(Debug, Clone)]
pub struct TimestampedMetrics {
    pub timestamp: DateTime<Utc>,
    pub metrics: InferenceMetrics,
}

/// Alert thresholds
#[derive(Debug, Clone, Default)]
pub struct AlertThresholds {
    /// Maximum error rate (percentage)
    pub max_error_rate: f64,
    /// Maximum latency (ms)
    pub max_latency_ms: f64,
    /// Minimum throughput (tokens/second)
    pub min_throughput_tokens_per_second: f64,
    /// Maximum CPU usage (percentage)
    pub max_cpu_usage_percent: f64,
    /// Maximum memory usage (percentage)
    pub max_memory_usage_percent: f64,
    /// Maximum queue depth
    pub max_queue_depth: usize,
}

/// Metric alert
#[derive(Debug, Clone)]
pub struct MetricAlert {
    /// Alert ID
    pub id: Uuid,
    /// Alert type
    pub alert_type: AlertType,
    /// Metric name
    pub metric_name: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert message
    pub message: String,
    /// Severity
    pub severity: AlertSeverity,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Resolved timestamp
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Alert type
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    ErrorRateHigh,
    LatencyHigh,
    ThroughputLow,
    CpuHigh,
    MemoryHigh,
    QueueDepthHigh,
    Custom(String),
}

/// Alert severity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self::with_config(MetricsConfig::default())
    }
    
    /// Create metrics collector with configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(InferenceMetrics::default())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            alert_thresholds: Arc::new(RwLock::new(AlertThresholds::default())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Record request completed
    pub async fn record_request_completed(
        &self,
        request_id: Uuid,
        success: bool,
        latency_ms: u64,
        input_tokens: usize,
        output_tokens: usize,
    ) -> Result<()> {
        debug!("Recording request completion: {}", request_id);
        
        let mut metrics = self.metrics.write().await;
        
        // Update request metrics
        metrics.requests.total_requests += 1;
        if success {
            metrics.requests.successful_requests += 1;
        } else {
            metrics.requests.failed_requests += 1;
        }
        
        // Update averages
        let total_requests = metrics.requests.total_requests;
        metrics.requests.avg_request_size = 
            (metrics.requests.avg_request_size * (total_requests - 1) as f64 + input_tokens as f64) / total_requests as f64;
        metrics.requests.avg_response_size = 
            (metrics.requests.avg_response_size * (total_requests - 1) as f64 + output_tokens as f64) / total_requests as f64;
        
        // Update performance metrics
        let latency = latency_ms as f64;
        metrics.performance.avg_latency_ms = 
            (metrics.performance.avg_latency_ms * (total_requests - 1) as f64 + latency) / total_requests as f64;
        
        // Update throughput
        if output_tokens > 0 {
            let tokens_per_second = (output_tokens as f64 * 1000.0) / latency;
            metrics.performance.throughput_tokens_per_second = 
                (metrics.performance.throughput_tokens_per_second * (total_requests - 1) as f64 + tokens_per_second) / total_requests as f64;
        }
        
        metrics.last_updated = Utc::now();
        
        // Check for alerts
        if self.config.enable_alerts {
            drop(metrics);
            self.check_alerts().await?;
        }
        
        Ok(())
    }
    
    /// Record error
    pub async fn record_error(&self, error_type: String, message: String, request_id: Option<Uuid>) -> Result<()> {
        debug!("Recording error: {}", error_type);
        
        let mut metrics = self.metrics.write().await;
        
        // Update error metrics
        metrics.errors.total_errors += 1;
        *metrics.errors.errors_by_type.entry(error_type.clone()).or_insert(0) += 1;
        
        // Add to recent errors
        let error_entry = ErrorEntry {
            timestamp: Utc::now(),
            error_type: error_type.clone(),
            message,
            request_id,
        };
        
        metrics.errors.recent_errors.push_back(error_entry);
        
        // Keep only last 100 errors
        if metrics.errors.recent_errors.len() > 100 {
            metrics.errors.recent_errors.pop_front();
        }
        
        // Update error rate
        let total_requests = metrics.requests.total_requests;
        if total_requests > 0 {
            metrics.errors.error_rate = metrics.errors.total_errors as f64 / total_requests as f64;
        }
        
        metrics.last_updated = Utc::now();
        
        Ok(())
    }
    
    /// Update resource metrics
    pub async fn update_resource_metrics(&self, resources: ResourceMetrics) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.resources = resources;
        metrics.last_updated = Utc::now();
        
        // Check for alerts
        if self.config.enable_alerts {
            drop(metrics);
            self.check_alerts().await?;
        }
        
        Ok(())
    }
    
    /// Update model metrics
    pub async fn update_model_metrics(&self, models: ModelMetrics) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.models = models;
        metrics.last_updated = Utc::now();
        Ok(())
    }
    
    /// Set custom metric
    pub async fn set_custom_metric(&self, name: String, value: f64) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.custom.insert(name, value);
        metrics.last_updated = Utc::now();
        Ok(())
    }
    
    /// Get current metrics
    pub async fn get_current_metrics(&self) -> InferenceMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get metrics history
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Vec<TimestampedMetrics> {
        let history = self.history.read().await;
        match limit {
            Some(limit) => history.iter().rev().take(limit).cloned().collect(),
            None => history.iter().rev().cloned().collect(),
        }
    }
    
    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<MetricAlert> {
        self.active_alerts.read().await.clone()
    }
    
    /// Resolve alert
    pub async fn resolve_alert(&self, alert_id: Uuid) -> Result<bool> {
        let mut alerts = self.active_alerts.write().await;
        
        if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.resolved_at = Some(Utc::now());
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Collect metrics snapshot
    pub async fn collect_snapshot(&self) -> Result<()> {
        debug!("Collecting metrics snapshot");
        
        let current_metrics = self.metrics.read().await.clone();
        let snapshot = TimestampedMetrics {
            timestamp: Utc::now(),
            metrics: current_metrics,
        };
        
        // Add to history
        {
            let mut history = self.history.write().await;
            history.push_back(snapshot);
            
            // Maintain size limit
            if history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }
        
        // Clean up old data if needed
        if self.config.retention_hours > 0 {
            self.cleanup_old_data().await?;
        }
        
        Ok(())
    }
    
    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(&self, window_seconds: u64) -> Result<InferenceMetrics> {
        let history = self.history.read().await;
        let cutoff_time = Utc::now() - chrono::Duration::seconds(window_seconds as i64);
        
        let relevant_metrics: Vec<&InferenceMetrics> = history.iter()
            .filter(|tm| tm.timestamp > cutoff_time)
            .map(|tm| &tm.metrics)
            .collect();
        
        if relevant_metrics.is_empty() {
            return Ok(InferenceMetrics::default());
        }
        
        // Aggregate metrics
        let count = relevant_metrics.len();
        let mut aggregated = InferenceMetrics::default();
        
        // Aggregate request metrics
        aggregated.requests.total_requests = relevant_metrics.iter()
            .map(|m| m.requests.total_requests)
            .sum();
        aggregated.requests.successful_requests = relevant_metrics.iter()
            .map(|m| m.requests.successful_requests)
            .sum();
        aggregated.requests.failed_requests = relevant_metrics.iter()
            .map(|m| m.requests.failed_requests)
            .sum();
        aggregated.requests.avg_request_size = relevant_metrics.iter()
            .map(|m| m.requests.avg_request_size)
            .sum::<f64>() / count as f64;
        aggregated.requests.avg_response_size = relevant_metrics.iter()
            .map(|m| m.requests.avg_response_size)
            .sum::<f64>() / count as f64;
        
        // Aggregate performance metrics
        aggregated.performance.avg_latency_ms = relevant_metrics.iter()
            .map(|m| m.performance.avg_latency_ms)
            .sum::<f64>() / count as f64;
        aggregated.performance.throughput_tokens_per_second = relevant_metrics.iter()
            .map(|m| m.performance.throughput_tokens_per_second)
            .sum::<f64>() / count as f64;
        
        // Aggregate resource metrics
        aggregated.resources.cpu_usage_percent = relevant_metrics.iter()
            .map(|m| m.resources.cpu_usage_percent)
            .sum::<f64>() / count as f64;
        aggregated.resources.memory_usage_percent = relevant_metrics.iter()
            .map(|m| m.resources.memory_usage_percent)
            .sum::<f64>() / count as f64;
        
        aggregated.last_updated = Utc::now();
        
        Ok(aggregated)
    }
    
    /// Check for alerts
    async fn check_alerts(&self) -> Result<()> {
        let metrics = self.metrics.read().await;
        let thresholds = self.alert_thresholds.read().await;
        
        let mut new_alerts = Vec::with_capacity(5);
        
        // Check error rate
        if metrics.errors.error_rate > thresholds.max_error_rate {
            new_alerts.push(MetricAlert {
                id: Uuid::new_v4(),
                alert_type: AlertType::ErrorRateHigh,
                metric_name: "error_rate".to_string(),
                current_value: metrics.errors.error_rate,
                threshold_value: thresholds.max_error_rate,
                message: format!("Error rate ({:.2}%) exceeds threshold ({:.2}%)", 
                                metrics.errors.error_rate * 100.0, thresholds.max_error_rate * 100.0),
                severity: if metrics.errors.error_rate > thresholds.max_error_rate * 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                created_at: Utc::now(),
                resolved_at: None,
            });
        }
        
        // Check latency
        if metrics.performance.avg_latency_ms > thresholds.max_latency_ms {
            new_alerts.push(MetricAlert {
                id: Uuid::new_v4(),
                alert_type: AlertType::LatencyHigh,
                metric_name: "avg_latency_ms".to_string(),
                current_value: metrics.performance.avg_latency_ms,
                threshold_value: thresholds.max_latency_ms,
                message: format!("Average latency ({:.1}ms) exceeds threshold ({:.1}ms)", 
                                metrics.performance.avg_latency_ms, thresholds.max_latency_ms),
                severity: if metrics.performance.avg_latency_ms > thresholds.max_latency_ms * 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                created_at: Utc::now(),
                resolved_at: None,
            });
        }
        
        // Check CPU usage
        if metrics.resources.cpu_usage_percent > thresholds.max_cpu_usage_percent {
            new_alerts.push(MetricAlert {
                id: Uuid::new_v4(),
                alert_type: AlertType::CpuHigh,
                metric_name: "cpu_usage_percent".to_string(),
                current_value: metrics.resources.cpu_usage_percent,
                threshold_value: thresholds.max_cpu_usage_percent,
                message: format!("CPU usage ({:.1}%) exceeds threshold ({:.1}%)", 
                                metrics.resources.cpu_usage_percent, thresholds.max_cpu_usage_percent),
                severity: AlertSeverity::Warning,
                created_at: Utc::now(),
                resolved_at: None,
            });
        }
        
        // Add new alerts to active alerts
        if !new_alerts.is_empty() {
            let mut active_alerts = self.active_alerts.write().await;
            for alert in new_alerts {
                // Check if similar alert already exists
                if !active_alerts.iter().any(|a| a.alert_type == alert.alert_type && a.resolved_at.is_none()) {
                    active_alerts.push(alert);
                }
            }
        }
        
        Ok(())
    }
    
    /// Clean up old data
    async fn cleanup_old_data(&self) -> Result<()> {
        let cutoff_time = Utc::now() - chrono::Duration::hours(self.config.retention_hours as i64);
        
        // Clean up history
        {
            let mut history = self.history.write().await;
            let initial_count = history.len();
            history.retain(|tm| tm.timestamp > cutoff_time);
            
            if history.len() < initial_count {
                debug!("Cleaned up {} old metric entries", initial_count - history.len());
            }
        }
        
        // Clean up resolved alerts
        {
            let mut alerts = self.active_alerts.write().await;
            let initial_count = alerts.len();
            alerts.retain(|a| a.resolved_at.is_none() || a.resolved_at.unwrap_or(Utc::now()) > cutoff_time);
            
            if alerts.len() < initial_count {
                debug!("Cleaned up {} old alerts", initial_count - alerts.len());
            }
        }
        
        Ok(())
    }
    
    /// Set alert thresholds
    pub async fn set_alert_thresholds(&self, thresholds: AlertThresholds) -> Result<()> {
        let mut alert_thresholds = self.alert_thresholds.write().await;
        *alert_thresholds = thresholds;
        Ok(())
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for metrics analysis
pub mod analysis {
    use super::*;
    
    /// Calculate growth rate
    pub fn calculate_growth_rate(current: f64, previous: f64) -> f64 {
        if previous == 0.0 {
            return 0.0;
        }
        ((current - previous) / previous) * 100.0
    }
    
    /// Detect anomalies in metrics
    pub fn detect_anomalies(
        metrics: &[InferenceMetrics],
        threshold: f64,
    ) -> Vec<usize> {
        if metrics.len() < 3 {
            return Vec::new();
        }
        
        let mut anomalies = Vec::with_capacity(metrics.len());
        
        for i in 1..metrics.len() - 1 {
            let prev = &metrics[i - 1];
            let curr = &metrics[i];
            let next = &metrics[i + 1];
            
            let avg_latency = (prev.performance.avg_latency_ms + curr.performance.avg_latency_ms + next.performance.avg_latency_ms) / 3.0;
            let std_dev = ((prev.performance.avg_latency_ms - avg_latency).powi(2) +
                           (curr.performance.avg_latency_ms - avg_latency).powi(2) +
                           (next.performance.avg_latency_ms - avg_latency).powi(2) / 3.0).sqrt();
            
            let z_score = (curr.performance.avg_latency_ms - avg_latency) / std_dev;
            
            if z_score.abs() > threshold {
                anomalies.push(i);
            }
        }
        
        anomalies
    }
    
    /// Generate metrics summary
    pub fn generate_summary(metrics: &InferenceMetrics) -> MetricsSummary {
        MetricsSummary {
            total_requests: metrics.requests.total_requests,
            success_rate: if metrics.requests.total_requests > 0 {
                metrics.requests.successful_requests as f64 / metrics.requests.total_requests as f64
            } else {
                0.0
            },
            avg_latency: metrics.performance.avg_latency_ms,
            throughput: metrics.performance.throughput_tokens_per_second,
            error_rate: metrics.errors.error_rate,
            cpu_usage: metrics.resources.cpu_usage_percent,
            memory_usage: metrics.resources.memory_usage_percent,
            active_alerts: 0, // Would be populated separately
        }
    }
    
    /// Metrics summary
    #[derive(Debug, Clone)]
    pub struct MetricsSummary {
        pub total_requests: u64,
        pub success_rate: f64,
        pub avg_latency: f64,
        pub throughput: f64,
        pub error_rate: f64,
        pub cpu_usage: f64,
        pub memory_usage: f64,
        pub active_alerts: usize,
    }
}
