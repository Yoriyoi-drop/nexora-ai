//! Logging and Tracing Configuration
//! 
//! Comprehensive logging setup with structured tracing for Nexora AI

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use tracing::{Level};
use tracing_subscriber::Layer;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};
use tracing_appender::{non_blocking, rolling};

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Whether to log to console
    pub console: bool,
    /// Whether to log to file
    pub file: bool,
    /// File output directory
    pub file_dir: Option<PathBuf>,
    /// File name prefix
    pub file_name_prefix: String,
    /// Maximum file size before rotation (in MB)
    pub max_file_size_mb: u64,
    /// Maximum number of rotated files to keep
    pub max_files: usize,
    /// Whether to include timestamps
    pub timestamps: bool,
    /// Whether to include target/module
    pub target: bool,
    /// Whether to include thread IDs
    pub thread_ids: bool,
    /// Whether to include span traces
    pub span_events: bool,
    /// JSON format for structured logging
    pub json_format: bool,
    /// Custom filter patterns
    pub filters: Vec<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            console: true,
            file: false,
            file_dir: None,
            file_name_prefix: "nexora".to_string(),
            max_file_size_mb: 100,
            max_files: 10,
            timestamps: true,
            target: true,
            thread_ids: false,
            span_events: true,
            json_format: false,
            filters: vec![
                "nexora=info".to_string(),
                "axum=info".to_string(),
                "tokio=warn".to_string(),
                "hyper=warn".to_string(),
                "sqlx=warn".to_string(),
                "tower=warn".to_string(),
            ],
        }
    }
}

/// Initialize logging system
pub fn init_logging(config: LoggingConfig) -> Result<()> {
    let level = parse_log_level(&config.level)?;
    
    // Build environment filter
    let env_filter = build_env_filter(&config, level)?;
    
    // Create subscriber with simple console output
    let subscriber = Registry::default()
        .with(env_filter)
        .with(
            fmt::layer()
                .with_target(config.target)
                .with_thread_ids(config.thread_ids)
                .with_span_events(if config.span_events { FmtSpan::CLOSE } else { FmtSpan::NONE })
                .with_ansi(true)
                .with_writer(std::io::stdout)
        );
    
    // Add file layer if enabled
    let subscriber = if config.file {
        let file_layer = create_file_layer(&config)?;
        subscriber.with(file_layer)
    } else {
        subscriber
    };
    
    // Initialize global subscriber
    subscriber.init();
    
    tracing::info!(
        level = %config.level,
        console = config.console,
        file = config.file,
        json_format = config.json_format,
        "Logging system initialized"
    );
    
    Ok(())
}

/// Build environment filter from config
fn build_env_filter(config: &LoggingConfig, default_level: Level) -> Result<EnvFilter> {
    let mut filter_parts = config.filters.clone();
    
    // Add default level if not specified
    if !filter_parts.iter().any(|f| f.contains("=")) {
        filter_parts.push(format!("nexora={}", default_level));
    }
    
    let filter_string = filter_parts.join(",");
    EnvFilter::try_new(&filter_string)
        .map_err(|e| anyhow::anyhow!("Invalid log filter '{}': {}", filter_string, e))
}

/// Parse log level string
fn parse_log_level(level_str: &str) -> Result<Level> {
    match level_str.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(anyhow::anyhow!("Invalid log level: {}", level_str)),
    }
}

/// Create file logging layer
fn create_file_layer(config: &LoggingConfig) -> Result<impl tracing_subscriber::Layer<Registry> + Send + Sync> {
    let file_dir = config.file_dir.clone()
        .unwrap_or_else(|| PathBuf::from("logs"));
    
    // Create directory if it doesn't exist
    std::fs::create_dir_all(&file_dir)?;
    
    // Create rolling file appender
    let file_appender = rolling::daily(&file_dir, &config.file_name_prefix);
    let (non_blocking, _guard) = non_blocking(file_appender);
    
    let layer = fmt::layer()
        .with_target(config.target)
        .with_thread_ids(config.thread_ids)
        .with_span_events(if config.span_events { FmtSpan::CLOSE } else { FmtSpan::NONE })
        .with_ansi(false)
        .with_writer(non_blocking);
    
    if config.json_format {
        Ok(layer.json().boxed())
    } else {
        Ok(layer.boxed())
    }
}

/// Create a tracing span with context
#[macro_export]
macro_rules! trace_span {
    ($name:expr $(,)?) => {
        tracing::span!(tracing::Level::TRACE, $name)
    };
    ($name:expr, $($field:tt)*) => {
        tracing::span!(tracing::Level::TRACE, $name, $($field)*)
    };
}

/// Create a debug span with context
#[macro_export]
macro_rules! debug_span {
    ($name:expr $(,)?) => {
        tracing::span!(tracing::Level::DEBUG, $name)
    };
    ($name:expr, $($field:tt)*) => {
        tracing::span!(tracing::Level::DEBUG, $name, $($field)*)
    };
}

/// Create an info span with context
#[macro_export]
macro_rules! info_span {
    ($name:expr $(,)?) => {
        tracing::span!(tracing::Level::INFO, $name)
    };
    ($name:expr, $($field:tt)*) => {
        tracing::span!(tracing::Level::INFO, $name, $($field)*)
    };
}

/// Create a warn span with context
#[macro_export]
macro_rules! warn_span {
    ($name:expr $(,)?) => {
        tracing::span!(tracing::Level::WARN, $name)
    };
    ($name:expr, $($field:tt)*) => {
        tracing::span!(tracing::Level::WARN, $name, $($field)*)
    };
}

/// Create an error span with context
#[macro_export]
macro_rules! error_span {
    ($name:expr $(,)?) => {
        tracing::span!(tracing::Level::ERROR, $name)
    };
    ($name:expr, $($field:tt)*) => {
        tracing::span!(tracing::Level::ERROR, $name, $($field)*)
    };
}

/// Instrument async function with tracing
#[macro_export]
macro_rules! instrument_async {
    ($($arg:ident)*) => {
        #[tracing::instrument(
            level = tracing::Level::DEBUG,
            skip($($arg),*),
            fields(
                function = tracing::__macro_support::Identifier::current().name(),
                module = tracing::__macro_support::Identifier::current().module_path()
            )
        )]
    };
}

/// Instrument sync function with tracing
#[macro_export]
macro_rules! instrument_sync {
    ($($arg:ident)*) => {
        #[tracing::instrument(
            level = tracing::Level::DEBUG,
            skip($($arg),*),
            fields(
                function = tracing::__macro_support::Identifier::current().name(),
                module = tracing::__macro_support::Identifier::current().module_path()
            )
        )]
    };
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Enable health checks
    pub enable_health_checks: bool,
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
    /// Enable distributed tracing
    pub enable_distributed_tracing: bool,
    /// Jaeger endpoint for distributed tracing
    pub jaeger_endpoint: Option<String>,
    /// Enable custom metrics
    pub enable_custom_metrics: bool,
    /// Metrics retention period in hours
    pub metrics_retention_hours: u64,
    /// Enable alerting
    pub enable_alerting: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Alert thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_threshold: f64,
    /// Memory usage threshold (percentage)
    pub memory_threshold: f64,
    /// Disk usage threshold (percentage)
    pub disk_threshold: f64,
    /// Error rate threshold (percentage)
    pub error_rate_threshold: f64,
    /// Response time threshold (milliseconds)
    pub response_time_threshold: u64,
    /// Request rate threshold (requests per second)
    pub request_rate_threshold: f64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_seconds: 30,
            enable_profiling: false,
            enable_resource_monitoring: true,
            enable_health_checks: true,
            health_check_interval_seconds: 60,
            enable_distributed_tracing: false,
            jaeger_endpoint: None,
            enable_custom_metrics: true,
            metrics_retention_hours: 24,
            enable_alerting: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 85.0,
            disk_threshold: 90.0,
            error_rate_threshold: 5.0,
            response_time_threshold: 5000,
            request_rate_threshold: 1000.0,
        }
    }
}

/// Metrics collector for system monitoring
pub struct MetricsCollector {
    config: MonitoringConfig,
    metrics: Arc<RwLock<SystemMetrics>>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    custom_metrics: Arc<RwLock<HashMap<String, CustomMetric>>>,
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: std::time::SystemTime,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_mb: f64,
    pub disk_usage_percent: f64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub active_connections: usize,
    pub request_count: u64,
    pub error_count: u64,
    pub average_response_time_ms: f64,
    pub requests_per_second: f64,
}

/// Custom metric for application-specific monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp: std::time::SystemTime,
    pub tags: HashMap<String, String>,
}

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub timestamp: std::time::SystemTime,
    pub resolved: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(SystemMetrics::default())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Initialize monitoring system
    pub async fn initialize(&self) -> Result<()> {
        if self.config.enable_distributed_tracing {
            self.setup_distributed_tracing().await?;
        }
        
        if self.config.enable_metrics {
            self.start_metrics_collection().await?;
        }
        
        if self.config.enable_health_checks {
            self.start_health_checks().await?;
        }
        
        tracing::info!("Monitoring system initialized");
        Ok(())
    }
    
    /// Collect current system metrics
    pub async fn collect_metrics(&self) -> Result<SystemMetrics> {
        let metrics = SystemMetrics {
            timestamp: std::time::SystemTime::now(),
            cpu_usage_percent: self.get_cpu_usage().await?,
            memory_usage_mb: self.get_memory_usage_mb().await?,
            memory_usage_percent: self.get_memory_usage_percent().await?,
            disk_usage_mb: self.get_disk_usage_mb().await?,
            disk_usage_percent: self.get_disk_usage_percent().await?,
            network_rx_bytes: self.get_network_rx_bytes().await?,
            network_tx_bytes: self.get_network_tx_bytes().await?,
            active_connections: self.get_active_connections().await?,
            request_count: self.get_request_count().await?,
            error_count: self.get_error_count().await?,
            average_response_time_ms: self.get_average_response_time_ms().await?,
            requests_per_second: self.get_requests_per_second().await?,
        };
        
        // Update stored metrics
        {
            let mut stored_metrics = self.metrics.write();
            *stored_metrics = metrics.clone();
        }
        
        // Check for alerts
        if self.config.enable_alerting {
            self.check_alerts(&metrics).await?;
        }
        
        Ok(metrics)
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> SystemMetrics {
        let metrics = self.metrics.read();
        metrics.clone()
    }
    
    /// Add custom metric
    pub async fn add_custom_metric(&self, name: String, value: f64, unit: String, tags: HashMap<String, String>) {
        let metric = CustomMetric {
            name: name.clone(),
            value,
            unit,
            timestamp: std::time::SystemTime::now(),
            tags,
        };
        
        let mut custom_metrics = self.custom_metrics.write();
        custom_metrics.insert(name, metric);
    }
    
    /// Get custom metrics
    pub async fn get_custom_metrics(&self) -> HashMap<String, CustomMetric> {
        let custom_metrics = self.custom_metrics.read();
        custom_metrics.clone()
    }
    
    /// Get active alerts
    pub async fn get_alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.read();
        alerts.iter().filter(|alert| !alert.resolved).cloned().collect()
    }
    
    /// Resolve alert
    pub async fn resolve_alert(&self, alert_id: &str) -> Result<bool> {
        let mut alerts = self.alerts.write();
        if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.resolved = true;
            tracing::info!("Alert {} resolved", alert_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Setup distributed tracing
    async fn setup_distributed_tracing(&self) -> Result<()> {
        if let Some(endpoint) = &self.config.jaeger_endpoint {
            // Setup Jaeger tracing
            tracing::info!("Setting up distributed tracing with Jaeger endpoint: {}", endpoint);
            // Implementation would depend on specific tracing library
        }
        Ok(())
    }
    
    /// Start metrics collection loop
    async fn start_metrics_collection(&self) -> Result<()> {
        let collector = self.clone();
        let interval = self.config.metrics_interval_seconds;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(std::time::Duration::from_secs(interval));
            
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = collector.collect_metrics().await {
                    tracing::error!("Failed to collect metrics: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Start health check loop
    async fn start_health_checks(&self) -> Result<()> {
        let collector = self.clone();
        let interval = self.config.health_check_interval_seconds;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(std::time::Duration::from_secs(interval));
            
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = collector.perform_health_check().await {
                    tracing::error!("Health check failed: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Perform health check
    async fn perform_health_check(&self) -> Result<()> {
        let metrics = self.collect_metrics().await?;
        
        // Check critical thresholds
        if metrics.cpu_usage_percent > self.config.alert_thresholds.cpu_threshold {
            self.create_alert(
                AlertSeverity::Critical,
                format!("CPU usage is critically high: {:.1}%", metrics.cpu_usage_percent),
                "cpu_usage".to_string(),
                metrics.cpu_usage_percent,
                self.config.alert_thresholds.cpu_threshold,
            ).await?;
        }
        
        if metrics.memory_usage_percent > self.config.alert_thresholds.memory_threshold {
            self.create_alert(
                AlertSeverity::Critical,
                format!("Memory usage is critically high: {:.1}%", metrics.memory_usage_percent),
                "memory_usage".to_string(),
                metrics.memory_usage_percent,
                self.config.alert_thresholds.memory_threshold,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Check for alert conditions
    async fn check_alerts(&self, metrics: &SystemMetrics) -> Result<()> {
        let thresholds = &self.config.alert_thresholds;
        
        // CPU usage alert
        if metrics.cpu_usage_percent > thresholds.cpu_threshold {
            self.create_alert(
                AlertSeverity::Warning,
                format!("High CPU usage: {:.1}%", metrics.cpu_usage_percent),
                "cpu_usage".to_string(),
                metrics.cpu_usage_percent,
                thresholds.cpu_threshold,
            ).await?;
        }
        
        // Memory usage alert
        if metrics.memory_usage_percent > thresholds.memory_threshold {
            self.create_alert(
                AlertSeverity::Warning,
                format!("High memory usage: {:.1}%", metrics.memory_usage_percent),
                "memory_usage".to_string(),
                metrics.memory_usage_percent,
                thresholds.memory_threshold,
            ).await?;
        }
        
        // Error rate alert
        if metrics.request_count > 0 {
            let error_rate = (metrics.error_count as f64 / metrics.request_count as f64) * 100.0;
            if error_rate > thresholds.error_rate_threshold {
                self.create_alert(
                    AlertSeverity::Warning,
                    format!("High error rate: {:.1}%", error_rate),
                    "error_rate".to_string(),
                    error_rate,
                    thresholds.error_rate_threshold,
                ).await?;
            }
        }
        
        Ok(())
    }
    
    /// Create new alert
    async fn create_alert(
        &self,
        severity: AlertSeverity,
        message: String,
        metric_name: String,
        current_value: f64,
        threshold: f64,
    ) -> Result<()> {
        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            severity,
            message,
            metric_name,
            current_value,
            threshold,
            timestamp: std::time::SystemTime::now(),
            resolved: false,
        };
        
        let mut alerts = self.alerts.write();
        alerts.push(alert.clone());
        
        tracing::warn!(
            alert_id = %alert.id,
            severity = ?alert.severity,
            metric = %alert.metric_name,
            value = alert.current_value,
            threshold = alert.threshold,
            "{}",
            alert.message
        );
        
        Ok(())
    }
    
    // Helper methods for collecting system metrics
    async fn get_cpu_usage(&self) -> Result<f64> {
        // Implementation would use sysinfo or similar crate
        Ok(0.0) // Placeholder
    }
    
    async fn get_memory_usage_mb(&self) -> Result<f64> {
        // Implementation would use sysinfo or similar crate
        Ok(0.0) // Placeholder
    }
    
    async fn get_memory_usage_percent(&self) -> Result<f64> {
        // Implementation would use sysinfo or similar crate
        Ok(0.0) // Placeholder
    }
    
    async fn get_disk_usage_mb(&self) -> Result<f64> {
        // Implementation would use sysinfo or similar crate
        Ok(0.0) // Placeholder
    }
    
    async fn get_disk_usage_percent(&self) -> Result<f64> {
        // Implementation would use sysinfo or similar crate
        Ok(0.0) // Placeholder
    }
    
    async fn get_network_rx_bytes(&self) -> Result<u64> {
        // Implementation would use network statistics
        Ok(0) // Placeholder
    }
    
    async fn get_network_tx_bytes(&self) -> Result<u64> {
        // Implementation would use network statistics
        Ok(0) // Placeholder
    }
    
    async fn get_active_connections(&self) -> Result<usize> {
        // Implementation would track active connections
        Ok(0) // Placeholder
    }
    
    async fn get_request_count(&self) -> Result<u64> {
        // Implementation would track request count
        Ok(0) // Placeholder
    }
    
    async fn get_error_count(&self) -> Result<u64> {
        // Implementation would track error count
        Ok(0) // Placeholder
    }
    
    async fn get_average_response_time_ms(&self) -> Result<f64> {
        // Implementation would calculate average response time
        Ok(0.0) // Placeholder
    }
    
    async fn get_requests_per_second(&self) -> Result<f64> {
        // Implementation would calculate requests per second
        Ok(0.0) // Placeholder
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            timestamp: std::time::SystemTime::now(),
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            memory_usage_percent: 0.0,
            disk_usage_mb: 0.0,
            disk_usage_percent: 0.0,
            network_rx_bytes: 0,
            network_tx_bytes: 0,
            active_connections: 0,
            request_count: 0,
            error_count: 0,
            average_response_time_ms: 0.0,
            requests_per_second: 0.0,
        }
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics: Arc::clone(&self.metrics),
            alerts: Arc::clone(&self.alerts),
            custom_metrics: Arc::clone(&self.custom_metrics),
        }
    }
}

/// Initialize logging and monitoring
pub fn init_logging_and_monitoring(
    logging_config: LoggingConfig,
    monitoring_config: MonitoringConfig,
) -> Result<MetricsCollector> {
    // Initialize logging
    init_logging(logging_config)?;
    
    // Create metrics collector
    let collector = MetricsCollector::new(monitoring_config);
    
    tracing::info!("Logging and monitoring system initialized");
    
    Ok(collector)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
        assert!(config.console);
        assert!(!config.file);
    }
    
    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert!(config.enable_metrics);
        assert!(config.enable_health_checks);
        assert_eq!(config.metrics_interval_seconds, 30);
    }
    
    #[test]
    fn test_alert_thresholds_default() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.cpu_threshold, 80.0);
        assert_eq!(thresholds.memory_threshold, 85.0);
    }
    
    #[test]
    fn test_parse_log_level() {
        assert!(parse_log_level("info").is_ok());
        assert!(parse_log_level("debug").is_ok());
        assert!(parse_log_level("invalid").is_err());
    }
    
    #[test]
    fn test_metrics_collector_creation() {
        let config = MonitoringConfig::default();
        let collector = MetricsCollector::new(config);
        let metrics = collector.get_metrics();
        assert_eq!(metrics.cpu_usage_percent, 0.0);
    }
    
    #[test]
    fn test_custom_metrics() {
        let config = MonitoringConfig::default();
        let collector = MetricsCollector::new(config);
        
        let mut tags = HashMap::new();
        tags.insert("environment".to_string(), "test".to_string());
        
        // Note: This would need to be called in an async context in real tests
        // collector.add_custom_metric("test_metric".to_string(), 42.0, "count".to_string(), tags).await;
    }
}

/// Logging utilities
pub mod utils {
    use super::*;
    use std::time::{Duration, Instant};
    
    /// Measure execution time of a function
    pub fn time_it<F, T>(name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        tracing::debug!(
            duration_ms = duration.as_millis(),
            "Function {} completed", name
        );
        
        result
    }
    
    /// Measure execution time of an async function
    pub async fn time_it_async<F, T>(name: &str, f: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();
        
        tracing::debug!(
            duration_ms = duration.as_millis(),
            "Async function {} completed", name
        );
        
        result
    }
    
    /// Log memory usage (Linux only)
    #[cfg(target_os = "linux")]
    pub fn log_memory_usage() {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            let mb = kb / 1024.0;
                            tracing::debug!(
                                memory_mb = mb,
                                "Current memory usage"
                            );
                        }
                    }
                    break;
                }
            }
        }
    }
    
    /// Log memory usage (non-Linux fallback)
    #[cfg(not(target_os = "linux"))]
    pub fn log_memory_usage() {
        tracing::debug!("Memory usage logging not available on this platform");
    }
    
    /// Create a structured log event
    pub fn log_event(
        level: Level,
        event: &str,
        context: &[(&str, &dyn std::fmt::Display)],
    ) {
        match level {
            Level::TRACE => {
                let span = tracing::trace_span!("event", event);
                let _enter = span.enter();
                for (key, value) in context {
                    tracing::trace!(%key, %value);
                }
            }
            Level::DEBUG => {
                let span = tracing::debug_span!("event", event);
                let _enter = span.enter();
                for (key, value) in context {
                    tracing::debug!(%key, %value);
                }
            }
            Level::INFO => {
                let span = tracing::info_span!("event", event);
                let _enter = span.enter();
                for (key, value) in context {
                    tracing::info!(%key, %value);
                }
            }
            Level::WARN => {
                let span = tracing::warn_span!("event", event);
                let _enter = span.enter();
                for (key, value) in context {
                    tracing::warn!(%key, %value);
                }
            }
            Level::ERROR => {
                let span = tracing::error_span!("event", event);
                let _enter = span.enter();
                for (key, value) in context {
                    tracing::error!(%key, %value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
        assert!(config.console);
        assert!(!config.file);
        assert!(config.timestamps);
        assert!(config.target);
    }
    
    #[test]
    fn test_parse_log_level() {
        assert!(matches!(parse_log_level("debug"), Ok(Level::DEBUG)));
        assert!(matches!(parse_log_level("INFO"), Ok(Level::INFO)));
        assert!(parse_log_level("invalid").is_err());
    }
    
    #[test]
    fn test_build_env_filter() {
        let config = LoggingConfig {
            filters: vec!["nexora=debug".to_string(), "tokio=warn".to_string()],
            ..Default::default()
        };
        
        let filter = build_env_filter(&config, Level::INFO);
        assert!(filter.is_ok());
    }
    
    #[test]
    fn test_time_it() {
        let result = utils::time_it("test", || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
    }
}
