//! Performance Metrics dan Monitoring System
//! 
//! Implementasi comprehensive monitoring dengan metrics collection, alerting, dan reporting

use crate::types::{ModelId, IntentType};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Mutex};
use parking_lot::RwLock as ParkingRwLock;
use tracing::{debug, info, warn};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Performance metrics collector
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    // Request metrics
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_request_time_ms: f64,
    pub p95_request_time_ms: f64,
    pub p99_request_time_ms: f64,
    
    // Model-specific metrics
    pub model_metrics: HashMap<ModelId, ModelMetrics>,
    
    // Intent-specific metrics
    pub intent_metrics: HashMap<IntentType, IntentMetrics>,
    
    // System metrics
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub active_connections: usize,
    pub queue_size: usize,
    
    // Timestamps
    pub last_updated: u64,
    pub start_time: u64,
}

/// Model-specific metrics
#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    pub model_id: ModelId,
    pub request_count: u64,
    pub success_count: u64,
    pub avg_response_time_ms: f64,
    pub error_rate: f64,
    pub last_request_time: u64,
}

/// Intent-specific metrics
#[derive(Debug, Clone, Default)]
pub struct IntentMetrics {
    pub intent_type: IntentType,
    pub detection_count: u64,
    pub correct_detections: u64,
    pub avg_confidence: f64,
    pub false_positive_rate: f64,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    pub enabled: bool,
    pub thresholds: AlertThresholds,
    pub cooldown_ms: u64,
    pub notification_channels: Vec<NotificationChannel>,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub error_rate_percent: f64,
    pub avg_response_time_ms: f64,
    pub queue_size: usize,
    pub memory_usage_percent: f64,
    pub cpu_usage_percent: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            error_rate_percent: 10.0,
            avg_response_time_ms: 5000.0,
            queue_size: 1000,
            memory_usage_percent: 80.0,
            cpu_usage_percent: 85.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum NotificationChannel {
    Log,
    Console,
    Email(String),
    Webhook(String),
}

/// Alert information
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: u64,
    pub resolved: bool,
    pub resolved_at: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlertType {
    HighErrorRate,
    SlowResponse,
    HighMemoryUsage,
    HighCpuUsage,
    QueueOverflow,
    ModelUnavailable,
    SystemError,
}

impl std::fmt::Display for AlertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertType::HighErrorRate => write!(f, "HighErrorRate"),
            AlertType::SlowResponse => write!(f, "SlowResponse"),
            AlertType::HighMemoryUsage => write!(f, "HighMemoryUsage"),
            AlertType::HighCpuUsage => write!(f, "HighCpuUsage"),
            AlertType::QueueOverflow => write!(f, "QueueOverflow"),
            AlertType::ModelUnavailable => write!(f, "ModelUnavailable"),
            AlertType::SystemError => write!(f, "SystemError"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info = 0,
    Warning = 1,
    Error = 2,
    Critical = 3,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "Info"),
            AlertSeverity::Warning => write!(f, "Warning"),
            AlertSeverity::Error => write!(f, "Error"),
            AlertSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// Performance monitoring system
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    metrics: Arc<ParkingRwLock<PerformanceMetrics>>,
    request_times: Arc<Mutex<VecDeque<f64>>>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
    alert_cooldowns: Arc<RwLock<HashMap<AlertType, u64>>>,
    start_time: Instant,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub metrics_retention_hours: u64,
    pub alert_history_size: usize,
    pub request_time_samples: usize,
    pub update_interval_ms: u64,
    pub alert_config: AlertConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_retention_hours: 24,
            alert_history_size: 1000,
            request_time_samples: 10000,
            update_interval_ms: 1000,
            alert_config: AlertConfig {
                enabled: true,
                thresholds: AlertThresholds::default(),
                cooldown_ms: 300000, // 5 minutes
                notification_channels: vec![NotificationChannel::Log, NotificationChannel::Console],
            },
        }
    }
}

impl PerformanceMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            metrics: Arc::new(ParkingRwLock::new(PerformanceMetrics {
                start_time: Self::current_timestamp(),
                ..Default::default()
            })),
            request_times: Arc::new(Mutex::new(VecDeque::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            config,
            start_time: Instant::now(),
        }
    }
    
    /// Record request completion
    pub async fn record_request(&self, success: bool, response_time_ms: f64, model_id: ModelId, intent_type: IntentType) {
        let mut metrics = self.metrics.write();
        
        // Update global metrics
        metrics.total_requests += 1;
        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }
        
        // Update request time samples
        {
            let mut request_times = self.request_times.lock().await;
            request_times.push_back(response_time_ms);
            
            // Keep only recent samples
            while request_times.len() > self.config.request_time_samples {
                request_times.pop_front();
            }
            
            // Calculate percentiles
            if !request_times.is_empty() {
                let mut sorted_times: Vec<f64> = request_times.iter().cloned().collect();
                sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                metrics.avg_request_time_ms = sorted_times.iter().sum::<f64>() / sorted_times.len() as f64;
                
                let p95_index = (sorted_times.len() as f64 * 0.95) as usize;
                let p99_index = (sorted_times.len() as f64 * 0.99) as usize;
                
                metrics.p95_request_time_ms = sorted_times[p95_index.min(sorted_times.len() - 1)];
                metrics.p99_request_time_ms = sorted_times[p99_index.min(sorted_times.len() - 1)];
            }
        }
        
        // Update model metrics
        let model_metrics = metrics.model_metrics.entry(model_id).or_insert_with(|| ModelMetrics {
            model_id,
            ..Default::default()
        });
        
        model_metrics.request_count += 1;
        if success {
            model_metrics.success_count += 1;
        }
        
        // Update moving average for response time
        let alpha = 0.1; // Smoothing factor
        if model_metrics.avg_response_time_ms == 0.0 {
            model_metrics.avg_response_time_ms = response_time_ms;
        } else {
            model_metrics.avg_response_time_ms = 
                alpha * response_time_ms + (1.0 - alpha) * model_metrics.avg_response_time_ms;
        }
        
        model_metrics.error_rate = if model_metrics.request_count > 0 {
            (model_metrics.request_count - model_metrics.success_count) as f64 / model_metrics.request_count as f64 * 100.0
        } else {
            0.0
        };
        
        model_metrics.last_request_time = Self::current_timestamp();
        
        // Update intent metrics
        let intent_metrics = metrics.intent_metrics.entry(intent_type).or_insert_with(|| IntentMetrics {
            intent_type,
            ..Default::default()
        });
        
        intent_metrics.detection_count += 1;
        
        metrics.last_updated = Self::current_timestamp();
        
        // Check for alerts
        if self.config.alert_config.enabled {
            self.check_alerts();
        }
    }
    
    /// Update system metrics
    pub fn update_system_metrics(&self, cpu_usage: f64, memory_usage_mb: f64, active_connections: usize, queue_size: usize) {
        let mut metrics = self.metrics.write();
        
        metrics.cpu_usage_percent = cpu_usage;
        metrics.memory_usage_mb = memory_usage_mb;
        metrics.active_connections = active_connections;
        metrics.queue_size = queue_size;
        metrics.last_updated = Self::current_timestamp();
        
        // Check for alerts
        if self.config.alert_config.enabled {
            self.check_alerts();
        }
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().clone()
    }
    
    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.alerts.read().await.clone()
    }
    
    /// Get total system memory in MB
    fn get_total_system_memory(&self) -> f64 {
        // Try to get total memory from /proc/meminfo on Linux
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
        
        // Fallback: return a reasonable default (4GB)
        4096.0
    }
    
    /// Get alert history
    pub async fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let history = self.alert_history.read().await;
        if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.iter().rev().cloned().collect()
        }
    }
    
    /// Resolve alert
    pub async fn resolve_alert(&self, alert_id: &str) -> bool {
        let mut alerts = self.alerts.write().await;
        
        if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.resolved = true;
            alert.resolved_at = Some(Self::current_timestamp());
            
            // Move to history
            let mut history = self.alert_history.write().await;
            history.push_back(alert.clone());
            
            // Keep history size limited
            while history.len() > self.config.alert_history_size {
                history.pop_front();
            }
            
            info!("Alert resolved: {}", alert_id);
            true
        } else {
            false
        }
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read();
        let uptime = self.start_time.elapsed().as_secs();
        
        let error_rate = if metrics.total_requests > 0 {
            (metrics.failed_requests as f64 / metrics.total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        let success_rate = if metrics.total_requests > 0 {
            (metrics.successful_requests as f64 / metrics.total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        let requests_per_second = if uptime > 0 {
            metrics.total_requests as f64 / uptime as f64
        } else {
            0.0
        };
        
        PerformanceReport {
            timestamp: Self::current_timestamp(),
            uptime_seconds: uptime,
            total_requests: metrics.total_requests,
            successful_requests: metrics.successful_requests,
            failed_requests: metrics.failed_requests,
            success_rate,
            error_rate,
            requests_per_second,
            avg_response_time_ms: metrics.avg_request_time_ms,
            p95_response_time_ms: metrics.p95_request_time_ms,
            p99_response_time_ms: metrics.p99_request_time_ms,
            cpu_usage_percent: metrics.cpu_usage_percent,
            memory_usage_mb: metrics.memory_usage_mb,
            active_connections: metrics.active_connections,
            queue_size: metrics.queue_size,
            model_count: metrics.model_metrics.len(),
            active_alerts: self.alerts.blocking_read().len(),
        }
    }
    
    /// Check for alert conditions
    fn check_alerts(&self) {
        let metrics = self.metrics.read();
        let current_time = Self::current_timestamp();
        let mut cooldowns = self.alert_cooldowns.blocking_write();
        
        // Check error rate
        let error_rate = if metrics.total_requests > 0 {
            (metrics.failed_requests as f64 / metrics.total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        if error_rate > self.config.alert_config.thresholds.error_rate_percent {
            if self.should_send_alert(&mut cooldowns, AlertType::HighErrorRate, current_time) {
                self.create_alert(
                    AlertType::HighErrorRate,
                    AlertSeverity::Error,
                    format!("High error rate detected: {:.2}%", error_rate)
                );
            }
        }
        
        // Check response time
        if metrics.avg_request_time_ms > self.config.alert_config.thresholds.avg_response_time_ms {
            if self.should_send_alert(&mut cooldowns, AlertType::SlowResponse, current_time) {
                self.create_alert(
                    AlertType::SlowResponse,
                    AlertSeverity::Warning,
                    format!("Slow response time detected: {:.2}ms", metrics.avg_request_time_ms)
                );
            }
        }
        
        // Check queue size
        if metrics.queue_size > self.config.alert_config.thresholds.queue_size {
            if self.should_send_alert(&mut cooldowns, AlertType::QueueOverflow, current_time) {
                self.create_alert(
                    AlertType::QueueOverflow,
                    AlertSeverity::Error,
                    format!("Queue overflow detected: {} items", metrics.queue_size)
                );
            }
        }
        
        // Check memory usage - get actual system memory
        let total_memory_mb = self.get_total_system_memory();
        let memory_usage_percent = if total_memory_mb > 0.0 {
            (metrics.memory_usage_mb / total_memory_mb) * 100.0
        } else {
            0.0
        };
        if memory_usage_percent > self.config.alert_config.thresholds.memory_usage_percent {
            if self.should_send_alert(&mut cooldowns, AlertType::HighMemoryUsage, current_time) {
                self.create_alert(
                    AlertType::HighMemoryUsage,
                    AlertSeverity::Warning,
                    format!("High memory usage detected: {:.1}%", memory_usage_percent)
                );
            }
        }
        
        // Check CPU usage
        if metrics.cpu_usage_percent > self.config.alert_config.thresholds.cpu_usage_percent {
            if self.should_send_alert(&mut cooldowns, AlertType::HighCpuUsage, current_time) {
                self.create_alert(
                    AlertType::HighCpuUsage,
                    AlertSeverity::Warning,
                    format!("High CPU usage detected: {:.1}%", metrics.cpu_usage_percent)
                );
            }
        }
    }
    
    fn should_send_alert(&self, cooldowns: &mut HashMap<AlertType, u64>, alert_type: AlertType, current_time: u64) -> bool {
        if let Some(last_alert_time) = cooldowns.get(&alert_type) {
            if current_time - last_alert_time < self.config.alert_config.cooldown_ms {
                return false;
            }
        }
        
        cooldowns.insert(alert_type, current_time);
        true
    }
    
    fn create_alert(&self, alert_type: AlertType, severity: AlertSeverity, message: String) {
        let alert = Alert {
            id: Uuid::new_v4().to_string(),
            alert_type,
            severity: severity.clone(),
            message,
            timestamp: Self::current_timestamp(),
            resolved: false,
            resolved_at: None,
        };
        
        // Add to active alerts
        {
            let mut alerts = self.alerts.blocking_write();
            alerts.push(alert.clone());
        }
        
        // Send notifications
        for channel in &self.config.alert_config.notification_channels {
            match channel {
                NotificationChannel::Log => {
                    warn!("ALERT [{}]: {}", severity, alert.message);
                }
                NotificationChannel::Console => {
                    eprintln!("🚨 ALERT [{}]: {}", severity, alert.message);
                }
                NotificationChannel::Email(email_config) => {
                    if let Err(e) = self.send_email_notification(email_config, &alert, &severity).await {
                        error!("Failed to send email notification: {}", e);
                    } else {
                        debug!("Email notification sent successfully for: {}", alert.message);
                    }
                }
                NotificationChannel::Webhook(webhook_url) => {
                    if let Err(e) = self.send_webhook_notification(webhook_url, &alert, &severity).await {
                        error!("Failed to send webhook notification: {}", e);
                    } else {
                        debug!("Webhook notification sent successfully for: {}", alert.message);
                    }
                }
            }
        }
        
        warn!("Alert created: {} - {}", alert.alert_type, alert.message);
    }
    
    /// Send email notification using SMTP
    async fn send_email_notification(&self, email_config: &str, alert: &Alert, severity: &str) -> Result<()> {
        use lettre::{Message, SmtpTransport, Transport, Tokio1Executor};
        use lettre::message::{header::ContentType, MultiPart, SinglePart};
        
        // Parse email configuration (format: "smtp://user:pass@host:port?from=from@example.com&to=to@example.com")
        let config = self.parse_email_config(email_config)?;
        
        // Create email message
        let email = Message::builder()
            .from(config.from.parse()?)
            .to(config.to.parse()?)
            .subject(&format!("Nexora AI Alert [{}]: {}", severity, alert.alert_type))
            .multipart(
                MultiPart::alternative()
                    .singlepart(
                        SinglePart::builder()
                            .header(ContentType::TEXT_PLAIN)
                            .body(self.format_email_text(alert, severity))
                    )
                    .singlepart(
                        SinglePart::builder()
                            .header(ContentType::TEXT_HTML)
                            .body(self.format_email_html(alert, severity))
                    )
            )?;
        
        // Create SMTP transport
        let transport = SmtpTransport::relay_dangerous(&config.host)?
            .port(config.port)
            .credentials(config.credentials)
            .build();
        
        // Send email
        match transport.send(&email).await {
            Ok(_) => {
                info!("Email alert sent successfully to {}", config.to);
                Ok(())
            }
            Err(e) => {
                error!("Failed to send email alert: {}", e);
                Err(anyhow::anyhow!("Email send failed: {}", e))
            }
        }
    }
    
    /// Parse email configuration string
    fn parse_email_config(&self, config: &str) -> Result<EmailConfig> {
        // Expected format: "smtp://user:pass@host:port?from=from@example.com&to=to@example.com"
        let url = url::Url::parse(config)
            .map_err(|e| anyhow::anyhow!("Invalid email config URL: {}", e))?;
        
        if url.scheme() != "smtp" {
            return Err(anyhow::anyhow!("Email config must use smtp:// scheme"));
        }
        
        let host = url.host_str()
            .ok_or_else(|| anyhow::anyhow!("Missing host in email config"))?
            .to_string();
        
        let port = url.port().unwrap_or(587);
        
        let username = url.username();
        let password = url.password()
            .ok_or_else(|| anyhow::anyhow!("Missing password in email config"))?;
        
        let from = url.query_pairs()
            .find(|(k, _)| k == "from")
            .map(|(_, v)| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("Missing 'from' parameter in email config"))?;
        
        let to = url.query_pairs()
            .find(|(k, _)| k == "to")
            .map(|(_, v)| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("Missing 'to' parameter in email config"))?;
        
        Ok(EmailConfig {
            host,
            port,
            username: username.to_string(),
            password: password.to_string(),
            from,
            to,
            credentials: lettre::transport::smtp::authentication::Credentials::new(
                username.to_string(),
                password.to_string()
            ),
        })
    }
    
    /// Format email as plain text
    fn format_email_text(&self, alert: &Alert, severity: &str) -> String {
        format!(
            "Nexora AI Alert Notification\n\n\
            Severity: {}\n\
            Alert Type: {}\n\
            Message: {}\n\n\
            Timestamp: {}\n\
            Alert ID: {}\n\n\
            This is an automated alert from the Nexora AI monitoring system.\n\
            Please check the system status and take appropriate action if necessary.\n\
            \n\
            --\n\
            Nexora AI Monitoring System",
            severity,
            alert.alert_type,
            alert.message,
            alert.timestamp,
            alert.alert_id
        )
    }
    
    /// Format email as HTML
    fn format_email_html(&self, alert: &Alert, severity: &str) -> String {
        let severity_color = match severity {
            "Critical" => "#dc3545",
            "Warning" => "#ffc107",
            "Info" => "#17a2b8",
            _ => "#6c757d",
        };
        
        format!(
            r#"
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Nexora AI Alert</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
                    .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ background-color: #007bff; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                    .content {{ padding: 20px; }}
                    .severity {{ color: {}; font-weight: bold; }}
                    .alert-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                    .footer {{ background-color: #f8f9fa; padding: 15px; border-radius: 0 0 8px 8px; font-size: 12px; color: #6c757d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚨 Nexora AI Alert</h1>
                    </div>
                    <div class="content">
                        <div class="alert-info">
                            <p><strong>Severity:</strong> <span class="severity">{}</span></p>
                            <p><strong>Alert Type:</strong> {}</p>
                            <p><strong>Message:</strong> {}</p>
                            <p><strong>Timestamp:</strong> {}</p>
                            <p><strong>Alert ID:</strong> {}</p>
                        </div>
                        <p>This is an automated alert from the Nexora AI monitoring system. Please check the system status and take appropriate action if necessary.</p>
                    </div>
                    <div class="footer">
                        <p>Nexora AI Monitoring System</p>
                    </div>
                </div>
            </body>
            </html>
            "#,
            severity_color,
            severity,
            alert.alert_type,
            alert.message,
            alert.timestamp,
            alert.alert_id
        )
    }
    
    /// Send webhook notification
    async fn send_webhook_notification(&self, webhook_url: &str, alert: &Alert, severity: &str) -> Result<()> {
        use reqwest::Client;
        use serde_json::json;
        
        // Create webhook payload
        let payload = json!({
            "alert": {
                "id": alert.alert_id,
                "type": alert.alert_type,
                "severity": severity,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metadata": alert.metadata
            },
            "source": "nexora-ai",
            "version": env!("CARGO_PKG_VERSION"),
            "sent_at": self.current_timestamp()
        });
        
        // Create HTTP client with timeout
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        
        // Send webhook request
        let response = client
            .post(webhook_url)
            .header("Content-Type", "application/json")
            .header("User-Agent", format!("Nexora-AI/{}", env!("CARGO_PKG_VERSION")))
            .json(&payload)
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    info!("Webhook alert sent successfully to {}", webhook_url);
                    Ok(())
                } else {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    error!("Webhook failed with status {}: {}", status, body);
                    Err(anyhow::anyhow!("Webhook request failed: {} - {}", status, body))
                }
            }
            Err(e) => {
                error!("Failed to send webhook request: {}", e);
                Err(anyhow::anyhow!("Webhook request error: {}", e))
            }
        }
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Email configuration
#[derive(Debug)]
struct EmailConfig {
    host: String,
    port: u16,
    username: String,
    password: String,
    from: String,
    to: String,
    credentials: lettre::transport::smtp::authentication::Credentials,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: u64,
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub requests_per_second: f64,
    pub avg_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub active_connections: usize,
    pub queue_size: usize,
    pub model_count: usize,
    pub active_alerts: usize,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new(MonitoringConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelId;
    
    #[tokio::test]
    async fn test_performance_monitoring() {
        let config = MonitoringConfig {
            alert_config: AlertConfig {
                enabled: true,
                thresholds: AlertThresholds {
                    error_rate_percent: 5.0,
                    avg_response_time_ms: 100.0,
                    queue_size: 10,
                    memory_usage_percent: 50.0,
                    cpu_usage_percent: 50.0,
                },
                cooldown_ms: 1000,
                notification_channels: vec![NotificationChannel::Log],
            },
            ..Default::default()
        };
        
        let monitor = PerformanceMonitor::new(config);
        
        // Record some requests
        monitor.record_request(true, 50.0, ModelId::Coding, IntentType::Coding);
        monitor.record_request(true, 75.0, ModelId::Logic, IntentType::Reasoning);
        monitor.record_request(false, 200.0, ModelId::Coding, IntentType::Coding);
        
        // Check metrics
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
        
        // Generate report
        let report = monitor.generate_report();
        assert!(report.total_requests > 0);
        assert!(report.success_rate > 0.0);
    }
    
    #[tokio::test]
    async fn test_alerts() {
        let config = MonitoringConfig {
            alert_config: AlertConfig {
                enabled: true,
                thresholds: AlertThresholds {
                    error_rate_percent: 1.0, // Very low threshold
                    ..Default::default()
                },
                cooldown_ms: 100,
                notification_channels: vec![NotificationChannel::Log],
            },
            ..Default::default()
        };
        
        let monitor = PerformanceMonitor::new(config);
        
        // Record failed requests to trigger alert
        for _ in 0..10 {
            monitor.record_request(false, 100.0, ModelId::Coding, IntentType::Coding);
        }
        
        // Check for alerts
        let alerts = monitor.get_active_alerts().await;
        assert!(!alerts.is_empty());
        
        // Resolve alert
        let resolved = monitor.resolve_alert(&alerts[0].id).await;
        assert!(resolved);
    }
    
    #[test]
    fn test_performance_report() {
        let monitor = PerformanceMonitor::new(MonitoringConfig::default());
        
        // Record some metrics
        monitor.record_request(true, 100.0, ModelId::Coding, IntentType::Coding);
        monitor.update_system_metrics(25.5, 512.0, 10, 5);
        
        let report = monitor.generate_report();
        assert_eq!(report.total_requests, 1);
        assert_eq!(report.success_rate, 100.0);
        assert_eq!(report.error_rate, 0.0);
        assert_eq!(report.cpu_usage_percent, 25.5);
        assert_eq!(report.memory_usage_mb, 512.0);
    }
}
