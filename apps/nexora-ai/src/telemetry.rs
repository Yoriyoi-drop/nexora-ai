use chrono::Utc;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize)]
pub struct DashboardStats {
    pub total_requests: u64,
    pub active_models: usize,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestRecord {
    pub id: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub duration_ms: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyticsEvent {
    pub event_type: String,
    pub source: String,
    pub data: Value,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricSnapshot {
    pub name: String,
    pub value: f64,
    pub labels: Value,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct TelemetryStore {
    requests: Arc<RwLock<Vec<RequestRecord>>>,
    events: Arc<RwLock<Vec<AnalyticsEvent>>>,
    snapshots: Arc<RwLock<Vec<MetricSnapshot>>>,
}

impl TelemetryStore {
    pub fn new() -> Self {
        Self {
            requests: Arc::new(RwLock::new(Vec::new())),
            events: Arc::new(RwLock::new(Vec::new())),
            snapshots: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn get_dashboard_stats(&self) -> DashboardStats {
        DashboardStats {
            total_requests: self.requests.read().await.len() as u64,
            active_models: 0,
            uptime_seconds: 0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
        }
    }

    pub async fn get_aggregate(&self) -> Value {
        serde_json::json!({
            "total_requests": self.requests.read().await.len(),
            "total_events": self.events.read().await.len(),
            "total_snapshots": self.snapshots.read().await.len(),
        })
    }

    pub async fn get_events(&self) -> Vec<AnalyticsEvent> {
        self.events.read().await.clone()
    }

    pub async fn get_requests(&self) -> Vec<RequestRecord> {
        self.requests.read().await.clone()
    }

    pub async fn get_metric_snapshots(&self) -> Vec<MetricSnapshot> {
        self.snapshots.read().await.clone()
    }
}

impl Default for TelemetryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct TelemetrySystem {
    pub store: TelemetryStore,
    pub recorder: TelemetryRecorder,
}

impl TelemetrySystem {
    pub fn new() -> Self {
        Self {
            store: TelemetryStore::new(),
            recorder: TelemetryRecorder,
        }
    }
}

#[derive(Clone)]
pub struct TelemetryRecorder;

impl TelemetryRecorder {
    pub async fn record_event(&self, event_type: &str, source: Option<String>, data: HashMap<String, String>) {
        tracing::debug!("Telemetry event: {} source={:?} data={:?}", event_type, source, data);
    }

    pub async fn record_login(&self, _email: &str, _success: bool) {
        tracing::debug!("Telemetry login: {}", _email);
    }
}

impl Default for TelemetrySystem {
    fn default() -> Self {
        Self::new()
    }
}
