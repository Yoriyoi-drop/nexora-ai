use crate::store::{AnalyticsEvent, MetricSnapshot, RequestRecord, TelemetryStore};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone)]
pub struct TelemetryRecorder {
    store: Arc<TelemetryStore>,
}

impl TelemetryRecorder {
    pub fn new(store: Arc<TelemetryStore>) -> Self {
        Self { store }
    }

    pub async fn record_request(
        &self,
        method: &str,
        path: &str,
        status: u16,
        latency_ms: u64,
        api_key: Option<String>,
        user_id: Option<String>,
    ) {
        let record = RequestRecord {
            id: Uuid::new_v4().to_string(),
            method: method.to_string(),
            path: path.to_string(),
            status,
            latency_ms,
            api_key,
            user_id,
            timestamp: Utc::now(),
        };
        self.store.record_request(record).await;
    }

    pub async fn record_event(
        &self,
        event_type: &str,
        user_id: Option<String>,
        metadata: HashMap<String, String>,
    ) {
        let event = AnalyticsEvent {
            id: Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            user_id,
            metadata,
            timestamp: Utc::now(),
        };
        self.store.record_event(event).await;
    }

    pub async fn record_login(&self, user_id: &str, success: bool) {
        let mut meta = HashMap::new();
        meta.insert("success".to_string(), success.to_string());
        self.record_event("auth.login", Some(user_id.to_string()), meta).await;
    }

    pub async fn record_api_usage(&self, api_key: &str, endpoint: &str, status: u16, latency_ms: u64) {
        let mut meta = HashMap::new();
        meta.insert("endpoint".to_string(), endpoint.to_string());
        meta.insert("status".to_string(), status.to_string());
        self.record_event("api.usage", None, meta).await;

        self.record_request("POST", endpoint, status, latency_ms, Some(api_key.to_string()), None).await;
    }

    pub async fn record_usage(&self, api_key: &str, tier: &str, tokens_input: u64, tokens_output: u64) {
        let mut meta = HashMap::new();
        meta.insert("tier".to_string(), tier.to_string());
        meta.insert("tokens_input".to_string(), tokens_input.to_string());
        meta.insert("tokens_output".to_string(), tokens_output.to_string());
        self.record_event("billing.usage", None, meta).await;

        let mut usage_meta = HashMap::new();
        usage_meta.insert("api_key".to_string(), api_key.to_string());
        usage_meta.insert("tokens".to_string(), (tokens_input + tokens_output).to_string());
        self.record_event("api.usage.tracked", None, usage_meta).await;
    }

    pub async fn record_error(&self, error_type: &str, detail: &str, user_id: Option<String>) {
        let mut meta = HashMap::new();
        meta.insert("error_type".to_string(), error_type.to_string());
        meta.insert("detail".to_string(), detail.to_string());
        self.record_event("error", user_id, meta).await;
    }

    pub async fn record_system_metrics(
        &self,
        gpu_alive: bool,
        gpu_utilization_pct: f64,
        gpu_tokens: u64,
        cpu_tokens: u64,
        tokens_per_sec: f64,
        kv_cache_pressure: f64,
        kv_hit_ratio: f64,
        scheduler_depth: i64,
        batching_efficiency_pct: f64,
        pcie_read_bytes: u64,
        pcie_write_bytes: u64,
        kv_internal_frag: f64,
        kv_external_frag: f64,
        memory_fragmentation: f64,
        gpu_memory_bytes: u64,
        gpu_memory_pct: f64,
    ) {
        let snapshot = MetricSnapshot {
            timestamp: Utc::now(),
            gpu_alive,
            gpu_utilization_pct,
            gpu_tokens,
            cpu_tokens,
            tokens_per_sec,
            kv_cache_pressure,
            kv_hit_ratio,
            scheduler_depth,
            batching_efficiency_pct,
            pcie_read_bytes,
            pcie_write_bytes,
            kv_internal_frag,
            kv_external_frag,
            memory_fragmentation,
            gpu_memory_bytes,
            gpu_memory_pct,
        };
        self.store.record_metric_snapshot(snapshot).await;
    }
}
