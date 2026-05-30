use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub id: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub latency_ms: u64,
    pub api_key: Option<String>,
    pub user_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAggregate {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_errors: u64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: u64,
    pub requests_by_endpoint: HashMap<String, u64>,
    pub requests_by_status: HashMap<u16, u64>,
    pub requests_by_user: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyUsage {
    pub date: String,
    pub requests: u64,
    pub tokens: u64,
    pub errors: u64,
    pub unique_users: u64,
    pub unique_api_keys: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    pub id: String,
    pub event_type: String,
    pub user_id: Option<String>,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStats {
    pub total_requests: u64,
    pub total_api_keys: u64,
    pub total_users: u64,
    pub active_users_24h: u64,
    pub errors_24h: u64,
    pub avg_latency_ms: f64,
    pub requests_per_minute: f64,
    pub daily_usage: Vec<DailyUsage>,
    pub top_endpoints: Vec<(String, u64)>,
    pub recent_errors: Vec<RequestRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    pub timestamp: DateTime<Utc>,
    pub gpu_alive: bool,
    pub gpu_utilization_pct: f64,
    pub gpu_tokens: u64,
    pub cpu_tokens: u64,
    pub tokens_per_sec: f64,
    pub kv_cache_pressure: f64,
    pub kv_hit_ratio: f64,
    pub scheduler_depth: i64,
    pub batching_efficiency_pct: f64,
    pub pcie_read_bytes: u64,
    pub pcie_write_bytes: u64,
    pub kv_internal_frag: f64,
    pub kv_external_frag: f64,
    pub memory_fragmentation: f64,
    pub gpu_memory_bytes: u64,
    pub gpu_memory_pct: f64,
}

pub struct TelemetryStore {
    requests: Arc<RwLock<Vec<RequestRecord>>>,
    events: Arc<RwLock<Vec<AnalyticsEvent>>>,
    snapshots: Arc<RwLock<Vec<MetricSnapshot>>>,
    max_requests: usize,
    max_events: usize,
    max_snapshots: usize,
}

impl TelemetryStore {
    pub fn new(max_requests: usize, max_events: usize, max_snapshots: usize) -> Self {
        Self {
            requests: Arc::new(RwLock::new(Vec::with_capacity(max_requests))),
            events: Arc::new(RwLock::new(Vec::with_capacity(max_events))),
            snapshots: Arc::new(RwLock::new(Vec::with_capacity(max_snapshots))),
            max_requests,
            max_events,
            max_snapshots,
        }
    }

    pub async fn record_request(&self, record: RequestRecord) {
        let mut requests = self.requests.write().await;
        if requests.len() >= self.max_requests {
            requests.remove(0);
        }
        requests.push(record);
    }

    pub async fn record_event(&self, event: AnalyticsEvent) {
        let mut events = self.events.write().await;
        if events.len() >= self.max_events {
            events.remove(0);
        }
        events.push(event);
    }

    pub async fn get_requests(&self) -> Vec<RequestRecord> {
        self.requests.read().await.clone()
    }

    pub async fn get_events(&self) -> Vec<AnalyticsEvent> {
        self.events.read().await.clone()
    }

    pub async fn get_aggregate(&self) -> UsageAggregate {
        let requests = self.requests.read().await;
        if requests.is_empty() {
            return UsageAggregate::default();
        }

        let total = requests.len() as u64;
        let mut total_latency: u64 = 0;
        let mut total_errors: u64 = 0;
        let mut latencies: Vec<u64> = Vec::with_capacity(requests.len());
        let mut by_endpoint: HashMap<String, u64> = HashMap::new();
        let mut by_status: HashMap<u16, u64> = HashMap::new();
        let mut by_user: HashMap<String, u64> = HashMap::new();

        for req in requests.iter() {
            total_latency += req.latency_ms;
            latencies.push(req.latency_ms);
            *by_endpoint.entry(req.path.clone()).or_default() += 1;
            *by_status.entry(req.status).or_default() += 1;
            if req.status >= 400 {
                total_errors += 1;
            }
            if let Some(ref uid) = req.user_id {
                *by_user.entry(uid.clone()).or_default() += 1;
            }
        }

        latencies.sort_unstable();
        let p99_idx = ((latencies.len() as f64) * 0.99).ceil() as usize - 1;
        let p99_idx = p99_idx.min(latencies.len().saturating_sub(1));

        UsageAggregate {
            total_requests: total,
            total_tokens: 0,
            total_errors,
            avg_latency_ms: total_latency as f64 / total as f64,
            p99_latency_ms: latencies[p99_idx],
            requests_by_endpoint: by_endpoint,
            requests_by_status: by_status,
            requests_by_user: by_user,
        }
    }

    pub async fn get_dashboard_stats(&self) -> DashboardStats {
        let requests = self.requests.read().await;
        let _events = self.events.read().await;

        let total = requests.len() as u64;
        let mut total_latency: u64 = 0;
        let mut errors_24h: u64 = 0;
        let mut active_users: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut daily: HashMap<String, DailyUsage> = HashMap::new();
        let mut endpoints: HashMap<String, u64> = HashMap::new();
        let mut recent_errors: Vec<RequestRecord> = Vec::new();

        let now = Utc::now();
        let one_day_ago = now - chrono::Duration::hours(24);

        for req in requests.iter() {
            total_latency += req.latency_ms;
            *endpoints.entry(req.path.clone()).or_default() += 1;

            if req.timestamp > one_day_ago {
                if req.status >= 400 {
                    errors_24h += 1;
                    if recent_errors.len() < 10 {
                        recent_errors.push(req.clone());
                    }
                }
                if let Some(ref uid) = req.user_id {
                    active_users.insert(uid.clone());
                }
            }

            let date_key = req.timestamp.format("%Y-%m-%d").to_string();
            let entry = daily.entry(date_key.clone()).or_insert(DailyUsage {
                date: date_key,
                requests: 0,
                tokens: 0,
                errors: 0,
                unique_users: 0,
                unique_api_keys: 0,
            });
            entry.requests += 1;
            if req.status >= 400 {
                entry.errors += 1;
            }
        }

        let mut top_endpoints: Vec<(String, u64)> = endpoints.into_iter().collect();
        top_endpoints.sort_by(|a, b| b.1.cmp(&a.1));
        top_endpoints.truncate(10);

        let mut daily_usage: Vec<DailyUsage> = daily.into_values().collect();
        daily_usage.sort_by(|a, b| b.date.cmp(&a.date));
        daily_usage.truncate(30);

        let avg_latency = if total > 0 {
            total_latency as f64 / total as f64
        } else {
            0.0
        };

        DashboardStats {
            total_requests: total,
            total_api_keys: 0,
            total_users: active_users.len() as u64,
            active_users_24h: active_users.len() as u64,
            errors_24h,
            avg_latency_ms: avg_latency,
            requests_per_minute: 0.0,
            daily_usage,
            top_endpoints,
            recent_errors,
        }
    }

    pub async fn record_metric_snapshot(&self, snapshot: MetricSnapshot) {
        let mut snapshots = self.snapshots.write().await;
        if snapshots.len() >= self.max_snapshots {
            snapshots.remove(0);
        }
        snapshots.push(snapshot);
    }

    pub async fn get_metric_snapshots(&self) -> Vec<MetricSnapshot> {
        self.snapshots.read().await.clone()
    }

    pub async fn prune(&self) {
        let mut requests = self.requests.write().await;
        if requests.len() > self.max_requests {
            let excess = requests.len() - self.max_requests;
            requests.drain(0..excess);
        }
        drop(requests);

        let mut events = self.events.write().await;
        if events.len() > self.max_events {
            let excess = events.len() - self.max_events;
            events.drain(0..excess);
        }
        drop(events);

        let mut snapshots = self.snapshots.write().await;
        if snapshots.len() > self.max_snapshots {
            let excess = snapshots.len() - self.max_snapshots;
            snapshots.drain(0..excess);
        }
    }

    pub async fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        #[derive(Serialize)]
        struct Snapshot<'a> {
            requests: &'a [RequestRecord],
            events: &'a [AnalyticsEvent],
            snapshots: &'a [MetricSnapshot],
        }
        let data = Snapshot {
            requests: &*self.requests.read().await,
            events: &*self.events.read().await,
            snapshots: &*self.snapshots.read().await,
        };
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)
    }

    pub async fn load_from_file(path: &str) -> Self {
        if let Ok(json) = std::fs::read_to_string(path) {
            #[derive(Deserialize)]
            struct Snapshot {
                requests: Vec<RequestRecord>,
                events: Vec<AnalyticsEvent>,
                snapshots: Vec<MetricSnapshot>,
            }
            if let Ok(data) = serde_json::from_str::<Snapshot>(&json) {
                let store = Self::new(data.requests.len().max(10000), data.events.len().max(5000), data.snapshots.len().max(1000));
                {
                    let mut r = store.requests.write().await;
                    *r = data.requests;
                }
                {
                    let mut e = store.events.write().await;
                    *e = data.events;
                }
                {
                    let mut s = store.snapshots.write().await;
                    *s = data.snapshots;
                }
                return store;
            }
        }
        Self::new(10000, 5000, 1000)
    }
}

impl Default for UsageAggregate {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_tokens: 0,
            total_errors: 0,
            avg_latency_ms: 0.0,
            p99_latency_ms: 0,
            requests_by_endpoint: HashMap::new(),
            requests_by_status: HashMap::new(),
            requests_by_user: HashMap::new(),
        }
    }
}
