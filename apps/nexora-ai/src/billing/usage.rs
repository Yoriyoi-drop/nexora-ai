use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::debug;

const MAX_RECORDS: usize = 10_000;
const RECORD_TTL_SECS: u64 = 90 * 86400;
const DAILY_TTL_SECS: u64 = 7 * 86400;

#[derive(Debug, Clone)]
pub struct UsageRecord {
    pub api_key: String,
    pub tier_name: String,
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub tokens_total: u64,
    pub request_count: u64,
    pub window_start: u64,
    pub last_access: u64,
}

impl UsageRecord {
    fn new(api_key: String, tier_name: String) -> Self {
        let now = now_secs();
        Self {
            api_key,
            tier_name,
            tokens_input: 0,
            tokens_output: 0,
            tokens_total: 0,
            request_count: 0,
            window_start: monthly_window(now),
            last_access: now,
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn monthly_window(t: u64) -> u64 {
    t / (86400 * 30)
}

fn daily_window(t: u64) -> u64 {
    t / 86400
}

#[derive(Debug, Clone)]
struct DailyCounter {
    window: u64,
    count: u32,
    last_access: u64,
}

#[derive(Debug)]
pub struct UsageTracker {
    records: Arc<RwLock<HashMap<String, UsageRecord>>>,
    daily_counts: Arc<RwLock<HashMap<String, DailyCounter>>>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
            daily_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn spawn_eviction_task(self: &Arc<Self>) {
        let this = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600));
            loop {
                interval.tick().await;
                this.evict_stale().await;
            }
        });
    }

    async fn evict_stale(&self) {
        let now = now_secs();
        let record_cutoff = now.saturating_sub(RECORD_TTL_SECS);
        let daily_cutoff = now.saturating_sub(DAILY_TTL_SECS);

        {
            let mut records = self.records.write().await;
            records.retain(|_, r| r.last_access >= record_cutoff);
        }
        {
            let mut daily = self.daily_counts.write().await;
            daily.retain(|_, d| d.last_access >= daily_cutoff);
        }
    }

    async fn evict_if_needed(&self) {
        let now = now_secs();
        let record_cutoff = now.saturating_sub(RECORD_TTL_SECS);
        let daily_cutoff = now.saturating_sub(DAILY_TTL_SECS);

        {
            let mut records = self.records.write().await;
            if records.len() >= MAX_RECORDS {
                records.retain(|_, r| r.last_access >= record_cutoff);
                if records.len() >= MAX_RECORDS {
                    if let Some(oldest_key) = records
                        .iter()
                        .min_by_key(|(_, r)| r.last_access)
                        .map(|(k, _)| k.clone())
                    {
                        records.remove(&oldest_key);
                    }
                }
            }
        }
        {
            let mut daily = self.daily_counts.write().await;
            if daily.len() >= MAX_RECORDS {
                daily.retain(|_, d| d.last_access >= daily_cutoff);
                if daily.len() >= MAX_RECORDS {
                    if let Some(oldest_key) = daily
                        .iter()
                        .min_by_key(|(_, d)| d.last_access)
                        .map(|(k, _)| k.clone())
                    {
                        daily.remove(&oldest_key);
                    }
                }
            }
        }
    }

    pub async fn track_usage(
        &self,
        api_key: &str,
        tier_name: &str,
        tokens_input: u32,
        tokens_output: u32,
    ) {
        let now = now_secs();
        let month_win = monthly_window(now);
        let day_win = daily_window(now);
        let tokens = (tokens_input + tokens_output) as u64;

        let is_new = {
            let records = self.records.read().await;
            !records.contains_key(api_key)
        };

        if is_new {
            self.evict_if_needed().await;
        }

        {
            let mut records = self.records.write().await;
            let record = records.entry(api_key.to_string()).or_insert_with(|| {
                UsageRecord::new(api_key.to_string(), tier_name.to_string())
            });

            if record.window_start != month_win {
                record.tokens_input = 0;
                record.tokens_output = 0;
                record.tokens_total = 0;
                record.request_count = 0;
                record.window_start = month_win;
            }
            record.last_access = now;
            record.tier_name = tier_name.to_string();
            record.tokens_input += tokens_input as u64;
            record.tokens_output += tokens_output as u64;
            record.tokens_total += tokens;
            record.request_count += 1;
        }

        {
            let mut daily = self.daily_counts.write().await;
            let dc = daily.entry(api_key.to_string()).or_insert(DailyCounter {
                window: day_win,
                count: 0,
                last_access: now,
            });
            dc.last_access = now;
            if dc.window != day_win {
                dc.window = day_win;
                dc.count = 0;
            }
            dc.count += 1;
        }

        debug!(
            "Usage tracked: key={}, tier={}, tokens_in={}, tokens_out={}",
            &api_key[..api_key.len().min(8)],
            tier_name,
            tokens_input,
            tokens_output
        );
    }

    pub async fn get_usage(&self, api_key: &str) -> Option<UsageRecord> {
        self.records.read().await.get(api_key).cloned()
    }

    pub async fn get_daily_count(&self, api_key: &str) -> u32 {
        let now = now_secs();
        let day_win = daily_window(now);
        let daily = self.daily_counts.read().await;
        daily.get(api_key).map(|d| {
            if d.window == day_win { d.count } else { 0 }
        }).unwrap_or(0)
    }

    pub async fn get_monthly_tokens(&self, api_key: &str) -> u64 {
        let now = now_secs();
        let month_win = monthly_window(now);
        let records = self.records.read().await;
        records.get(api_key).map(|r| {
            if r.window_start == month_win { r.tokens_total } else { 0 }
        }).unwrap_or(0)
    }

    pub async fn get_daily_remaining(&self, api_key: &str, daily_limit: u32) -> u32 {
        let used = self.get_daily_count(api_key).await;
        daily_limit.saturating_sub(used)
    }

    pub async fn get_monthly_remaining(&self, api_key: &str, monthly_quota: u64) -> u64 {
        let used = self.get_monthly_tokens(api_key).await;
        monthly_quota.saturating_sub(used)
    }

    pub async fn usage_summary(&self, api_key: &str) -> UsageSummary {
        let monthly_tokens = self.get_monthly_tokens(api_key).await;
        let daily_count = self.get_daily_count(api_key).await;
        UsageSummary {
            monthly_tokens_used: monthly_tokens,
            daily_requests: daily_count,
            timestamp: now_secs(),
        }
    }

    pub async fn get_all_usage_snapshots(&self) -> Vec<UsageRecord> {
        self.records.read().await.values().cloned().collect()
    }
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageSummary {
    pub monthly_tokens_used: u64,
    pub daily_requests: u32,
    pub timestamp: u64,
}
