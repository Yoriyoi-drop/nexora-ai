use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::debug;

#[derive(Debug, Clone)]
pub struct UsageRecord {
    pub api_key: String,
    pub tier_name: String,
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub tokens_total: u64,
    pub request_count: u64,
    pub window_start: u64,
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
            });
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
