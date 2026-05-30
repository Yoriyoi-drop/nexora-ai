use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    default_rps: f64,
    default_burst: f64,
}

struct TokenBucket {
    tokens: f64,
    last_refill: u64,
    max_tokens: f64,
    refill_rate: f64,
}

impl RateLimiter {
    pub fn new(default_rps: f64, default_burst: f64) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            default_rps,
            default_burst,
        }
    }

    pub async fn check(&self, key: &str) -> RateLimitResult {
        self.check_with_limits(key, self.default_rps, self.default_burst).await
    }

    pub async fn check_with_limits(&self, key: &str, rps: f64, burst: f64) -> RateLimitResult {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut buckets = self.buckets.write().await;
        let bucket = buckets.entry(key.to_string()).or_insert(TokenBucket {
            tokens: burst,
            last_refill: now,
            max_tokens: burst,
            refill_rate: rps,
        });

        let elapsed = now.saturating_sub(bucket.last_refill);
        let refill = elapsed as f64 * bucket.refill_rate;
        bucket.tokens = (bucket.tokens + refill).min(bucket.max_tokens);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            RateLimitResult::Allowed {
                remaining: bucket.tokens as u64,
                reset_at: now + 1,
            }
        } else {
            RateLimitResult::Denied {
                retry_after: (1.0 / bucket.refill_rate).ceil() as u64,
            }
        }
    }

    pub async fn reset(&self, key: &str) {
        self.buckets.write().await.remove(key);
    }
}

#[derive(Debug, Clone)]
pub enum RateLimitResult {
    Allowed { remaining: u64, reset_at: u64 },
    Denied { retry_after: u64 },
}

impl RateLimitResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitResult::Allowed { .. })
    }
}
