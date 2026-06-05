use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

const DEFAULT_MAX_BUCKETS: usize = 10_000;
const TTL_SECS: u64 = 3600;

#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    default_rps: f64,
    default_burst: f64,
    max_buckets: usize,
}

struct TokenBucket {
    tokens: f64,
    last_refill: u64,
    max_tokens: f64,
    refill_rate: f64,
    last_access: u64,
}

impl RateLimiter {
    pub fn new(default_rps: f64, default_burst: f64) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            default_rps,
            default_burst,
            max_buckets: DEFAULT_MAX_BUCKETS,
        }
    }

    pub fn with_max_buckets(mut self, max: usize) -> Self {
        self.max_buckets = max;
        self
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

        if let Some(bucket) = buckets.get_mut(key) {
            let elapsed = now.saturating_sub(bucket.last_refill);
            let refill = elapsed as f64 * bucket.refill_rate;
            bucket.tokens = (bucket.tokens + refill).min(bucket.max_tokens);
            bucket.last_refill = now;
            bucket.last_access = now;

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
        } else {
            self.evict_stale(&mut buckets, now);
            let bucket = TokenBucket {
                tokens: burst - 1.0,
                last_refill: now,
                max_tokens: burst,
                refill_rate: rps,
                last_access: now,
            };
            buckets.insert(key.to_string(), bucket);
            RateLimitResult::Allowed {
                remaining: burst as u64 - 1,
                reset_at: now + 1,
            }
        }
    }

    pub async fn reset(&self, key: &str) {
        self.buckets.write().await.remove(key);
    }

    fn evict_stale(&self, buckets: &mut HashMap<String, TokenBucket>, now: u64) {
        if buckets.len() < self.max_buckets {
            return;
        }
        let cutoff = now.saturating_sub(TTL_SECS);
        buckets.retain(|_, b| b.last_access >= cutoff);
        if buckets.len() >= self.max_buckets {
            if let Some(oldest) = buckets.iter().min_by_key(|(_, b)| b.last_access).map(|(k, _)| k.clone()) {
                buckets.remove(&oldest);
            }
        }
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
