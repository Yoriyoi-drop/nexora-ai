//! Rate limiting for API requests — token bucket per client, TTL eviction.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::config::api::RateLimitConfig;

const MAX_CLIENTS: usize = 10_000;
const CLIENT_TTL: Duration = Duration::from_secs(3600);

#[derive(Debug)]
struct ClientBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
    last_access: Instant,
}

/// Rate limiter for API requests — token bucket per client.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    clients: Arc<RwLock<HashMap<String, ClientBucket>>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            clients: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn is_allowed(&self, client_id: &str) -> Result<RateLimitStatus> {
        if !self.config.enabled {
            return Ok(RateLimitStatus::Allowed);
        }
        let burst = self.config.burst_size.max(1) as f64;
        let rps = (self.config.requests_per_minute as f64) / 60.0;
        let now = Instant::now();

        let mut clients = self.clients.write().await;

        if let Some(bucket) = clients.get_mut(client_id) {
            let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
            let refill = elapsed * bucket.refill_rate;
            bucket.tokens = (bucket.tokens + refill).min(bucket.max_tokens);
            bucket.last_refill = now;
            bucket.last_access = now;

            if bucket.tokens >= 1.0 {
                bucket.tokens -= 1.0;
                Ok(RateLimitStatus::Allowed)
            } else {
                debug!("Rate limit exceeded for client {}", client_id);
                Ok(RateLimitStatus::RateLimited {
                    retry_after: Duration::from_secs_f64(1.0 / bucket.refill_rate),
                })
            }
        } else {
            self.evict_stale(&mut clients, now);
            clients.insert(client_id.to_string(), ClientBucket {
                tokens: burst - 1.0,
                max_tokens: burst,
                refill_rate: rps,
                last_refill: now,
                last_access: now,
            });
            Ok(RateLimitStatus::Allowed)
        }
    }

    pub async fn get_status(&self, client_id: &str) -> Result<RateLimitDetails> {
        let clients = self.clients.read().await;
        if let Some(bucket) = clients.get(client_id) {
            Ok(RateLimitDetails {
                allowed: bucket.tokens >= 1.0,
                current_requests: (bucket.max_tokens - bucket.tokens) as usize,
                max_requests: bucket.max_tokens as usize,
                reset_time: Instant::now() + Duration::from_secs_f64(1.0 / bucket.refill_rate),
            })
        } else {
            Ok(RateLimitDetails {
                allowed: true,
                current_requests: 0,
                max_requests: self.config.requests_per_minute,
                reset_time: Instant::now() + Duration::from_secs(60),
            })
        }
    }

    pub async fn reset_client(&self, client_id: &str) -> Result<()> {
        self.clients.write().await.remove(client_id);
        Ok(())
    }

    pub async fn get_stats(&self) -> Result<RateLimitStats> {
        let clients = self.clients.read().await;
        let total_clients = clients.len();
        let mut active_clients = 0;
        let mut total_requests = 0;
        for bucket in clients.values() {
            let used = (bucket.max_tokens - bucket.tokens) as usize;
            total_requests += used;
            if used > 0 {
                active_clients += 1;
            }
        }
        Ok(RateLimitStats {
            total_clients,
            active_clients,
            total_requests,
            max_requests_per_client: self.config.requests_per_minute,
        })
    }

    fn evict_stale(&self, clients: &mut HashMap<String, ClientBucket>, now: Instant) {
        if clients.len() < MAX_CLIENTS {
            return;
        }
        let cutoff = now - CLIENT_TTL;
        clients.retain(|_, b| b.last_access >= cutoff);
        if clients.len() >= MAX_CLIENTS {
            if let Some(oldest) = clients.iter().min_by_key(|(_, b)| b.last_access).map(|(k, _)| k.clone()) {
                clients.remove(&oldest);
            }
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitStatus {
    Allowed,
    RateLimited { retry_after: Duration },
}

#[derive(Debug, Clone)]
pub struct RateLimitDetails {
    pub allowed: bool,
    pub current_requests: usize,
    pub max_requests: usize,
    pub reset_time: Instant,
}

#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub total_clients: usize,
    pub active_clients: usize,
    pub total_requests: usize,
    pub max_requests_per_client: usize,
}
