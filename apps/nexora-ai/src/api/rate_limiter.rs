//! Rate limiting for API requests

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use super::config::RateLimitConfig;

/// Rate limiter for API requests
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    clients: Arc<Mutex<HashMap<String, ClientInfo>>>,
}

#[derive(Debug)]
struct ClientInfo {
    requests: Vec<Instant>,
    last_cleanup: Instant,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Check if request is allowed
    pub async fn is_allowed(&self, client_id: &str) -> Result<bool> {
        if !self.config.enabled {
            return Ok(true);
        }
        
        let mut clients = self.clients.lock().await;
        let now = Instant::now();
        
        // Get or create client info
        let client_info = clients.entry(client_id.to_string()).or_insert_with(|| ClientInfo {
            requests: Vec::new(),
            last_cleanup: now,
        });
        
        // Cleanup old requests
        self.cleanup_old_requests(client_info, now);
        
        // Check rate limit
        let window_start = now - Duration::from_secs(60);
        let recent_requests = client_info.requests.iter()
            .filter(|&req_time| *req_time > window_start)
            .count();
        
        if recent_requests >= self.config.requests_per_minute {
            debug!("Rate limit exceeded for client {}: {}/{} requests", 
                   client_id, recent_requests, self.config.requests_per_minute);
            return Ok(false);
        }
        
        // Add current request
        client_info.requests.push(now);
        
        // Check burst limit
        if client_info.requests.len() > self.config.burst_size {
            warn!("Burst limit exceeded for client {}: {} requests", 
                  client_id, client_info.requests.len());
            // Remove oldest request to stay within burst limit
            client_info.requests.remove(0);
        }
        
        Ok(true)
    }
    
    /// Cleanup old requests
    fn cleanup_old_requests(&self, client_info: &mut ClientInfo, now: Instant) {
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);
        
        if now.duration_since(client_info.last_cleanup) >= cleanup_interval {
            let window_start = now - Duration::from_secs(60);
            client_info.requests.retain(|&req_time| req_time > window_start);
            client_info.last_cleanup = now;
        }
    }
    
    /// Get current rate limit status
    pub async fn get_status(&self, client_id: &str) -> Result<RateLimitStatus> {
        let clients = self.clients.lock().await;
        let now = Instant::now();
        
        if let Some(client_info) = clients.get(client_id) {
            let window_start = now - Duration::from_secs(60);
            let recent_requests = client_info.requests.iter()
                .filter(|&req_time| *req_time > window_start)
                .count();
            
            Ok(RateLimitStatus {
                allowed: recent_requests < self.config.requests_per_minute,
                current_requests: recent_requests,
                max_requests: self.config.requests_per_minute,
                reset_time: now + Duration::from_secs(60),
            })
        } else {
            Ok(RateLimitStatus {
                allowed: true,
                current_requests: 0,
                max_requests: self.config.requests_per_minute,
                reset_time: now + Duration::from_secs(60),
            })
        }
    }
    
    /// Reset rate limit for a client
    pub async fn reset_client(&self, client_id: &str) -> Result<()> {
        let mut clients = self.clients.lock().await;
        clients.remove(client_id);
        Ok(())
    }
    
    /// Get statistics
    pub async fn get_stats(&self) -> Result<RateLimitStats> {
        let clients = self.clients.lock().await;
        let now = Instant::now();
        
        let total_clients = clients.len();
        let mut total_requests = 0;
        let mut active_clients = 0;
        
        for client_info in clients.values() {
            let window_start = now - Duration::from_secs(60);
            let recent_requests = client_info.requests.iter()
                .filter(|&req_time| *req_time > window_start)
                .count();
            
            total_requests += recent_requests;
            if recent_requests > 0 {
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
}

/// Rate limit status
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub allowed: bool,
    pub current_requests: usize,
    pub max_requests: usize,
    pub reset_time: Instant,
}

/// Rate limit statistics
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub total_clients: usize,
    pub active_clients: usize,
    pub total_requests: usize,
    pub max_requests_per_client: usize,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}
