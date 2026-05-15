//! Nexora AI API Layer - Rust implementation
//! 
//! High-performance HTTP API server replacing runtime_http_server.c

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

pub mod server;
pub mod handlers;
pub mod middleware;
pub mod routing;
pub mod metrics;
pub mod security;

pub use server::ApiServer;
pub use handlers::*;
pub use middleware::*;
pub use routing::*;
pub use metrics::*;
pub use security::*;

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout_seconds: u64,
    pub enable_tls: bool,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    pub enable_cors: bool,
    pub cors_origins: Vec<String>,
    pub enable_metrics: bool,
    pub enable_logging: bool,
}

/// API request context
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub timestamp: u64,
    pub client_ip: String,
    pub user_agent: Option<String>,
    pub method: String,
    pub path: String,
    pub query_params: HashMap<String, String>,
    pub headers: HashMap<String, String>,
}

/// API response
#[derive(Debug, Clone, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ApiError>,
    pub metadata: ResponseMetadata,
}

/// API error
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub details: Option<String>,
}

/// Response metadata
#[derive(Debug, Clone, Serialize)]
pub struct ResponseMetadata {
    pub request_id: String,
    pub timestamp: u64,
    pub processing_time_ms: u64,
    pub version: String,
}

/// API statistics
#[derive(Debug, Clone, Default)]
pub struct ApiStatistics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub requests_per_second: f64,
    pub active_connections: usize,
    pub route_counts: HashMap<String, u64>,
    pub error_counts: HashMap<String, u64>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 1000,
            request_timeout_seconds: 30,
            enable_tls: false,
            cert_path: None,
            key_path: None,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            enable_metrics: true,
            enable_logging: true,
        }
    }
}

impl<T> ApiResponse<T> {
    /// Create successful response
    pub fn success(data: T, request_id: String) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            metadata: ResponseMetadata {
                request_id,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                    .as_secs(),
                processing_time_ms: 0,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }
    
    /// Create error response
    pub fn error(code: String, message: String, request_id: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(ApiError {
                code,
                message,
                details: None,
            }),
            metadata: ResponseMetadata {
                request_id,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                    .as_secs(),
                processing_time_ms: 0,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }
    
    /// Set processing time
    pub fn with_processing_time(mut self, time_ms: u64) -> Self {
        self.metadata.processing_time_ms = time_ms;
        self
    }
}

/// Common API data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub timestamp: u64,
    pub version: String,
    pub uptime_seconds: u64,
    pub active_connections: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInfo {
    pub path: String,
    pub method: String,
    pub handler: String,
    pub middleware: Vec<String>,
    pub rate_limit: Option<u32>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsData {
    pub timestamp: u64,
    pub requests_total: u64,
    pub requests_per_second: f64,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub active_connections: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub top_routes: Vec<RouteMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMetrics {
    pub path: String,
    pub method: String,
    pub requests: u64,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
}

/// API trait for handlers
#[async_trait::async_trait]
pub trait ApiHandler: Send + Sync {
    /// Handle API request
    async fn handle(&self, ctx: RequestContext, body: Vec<u8>) -> Result<Vec<u8>>;
    
    /// Get handler name
    fn name(&self) -> &str;
    
    /// Get handler version
    fn version(&self) -> &str {
        "1.0"
    }
    
    /// Get current process memory usage in MB
    fn get_process_memory_usage(&self) -> f64 {
        get_process_memory_mb()
    }
}

/// Middleware trait
#[async_trait::async_trait]
pub trait Middleware: Send + Sync {
    /// Process request
    async fn process_request(&self, ctx: &mut RequestContext, body: &mut Vec<u8>) -> Result<()>;
    
    /// Process response
    async fn process_response(&self, ctx: &mut RequestContext, response: &mut Vec<u8>) -> Result<()>;
    
    /// Get middleware name
    fn name(&self) -> &str;
}

/// Rate limiter
#[derive(Debug, Clone)]
pub struct RateLimiter {
    limits: Arc<RwLock<HashMap<String, RateLimit>>>,
    requests: Arc<RwLock<HashMap<String, Vec<u64>>>>,
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub max_requests: u32,
    pub window_seconds: u64,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limits: Arc::new(RwLock::new(HashMap::new())),
            requests: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn add_limit(&self, key: String, limit: RateLimit) {
        let mut limits = self.limits.write().await;
        limits.insert(key, limit);
    }
    
    pub async fn check_rate_limit(&self, key: &str) -> Result<bool> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();
        
        let limits = self.limits.read().await;
        if let Some(limit) = limits.get(key) {
            let mut requests = self.requests.write().await;
            let request_times = requests.entry(key.to_string()).or_insert_with(Vec::new);
            
            // Remove old requests outside the window
            request_times.retain(|&time| time > now - limit.window_seconds);
            
            // Check if under limit
            if request_times.len() < limit.max_requests as usize {
                request_times.push(now);
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(true) // No limit configured
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Read actual RSS memory from /proc/self/statm (Linux) or return a reasonable default
fn get_process_memory_mb() -> f64 {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(page_count_str) = contents.split_whitespace().nth(1) {
            if let Ok(pages) = page_count_str.parse::<u64>() {
                let page_size = 4096u64;
                return (pages * page_size) as f64 / (1024.0 * 1024.0);
            }
        }
    }
    50.0
}
