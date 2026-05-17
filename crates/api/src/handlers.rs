//! API Handlers - Rust implementation
//! 
//! Request handlers for various API endpoints

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::Serialize;
use serde_json::json;

use crate::{ApiHandler, RequestContext, ApiResponse, RouteInfo};

/// Handler registry for managing API handlers
pub struct HandlerRegistry {
    handlers: HashMap<String, Arc<dyn ApiHandler>>,
    routes: HashMap<String, RouteInfo>,
}

impl HandlerRegistry {
    /// Create new handler registry
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            routes: HashMap::new(),
        }
    }
    
    /// Register a handler
    pub fn register_handler(&mut self, path: String, method: String, handler: Arc<dyn ApiHandler>) {
        let route_key = format!("{} {}", method, path);
        self.handlers.insert(route_key.clone(), handler.clone());
        
        let route_info = RouteInfo {
            path: path.clone(),
            method,
            handler: handler.name().to_string(),
            middleware: Vec::new(),
            rate_limit: None,
            timeout_ms: None,
        };
        
        self.routes.insert(route_key, route_info);
    }
    
    /// Get handler for route
    pub fn get_handler(&self, method: &str, path: &str) -> Option<&Arc<dyn ApiHandler>> {
        let route_key = format!("{} {}", method, path);
        self.handlers.get(&route_key)
    }
    
    /// List all routes
    pub async fn list_routes(&self) -> Vec<RouteInfo> {
        self.routes.values().cloned().collect()
    }
    
    /// Get handler count
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }
}

impl Default for HandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check handler
#[derive(Debug)]
pub struct HealthHandler {
    version: String,
}

impl HealthHandler {
    pub fn new(version: String) -> Self {
        Self { version }
    }
}

#[async_trait::async_trait]
impl ApiHandler for HealthHandler {
    async fn handle(&self, ctx: RequestContext, _body: Vec<u8>) -> Result<Vec<u8>> {
        let health_data = json!({
            "healthy": true,
            "timestamp": ctx.timestamp,
            "version": self.version,
            "request_id": ctx.request_id,
        });
        
        let response = ApiResponse::success(health_data, ctx.request_id);
        Ok(serde_json::to_vec(&response)?)
    }
    
    fn name(&self) -> &str {
        "health_handler"
    }
}

/// Echo handler for testing
#[derive(Debug)]
pub struct EchoHandler;

#[async_trait::async_trait]
impl ApiHandler for EchoHandler {
    async fn handle(&self, ctx: RequestContext, body: Vec<u8>) -> Result<Vec<u8>> {
        let echo_data = json!({
            "request_id": ctx.request_id,
            "method": ctx.method,
            "path": ctx.path,
            "query_params": ctx.query_params,
            "headers": ctx.headers,
            "body_length": body.len(),
            "timestamp": ctx.timestamp,
        });
        
        let response = ApiResponse::success(echo_data, ctx.request_id);
        Ok(serde_json::to_vec(&response)?)
    }
    
    fn name(&self) -> &str {
        "echo_handler"
    }
}

/// Status handler
#[derive(Debug)]
pub struct StatusHandler {
    start_time: std::time::Instant,
}

impl StatusHandler {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }
}

#[async_trait::async_trait]
impl ApiHandler for StatusHandler {
    async fn handle(&self, ctx: RequestContext, _body: Vec<u8>) -> Result<Vec<u8>> {
        let uptime = self.start_time.elapsed();
        
        let status_data = json!({
            "status": "running",
            "uptime_seconds": uptime.as_secs(),
            "uptime_human": format!("{}h {}m {}s", 
                uptime.as_secs() / 3600,
                (uptime.as_secs() % 3600) / 60,
                uptime.as_secs() % 60
            ),
            "request_id": ctx.request_id,
            "timestamp": ctx.timestamp,
        });
        
        let response = ApiResponse::success(status_data, ctx.request_id);
        Ok(serde_json::to_vec(&response)?)
    }
    
    fn name(&self) -> &str {
        "status_handler"
    }
}

/// Metrics handler
#[derive(Debug)]
pub struct MetricsHandler {
    metrics: Arc<RwLock<HandlerMetrics>>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct HandlerMetrics {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    average_response_time_ms: f64,
    last_request_time: Option<u64>,
}

impl MetricsHandler {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HandlerMetrics::default())),
        }
    }
    
    pub async fn record_request(&self, success: bool, response_time_ms: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        
        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }
        
        // Update average response time
        let total = metrics.total_requests as f64;
        let current_avg = metrics.average_response_time_ms;
        let new_avg = (current_avg * (total - 1.0) + response_time_ms as f64) / total;
        metrics.average_response_time_ms = new_avg;
        
        metrics.last_request_time = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_secs()
        );
    }
    
    pub async fn get_metrics(&self) -> HandlerMetrics {
        self.metrics.read().await.clone()
    }
}

#[async_trait::async_trait]
impl ApiHandler for MetricsHandler {
    async fn handle(&self, ctx: RequestContext, _body: Vec<u8>) -> Result<Vec<u8>> {
        let metrics = self.get_metrics().await;
        
        let metrics_data = json!({
            "handler_metrics": metrics,
            "request_id": ctx.request_id,
            "timestamp": ctx.timestamp,
        });
        
        let response = ApiResponse::success(metrics_data, ctx.request_id);
        Ok(serde_json::to_vec(&response)?)
    }
    
    fn name(&self) -> &str {
        "metrics_handler"
    }
}

/// Config handler
#[derive(Debug)]
pub struct ConfigHandler {
    config: Arc<RwLock<serde_json::Value>>,
}

impl ConfigHandler {
    pub fn new(config: serde_json::Value) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
        }
    }
    
    pub async fn update_config(&self, new_config: serde_json::Value) -> Result<()> {
        let mut config = self.config.write().await;
        *config = new_config;
        Ok(())
    }
    
    pub async fn get_config(&self) -> serde_json::Value {
        self.config.read().await.clone()
    }
}

#[async_trait::async_trait]
impl ApiHandler for ConfigHandler {
    async fn handle(&self, ctx: RequestContext, body: Vec<u8>) -> Result<Vec<u8>> {
        match ctx.method.as_str() {
            "GET" => {
                let config = self.get_config().await;
                let response = ApiResponse::success(config, ctx.request_id);
                Ok(serde_json::to_vec(&response)?)
            }
            "PUT" | "POST" => {
                let new_config: serde_json::Value = serde_json::from_slice(&body)?;
                self.update_config(new_config).await?;
                
                let config = self.get_config().await;
                let response = ApiResponse::success(config, ctx.request_id);
                Ok(serde_json::to_vec(&response)?)
            }
            _ => {
                let error: ApiResponse<serde_json::Value> = ApiResponse::error(
                    "METHOD_NOT_ALLOWED".to_string(),
                    "Method not allowed".to_string(),
                    ctx.request_id,
                );
                Ok(serde_json::to_vec(&error)?)
            }
        }
    }
    
    fn name(&self) -> &str {
        "config_handler"
    }
}

/// Info handler
#[derive(Debug)]
pub struct InfoHandler {
    build_info: BuildInfo,
}

#[derive(Debug, Clone, Serialize)]
struct BuildInfo {
    version: String,
    git_commit: Option<String>,
    build_time: String,
    rust_version: String,
    target: String,
    features: Vec<String>,
}

impl InfoHandler {
    pub fn new() -> Self {
        Self {
            build_info: BuildInfo {
                version: env!("CARGO_PKG_VERSION").to_string(),
                git_commit: option_env!("GIT_COMMIT").map(|s| s.to_string()),
                build_time: std::env::var("VERGEN_BUILD_TIMESTAMP").unwrap_or_else(|_| "unknown".to_string()),
                rust_version: "unknown".to_string(), // rustc_version_runtime::rust_version() not available
                target: std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
                features: vec![
                    "default".to_string(),
                    // Add actual features based on compilation
                ],
            },
        }
    }
}

#[async_trait::async_trait]
impl ApiHandler for InfoHandler {
    async fn handle(&self, ctx: RequestContext, _body: Vec<u8>) -> Result<Vec<u8>> {
        let info_data = json!({
            "build_info": self.build_info,
            "runtime_info": {
                "pid": std::process::id(),
                "thread_count": std::thread::available_parallelism()?.get(),
                "memory_usage_mb": self.get_process_memory_usage(),
            },
            "request_id": ctx.request_id,
            "timestamp": ctx.timestamp,
        });
        
        let response = ApiResponse::success(info_data, ctx.request_id);
        Ok(serde_json::to_vec(&response)?)
    }
    
    fn name(&self) -> &str {
        "info"
    }
    
    fn get_process_memory_usage(&self) -> f64 {
        nexora_infrastructure::common::get_process_memory_mb()
    }
}

/// Test handler for development
#[derive(Debug)]
pub struct TestHandler {
    test_data: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl TestHandler {
    pub fn new() -> Self {
        Self {
            test_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn set_test_data(&self, key: String, value: serde_json::Value) {
        let mut data = self.test_data.write().await;
        data.insert(key, value);
    }
    
    pub async fn get_test_data(&self, key: &str) -> Option<serde_json::Value> {
        let data = self.test_data.read().await;
        data.get(key).cloned()
    }
    
    pub async fn clear_test_data(&self) {
        let mut data = self.test_data.write().await;
        data.clear();
    }
}

#[async_trait::async_trait]
impl ApiHandler for TestHandler {
    async fn handle(&self, ctx: RequestContext, body: Vec<u8>) -> Result<Vec<u8>> {
        match (ctx.method.as_str(), ctx.path.as_str()) {
            ("GET", "/test/data") => {
                let data = self.test_data.read().await;
                let response = ApiResponse::success(json!({"data": *data}), ctx.request_id);
                Ok(serde_json::to_vec(&response)?)
            }
            ("POST", "/test/data") => {
                let new_data: serde_json::Value = serde_json::from_slice(&body)?;
                if let Some(obj) = new_data.as_object() {
                    for (key, value) in obj {
                        self.set_test_data(key.clone(), value.clone()).await;
                    }
                }
                
                let data = self.test_data.read().await;
                let response = ApiResponse::success(json!({"data": *data}), ctx.request_id);
                Ok(serde_json::to_vec(&response)?)
            }
            ("DELETE", "/test/data") => {
                self.clear_test_data().await;
                let response = ApiResponse::success(json!({"message": "Test data cleared"}), ctx.request_id);
                Ok(serde_json::to_vec(&response)?)
            }
            _ => {
                let error: ApiResponse<serde_json::Value> = ApiResponse::error(
                    "NOT_FOUND".to_string(),
                    "Endpoint not found".to_string(),
                    ctx.request_id,
                );
                Ok(serde_json::to_vec(&error)?)
            }
        }
    }
    
    fn name(&self) -> &str {
        "test_handler"
    }
}

/// Utility function to create default handler registry
pub fn create_default_handlers() -> HandlerRegistry {
    let mut registry = HandlerRegistry::new();
    
    // Register default handlers
    registry.register_handler(
        "/health".to_string(),
        "GET".to_string(),
        Arc::new(HealthHandler::new(env!("CARGO_PKG_VERSION").to_string())),
    );
    
    registry.register_handler(
        "/echo".to_string(),
        "POST".to_string(),
        Arc::new(EchoHandler),
    );
    
    registry.register_handler(
        "/status".to_string(),
        "GET".to_string(),
        Arc::new(StatusHandler::new()),
    );
    
    registry.register_handler(
        "/metrics".to_string(),
        "GET".to_string(),
        Arc::new(MetricsHandler::new()),
    );
    
    let default_config = json!({
        "api": {
            "version": env!("CARGO_PKG_VERSION"),
            "max_connections": 1000,
            "timeout_seconds": 30
        }
    });
    
    registry.register_handler(
        "/config".to_string(),
        "GET".to_string(),
        Arc::new(ConfigHandler::new(default_config)),
    );
    
    registry.register_handler(
        "/config".to_string(),
        "PUT".to_string(),
        Arc::new(ConfigHandler::new(json!({}))),
    );
    
    registry.register_handler(
        "/info".to_string(),
        "GET".to_string(),
        Arc::new(InfoHandler::new()),
    );
    
    registry.register_handler(
        "/test/data".to_string(),
        "GET".to_string(),
        Arc::new(TestHandler::new()),
    );
    
    registry.register_handler(
        "/test/data".to_string(),
        "POST".to_string(),
        Arc::new(TestHandler::new()),
    );
    
    registry.register_handler(
        "/test/data".to_string(),
        "DELETE".to_string(),
        Arc::new(TestHandler::new()),
    );
    
    registry
}
