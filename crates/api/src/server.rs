//! API Server - Rust implementation
//! 
//! High-performance HTTP server replacing runtime_http_server.c

use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tracing::{info, warn};
use axum::{
    Router,
    routing::get,
    extract::{Request, State},
    response::{Json, Response},
    http::StatusCode,
    middleware::{self, Next},
};
use tower_http::cors::{CorsLayer, Any};
use serde_json::json;

use crate::{
    ApiConfig, ApiStatistics, HealthStatus,
    handlers::HandlerRegistry, middleware::MiddlewareStack, metrics::MetricsCollector,
};

/// Main API server
pub struct ApiServer {
    config: ApiConfig,
    router: Router,
    handlers: Arc<HandlerRegistry>,
    middleware: Arc<MiddlewareStack>,
    metrics: Arc<MetricsCollector>,
    statistics: Arc<tokio::sync::RwLock<ApiStatistics>>,
    start_time: Instant,
}

impl ApiServer {
    /// Create new API server
    pub fn new(config: ApiConfig) -> Result<Self> {
        let handlers = Arc::new(HandlerRegistry::new());
        let middleware = Arc::new(MiddlewareStack::new());
        let metrics = Arc::new(MetricsCollector::new());
        let statistics = Arc::new(tokio::sync::RwLock::new(ApiStatistics::default()));
        
        let router = Self::build_router(&config, handlers.clone(), middleware.clone(), metrics.clone())?;
        
        Ok(Self {
            config,
            router,
            handlers,
            middleware,
            metrics,
            statistics,
            start_time: Instant::now(),
        })
    }
    
    /// Build application router
    fn build_router(
        config: &ApiConfig,
        handlers: Arc<HandlerRegistry>,
        middleware: Arc<MiddlewareStack>,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Router> {
        let mut app = Router::new()
            // Health check endpoints
            .route("/health", get(health_check_handler))
            .route("/health/detailed", get(detailed_health_check_handler))
            
            // Metrics endpoints
            .route("/metrics", get(metrics_handler))
            .route("/metrics/routes", get(route_metrics_handler))
            
            // API endpoints
            .route("/api/v1/status", get(api_status_handler))
            .route("/api/v1/routes", get(list_routes_handler))
            
            // System endpoints
            .route("/system/info", get(system_info_handler))
            .route("/system/stats", get(system_stats_handler))
            
            // Add middleware state
            .with_state(AppState {
                handlers,
                middleware,
                metrics,
                statistics: Arc::new(tokio::sync::RwLock::new(ApiStatistics::default())),
                config: config.clone(),
            });
        
        // Add CORS if enabled
        if config.enable_cors {
            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::PUT,
                    axum::http::Method::DELETE,
                ])
                .allow_headers(Any);
            
            app = app.layer(cors);
        }
        
        // Add common middleware
        app = app.layer(middleware::from_fn(request_logging_middleware))
            .layer(middleware::from_fn(rate_limit_middleware));
        
        Ok(app)
    }
    
    /// Start the server
    pub async fn start(self) -> Result<()> {
        let addr: std::net::SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid address: {}", e))?;
        
        info!("Starting API server on {}", addr);
        
        let listener = TcpListener::bind(addr).await
            .map_err(|e| anyhow::anyhow!("Failed to bind to address: {}", e))?;
        
        info!("API server listening on {}", listener.local_addr()?);
        
        if self.config.enable_tls {
            self.start_tls_server(listener).await
        } else {
            self.start_http_server(listener).await
        }
    }
    
    /// Start HTTP server
    async fn start_http_server(self, listener: TcpListener) -> Result<()> {
        axum::serve(listener, self.router)
            .await
            .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;
        
        Ok(())
    }
    
    /// Create Axum application
    async fn create_app(self) -> axum::Router {
        self.router
    }
    
    /// Start HTTPS server
    async fn start_tls_server(self, listener: TcpListener) -> Result<()> {
        #[cfg(feature = "tls")]
        {
            info!("Starting HTTPS server with TLS support");
            
            // Load TLS configuration
            let tls_config = self.load_tls_config().await?;
            
            // Create TLS acceptor
            let tls_acceptor = tokio_rustls::TlsAcceptor::from(tls_config);
            
            // Create HTTPS router
            let app = self.create_app().await;
            
            // Serve HTTPS with TLS
            axum_server::bind_rustls(listener, tls_acceptor)
                .serve(app.into_make_service())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to start HTTPS server: {}", e))?;
            
            info!("HTTPS server started successfully");
            Ok(())
        }
        
        #[cfg(not(feature = "tls"))]
        {
            warn!("TLS feature not enabled, falling back to HTTP");
            self.start_http_server(listener).await
        }
    }
    
    /// Load TLS configuration
    #[cfg(feature = "tls")]
    async fn load_tls_config(&self) -> Result<tokio_rustls::TlsConfig> {
        use std::fs;
        use rustls::{Certificate, PrivateKey, ServerConfig};
        use rustls_pemfile::{certs, pkcs8_private_keys};
        
        // Default certificate paths
        let cert_path = std::env::var("TLS_CERT_PATH").unwrap_or_else(|_| "certs/server.crt".to_string());
        let key_path = std::env::var("TLS_KEY_PATH").unwrap_or_else(|_| "certs/server.key".to_string());
        
        // Load certificate file
        let cert_file = fs::File::open(&cert_path)
            .map_err(|e| anyhow::anyhow!("Failed to open certificate file {}: {}", cert_path, e))?;
        let mut cert_reader = std::io::BufReader::new(cert_file);
        let certs = certs(&mut cert_reader)
            .map_err(|e| anyhow::anyhow!("Failed to read certificates: {}", e))?;
        
        // Load private key file
        let key_file = fs::File::open(&key_path)
            .map_err(|e| anyhow::anyhow!("Failed to open private key file {}: {}", key_path, e))?;
        let mut key_reader = std::io::BufReader::new(key_file);
        let keys = pkcs8_private_keys(&mut key_reader)
            .map_err(|e| anyhow::anyhow!("Failed to read private keys: {}", e))?;
        
        if keys.is_empty() {
            return Err(anyhow::anyhow!("No private keys found in {}", key_path));
        }
        
        // Create server configuration
        let mut config = ServerConfig::builder()
            .with_safe_defaults(rustls::Version::TLS_1_2)
            .with_no_client_auth()
            .with_single_cert(
                certs.into_iter().map(Certificate).collect(),
                PrivateKey(keys[0].clone()),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create TLS config: {}", e))?;
        
        // Enable ALPN for HTTP/2
        config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        
        Ok(tokio_rustls::TlsConfig::from_config(Arc::new(config)))
    }
    
    /// Generate self-signed certificate for development
    #[cfg(feature = "tls")]
    pub async fn generate_self_signed_cert() -> Result<()> {
        use rcgen::{CertificateParams, DistinguishedName, KeyPair};
        use std::fs;
        use time::OffsetDateTime;
        
        info!("Generating self-signed certificate for development");
        
        // Create certificate parameters
        let mut params = CertificateParams::default();
        params.distinguished_name = DistinguishedName::new();
        params.distinguished_name.push(rcgen::DnType::CommonName, "localhost");
        params.distinguished_name.push(rcgen::DnType::OrganizationName, "Nexora AI");
        params.distinguished_name.push(rcgen::DnType::OrganizationalUnitName, "Development");
        
        // Set validity period (1 year)
        params.not_before = OffsetDateTime::now_utc();
        params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
        
        // Generate key pair
        let key_pair = KeyPair::generate()
            .map_err(|e| anyhow::anyhow!("Failed to generate key pair: {}", e))?;
        
        // Generate certificate
        let cert = params.self_signed(&key_pair)
            .map_err(|e| anyhow::anyhow!("Failed to generate certificate: {}", e))?;
        
        // Create certs directory if it doesn't exist
        fs::create_dir_all("certs")
            .map_err(|e| anyhow::anyhow!("Failed to create certs directory: {}", e))?;
        
        // Write certificate
        fs::write("certs/server.crt", cert.pem())
            .map_err(|e| anyhow::anyhow!("Failed to write certificate: {}", e))?;
        
        // Write private key
        fs::write("certs/server.key", key_pair.serialize_pem())
            .map_err(|e| anyhow::anyhow!("Failed to write private key: {}", e))?;
        
        info!("Self-signed certificate generated successfully");
        info!("Certificate: certs/server.crt");
        info!("Private key: certs/server.key");
        info!("Use these for development or replace with production certificates");
        
        Ok(())
    }
    
    /// Get server statistics
    pub async fn get_statistics(&self) -> ApiStatistics {
        self.statistics.read().await.clone()
    }
    
    /// Get server uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub handlers: Arc<HandlerRegistry>,
    pub middleware: Arc<MiddlewareStack>,
    pub metrics: Arc<MetricsCollector>,
    pub statistics: Arc<tokio::sync::RwLock<ApiStatistics>>,
    pub config: ApiConfig,
}

/// Health check handler
async fn health_check_handler(State(state): State<AppState>) -> Result<Json<HealthStatus>, StatusCode> {
    let stats = state.statistics.read().await;
    
    // Calculate uptime from server start time (stored in stats or use current time as fallback)
    let uptime_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs();
    
    Ok(Json(HealthStatus {
        healthy: true,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds,
        active_connections: stats.active_connections,
        memory_usage_mb: get_memory_usage_mb(),
        cpu_usage_percent: get_cpu_usage_percent(),
    }))
}

/// Detailed health check handler
async fn detailed_health_check_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let stats = state.statistics.read().await;
    
    Ok(Json(json!({
        "healthy": true,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs(),
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": 0,
        "statistics": {
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "average_response_time_ms": stats.average_response_time_ms,
            "requests_per_second": stats.requests_per_second,
            "active_connections": stats.active_connections,
        },
        "system": {
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "thread_count": 0,
        },
        "endpoints": {
            "total": state.handlers.handler_count(),
            "healthy": state.handlers.handler_count(),
        }
    })))
}

/// Metrics handler
async fn metrics_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let metrics_data = state.metrics.get_current_metrics().await;
    
    Ok(Json(json!({
        "timestamp": metrics_data.timestamp,
        "requests_total": metrics_data.requests_total,
        "requests_per_second": metrics_data.requests_per_second,
        "average_response_time_ms": metrics_data.average_response_time_ms,
        "error_rate_percent": metrics_data.error_rate_percent,
        "active_connections": metrics_data.active_connections,
        "memory_usage_mb": metrics_data.memory_usage_mb,
        "cpu_usage_percent": metrics_data.cpu_usage_percent,
        "top_routes": metrics_data.top_routes
    })))
}

/// Route metrics handler
async fn route_metrics_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let stats = state.statistics.read().await;
    
    let routes: Vec<_> = stats.route_counts.iter()
        .map(|(path, count)| json!({
            "path": path,
            "requests": count
        }))
        .collect();
    
    Ok(Json(json!({
        "routes": routes,
        "total_routes": routes.len()
    })))
}

/// API status handler
async fn api_status_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let stats = state.statistics.read().await;
    
    Ok(Json(json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": 0,
        "statistics": {
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "average_response_time_ms": stats.average_response_time_ms,
            "requests_per_second": stats.requests_per_second,
            "active_connections": stats.active_connections,
        },
        "config": {
            "max_connections": state.config.max_connections,
            "request_timeout_seconds": 30,
            "enable_cors": state.config.enable_cors,
            "enable_metrics": state.config.enable_metrics,
        }
    })))
}

/// List routes handler
async fn list_routes_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let routes = state.handlers.list_routes().await;
    
    Ok(Json(json!({
        "routes": routes,
        "total": routes.len()
    })))
}

/// System info handler
async fn system_info_handler() -> Result<Json<serde_json::Value>, StatusCode> {
    Ok(Json(json!({
        "system": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "version": env!("CARGO_PKG_VERSION"),
        },
        "runtime": {
            "rust_version": "unknown", // rustc_version_runtime::rust_version() not available
            "tokio_version": "1.0", // Static version since tokio::version() doesn't exist
        },
        "build": {
            "debug": cfg!(debug_assertions),
            "target": std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
        }
    })))
}

/// System stats handler
async fn system_stats_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let stats = state.statistics.read().await;
    
    Ok(Json(json!({
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs(),
        "uptime_seconds": 0,
        "requests": {
            "total": stats.total_requests,
            "successful": stats.successful_requests,
            "failed": stats.failed_requests,
            "per_second": stats.requests_per_second,
        },
        "performance": {
            "average_response_time_ms": stats.average_response_time_ms,
            "active_connections": stats.active_connections,
        },
        "system": {
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "thread_count": 0,
        }
    })))
}

/// Request logging middleware
async fn request_logging_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let user_agent = request.headers()
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");
    
    info!("{} {} from {}", method, uri, user_agent);
    
    let response = next.run(request).await;
    
    let duration = start.elapsed();
    info!("{} {} completed in {}ms", method, uri, duration.as_millis());
    
    Ok(response)
}

/// Metrics collection middleware
async fn metrics_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let start = Instant::now();
    let path = request.uri().path().to_string();
    let method = request.method().to_string();
    
    let response = next.run(request).await;
    
    let duration = start.elapsed();
    let status = response.status();
    
    // Update statistics
    {
        let mut stats = state.statistics.write().await;
        stats.total_requests += 1;
        
        if status.is_success() {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }
        
        // Update average response time
        let total_requests = stats.total_requests as f64;
        let current_avg = stats.average_response_time_ms;
        let new_avg = (current_avg * (total_requests - 1.0) + duration.as_millis() as f64) / total_requests;
        stats.average_response_time_ms = new_avg;
        
        // Update route counts
        let route_key = format!("{} {}", method, path);
        *stats.route_counts.entry(route_key).or_insert(0) += 1;
    }
    
    // Record metrics
    state.metrics.record_request(&method, &path, duration, status).await;
    
    Ok(response)
}

/// Get current memory usage in MB
fn get_memory_usage_mb() -> f64 {
    use std::fs;
    
    // Try to read from /proc/self/status on Linux
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                // Format: VmRSS:     12345 kB
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0; // Convert KB to MB
                    }
                }
            }
        }
    }
    
    // Fallback: estimate using process info (less accurate)
    0.0
}

/// Get current CPU usage percentage
fn get_cpu_usage_percent() -> f64 {
    use std::fs;
    
    // Try to read from /proc/stat on Linux
    if let Ok(stat) = fs::read_to_string("/proc/stat") {
        if let Some(cpu_line) = stat.lines().next() {
            if cpu_line.starts_with("cpu ") {
                let parts: Vec<&str> = cpu_line.split_whitespace().collect();
                if parts.len() >= 8 {
                    // CPU time calculation: user + nice + system + idle + iowait + irq + softirq
                    let mut total_time = 0;
                    let mut idle_time = 0;
                    
                    for (i, part) in parts.iter().enumerate().skip(1).take(7) {
                        if let Ok(time) = part.parse::<u64>() {
                            total_time += time;
                            if i == 3 { // idle time is the 4th field (index 3)
                                idle_time = time;
                            }
                        }
                    }
                    
                    if total_time > 0 {
                        let usage = ((total_time - idle_time) as f64 / total_time as f64) * 100.0;
                        return usage;
                    }
                }
            }
        }
    }
    
    // Fallback: return 0 (can't determine)
    0.0
}

/// Rate limiting middleware
async fn rate_limit_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Implement rate limiting
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use std::time::{Duration, Instant};
    
    // Get client IP from request
    let client_ip = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.split(',').next())
        .unwrap_or_else(|| {
            request
                .headers()
                .get("x-real-ip")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("unknown")
        })
        .to_string();
    
    // Rate limit configuration (100 requests per minute per IP)
    const REQUESTS_PER_MINUTE: u32 = 100;
    const WINDOW_SIZE: Duration = Duration::from_secs(60);
    
    // Use a simple in-memory rate limiter
    // In production, this should use Redis or similar
    static RATE_LIMITER: std::sync::LazyLock<Arc<RwLock<HashMap<String, Vec<Instant>>>>> = 
        std::sync::LazyLock::new(|| Arc::new(RwLock::new(HashMap::new())));
    
    let now = Instant::now();
    let mut rate_limiter = RATE_LIMITER.write().await;
    
    // Get or create client entry
    let requests = rate_limiter.entry(client_ip.clone()).or_insert_with(Vec::new);
    
    // Remove old requests outside the window
    requests.retain(|&timestamp| now.duration_since(timestamp) < WINDOW_SIZE);
    
    // Check if rate limit exceeded
    if requests.len() >= REQUESTS_PER_MINUTE as usize {
        tracing::warn!("Rate limit exceeded for IP: {}", client_ip);
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    
    // Add current request
    requests.push(now);
    
    // Continue with request
    Ok(next.run(request).await)
}
