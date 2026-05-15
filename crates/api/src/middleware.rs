//! API Middleware - Rust implementation
//! 
//! Middleware stack for request/response processing

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::Serialize;
use crate::security::RateLimitStatistics;

use crate::{Middleware, RequestContext, RateLimiter};

/// Middleware stack for processing requests and responses
pub struct MiddlewareStack {
    middlewares: Vec<Arc<dyn Middleware>>,
    rate_limiter: Arc<RateLimiter>,
}

impl MiddlewareStack {
    /// Create new middleware stack
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
            rate_limiter: Arc::new(RateLimiter::new()),
        }
    }
    
    /// Add middleware to stack
    pub fn add_middleware(&mut self, middleware: Arc<dyn Middleware>) {
        self.middlewares.push(middleware);
    }
    
    /// Process request through middleware stack
    pub async fn process_request(&self, ctx: &mut RequestContext, body: &mut Vec<u8>) -> Result<()> {
        for middleware in &self.middlewares {
            middleware.process_request(ctx, body).await?;
        }
        Ok(())
    }
    
    /// Process response through middleware stack
    pub async fn process_response(&self, ctx: &mut RequestContext, response: &mut Vec<u8>) -> Result<()> {
        for middleware in &self.middlewares {
            middleware.process_response(ctx, response).await?;
        }
        Ok(())
    }
    
    /// Get rate limiter
    pub fn rate_limiter(&self) -> &Arc<RateLimiter> {
        &self.rate_limiter
    }
}

impl Default for MiddlewareStack {
    fn default() -> Self {
        Self::new()
    }
}

/// Authentication middleware
#[derive(Debug)]
pub struct AuthMiddleware {
    api_keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
    enable_auth: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ApiKeyInfo {
    key: String,
    name: String,
    permissions: Vec<String>,
    created_at: u64,
    last_used: Option<u64>,
    usage_count: u64,
}

impl AuthMiddleware {
    pub fn new(enable_auth: bool) -> Self {
        Self {
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            enable_auth,
        }
    }
    
    pub async fn add_api_key(&self, key: String, name: String, permissions: Vec<String>) -> Result<()> {
        let info = ApiKeyInfo {
            key: key.clone(),
            name,
            permissions,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_secs(),
            last_used: None,
            usage_count: 0,
        };
        
        let mut keys = self.api_keys.write().await;
        keys.insert(key, info);
        Ok(())
    }
    
    pub async fn validate_api_key(&self, key: &str) -> Result<bool> {
        let keys = self.api_keys.read().await;
        Ok(keys.contains_key(key))
    }
    
    pub async fn update_usage(&self, key: &str) -> Result<()> {
        let mut keys = self.api_keys.write().await;
        if let Some(info) = keys.get_mut(key) {
            info.last_used = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                    .as_secs()
            );
            info.usage_count += 1;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Middleware for AuthMiddleware {
    async fn process_request(&self, ctx: &mut RequestContext, _body: &mut Vec<u8>) -> Result<()> {
        if !self.enable_auth {
            return Ok(());
        }
        
        // Check for API key in headers
        let api_key = ctx.headers.get("authorization")
            .or_else(|| ctx.headers.get("x-api-key"))
            .and_then(|k| k.strip_prefix("Bearer "))
            .unwrap_or("");
        
        if !self.validate_api_key(api_key).await? {
            return Err(anyhow::anyhow!("Invalid API key"));
        }
        
        // Update usage statistics
        self.update_usage(api_key).await?;
        
        // Add user info to context
        ctx.headers.insert("authenticated".to_string(), "true".to_string());
        
        Ok(())
    }
    
    async fn process_response(&self, _ctx: &mut RequestContext, _response: &mut Vec<u8>) -> Result<()> {
        // No post-processing needed for auth
        Ok(())
    }
    
    fn name(&self) -> &str {
        "auth_middleware"
    }
}

/// Logging middleware
#[derive(Debug)]
pub struct LoggingMiddleware {
    log_level: LogLevel,
    include_body: bool,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LoggingMiddleware {
    pub fn new(log_level: LogLevel, include_body: bool) -> Self {
        Self {
            log_level,
            include_body,
        }
    }
    
    fn should_log(&self, level: LogLevel) -> bool {
        match (&self.log_level, level) {
            (LogLevel::Debug, _) => true,
            (LogLevel::Info, LogLevel::Info | LogLevel::Warn | LogLevel::Error) => true,
            (LogLevel::Warn, LogLevel::Warn | LogLevel::Error) => true,
            (LogLevel::Error, LogLevel::Error) => true,
            _ => false,
        }
    }
}

#[async_trait::async_trait]
impl Middleware for LoggingMiddleware {
    async fn process_request(&self, ctx: &mut RequestContext, body: &mut Vec<u8>) -> Result<()> {
        if self.should_log(LogLevel::Info) {
            tracing::info!(
                request_id = %ctx.request_id,
                method = %ctx.method,
                path = %ctx.path,
                client_ip = %ctx.client_ip,
                body_length = body.len(),
                "Incoming request"
            );
        }
        
        if self.should_log(LogLevel::Debug) && self.include_body {
            tracing::debug!(
                request_id = %ctx.request_id,
                body = %String::from_utf8_lossy(&body),
                "Request body"
            );
        }
        
        Ok(())
    }
    
    async fn process_response(&self, ctx: &mut RequestContext, response: &mut Vec<u8>) -> Result<()> {
        if self.should_log(LogLevel::Info) {
            tracing::info!(
                request_id = %ctx.request_id,
                response_length = response.len(),
                "Response sent"
            );
        }
        
        if self.should_log(LogLevel::Debug) && self.include_body {
            tracing::debug!(
                request_id = %ctx.request_id,
                response = %String::from_utf8_lossy(&response),
                "Response body"
            );
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "logging_middleware"
    }
}

/// Rate limiting middleware
#[derive(Debug)]
pub struct RateLimitingMiddleware {
    rate_limiter: Arc<RateLimiter>,
    default_limit: u32,
    window_seconds: u64,
}

impl RateLimitingMiddleware {
    pub fn new(rate_limiter: Arc<RateLimiter>, default_limit: u32, window_seconds: u64) -> Self {
        Self {
            rate_limiter,
            default_limit,
            window_seconds,
        }
    }

    pub async fn init_default_limit(&self) {
        let _ = self.rate_limiter.add_limit(
            "default".to_string(),
            crate::RateLimit {
                max_requests: self.default_limit,
                window_seconds: self.window_seconds,
            }
        );
    }
    
    pub fn with_custom_limits(self, limits: HashMap<String, crate::RateLimit>) -> Self {
        for (key, limit) in limits {
            let _ = self.rate_limiter.add_limit(key, limit);
        }
        self
    }
    
    fn get_client_key(&self, ctx: &RequestContext) -> String {
        // Use IP address as default key, fall back to default if no IP
        if ctx.client_ip.is_empty() {
            "default".to_string()
        } else {
            ctx.client_ip.clone()
        }
    }
    
    /// Add specific rate limit for an IP or client
    pub fn add_client_limit(&mut self, client_key: String, max_requests: u32, window_seconds: u64) {
        let _ = self.rate_limiter.add_limit(
            client_key.clone(),
            crate::RateLimit {
                max_requests,
                window_seconds,
            }
        );
    }
    
    /// Check if a specific client would be rate limited
    pub async fn is_rate_limited(&self, client_key: &str) -> Result<bool> {
        self.rate_limiter.check_rate_limit(client_key).await
    }
    
    /// Get current request count for a client
    pub async fn get_request_count(&self, client_key: &str) -> Result<usize> {
        let requests = self.rate_limiter.requests.read().await;
        Ok(requests.get(client_key).map(|v| v.len()).unwrap_or(0))
    }
    
    /// Reset rate limit for a client
    pub async fn reset_rate_limit(&self, client_key: &str) -> Result<()> {
        let mut requests = self.rate_limiter.requests.write().await;
        requests.remove(client_key);
        Ok(())
    }
    
    /// Get rate limit statistics
    pub async fn get_statistics(&self) -> Result<RateLimitStatistics> {
        let requests = self.rate_limiter.requests.read().await;
        let total_clients = requests.len();
        let total_requests: usize = requests.values().map(|v| v.len()).sum();
        
        // Calculate requests per second
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let recent_requests: usize = requests.values()
            .map(|times| times.iter().filter(|&&timestamp| now - timestamp <= 60).count())
            .sum();
        
        Ok(RateLimitStatistics {
            total_clients,
            total_requests,
            requests_per_minute: recent_requests as f64,
            average_requests_per_client: if total_clients > 0 {
                total_requests as f64 / total_clients as f64
            } else {
                0.0
            },
        })
    }
    
    /// Clear rate limit data for a client
    pub async fn clear_client_data(&self, client_key: &str) -> Result<()> {
        let mut requests = self.rate_limiter.requests.write().await;
        requests.remove(client_key);
        Ok(())
    }
}

#[async_trait::async_trait]
impl Middleware for RateLimitingMiddleware {
    async fn process_request(&self, ctx: &mut RequestContext, _body: &mut Vec<u8>) -> Result<()> {
        let client_key = self.get_client_key(ctx);
        
        // Check rate limit with proper error handling
        match self.rate_limiter.check_rate_limit(&client_key).await {
            Ok(true) => {
                // Request allowed - add rate limit headers
                if let Ok(count) = self.get_request_count(&client_key).await {
                    ctx.headers.insert(
                        "X-RateLimit-Limit".to_string(),
                        self.default_limit.to_string()
                    );
                    ctx.headers.insert(
                        "X-RateLimit-Remaining".to_string(),
                        (self.default_limit.saturating_sub(count as u32)).to_string()
                    );
                    ctx.headers.insert(
                        "X-RateLimit-Reset".to_string(),
                        (std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() + self.window_seconds).to_string()
                    );
                }
                Ok(())
            },
            Ok(false) => {
                // Rate limit exceeded
                ctx.headers.insert(
                    "X-RateLimit-Limit".to_string(),
                    self.default_limit.to_string()
                );
                ctx.headers.insert(
                    "X-RateLimit-Remaining".to_string(),
                    "0".to_string()
                );
                ctx.headers.insert(
                    "X-RateLimit-Retry-After".to_string(),
                    self.window_seconds.to_string()
                );
                Err(anyhow::anyhow!("Rate limit exceeded for client: {}. Limit: {} requests per {} seconds", 
                    client_key, self.default_limit, self.window_seconds))
            },
            Err(e) => {
                // Log error but allow request to continue (fail open)
                tracing::warn!(
                    client_key = %client_key,
                    error = %e,
                    "Rate limit check failed, allowing request"
                );
                Ok(())
            }
        }
    }
    
    async fn process_response(&self, _ctx: &mut RequestContext, _response: &mut Vec<u8>) -> Result<()> {
        // No response processing needed for rate limiting
        Ok(())
    }
    
    fn name(&self) -> &str {
        "rate_limiting_middleware"
    }
}

/// CORS middleware
#[derive(Debug)]
pub struct CorsMiddleware {
    allowed_origins: Vec<String>,
    allowed_methods: Vec<String>,
    allowed_headers: Vec<String>,
    expose_headers: Vec<String>,
    allow_credentials: bool,
    max_age: u64,
}

impl CorsMiddleware {
    pub fn new() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Requested-With".to_string(),
            ],
            expose_headers: Vec::new(),
            allow_credentials: false,
            max_age: 86400, // 24 hours
        }
    }
    
    fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|allowed| {
            allowed == "*" || allowed == origin
        })
    }
}

#[async_trait::async_trait]
impl Middleware for CorsMiddleware {
    async fn process_request(&self, ctx: &mut RequestContext, _body: &mut Vec<u8>) -> Result<()> {
        // Handle preflight requests
        if ctx.method == "OPTIONS" {
            return Ok(());
        }
        
        // Check Origin header
        if let Some(origin) = ctx.headers.get("origin") {
            if !self.is_origin_allowed(origin) {
                return Err(anyhow::anyhow!("Origin not allowed: {}", origin));
            }
        }
        
        Ok(())
    }
    
    async fn process_response(&self, ctx: &mut RequestContext, _response: &mut Vec<u8>) -> Result<()> {
        // Add CORS headers to response
        // Note: In a real implementation, this would modify HTTP headers
        // For now, we just log the CORS information
        
        if ctx.method == "OPTIONS" {
            tracing::debug!("CORS preflight request processed");
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "cors_middleware"
    }
}

/// Compression middleware
#[derive(Debug)]
pub struct CompressionMiddleware {
    enable_compression: bool,
    min_size: usize,
    compression_level: u32,
}

impl CompressionMiddleware {
    pub fn new(enable_compression: bool, min_size: usize, compression_level: u32) -> Self {
        Self {
            enable_compression,
            min_size,
            compression_level,
        }
    }
    
    fn should_compress(&self, content_length: usize) -> bool {
        self.enable_compression && content_length >= self.min_size
    }
}

#[async_trait::async_trait]
impl Middleware for CompressionMiddleware {
    async fn process_request(&self, ctx: &mut RequestContext, _body: &mut Vec<u8>) -> Result<()> {
        // Check if client accepts compression
        let accepts_encoding = ctx.headers.get("accept-encoding")
            .map(|s| s.contains("gzip"))
            .unwrap_or(false);
        
        if accepts_encoding {
            ctx.headers.insert("compression_supported".to_string(), "true".to_string());
        }
        
        Ok(())
    }
    
    async fn process_response(&self, ctx: &mut RequestContext, response: &mut Vec<u8>) -> Result<()> {
        if self.should_compress(response.len()) {
            let original_size = response.len();
            
            // Perform compression using gzip
            match self.compress_data(response) {
                Ok(compressed_data) => {
                    *response = compressed_data;
                    
                    // Update response headers
                    ctx.headers.insert("Content-Encoding".to_string(), "gzip".to_string());
                    ctx.headers.insert("Content-Length".to_string(), response.len().to_string());
                    
                    tracing::info!(
                        request_id = %ctx.request_id,
                        original_size = original_size,
                        compressed_size = response.len(),
                        compression_ratio = format!("{:.2}%", (1.0 - response.len() as f64 / original_size as f64) * 100.0),
                        "Response compressed successfully"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %ctx.request_id,
                        error = %e,
                        "Failed to compress response, sending uncompressed"
                    );
                }
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "compression_middleware"
    }
}

impl CompressionMiddleware {
    /// Compress data using gzip
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::prelude::*;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        encoder.finish().map_err(|e| anyhow::anyhow!("Compression failed: {}", e))
    }
}

/// Security middleware
#[derive(Debug)]
pub struct SecurityMiddleware {
    max_request_size: usize,
    block_suspicious_ips: bool,
    suspicious_patterns: Vec<String>,
}

impl SecurityMiddleware {
    pub fn new(max_request_size: usize, block_suspicious_ips: bool) -> Self {
        Self {
            max_request_size,
            block_suspicious_ips,
            suspicious_patterns: vec![
                "<script".to_string(),
                "javascript:".to_string(),
                "data:text/html".to_string(),
                "..\\".to_string(),
                "../".to_string(),
            ],
        }
    }
    
    fn is_suspicious_content(&self, content: &str) -> bool {
        let content_lower = content.to_lowercase();
        self.suspicious_patterns.iter()
            .any(|pattern| content_lower.contains(pattern))
    }
    
    fn is_suspicious_ip(&self, ip: &str) -> bool {
        // Basic IP reputation checking
        self.is_in_blacklist(ip) || self.is_suspicious_ip_pattern(ip)
    }
    
    /// Check if IP is in blacklist
    fn is_in_blacklist(&self, ip: &str) -> bool {
        // Common malicious IP ranges and patterns
        let blacklist_patterns = vec![
            "0.0.0.0", // Unspecified address
            "127.0.0.1", // Localhost (should be handled separately)
            "::1", // IPv6 localhost
        ];
        
        // Check exact matches
        if blacklist_patterns.contains(&ip) {
            return true;
        }
        
        // Check for suspicious IP patterns
        self.is_suspicious_ip_pattern(ip)
    }
    
    /// Check for suspicious IP patterns
    fn is_suspicious_ip_pattern(&self, ip: &str) -> bool {
        // Check for private IP ranges that shouldn't be accessing public endpoints
        if ip.starts_with("10.") || ip.starts_with("192.168.") || 
           (ip.starts_with("172.") && self.is_in_172_16_31_range(ip)) {
            return true;
        }
        
        // Check for known proxy/VPN patterns (simplified)
        if self.is_known_proxy_pattern(ip) {
            return true;
        }
        
        false
    }
    
    /// Check if IP is in 172.16.0.0/12 range
    fn is_in_172_16_31_range(&self, ip: &str) -> bool {
        if let Some(octets) = self.parse_ipv4(ip) {
            octets[0] == 172 && (16..=32).contains(&octets[1])
        } else {
            false
        }
    }
    
    /// Check for known proxy/VPN patterns
    fn is_known_proxy_pattern(&self, ip: &str) -> bool {
        // Simplified proxy detection - in production, use a proper IP reputation service
        let suspicious_asn_ranges = vec![
            "1.0.0.0/8",   // APNIC
            "2.0.0.0/8",   // RIPE NCC
            "5.0.0.0/8",   // RIPE NCC
        ];
        
        // Check if IP falls in suspicious ranges
        for range in &suspicious_asn_ranges {
            if self.ip_in_range(ip, range) {
                return true;
            }
        }
        
        false
    }
    
    /// Parse IPv4 address into octets
    fn parse_ipv4(&self, ip: &str) -> Option<[u8; 4]> {
        let parts: Vec<&str> = ip.split('.').collect();
        if parts.len() != 4 {
            return None;
        }
        
        let mut octets = [0u8; 4];
        for (i, part) in parts.iter().enumerate() {
            match part.parse::<u8>() {
                Ok(octet) => octets[i] = octet,
                Err(_) => return None,
            }
        }
        
        Some(octets)
    }
    
    /// Check if IP is in CIDR range (simplified implementation)
    fn ip_in_range(&self, ip: &str, cidr: &str) -> bool {
        if let Some((network_str, prefix_str)) = cidr.split_once('/') {
            if let (Some(network), Ok(prefix)) = (self.parse_ipv4(network_str), prefix_str.parse::<u8>()) {
                if let Some(target) = self.parse_ipv4(ip) {
                    // Simple CIDR check (for demonstration)
                    let mask = self.create_netmask(prefix);
                    let network_masked = self.apply_mask(network, mask);
                    let target_masked = self.apply_mask(target, mask);
                    return network_masked == target_masked;
                }
            }
        }
        false
    }
    
    /// Create netmask for CIDR
    fn create_netmask(&self, prefix: u8) -> [u8; 4] {
        let mut mask = [0u8; 4];
        let mut bits = prefix;
        
        for i in 0..4 {
            if bits >= 8 {
                mask[i] = 255;
                bits -= 8;
            } else if bits > 0 {
                mask[i] = (256u32 - (1u32 << (8 - bits))) as u8;
                bits = 0;
            } else {
                mask[i] = 0;
            }
        }
        
        mask
    }
    
    /// Apply netmask to IP
    fn apply_mask(&self, ip: [u8; 4], mask: [u8; 4]) -> [u8; 4] {
        [
            ip[0] & mask[0],
            ip[1] & mask[1],
            ip[2] & mask[2],
            ip[3] & mask[3],
        ]
    }
}

#[async_trait::async_trait]
impl Middleware for SecurityMiddleware {
    async fn process_request(&self, ctx: &mut RequestContext, body: &mut Vec<u8>) -> Result<()> {
        // Check request size
        if body.len() > self.max_request_size {
            return Err(anyhow::anyhow!("Request too large: {} bytes", body.len()));
        }
        
        // Check for suspicious IP
        if self.block_suspicious_ips && self.is_suspicious_ip(&ctx.client_ip) {
            return Err(anyhow::anyhow!("Suspicious IP blocked: {}", ctx.client_ip));
        }
        
        // Check for suspicious content
        let body_str = String::from_utf8_lossy(body);
        if self.is_suspicious_content(&body_str) {
            return Err(anyhow::anyhow!("Suspicious content detected"));
        }
        
        Ok(())
    }
    
    async fn process_response(&self, _ctx: &mut RequestContext, _response: &mut Vec<u8>) -> Result<()> {
        // No post-processing needed for security
        Ok(())
    }
    
    fn name(&self) -> &str {
        "security_middleware"
    }
}

/// Utility function to create default middleware stack
pub async fn create_default_middleware_stack() -> MiddlewareStack {
    let mut stack = MiddlewareStack::new();
    
    // Add default middlewares in order
    stack.add_middleware(Arc::new(SecurityMiddleware::new(10 * 1024 * 1024, true))); // 10MB max
    stack.add_middleware(Arc::new(LoggingMiddleware::new(LogLevel::Info, false)));
    let rate_limit_mw = RateLimitingMiddleware::new(
        stack.rate_limiter().clone(),
        100, // 100 requests per window
        60,  // per minute
    );
    rate_limit_mw.init_default_limit().await;
    stack.add_middleware(Arc::new(rate_limit_mw));
    stack.add_middleware(Arc::new(CorsMiddleware::new()));
    stack.add_middleware(Arc::new(AuthMiddleware::new(false))); // Auth disabled by default
    stack.add_middleware(Arc::new(CompressionMiddleware::new(true, 1024, 6)));
    
    stack
}
