//! Advanced Security Middleware
//! 
//! Implementasi security middleware dengan fitur-fitur advanced seperti:
//! - IP whitelisting/blacklisting
//! - Request validation dan sanitization
//! - DDoS protection
//! - CORS handling
//! - Security headers
//! - Input validation

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use regex::Regex;
use tracing::{info, warn, error};

use crate::{Middleware, RequestContext};

/// Security middleware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable IP whitelisting
    pub enable_ip_whitelist: bool,
    /// Enable IP blacklisting
    pub enable_ip_blacklist: bool,
    /// Enable request size limiting
    pub enable_size_limit: bool,
    /// Maximum request size in bytes
    pub max_request_size: usize,
    /// Enable DDoS protection
    pub enable_ddos_protection: bool,
    /// DDoS threshold requests per minute
    pub ddos_threshold: u32,
    /// Enable CORS
    pub enable_cors: bool,
    /// CORS allowed origins
    pub cors_allowed_origins: Vec<String>,
    /// Enable security headers
    pub enable_security_headers: bool,
    /// Enable input validation
    pub enable_input_validation: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_ip_whitelist: false,
            enable_ip_blacklist: true,
            enable_size_limit: true,
            max_request_size: 10 * 1024 * 1024, // 10MB
            enable_ddos_protection: true,
            ddos_threshold: 1000,
            enable_cors: true,
            cors_allowed_origins: vec!["*".to_string()],
            enable_security_headers: true,
            enable_input_validation: true,
        }
    }
}

/// Advanced security middleware
#[derive(Debug)]
pub struct SecurityMiddleware {
    config: SecurityConfig,
    ip_whitelist: Arc<RwLock<HashSet<String>>>,
    ip_blacklist: Arc<RwLock<HashSet<String>>>,
    ddos_tracker: Arc<RwLock<HashMap<String, DdosTracker>>>,
    input_validators: Vec<InputValidator>,
    cors_handler: CorsHandler,
    security_headers: SecurityHeaders,
}

/// DDoS tracking information
#[derive(Debug, Clone)]
struct DdosTracker {
    request_count: u32,
    first_request: std::time::Instant,
    last_request: std::time::Instant,
    blocked: bool,
    block_until: Option<std::time::Instant>,
}

/// Input validator for different types of input
#[derive(Debug, Clone)]
struct InputValidator {
    name: String,
    pattern: Regex,
    description: String,
    required: bool,
}

fn build_validator_pattern(raw: &str) -> Result<Regex> {
    Regex::new(raw).map_err(|e| anyhow::anyhow!("Invalid security regex pattern '{}': {}", raw, e))
}

/// CORS handler
#[derive(Debug)]
struct CorsHandler {
    allowed_origins: Vec<String>,
    allow_all: bool,
    allowed_methods: Vec<String>,
    allowed_headers: Vec<String>,
    max_age: u64,
    allow_credentials: bool,
}

/// Security headers handler
#[derive(Debug)]
struct SecurityHeaders {
    headers: HashMap<String, String>,
}

/// Rate limit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatistics {
    pub total_clients: usize,
    pub total_requests: usize,
    pub requests_per_minute: f64,
    pub average_requests_per_client: f64,
}

impl SecurityMiddleware {
    /// Create new security middleware
    pub fn new(config: SecurityConfig) -> Self {
        let input_validators = Self::create_default_validators();
        let cors_handler = CorsHandler::new(&config);
        let security_headers = SecurityHeaders::new();
        
        Self {
            config,
            ip_whitelist: Arc::new(RwLock::new(HashSet::new())),
            ip_blacklist: Arc::new(RwLock::new(HashSet::new())),
            ddos_tracker: Arc::new(RwLock::new(HashMap::new())),
            input_validators,
            cors_handler,
            security_headers,
        }
    }
    
    /// Add IP to whitelist
    pub async fn add_ip_to_whitelist(&self, ip: &str) -> Result<()> {
        let mut whitelist = self.ip_whitelist.write().await;
        whitelist.insert(ip.to_string());
        info!("Added IP {} to whitelist", ip);
        Ok(())
    }
    
    /// Add IP to blacklist
    pub async fn add_ip_to_blacklist(&self, ip: &str) -> Result<()> {
        let mut blacklist = self.ip_blacklist.write().await;
        blacklist.insert(ip.to_string());
        info!("Added IP {} to blacklist", ip);
        Ok(())
    }
    
    /// Remove IP from whitelist
    pub async fn remove_ip_from_whitelist(&self, ip: &str) -> Result<()> {
        let mut whitelist = self.ip_whitelist.write().await;
        whitelist.remove(ip);
        info!("Removed IP {} from whitelist", ip);
        Ok(())
    }
    
    /// Remove IP from blacklist
    pub async fn remove_ip_from_blacklist(&self, ip: &str) -> Result<()> {
        let mut blacklist = self.ip_blacklist.write().await;
        blacklist.remove(ip);
        info!("Removed IP {} from blacklist", ip);
        Ok(())
    }
    
    /// Get security statistics
    pub async fn get_statistics(&self) -> Result<SecurityStatistics> {
        let ddos_tracker = self.ddos_tracker.read().await;
        let whitelist = self.ip_whitelist.read().await;
        let blacklist = self.ip_blacklist.read().await;
        
        let blocked_ips = ddos_tracker.values()
            .filter(|tracker| tracker.blocked)
            .count();
        
        let active_blocks = ddos_tracker.values()
            .filter(|tracker| {
                if let Some(block_until) = tracker.block_until {
                    block_until > std::time::Instant::now()
                } else {
                    false
                }
            })
            .count();
        
        Ok(SecurityStatistics {
            total_tracked_ips: ddos_tracker.len(),
            whitelisted_ips: whitelist.len(),
            blacklisted_ips: blacklist.len(),
            blocked_ips,
            active_blocks,
        })
    }
    
    /// Clear expired blocks
    pub async fn clear_expired_blocks(&self) -> Result<usize> {
        let mut ddos_tracker = self.ddos_tracker.write().await;
        let now = std::time::Instant::now();
        let mut cleared = 0;
        
        ddos_tracker.retain(|_, tracker| {
            if let Some(block_until) = tracker.block_until {
                if block_until <= now {
                    cleared += 1;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
        
        info!("Cleared {} expired DDoS blocks", cleared);
        Ok(cleared)
    }
    
    /// Create default input validators
    fn create_default_validators() -> Vec<InputValidator> {
        let raw_patterns = [
            ("sql_injection", r"(?i)\b(union\s+select|select\s+.*\bfrom\b|insert\s+into|delete\s+from|drop\s+table|alter\s+table|exec\s+\()"),
            ("xss", r"(?i)(<script[\s>]|javascript:\s*\(|onload\s*=|onerror\s*=|onclick\s*=)"),
            ("path_traversal", r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)"),
            ("command_injection", r"(?i)(;\s*(sh|bash|cmd|powershell)|[`$](\(|\{))"),
        ];

        let mut validators = Vec::with_capacity(raw_patterns.len());
        for (name, pattern_str) in &raw_patterns {
            if let Ok(pattern) = build_validator_pattern(pattern_str) {
                validators.push(InputValidator {
                    name: name.to_string(),
                    pattern,
                    description: format!("{} patterns", name.replace('_', " ")),
                    required: true,
                });
            }
        }
        validators
    }
    
    /// Validate IP address against whitelist/blacklist
    async fn validate_ip(&self, ip: &str) -> Result<bool> {
        // Check blacklist first (deny by default)
        if self.config.enable_ip_blacklist {
            let blacklist = self.ip_blacklist.read().await;
            if blacklist.contains(ip) {
                warn!("IP {} is blacklisted", ip);
                return Ok(false);
            }
        }
        
        // Check whitelist (allow by default if whitelist is empty)
        if self.config.enable_ip_whitelist {
            let whitelist = self.ip_whitelist.read().await;
            if !whitelist.is_empty() && !whitelist.contains(ip) {
                warn!("IP {} is not in whitelist", ip);
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Check for DDoS attack
    async fn check_ddos(&self, ip: &str) -> Result<bool> {
        if !self.config.enable_ddos_protection {
            return Ok(true);
        }
        
        let mut tracker = self.ddos_tracker.write().await;
        let now = std::time::Instant::now();
        
        let entry = tracker.entry(ip.to_string()).or_insert(DdosTracker {
            request_count: 0,
            first_request: now,
            last_request: now,
            blocked: false,
            block_until: None,
        });
        
        // Check if currently blocked
        if entry.blocked {
            if let Some(block_until) = entry.block_until {
                if block_until > now {
                    return Ok(false);
                } else {
                    // Block expired
                    entry.blocked = false;
                    entry.block_until = None;
                    info!("DDoS block expired for IP {}", ip);
                }
            }
        }
        
        // Update request tracking
        entry.request_count += 1;
        entry.last_request = now;
        
        // Check if threshold exceeded
        let window_duration = now.duration_since(entry.first_request);
        if window_duration >= std::time::Duration::from_secs(60) {
            // Reset counter if window expired
            entry.request_count = 1;
            entry.first_request = now;
        } else if entry.request_count > self.config.ddos_threshold {
            // Block the IP
            entry.blocked = true;
            entry.block_until = Some(now + std::time::Duration::from_secs(300)); // 5 minutes block
            error!("DDoS protection triggered for IP {} - {} requests in {:?}", 
                   ip, entry.request_count, window_duration);
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Validate request size
    fn validate_request_size(&self, body: &[u8]) -> Result<bool> {
        if self.config.enable_size_limit && body.len() > self.config.max_request_size {
            warn!("Request size {} exceeds limit {}", body.len(), self.config.max_request_size);
            return Ok(false);
        }
        Ok(true)
    }
    
    /// Validate input for security threats
    fn validate_input(&self, body: &[u8]) -> Result<bool> {
        if !self.config.enable_input_validation {
            return Ok(true);
        }
        
        let body_str = String::from_utf8_lossy(body);
        
        for validator in &self.input_validators {
            if validator.pattern.is_match(&body_str) {
                warn!("Input validation failed: {} detected", validator.description);
                if validator.required {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
    
    /// Handle CORS headers
    fn handle_cors(&self, ctx: &mut RequestContext) -> Result<()> {
        if !self.config.enable_cors {
            return Ok(());
        }
        
        // Check Origin header
        if let Some(origin) = ctx.headers.get("origin") {
            if self.cors_handler.is_origin_allowed(origin) {
                ctx.headers.insert("Access-Control-Allow-Origin".to_string(), origin.clone());
            } else {
                warn!("CORS: Origin {} not allowed", origin);
                return Err(anyhow::anyhow!("CORS: Origin not allowed"));
            }
        } else {
            // No Origin header, allow all
            ctx.headers.insert("Access-Control-Allow-Origin".to_string(), "*".to_string());
        }
        
        // Add other CORS headers
        ctx.headers.insert("Access-Control-Allow-Methods".to_string(), 
                         self.cors_handler.allowed_methods.join(", "));
        ctx.headers.insert("Access-Control-Allow-Headers".to_string(), 
                         self.cors_handler.allowed_headers.join(", "));
        ctx.headers.insert("Access-Control-Max-Age".to_string(), 
                         self.cors_handler.max_age.to_string());
        
        if self.cors_handler.allow_credentials {
            ctx.headers.insert("Access-Control-Allow-Credentials".to_string(), "true".to_string());
        }
        
        Ok(())
    }
    
    /// Add security headers
    fn add_security_headers(&self, ctx: &mut RequestContext) -> Result<()> {
        if !self.config.enable_security_headers {
            return Ok(());
        }
        
        for (key, value) in &self.security_headers.headers {
            ctx.headers.insert(key.clone(), value.clone());
        }
        
        Ok(())
    }
}

#[async_trait::async_trait]
impl Middleware for SecurityMiddleware {
    async fn process_request(&self, ctx: &mut RequestContext, body: &mut Vec<u8>) -> Result<()> {
        // Validate IP address
        if !self.validate_ip(&ctx.client_ip).await? {
            return Err(anyhow::anyhow!("IP address not allowed"));
        }
        
        // Check for DDoS
        if !self.check_ddos(&ctx.client_ip).await? {
            return Err(anyhow::anyhow!("DDoS protection triggered"));
        }
        
        // Validate request size
        if !self.validate_request_size(body)? {
            return Err(anyhow::anyhow!("Request too large"));
        }
        
        // Validate input
        if !self.validate_input(body)? {
            return Err(anyhow::anyhow!("Invalid input detected"));
        }
        
        // Handle CORS
        self.handle_cors(ctx)?;
        
        // Add security headers
        self.add_security_headers(ctx)?;
        
        // Add security metadata to context
        ctx.headers.insert("security-validated".to_string(), "true".to_string());
        ctx.headers.insert("security-timestamp".to_string(), 
                         std::time::SystemTime::now()
                             .duration_since(std::time::UNIX_EPOCH)
                             .unwrap_or_default()
                             .as_secs()
                             .to_string());
        
        Ok(())
    }
    
    async fn process_response(&self, _ctx: &mut RequestContext, _response: &mut Vec<u8>) -> Result<()> {
        // Security middleware typically doesn't modify responses
        Ok(())
    }
    
    fn name(&self) -> &str {
        "security_middleware"
    }
}

impl CorsHandler {
    fn new(config: &SecurityConfig) -> Self {
        let allow_all = config.cors_allowed_origins.iter().any(|o| o == "*");
        Self {
            allowed_origins: config.cors_allowed_origins.clone(),
            allow_all,
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
                "X-API-Key".to_string(),
                "X-Requested-With".to_string(),
            ],
            max_age: 86400,
            allow_credentials: false,
        }
    }
    
    fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allow_all || self.allowed_origins.contains(&origin.to_string())
    }
}

impl SecurityHeaders {
    fn new() -> Self {
        let mut headers = HashMap::new();
        
        // Content Security Policy
        headers.insert("Content-Security-Policy".to_string(), 
                       "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'".to_string());
        
        // X-Frame-Options
        headers.insert("X-Frame-Options".to_string(), "DENY".to_string());
        
        // X-Content-Type-Options
        headers.insert("X-Content-Type-Options".to_string(), "nosniff".to_string());
        
        // X-XSS-Protection
        headers.insert("X-XSS-Protection".to_string(), "1; mode=block".to_string());
        
        // Strict-Transport-Security
        headers.insert("Strict-Transport-Security".to_string(), 
                       "max-age=31536000; includeSubDomains".to_string());
        
        // Referrer-Policy
        headers.insert("Referrer-Policy".to_string(), "strict-origin-when-cross-origin".to_string());
        
        // Permissions-Policy
        headers.insert("Permissions-Policy".to_string(), 
                       "geolocation=(), microphone=(), camera=()".to_string());
        
        Self { headers }
    }
}

/// Security statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStatistics {
    pub total_tracked_ips: usize,
    pub whitelisted_ips: usize,
    pub blacklisted_ips: usize,
    pub blocked_ips: usize,
    pub active_blocks: usize,
}

/// Security middleware builder
pub struct SecurityMiddlewareBuilder {
    config: SecurityConfig,
}

impl SecurityMiddlewareBuilder {
    pub fn new() -> Self {
        Self {
            config: SecurityConfig::default(),
        }
    }
    
    pub fn with_ip_whitelist(mut self, enabled: bool) -> Self {
        self.config.enable_ip_whitelist = enabled;
        self
    }
    
    pub fn with_ip_blacklist(mut self, enabled: bool) -> Self {
        self.config.enable_ip_blacklist = enabled;
        self
    }
    
    pub fn with_max_request_size(mut self, size: usize) -> Self {
        self.config.max_request_size = size;
        self
    }
    
    pub fn with_ddos_protection(mut self, enabled: bool, threshold: u32) -> Self {
        self.config.enable_ddos_protection = enabled;
        self.config.ddos_threshold = threshold;
        self
    }
    
    pub fn with_cors(mut self, enabled: bool, origins: Vec<String>) -> Self {
        self.config.enable_cors = enabled;
        self.config.cors_allowed_origins = origins;
        self
    }
    
    pub fn with_security_headers(mut self, enabled: bool) -> Self {
        self.config.enable_security_headers = enabled;
        self
    }
    
    pub fn build(self) -> SecurityMiddleware {
        SecurityMiddleware::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert!(config.enable_ip_blacklist);
        assert!(config.enable_size_limit);
        assert_eq!(config.max_request_size, 10 * 1024 * 1024);
    }
    
    #[test]
    fn test_security_middleware_creation() {
        let config = SecurityConfig::default();
        let middleware = SecurityMiddleware::new(config);
        assert_eq!(middleware.name(), "security_middleware");
    }
    
    #[test]
    fn test_security_middleware_builder() {
        let middleware = SecurityMiddlewareBuilder::new()
            .with_ip_whitelist(true)
            .with_max_request_size(5 * 1024 * 1024)
            .with_ddos_protection(true, 500)
            .build();
        
        // Test that the middleware was built successfully
        assert_eq!(middleware.name(), "security_middleware");
    }
    
    #[test]
    fn test_input_validators() {
        let validators = SecurityMiddleware::create_default_validators();
        assert_eq!(validators.len(), 4);
        
        // Test SQL injection validator
        let sql_validator = &validators[0];
        assert_eq!(sql_validator.name, "sql_injection");
        assert!(sql_validator.pattern.is_match("SELECT * FROM users"));
    }
    
    #[test]
    fn test_cors_handler() {
        let config = SecurityConfig {
            cors_allowed_origins: vec!["https://example.com".to_string()],
            ..Default::default()
        };
        let cors_handler = CorsHandler::new(&config);
        
        assert!(cors_handler.is_origin_allowed("https://example.com"));
        assert!(!cors_handler.is_origin_allowed("https://evil.com"));
    }
    
    #[test]
    fn test_security_headers() {
        let headers = SecurityHeaders::new();
        assert!(headers.headers.contains_key("X-Frame-Options"));
        assert_eq!(headers.headers.get("X-Frame-Options"), Some(&"DENY".to_string()));
    }
    
    #[tokio::test]
    async fn test_ip_whitelist_blacklist() {
        let middleware = SecurityMiddleware::new(SecurityConfig::default());
        
        // Add IP to whitelist
        middleware.add_ip_to_whitelist("192.168.1.1").await.unwrap();
        
        // Add IP to blacklist
        middleware.add_ip_to_blacklist("192.168.1.2").await.unwrap();
        
        // Test validation
        assert!(middleware.validate_ip("192.168.1.1").await.unwrap());
        assert!(!middleware.validate_ip("192.168.1.2").await.unwrap());
        assert!(middleware.validate_ip("192.168.1.3").await.unwrap()); // Not in either list
    }
    
    #[tokio::test]
    async fn test_ddos_protection() {
        let config = SecurityConfig {
            enable_ddos_protection: true,
            ddos_threshold: 5,
            ..Default::default()
        };
        let middleware = SecurityMiddleware::new(config);
        
        let ip = "192.168.1.1";
        
        // First 5 requests should pass (threshold is 5, so fail happens at 6)
        for _ in 0..5 {
            assert!(middleware.check_ddos(ip).await.unwrap());
        }
        
        // Sixth request should trigger DDoS protection
        assert!(!middleware.check_ddos(ip).await.unwrap());
    }
    
    #[test]
    fn test_request_size_validation() {
        let config = SecurityConfig {
            enable_size_limit: true,
            max_request_size: 100,
            ..Default::default()
        };
        let middleware = SecurityMiddleware::new(config);
        
        // Small request should pass
        let small_body = vec![0u8; 50];
        assert!(middleware.validate_request_size(&small_body).unwrap());
        
        // Large request should fail
        let large_body = vec![0u8; 200];
        assert!(!middleware.validate_request_size(&large_body).unwrap());
    }
    
    #[test]
    fn test_input_validation() {
        let middleware = SecurityMiddleware::new(SecurityConfig::default());
        
        // Clean input should pass
        let clean_input = b"Hello, world!";
        assert!(middleware.validate_input(clean_input).unwrap());
        
        // SQL injection should fail
        let sql_input = b"SELECT * FROM users";
        assert!(!middleware.validate_input(sql_input).unwrap());
        
        // XSS should fail
        let xss_input = b"<script>alert('xss')</script>";
        assert!(!middleware.validate_input(xss_input).unwrap());
    }
}
