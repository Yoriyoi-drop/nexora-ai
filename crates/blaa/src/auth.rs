//! Authentication and authorization for BLAA API

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;

use super::{BlaaError, BlaaResult};

type HmacSha256 = Hmac<Sha256>;

/// Authentication method for BLAA API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    /// Bearer token authentication
    Bearer(String),
    
    /// API key authentication
    ApiKey(String),
    
    /// HMAC signature authentication
    HmacSignature {
        api_key: String,
        secret_key: String,
    },
}

impl AuthMethod {
    /// Create bearer token authentication
    pub fn bearer(token: impl Into<String>) -> Self {
        Self::Bearer(token.into())
    }
    
    /// Create API key authentication
    pub fn api_key(key: impl Into<String>) -> Self {
        Self::ApiKey(key.into())
    }
    
    /// Create HMAC signature authentication
    pub fn hmac_signature(api_key: impl Into<String>, secret_key: impl Into<String>) -> Self {
        Self::HmacSignature {
            api_key: api_key.into(),
            secret_key: secret_key.into(),
        }
    }
    
    /// Get authorization header value
    pub fn header_value(&self) -> String {
        match self {
            Self::Bearer(token) => format!("Bearer {}", token),
            Self::ApiKey(key) => format!("Api-Key {}", key),
            Self::HmacSignature { api_key, .. } => format!("Api-Key {}", api_key),
        }
    }
    
    /// Generate HMAC signature for request
    pub fn generate_signature(
        &self,
        method: &str,
        path: &str,
        body: &str,
        timestamp: u64,
    ) -> BlaaResult<String> {
        match self {
            Self::HmacSignature { secret_key, .. } => {
                let _message = format!(
                    "{}\n{}\n{}\n{}",
                    method,
                    path,
                    timestamp,
                    body
                );
                
                let mac = HmacSha256::new_from_slice(secret_key.as_bytes())
                    .map_err(|e| BlaaError::Authentication(
                        format!("Failed to create HMAC: {}", e)
                    ))?;
                
                let signature = mac.finalize().into_bytes();
                Ok(BASE64.encode(signature))
            }
            _ => Ok(String::new()),
        }
    }
    
    /// Get authentication headers for request
    pub fn headers(&self, method: &str, path: &str, body: &str) -> BlaaResult<std::collections::HashMap<String, String>> {
        let mut headers = std::collections::HashMap::new();
        
        match self {
            Self::Bearer(_) | Self::ApiKey(_) => {
                headers.insert("Authorization".to_string(), self.header_value());
            }
            Self::HmacSignature { .. } => {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| BlaaError::Authentication(
                        format!("Failed to get timestamp: {}", e)
                    ))?
                    .as_secs();
                
                let signature = self.generate_signature(method, path, body, timestamp)?;
                
                headers.insert("Authorization".to_string(), self.header_value());
                headers.insert("X-Blaa-Signature".to_string(), signature);
                headers.insert("X-Blaa-Timestamp".to_string(), timestamp.to_string());
            }
        }
        
        Ok(headers)
    }
}

/// Token manager for handling authentication tokens
#[derive(Debug, Clone)]
pub struct TokenManager {
    auth_method: AuthMethod,
    token_cache: Option<String>,
    expires_at: Option<std::time::SystemTime>,
}

impl TokenManager {
    /// Create new token manager
    pub fn new(auth_method: AuthMethod) -> Self {
        Self {
            auth_method,
            token_cache: None,
            expires_at: None,
        }
    }
    
    /// Get current authentication token
    ///
    /// Note: Caching is meaningful only for HMAC tokens (which have a 1-hour expiry).
    /// For Bearer and ApiKey tokens the value never changes, so caching is a no-op
    /// but kept for interface uniformity.
    pub async fn get_token(&mut self) -> BlaaResult<String> {
        // Check if cached token is still valid
        if let (Some(token), Some(expires_at)) = (&self.token_cache, &self.expires_at) {
            if std::time::SystemTime::now() < *expires_at {
                debug!("Using cached authentication token");
                return Ok(token.clone());
            }
        }
        
        // Generate new token
        let token = match &self.auth_method {
            AuthMethod::Bearer(token) => token.clone(),
            AuthMethod::ApiKey(key) => key.clone(),
            AuthMethod::HmacSignature { api_key, .. } => api_key.clone(),
        };
        
        // Cache the token (expires in 1 hour for HMAC, never for Bearer/ApiKey)
        let expires_at = match &self.auth_method {
            AuthMethod::HmacSignature { .. } => {
                Some(std::time::SystemTime::now() + std::time::Duration::from_secs(3600))
            }
            _ => None,
        };
        
        self.token_cache = Some(token.clone());
        self.expires_at = expires_at;
        
        Ok(token)
    }
    
    /// Refresh authentication token
    pub async fn refresh_token(&mut self) -> BlaaResult<()> {
        debug!("Refreshing authentication token");
        self.token_cache = None;
        self.expires_at = None;
        let _ = self.get_token().await?;
        Ok(())
    }
    
    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = &self.expires_at {
            std::time::SystemTime::now() >= *expires_at
        } else {
            false
        }
    }
    
    /// Get authentication headers for request
    pub fn get_headers(&self, method: &str, path: &str, body: &str) -> BlaaResult<std::collections::HashMap<String, String>> {
        self.auth_method.headers(method, path, body)
    }
}

/// Rate limiter for API requests
#[derive(Debug, Clone)]
pub struct RateLimiter {
    requests_per_second: u32,
    last_request_times: std::collections::VecDeque<std::time::Instant>,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(requests_per_second: u32) -> Self {
        Self {
            requests_per_second,
            last_request_times: std::collections::VecDeque::new(),
        }
    }
    
    /// Check if request is allowed and wait if necessary
    pub async fn acquire(&mut self) {
        let now = std::time::Instant::now();
        
        // Remove old requests (older than 1 second)
        while let Some(&front_time) = self.last_request_times.front() {
            if now.duration_since(front_time) >= std::time::Duration::from_secs(1) {
                self.last_request_times.pop_front();
            } else {
                break;
            }
        }
        
        // Check if we've hit the rate limit
        if self.last_request_times.len() >= self.requests_per_second as usize {
            if let Some(&oldest_time) = self.last_request_times.front() {
                let wait_time = std::time::Duration::from_secs(1)
                    .saturating_sub(now.duration_since(oldest_time));
                
                if wait_time > std::time::Duration::ZERO {
                    debug!("Rate limit reached, waiting {:?}", wait_time);
                    tokio::time::sleep(wait_time).await;
                }
            }
        }
        
        // Record this request
        self.last_request_times.push_back(now);
    }
    
    /// Get current rate limit status
    pub fn status(&self) -> (usize, u32) {
        let current_requests = self.last_request_times.len();
        let remaining = self.requests_per_second.saturating_sub(current_requests as u32);
        (current_requests, remaining)
    }
}
