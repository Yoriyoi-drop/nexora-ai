//! Configuration for BLAA API integration

use serde::{Deserialize, Serialize};
use std::time::Duration;
use anyhow::Result;

use super::{BlaaError, BlaaResult, defaults};

/// Configuration for BLAA API client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlaaConfig {
    /// API key for authentication
    pub api_key: String,
    
    /// Base URL for BLAA API
    #[serde(default = "default_base_url")]
    pub base_url: String,
    
    /// API version
    #[serde(default = "default_api_version")]
    pub api_version: String,
    
    /// Default model to use
    #[serde(default = "default_model")]
    pub default_model: String,
    
    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    
    /// Maximum number of retries for failed requests
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    
    /// Rate limit requests per second
    #[serde(default = "default_rate_limit")]
    pub rate_limit_rps: u32,
    
    /// Organization ID (if applicable)
    pub organization_id: Option<String>,
    
    /// Custom headers to include in requests
    #[serde(default)]
    pub custom_headers: std::collections::HashMap<String, String>,
}

impl Default for BlaaConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: default_base_url(),
            api_version: default_api_version(),
            default_model: default_model(),
            timeout_secs: defaults::DEFAULT_TIMEOUT_SECS,
            max_retries: defaults::DEFAULT_MAX_RETRIES,
            rate_limit_rps: defaults::DEFAULT_RATE_LIMIT_RPS,
            organization_id: None,
            custom_headers: std::collections::HashMap::new(),
        }
    }
}

impl BlaaConfig {
    /// Create new configuration from API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }
    
    /// Create configuration from environment variables
    pub fn from_env() -> BlaaResult<Self> {
        let api_key = std::env::var("BLAA_API_KEY")
            .map_err(|_| BlaaError::InvalidConfiguration(
                "BLAA_API_KEY environment variable not set".to_string()
            ))?;
        
        let mut config = Self::new(api_key);
        
        if let Ok(base_url) = std::env::var("BLAA_BASE_URL") {
            config.base_url = base_url;
        }
        
        if let Ok(api_version) = std::env::var("BLAA_API_VERSION") {
            config.api_version = api_version;
        }
        
        if let Ok(default_model) = std::env::var("BLAA_DEFAULT_MODEL") {
            config.default_model = default_model;
        }
        
        if let Ok(timeout) = std::env::var("BLAA_TIMEOUT_SECS") {
            config.timeout_secs = timeout.parse()
                .map_err(|_| BlaaError::InvalidConfiguration(
                    "Invalid BLAA_TIMEOUT_SECS value".to_string()
                ))?;
        }
        
        if let Ok(max_retries) = std::env::var("BLAA_MAX_RETRIES") {
            config.max_retries = max_retries.parse()
                .map_err(|_| BlaaError::InvalidConfiguration(
                    "Invalid BLAA_MAX_RETRIES value".to_string()
                ))?;
        }
        
        if let Ok(rate_limit) = std::env::var("BLAA_RATE_LIMIT_RPS") {
            config.rate_limit_rps = rate_limit.parse()
                .map_err(|_| BlaaError::InvalidConfiguration(
                    "Invalid BLAA_RATE_LIMIT_RPS value".to_string()
                ))?;
        }
        
        if let Ok(org_id) = std::env::var("BLAA_ORGANIZATION_ID") {
            config.organization_id = Some(org_id);
        }
        
        Ok(config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> BlaaResult<()> {
        if self.api_key.is_empty() {
            return Err(BlaaError::InvalidConfiguration(
                "API key cannot be empty".to_string()
            ));
        }
        
        if self.base_url.is_empty() {
            return Err(BlaaError::InvalidConfiguration(
                "Base URL cannot be empty".to_string()
            ));
        }
        
        if self.timeout_secs == 0 {
            return Err(BlaaError::InvalidConfiguration(
                "Timeout must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Get request timeout as Duration
    pub fn timeout_duration(&self) -> Duration {
        Duration::from_secs(self.timeout_secs)
    }
    
    /// Build full API endpoint URL
    pub fn endpoint_url(&self, path: &str) -> String {
        format!("{}/{}/{}", self.base_url.trim_end_matches('/'), self.api_version, path.trim_start_matches('/'))
    }
}

// Default value functions
fn default_base_url() -> String {
    super::BLAA_BASE_URL.to_string()
}

fn default_api_version() -> String {
    super::BLAA_API_VERSION.to_string()
}

fn default_model() -> String {
    defaults::DEFAULT_MODEL.to_string()
}

fn default_timeout() -> u64 {
    defaults::DEFAULT_TIMEOUT_SECS
}

fn default_max_retries() -> u32 {
    defaults::DEFAULT_MAX_RETRIES
}

fn default_rate_limit() -> u32 {
    defaults::DEFAULT_RATE_LIMIT_RPS
}
