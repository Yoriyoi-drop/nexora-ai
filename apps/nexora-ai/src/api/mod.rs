//! REST API module for Nexora-AI

pub mod config;
pub mod types;
pub mod client;
pub mod rate_limiter;

/// Main Nexora API client
pub struct NexoraApi {
    config: ApiConfig,
    client: reqwest::Client,
}

impl NexoraApi {
    pub fn new(config: ApiConfig) -> Result<Self, anyhow::Error> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()?;
        
        Ok(Self { config, client })
    }
    
    pub async fn process_text(&self, text: &str) -> Result<ApiResponse<String>, anyhow::Error> {
        // Validate input
        if text.is_empty() {
            return Err(anyhow::anyhow!("Input text cannot be empty"));
        }
        
        if text.len() > 10000 {
            return Err(anyhow::anyhow!("Input text too long (max 10000 characters)"));
        }
        
        let start_time = std::time::Instant::now();
        
        // Build request to internal processing endpoint
        let request_body = serde_json::json!({
            "input": text,
            "model": "default",
            "max_tokens": 1000,
            "temperature": 0.7
        });
        
        let url = format!("{}/process", self.config.base_url);
        let response = self.client
            .post(&url)
            .json(&request_body)
            .header("Content-Type", "application/json")
            .header("User-Agent", "nexora-api-client/1.0")
            .send()
            .await?;
        
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("API request failed with status {}: {}", status, error_text));
        }
        
        let response_text = response.text().await?;
        let _processing_time = start_time.elapsed();
        
        // Parse response
        let parsed_response: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| anyhow::anyhow!("Failed to parse API response: {}", e))?;
        
        let generated_text = parsed_response.get("response")
            .and_then(|v| v.as_str())
            .unwrap_or("No response generated")
            .to_string();
        
        Ok(ApiResponse {
            success: true,
            data: Some(generated_text),
            error: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }
}

// Re-export main API types
pub use config::{ApiConfig, ApiResponse, RateLimitConfig, HttpClientConfig};
pub use types::*;
pub use client::ApiClient;
pub use rate_limiter::{RateLimiter, RateLimitStatus, RateLimitStats};
