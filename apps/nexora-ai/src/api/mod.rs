//! REST API module for Nexora-AI

pub mod types;
pub mod client;
pub mod rate_limiter;

pub use crate::config::api::{ApiConfig, ApiResponse as ApiResp, RateLimitConfig, HttpClientConfig};
pub use types::*;
pub use client::ApiClient;
pub use rate_limiter::{RateLimiter, RateLimitStatus, RateLimitStats};
pub use crate::config::api::ApiResponse;

pub struct NexoraApi {
    client: ApiClient,
}

impl NexoraApi {
    pub fn new(config: ApiConfig) -> Result<Self, anyhow::Error> {
        Ok(Self {
            client: ApiClient::new(config)?,
        })
    }

    pub async fn process_text(&self, text: &str) -> Result<ApiResp<serde_json::Value>, anyhow::Error> {
        let response = self.client.make_request(
            reqwest::Method::POST,
            "process",
            Some(serde_json::json!({"text": text})),
        ).await?;
        let body = response.text().await?;
        if body.len() > 50_000_000 {
            return Err(anyhow::anyhow!("API response too large: {} bytes", body.len()));
        }
        Ok(serde_json::from_str(&body)?)
    }
}
