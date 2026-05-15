//! HTTP client for API requests

use std::sync::Arc;
use anyhow::Result;
use serde_json::Value;
use tracing::{info, warn};
use reqwest::{Client, Response};

use crate::NexoraAI;
use super::config::{ApiConfig, HttpClientConfig, ApiResponse};
use super::types::*;

/// HTTP client for making API requests
#[derive(Debug, Clone)]
pub struct ApiClient {
    client: Client,
    config: ApiConfig,
    #[allow(dead_code)]
    nexora: Arc<NexoraAI>,
}

impl ApiClient {
    /// Create new API client
    pub fn new(nexora: Arc<NexoraAI>, config: ApiConfig) -> Result<Self> {
        let http_config = HttpClientConfig::default();
        
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .user_agent(&http_config.user_agent)
            .build()?;
        
        Ok(Self {
            client,
            config,
            nexora,
        })
    }
    
    /// Make HTTP request with retry logic
    async fn make_request(&self, method: reqwest::Method, endpoint: &str, body: Option<Value>) -> Result<Response> {
        let url = format!("{}/{}", self.config.base_url.trim_end_matches('/'), endpoint.trim_start_matches('/'));
        
        let mut request = self.client.request(method, &url);
        
        // Add API key if present
        if let Some(api_key) = &self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        
        // Add JSON body if present
        if let Some(body) = body {
            request = request.json(&body);
        }
        
        // Retry logic
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            match request.try_clone()
                .ok_or_else(|| anyhow::anyhow!("Failed to clone request for retry"))?
                .send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(response);
                    } else if response.status().is_server_error() && attempt < self.config.max_retries {
                        warn!("Request failed (attempt {}/{}): {}, retrying...", attempt, self.config.max_retries, response.status());
                        tokio::time::sleep(std::time::Duration::from_millis(1000 * attempt as u64)).await;
                        continue;
                    } else {
                        last_error = Some(anyhow::anyhow!("HTTP error: {}", response.status()));
                        break;
                    }
                }
                Err(e) => {
                    if attempt < self.config.max_retries {
                        warn!("Request failed (attempt {}/{}): {}, retrying...", attempt, self.config.max_retries, e);
                        tokio::time::sleep(std::time::Duration::from_millis(1000 * attempt as u64)).await;
                    } else {
                        last_error = Some(anyhow::anyhow!("Request failed: {}", e));
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Request failed")))
    }
    
    /// Process request
    pub async fn process_request(&self, request: ProcessRequest) -> Result<ProcessResponse> {
        info!("Processing API request: {}", request.input);
        
        let body = serde_json::to_value(request)?;
        let response = self.make_request(reqwest::Method::POST, "process", Some(body)).await?;
        
        let api_response: ApiResponse<ProcessResponse> = response.json().await?;
        
        if let Some(data) = api_response.data {
            Ok(data)
        } else {
            Err(anyhow::anyhow!("Process request failed: {:?}", api_response.error))
        }
    }
    
    /// Generate text
    pub async fn generate_text(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        info!("Generating text: {} chars", request.prompt.len());
        
        let body = serde_json::to_value(request)?;
        let response = self.make_request(reqwest::Method::POST, "generate", Some(body)).await?;
        
        let api_response: ApiResponse<GenerateResponse> = response.json().await?;
        
        if let Some(data) = api_response.data {
            Ok(data)
        } else {
            Err(anyhow::anyhow!("Generate request failed: {:?}", api_response.error))
        }
    }
    
    /// Chat
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        info!("Chat request: {} (conversation_id: {:?})", request.message, request.conversation_id);
        
        let body = serde_json::to_value(request)?;
        let response = self.make_request(reqwest::Method::POST, "chat", Some(body)).await?;
        
        let api_response: ApiResponse<ChatResponse> = response.json().await?;
        
        if let Some(data) = api_response.data {
            Ok(data)
        } else {
            Err(anyhow::anyhow!("Chat request failed: {:?}", api_response.error))
        }
    }
    
    /// Analyze code
    pub async fn analyze_code(&self, request: AnalyzeCodeRequest) -> Result<AnalyzeCodeResponse> {
        info!("Analyzing code: {} chars, language: {}", request.code.len(), request.language);
        
        let body = serde_json::to_value(request)?;
        let response = self.make_request(reqwest::Method::POST, "code/analyze", Some(body)).await?;
        
        let api_response: ApiResponse<AnalyzeCodeResponse> = response.json().await?;
        
        if let Some(data) = api_response.data {
            Ok(data)
        } else {
            Err(anyhow::anyhow!("Code analysis request failed: {:?}", api_response.error))
        }
    }
    
    /// Generate code
    pub async fn generate_code(&self, request: GenerateCodeRequest) -> Result<GenerateCodeResponse> {
        info!("Generating code: {} in {}", request.description, request.language);
        
        let body = serde_json::to_value(request)?;
        let response = self.make_request(reqwest::Method::POST, "code/generate", Some(body)).await?;
        
        let api_response: ApiResponse<GenerateCodeResponse> = response.json().await?;
        
        if let Some(data) = api_response.data {
            Ok(data)
        } else {
            Err(anyhow::anyhow!("Code generation request failed: {:?}", api_response.error))
        }
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<Value> {
        info!("Performing health check");
        
        let response = self.make_request(reqwest::Method::GET, "health", None).await?;
        let health_data: Value = response.json().await?;
        
        Ok(health_data)
    }
    
    /// Get system information
    pub async fn get_system_info(&self) -> Result<Value> {
        info!("Getting system information");
        
        let response = self.make_request(reqwest::Method::GET, "info", None).await?;
        let info_data: Value = response.json().await?;
        
        Ok(info_data)
    }
}
