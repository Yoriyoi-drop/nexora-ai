//! BLAA API client implementation

use futures::Stream;
use reqwest::{Client, RequestBuilder, Response};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::time::{timeout, Duration};
use tracing::{debug, warn};
use uuid::Uuid;
use bytes::Bytes;

use super::{
    auth::{AuthMethod, RateLimiter, TokenManager},
    BlaaConfig, BlaaError, BlaaResult,
};
use super::models::*;

/// BLAA API client
#[derive(Debug, Clone)]
pub struct BlaaClient {
    config: BlaaConfig,
    http_client: Client,
    token_manager: TokenManager,
    rate_limiter: RateLimiter,
}

impl BlaaClient {
    /// Create new BLAA client with configuration
    pub fn new(config: BlaaConfig) -> BlaaResult<Self> {
        config.validate()?;
        
        let http_client = Client::builder()
            .timeout(config.timeout_duration())
            .user_agent(format!("nexora-blaa/{}", env!("CARGO_PKG_VERSION")))
            .build()?;
        
        let auth_method = AuthMethod::api_key(&config.api_key);
        let token_manager = TokenManager::new(auth_method);
        let rate_limiter = RateLimiter::new(config.rate_limit_rps);
        
        Ok(Self {
            config,
            http_client,
            token_manager,
            rate_limiter,
        })
    }
    
    /// Create client from environment variables
    pub async fn from_env() -> BlaaResult<Self> {
        let config = BlaaConfig::from_env()?;
        Self::new(config)
    }
    
    /// Get configuration reference
    pub fn config(&self) -> &BlaaConfig {
        &self.config
    }
    
    /// Create chat completion
    pub async fn create_chat_completion(&mut self, request: ChatCompletionRequest) -> BlaaResult<ChatCompletionResponse> {
        let endpoint = self.config.endpoint_url("chat/completions");
        let response = self
            .post(&endpoint, &request)
            .await?;
        
        let completion: ChatCompletionResponse = response.json().await?;
        Ok(completion)
    }
    
    /// Create streaming chat completion
    pub async fn create_chat_completion_stream(&mut self, request: ChatCompletionRequest) -> BlaaResult<impl Stream<Item = BlaaResult<ChatCompletionChunk>>> {
        let mut streaming_request = request.clone();
        streaming_request.stream = Some(true);
        
        let endpoint = self.config.endpoint_url("chat/completions");
        let response = self
            .post(&endpoint, &streaming_request)
            .await?;
        
        Ok(ChatCompletionStream::new(response))
    }
    
    /// Create embeddings
    pub async fn create_embeddings(&mut self, request: EmbeddingRequest) -> BlaaResult<EmbeddingResponse> {
        let endpoint = self.config.endpoint_url("embeddings");
        let response = self
            .post(&endpoint, &request)
            .await?;
        
        let embedding: EmbeddingResponse = response.json().await?;
        Ok(embedding)
    }
    
    /// List available models
    pub async fn list_models(&mut self) -> BlaaResult<Vec<ModelInfo>> {
        let endpoint = self.config.endpoint_url("models");
        let response = self.get(&endpoint).await?;
        
        let models_response: ModelsResponse = response.json().await?;
        Ok(models_response.data)
    }
    
    /// Get model information
    pub async fn get_model(&mut self, model_id: &str) -> BlaaResult<ModelInfo> {
        let endpoint = self.config.endpoint_url(&format!("models/{}", model_id));
        let response = self.get(&endpoint).await?;
        
        let model: ModelInfo = response.json().await?;
        Ok(model)
    }
    
    /// Internal POST request method
    async fn post<T: Serialize>(&mut self, url: &str, body: &T) -> BlaaResult<Response> {
        self.rate_limiter.acquire().await;
        
        let request_body = serde_json::to_string(body)?;
        let mut request = self.http_client.post(url).json(body);
        
        request = self.add_auth_headers(request, "POST", url, &request_body).await?;
        request = self.add_custom_headers(request);
        
        self.execute_request(request).await
    }
    
    /// Internal GET request method
    async fn get(&mut self, url: &str) -> BlaaResult<Response> {
        self.rate_limiter.acquire().await;
        
        let mut request = self.http_client.get(url);
        
        request = self.add_auth_headers(request, "GET", url, "").await?;
        request = self.add_custom_headers(request);
        
        self.execute_request(request).await
    }
    
    /// Add authentication headers to request
    async fn add_auth_headers(
        &mut self,
        mut request: RequestBuilder,
        method: &str,
        path: &str,
        body: &str,
    ) -> BlaaResult<RequestBuilder> {
        let _token = self.token_manager.get_token().await?;
        let auth_headers = self.token_manager.get_headers(method, path, body)?;
        
        for (key, value) in auth_headers {
            request = request.header(&key, &value);
        }
        
        Ok(request)
    }
    
    /// Add custom headers from configuration
    fn add_custom_headers(&self, mut request: RequestBuilder) -> RequestBuilder {
        for (key, value) in &self.config.custom_headers {
            request = request.header(key, value);
        }
        
        // Add organization header if present
        if let Some(org_id) = &self.config.organization_id {
            request = request.header("BLAA-Organization", org_id);
        }
        
        // Add request ID for tracing
        request = request.header("X-Request-ID", Uuid::new_v4().to_string());
        
        request
    }
    
    /// Execute HTTP request with retry logic
    async fn execute_request(&self, request: RequestBuilder) -> BlaaResult<Response> {
        let mut last_error = None;
        
        for attempt in 1..=self.config.max_retries {
            debug!("Executing request (attempt {}/{})", attempt, self.config.max_retries);
            
            // Clone the request for each attempt to avoid move issues
            let request_clone = request.try_clone()
                .ok_or_else(|| BlaaError::ApiRequest("Failed to clone request for retry".to_string()))?;
            
            match timeout(self.config.timeout_duration(), request_clone.send()).await {
                Ok(Ok(response)) => {
                    let status = response.status();
                    
                    if status.is_success() {
                        debug!("Request successful: {}", status);
                        return Ok(response);
                    } else if status.is_client_error() {
                        let error_text = response.text().await.unwrap_or_default();
                        let error = self.parse_api_error(&error_text);
                        
                        // Don't retry on client errors (4xx)
                        return Err(error);
                    } else if status.is_server_error() {
                        let error_text = response.text().await.unwrap_or_default();
                        let error = BlaaError::ApiRequest(format!(
                            "Server error {}: {}",
                            status,
                            error_text
                        ));
                        
                        if attempt < self.config.max_retries {
                            warn!("Server error (attempt {}): {}", attempt, error);
                            let delay = Duration::from_millis(1000 * attempt as u64);
                            tokio::time::sleep(delay).await;
                            last_error = Some(error);
                            continue;
                        } else {
                            return Err(error);
                        }
                    } else {
                        let error = BlaaError::ApiRequest(format!(
                            "Unexpected status: {}",
                            status
                        ));
                        return Err(error);
                    }
                }
                Ok(Err(e)) => {
                    let error = BlaaError::Network(e);
                    
                    if attempt < self.config.max_retries {
                        warn!("Network error (attempt {}): {}", attempt, error);
                        let delay = Duration::from_millis(1000 * attempt as u64);
                        tokio::time::sleep(delay).await;
                        last_error = Some(error);
                        continue;
                    } else {
                        return Err(error);
                    }
                }
                Err(_) => {
                    let error = BlaaError::ApiRequest("Request timeout".to_string());
                    
                    if attempt < self.config.max_retries {
                        warn!("Timeout error (attempt {}): {}", attempt, error);
                        let delay = Duration::from_millis(1000 * attempt as u64);
                        tokio::time::sleep(delay).await;
                        last_error = Some(error);
                        continue;
                    } else {
                        return Err(error);
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| {
            BlaaError::ApiRequest("All retry attempts failed".to_string())
        }))
    }
    
    /// Parse API error response
    fn parse_api_error(&self, error_text: &str) -> BlaaError {
        match serde_json::from_str::<BlaaErrorResponse>(error_text) {
            Ok(error_response) => {
                let error_info = error_response.error;
                
                match error_info.error_type.as_str() {
                    "authentication_error" => {
                        BlaaError::Authentication(error_info.message)
                    }
                    "rate_limit_error" => {
                        BlaaError::RateLimit(error_info.message)
                    }
                    "invalid_request_error" => {
                        BlaaError::InvalidConfiguration(error_info.message)
                    }
                    _ => {
                        BlaaError::ApiRequest(format!(
                            "API error: {}",
                            error_info.message
                        ))
                    }
                }
            }
            Err(_) => {
                // If we can't parse the error, return the raw text
                BlaaError::ApiRequest(format!("Unknown API error: {}", error_text))
            }
        }
    }
}

/// Streaming chat completion response
pub struct ChatCompletionStream {
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buffer: Vec<u8>,
}

impl ChatCompletionStream {
    fn new(response: Response) -> Self {
        let stream = Box::pin(response.bytes_stream());
        Self {
            stream,
            buffer: Vec::new(),
        }
    }
}

impl Stream for ChatCompletionStream {
    type Item = BlaaResult<ChatCompletionChunk>;
    
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            match self.stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);

                    if let Some(line_end) = self.buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = self.buffer.drain(..=line_end).collect();
                        let line = String::from_utf8(line_bytes)
                            .map_err(|e| BlaaError::InvalidResponse(
                                format!("Invalid UTF-8 in SSE stream: {}", e)
                            ))?;
                        let line = line.trim().to_string();

                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data.trim() == "[DONE]" {
                                continue;
                            }
                            match serde_json::from_str::<ChatCompletionChunk>(data.trim()) {
                                Ok(chunk) => return Poll::Ready(Some(Ok(chunk))),
                                Err(_) => debug!("Skipping non-chunk SSE line: {}", data.trim()),
                            }
                        } else {
                            continue;
                        }
                    } else {
                        return Poll::Pending;
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(BlaaError::ApiRequest(e.to_string()))));
                }
                Poll::Ready(None) => {
                    if !self.buffer.is_empty() {
                        let remaining = std::mem::take(&mut self.buffer);
                        let line = String::from_utf8(remaining)
                            .map_err(|e| BlaaError::InvalidResponse(
                                format!("Invalid UTF-8 in SSE stream: {}", e)
                            ))?;
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                return Poll::Ready(None);
                            }
                            match serde_json::from_str::<ChatCompletionChunk>(data.trim()) {
                                Ok(chunk) => return Poll::Ready(Some(Ok(chunk))),
                                Err(_) => debug!("Skipping non-chunk SSE line in buffer: {}", data.trim()),
                            }
                        }
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Models list response
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
    object: String,
}
