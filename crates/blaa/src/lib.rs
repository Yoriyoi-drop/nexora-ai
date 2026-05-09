//! BLAA (Black Language Model API) integration for Nexora AI
//! 
//! This crate provides integration with the BLAA API for advanced language model capabilities.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub mod client;
pub mod config;

pub mod auth;

pub use client::BlaaClient;
pub use config::BlaaConfig;

#[derive(Error, Debug)]
pub enum BlaaError {
    #[error("Authentication failed: {0}")]
    Authentication(String),
    
    #[error("API request failed: {0}")]
    ApiRequest(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Invalid response format: {0}")]
    InvalidResponse(String),
}

pub type BlaaResult<T> = Result<T, BlaaError>;

/// BLAA API version information
pub const BLAA_API_VERSION: &str = "v1";
pub const BLAA_BASE_URL: &str = "https://api.blaa.ai";

/// Default configuration values
pub struct DefaultConfig {
    pub timeout: u64,
    pub max_retries: usize,
}

impl Default for DefaultConfig {
    fn default() -> Self {
        Self {
            timeout: 30,
            max_retries: 3,
        }
    }
}

/// Default constants
pub mod defaults {
    pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
    pub const DEFAULT_MAX_RETRIES: u32 = 3;
    pub const DEFAULT_RATE_LIMIT_RPS: u32 = 10;
    pub const DEFAULT_MODEL: &str = "blaa-small";
}

/// API models
pub mod models {
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionRequest {
        pub model: String,
        pub messages: Vec<ChatMessage>,
        pub max_tokens: Option<u32>,
        pub temperature: Option<f32>,
        pub stream: Option<bool>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatMessage {
        pub role: String,
        pub content: String,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<ChatChoice>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatChoice {
        pub index: u32,
        pub message: ChatMessage,
        pub finish_reason: Option<String>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<ChatChoice>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingRequest {
        pub model: String,
        pub input: Vec<String>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        pub object: String,
        pub data: Vec<EmbeddingData>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingData {
        pub object: String,
        pub embedding: Vec<f32>,
        pub index: u32,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelInfo {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub owned_by: String,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BlaaErrorResponse {
        pub error: BlaaErrorInfo,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BlaaErrorInfo {
        pub message: String,
        pub error_type: String,
        pub code: Option<String>,
    }
}

