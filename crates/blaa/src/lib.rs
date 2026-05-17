//! BLAA (Black Language Model API) integration for Nexora AI
//! 
//! This crate provides integration with the BLAA API for advanced language model capabilities.

use thiserror::Error;

pub mod client;
pub mod config;

pub mod auth;

pub use client::BlaaClient;
pub use config::BlaaConfig;
pub use models::*;

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

/// Default constants
pub mod defaults {
    pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
    pub const DEFAULT_MAX_RETRIES: u32 = 3;
    pub const DEFAULT_RATE_LIMIT_RPS: u32 = 10;
    pub const DEFAULT_MODEL: &str = "blaa-small";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        assert_eq!(defaults::DEFAULT_TIMEOUT_SECS, 30);
        assert_eq!(defaults::DEFAULT_MAX_RETRIES, 3);
    }

    #[test]
    fn test_defaults_constants() {
        assert_eq!(defaults::DEFAULT_TIMEOUT_SECS, 30);
        assert_eq!(defaults::DEFAULT_MODEL, "blaa-small");
    }

    #[test]
    fn test_blaa_error_display() {
        let err = BlaaError::Authentication("bad key".into());
        assert_eq!(format!("{}", err), "Authentication failed: bad key");

        let err = BlaaError::InvalidConfiguration("missing field".into());
        assert_eq!(format!("{}", err), "Invalid configuration: missing field");
    }

    #[test]
    fn test_api_constants() {
        assert_eq!(BLAA_API_VERSION, "v1");
        assert!(!BLAA_BASE_URL.is_empty());
    }

    #[test]
    fn test_chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: "hello".into(),
            name: None,
            function_call: None,
            tool_calls: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content, "hello");
        assert!(matches!(deserialized.role, MessageRole::User));
    }

    #[test]
    fn test_chat_completion_request_builder() {
        let req = ChatCompletionRequest {
            model: "test-model".into(),
            messages: vec![ChatMessage {
                role: MessageRole::System,
                content: "be helpful".into(),
                name: None,
                function_call: None,
                tool_calls: None,
            }],
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            system: None,
        };
        let json = serde_json::to_string_pretty(&req).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains("be helpful"));
    }

    #[test]
    fn test_embedding_input_serde() {
        let single = EmbeddingInput::Single("hello".into());
        let json = serde_json::to_string(&single).unwrap();
        assert_eq!(json, "\"hello\"");

        let multiple = EmbeddingInput::Multiple(vec!["a".into(), "b".into()]);
        let json = serde_json::to_string(&multiple).unwrap();
        assert!(json.contains("a"));
    }

    #[test]
    fn test_model_info_deserialization() {
        let json = r#"{
            "id": "blaa-small",
            "object": "model",
            "object_type": "language",
            "created": 1700000000,
            "owned_by": "blaa"
        }"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.id, "blaa-small");
        assert_eq!(info.owned_by, "blaa");
    }

    #[test]
    fn test_stop_sequence_serde() {
        let single = StopSequence::Single("STOP".into());
        let json = serde_json::to_string(&single).unwrap();
        assert_eq!(json, "\"STOP\"");

        let multi = StopSequence::Multiple(vec!["STOP".into(), "END".into()]);
        let json = serde_json::to_string(&multi).unwrap();
        assert!(json.contains("END"));
    }
}

/// API models
pub mod models {
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum MessageRole {
        System,
        User,
        Assistant,
        Tool,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionRequest {
        pub model: String,
        pub messages: Vec<ChatMessage>,
        pub max_tokens: Option<u32>,
        pub temperature: Option<f32>,
        pub top_p: Option<f32>,
        pub n: Option<u32>,
        pub stream: Option<bool>,
        pub stop: Option<StopSequence>,
        pub presence_penalty: Option<f32>,
        pub frequency_penalty: Option<f32>,
        pub logit_bias: Option<HashMap<String, f32>>,
        pub user: Option<String>,
        pub system: Option<String>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatMessage {
        pub role: MessageRole,
        pub content: String,
        pub name: Option<String>,
        pub function_call: Option<serde_json::Value>,
        pub tool_calls: Option<Vec<serde_json::Value>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum StopSequence {
        Single(String),
        Multiple(Vec<String>),
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<ChatChoice>,
        pub usage: Option<Usage>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatChoice {
        pub index: u32,
        pub message: ChatMessage,
        pub finish_reason: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatDelta {
        pub role: Option<MessageRole>,
        pub content: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatChunkChoice {
        pub index: u32,
        pub delta: ChatDelta,
        pub finish_reason: Option<String>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<ChatChunkChoice>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Usage {
        pub prompt_tokens: u32,
        pub completion_tokens: u32,
        pub total_tokens: u32,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingRequest {
        pub model: String,
        pub input: EmbeddingInput,
        pub user: Option<String>,
        pub encoding_format: Option<EmbeddingFormat>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum EmbeddingInput {
        Single(String),
        Multiple(Vec<String>),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum EmbeddingFormat {
        Float,
        Base64,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        pub object: String,
        pub data: Vec<EmbeddingData>,
        pub model: Option<String>,
        pub usage: Option<Usage>,
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
        pub object_type: String,
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
