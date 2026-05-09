//! BLAA integration untuk inference engine

use anyhow::Result;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{InferenceEngine, InferenceRequest, InferenceResponse, InferenceError, FinishReason, GeneratedToken, Result as InferenceResult};
use nexora_blaa::{BlaaClient, BlaaConfig, ChatMessage, MessageRole, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, EmbeddingRequest, Usage};

/// BLAA integration untuk inference engine
#[derive(Debug, Clone)]
pub struct BlaaInferenceEngine {
    client: Arc<Mutex<BlaaClient>>,
    default_model: String,
    max_tokens: u32,
}

impl BlaaInferenceEngine {
    /// Create new BLAA inference engine
    pub async fn new(config: BlaaConfig) -> InferenceResult<Self> {
        let client = BlaaClient::new(config)?;
        let default_model = client.config().default_model.clone();
        
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            default_model,
            max_tokens: 4096,
        })
    }
    
    /// Create from environment variables
    pub async fn from_env() -> InferenceResult<Self> {
        let client = BlaaClient::from_env().await?;
        let default_model = client.config().default_model.clone();
        
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            default_model,
            max_tokens: 4096,
        })
    }
    
    /// Get BLAA client reference
    pub fn client(&self) -> Arc<Mutex<BlaaClient>> {
        self.client.clone()
    }
    
    /// Set default model
    pub fn with_default_model(mut self, model: String) -> Self {
        self.default_model = model;
        self
    }
    
    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
    
    /// Convert inference request to BLAA chat completion request
    fn to_blaa_request(&self, request: &InferenceRequest) -> ChatCompletionRequest {
        let mut messages = Vec::new();
        
        // Add system prompt if present in metadata
        if let Some(system_prompt) = request.metadata.get("system_prompt") {
            if let Some(system_text) = system_prompt.as_str() {
                messages.push(ChatMessage {
                    role: MessageRole::System,
                    content: system_text.to_string(),
                    name: None,
                    function_call: None,
                    tool_calls: None,
                });
            }
        }
        
        // Add user prompt
        messages.push(ChatMessage {
            role: MessageRole::User,
            content: request.prompt.clone(),
            name: None,
            function_call: None,
            tool_calls: None,
        });
        
        // Add conversation history if present
        if let Some(history) = request.metadata.get("conversation_history") {
            if let Some(history_array) = history.as_array() {
                for history_item in history_array {
                    if let Some(history_obj) = history_item.as_object() {
                        let role_str = history_obj.get("role")
                            .and_then(|r| r.as_str())
                            .unwrap_or("user");
                        
                        let content = history_obj.get("content")
                            .and_then(|c| c.as_str())
                            .unwrap_or("");
                        
                        let role = match role_str {
                            "system" => MessageRole::System,
                            "assistant" => MessageRole::Assistant,
                            _ => MessageRole::User,
                        };
                        
                        messages.push(ChatMessage {
                            role,
                            content: content.to_string(),
                            name: None,
                            function_call: None,
                            tool_calls: None,
                        });
                    }
                }
            }
        }
        
        ChatCompletionRequest {
            messages,
            model: request.model_id.clone(),
            max_tokens: Some(request.max_tokens.min(self.max_tokens)),
            temperature: Some(request.temperature),
            top_p: Some(request.top_p),
            n: Some(1),
            stream: Some(request.streaming),
            stop: if request.stop_sequences.is_empty() {
                None
            } else {
                Some(nexora_blaa::StopSequence::Multiple(request.stop_sequences.clone()))
            },
            presence_penalty: Some(request.presence_penalty),
            frequency_penalty: Some(request.frequency_penalty),
            logit_bias: None,
            user: Some(request.request_id.to_string()),
            system: None, // Already handled in messages
        }
    }
    
    /// Convert BLAA response to inference response
    fn from_blaa_response(&self, request: &InferenceRequest, blaa_response: ChatCompletionResponse) -> InferenceResponse {
        let mut response = InferenceResponse::new(request.request_id);
        
        if let Some(choice) = blaa_response.choices.first() {
            let content = &choice.message.content;
            
            // Generate tokens from content (simplified - in real implementation would use tokenizer)
            let tokens: Vec<GeneratedToken> = content
                .chars()
                .enumerate()
                .map(|(i, c)| GeneratedToken::new(
                    i as u32,
                    c.to_string(),
                    -1.0, // Default log prob
                    i,
                ))
                .collect();
            
            response.text = content.clone();
            response.tokens = tokens;
            response.total_tokens = content.chars().count();
            
            // Map finish reason
            response.finish_reason = match choice.finish_reason.as_deref() {
                Some("length") => FinishReason::MaxTokens,
                Some("stop") => FinishReason::StopSequence,
                Some("content_filter") => FinishReason::Error("Content filtered".to_string()),
                _ => FinishReason::EndOfSequence,
            };
            
            // Add usage information
            if let Some(usage) = &blaa_response.usage {
                response.metadata.insert("prompt_tokens".to_string(), json!(usage.prompt_tokens));
                response.metadata.insert("completion_tokens".to_string(), json!(usage.completion_tokens));
                response.metadata.insert("total_tokens".to_string(), json!(usage.total_tokens));
                response.metadata.insert("blaa_model".to_string(), json!(blaa_response.model));
            }
        }
        
        response
    }
    
    /// Convert streaming BLAA chunk to inference response
    fn from_blaa_chunk(&self, request: &InferenceRequest, chunk: ChatCompletionChunk) -> Option<InferenceResponse> {
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                let mut response = InferenceResponse::new(request.request_id);
                
                // Generate tokens from content chunk
                let tokens: Vec<GeneratedToken> = content
                    .chars()
                    .enumerate()
                    .map(|(i, c)| GeneratedToken::new(
                        i as u32,
                        c.to_string(),
                        -1.0, // Default log prob
                        i,
                    ))
                    .collect();
                
                response.text = content.clone();
                response.tokens = tokens;
                response.total_tokens = content.chars().count();
                
                // Mark as streaming response
                response.metadata.insert("streaming".to_string(), json!(true));
                response.metadata.insert("chunk_id".to_string(), json!(chunk.id));
                
                return Some(response);
            }
        }
        None
    }
}

#[async_trait::async_trait]
impl InferenceEngine for BlaaInferenceEngine {
    async fn generate(&self, request: InferenceRequest) -> InferenceResult<InferenceResponse> {
        let start_time = std::time::Instant::now();
        
        info!("Generating inference for request {} with model {}", 
               request.request_id, request.model_id);
        
        let blaa_request = self.to_blaa_request(&request);
        
        let result = if request.streaming {
            // Handle streaming request
            let mut stream = self.client.lock().await.create_chat_completion_stream(blaa_request).await?;
            let mut full_response = String::new();
            let mut chunk_count = 0;
            
            use futures::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        chunk_count += 1;
                        if let Some(response) = self.from_blaa_chunk(&request, chunk) {
                            full_response.push_str(&response.text);
                        }
                    }
                    Err(e) => {
                        error!("Error in streaming chunk: {}", e);
                        return Err(InferenceError::InternalError(e.to_string()));
                    }
                }
            }
            
            let mut final_response = InferenceResponse::new(request.request_id);
            final_response.text = full_response.clone();
            final_response.total_tokens = full_response.chars().count();
            final_response.finish_reason = FinishReason::EndOfSequence;
            final_response.metadata.insert("streaming".to_string(), json!(true));
            final_response.metadata.insert("chunk_count".to_string(), json!(chunk_count));
            
            final_response
        } else {
            // Handle non-streaming request
            let blaa_response = self.client.lock().await.create_chat_completion(blaa_request).await?;
            self.from_blaa_response(&request, blaa_response)
        };
        
        let inference_time = start_time.elapsed().as_millis() as u64;
        let mut response = result;
        response.inference_time_ms = inference_time;
        
        info!("Inference completed for request {} in {}ms", 
               request.request_id, inference_time);
        
        Ok(response)
    }
    
    async fn generate_stream(
        &self,
        request: InferenceRequest,
    ) -> InferenceResult<Pin<Box<dyn futures::Stream<Item = InferenceResult<InferenceResponse>> + Send>>> {
        let blaa_request = self.to_blaa_request(&request);
        let stream = self.client.lock().await.create_chat_completion_stream(blaa_request).await?;
        
        let request_clone = request.clone();
        let stream = stream.map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    if let Some(response) = Self::from_blaa_chunk_static(&request_clone, chunk) {
                        Ok(response)
                    } else {
                        Ok(InferenceResponse::new(request_clone.request_id))
                    }
                }
                Err(e) => Err(InferenceError::InternalError(e.to_string())),
            }
        });
        
        Ok(Box::pin(stream))
    }
    
    async fn health_check(&self) -> InferenceResult<bool> {
        match self.client.lock().await.list_models().await {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!("BLAA health check failed: {}", e);
                Ok(false)
            }
        }
    }
    
    async fn get_model_info(&self, model_id: &str) -> InferenceResult<HashMap<String, Value>> {
        match self.client.lock().await.get_model(model_id).await {
            Ok(model) => {
                let mut info = HashMap::new();
                info.insert("id".to_string(), json!(model.id));
                info.insert("object".to_string(), json!(model.object_type));
                info.insert("created".to_string(), json!(model.created));
                info.insert("owned_by".to_string(), json!(model.owned_by));
                info.insert("provider".to_string(), json!("blaa"));
                Ok(info)
            }
            Err(e) => Err(InferenceError::InternalError(e.to_string())),
        }
    }
    
    async fn list_models(&self) -> InferenceResult<Vec<String>> {
        match self.client.lock().await.list_models().await {
            Ok(models) => {
                let model_ids: Vec<String> = models.into_iter()
                    .map(|m| m.id)
                    .collect();
                Ok(model_ids)
            }
            Err(e) => Err(InferenceError::InternalError(e.to_string())),
        }
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_batching(&self) -> bool {
        false // BLAA doesn't support batching in this implementation
    }
    
    fn get_engine_type(&self) -> String {
        "blaa".to_string()
    }
    
    fn get_engine_version(&self) -> String {
        nexora_blaa::BLAA_API_VERSION.to_string()
    }
}

// Static helper methods for non-self usage
impl BlaaInferenceEngine {
    fn from_blaa_chunk_static(request: &InferenceRequest, chunk: ChatCompletionChunk) -> Option<InferenceResponse> {
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                let mut response = InferenceResponse::new(request.request_id);
                
                let tokens: Vec<GeneratedToken> = content
                    .chars()
                    .enumerate()
                    .map(|(i, c)| GeneratedToken::new(
                        i as u32,
                        c.to_string(),
                        -1.0,
                        i,
                    ))
                    .collect();
                
                response.text = content.clone();
                response.tokens = tokens;
                response.total_tokens = content.chars().count();
                response.metadata.insert("streaming".to_string(), json!(true));
                response.metadata.insert("chunk_id".to_string(), json!(chunk.id));
                
                return Some(response);
            }
        }
        None
    }
}

/// BLAA embeddings integration
#[derive(Debug, Clone)]
pub struct BlaaEmbeddingsEngine {
    client: Arc<Mutex<BlaaClient>>,
    default_model: String,
}

impl BlaaEmbeddingsEngine {
    /// Create new BLAA embeddings engine
    pub async fn new(config: BlaaConfig) -> InferenceResult<Self> {
        let client = BlaaClient::new(config)?;
        let default_model = client.config().default_model.clone();
        
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            default_model,
        })
    }
    
    /// Create from environment variables
    pub async fn from_env() -> InferenceResult<Self> {
        let client = BlaaClient::from_env().await?;
        let default_model = client.config().default_model.clone();
        
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            default_model,
        })
    }
    
    /// Generate embeddings for text
    pub async fn generate_embeddings(&self, texts: Vec<String>) -> InferenceResult<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            input: nexora_blaa::EmbeddingInput::Multiple(texts),
            model: self.default_model.clone(),
            user: None,
            encoding_format: Some(nexora_blaa::EmbeddingFormat::Float),
        };
        
        let response = self.client.lock().await.create_embeddings(request).await?;
        
        let embeddings: Vec<Vec<f32>> = response.data
            .into_iter()
            .map(|data| data.embedding)
            .collect();
        
        Ok(embeddings)
    }
    
    /// Generate single embedding
    pub async fn generate_embedding(&self, text: String) -> InferenceResult<Vec<f32>> {
        let embeddings = self.generate_embeddings(vec![text]).await?;
        embeddings.into_iter()
            .next()
            .ok_or_else(|| InferenceError::InternalError("No embedding generated".to_string()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_blaa_engine_creation() {
        // This test would require actual API key
        // For now, just test the structure
        let config = BlaaConfig::new("test-key");
        let result = BlaaInferenceEngine::new(config).await;
        
        // Should succeed with test key
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_request_conversion() {
        let config = BlaaConfig::new("test-key");
        let engine = BlaaInferenceEngine::new(config).await.unwrap();
        
        let request = InferenceRequest::new("Hello, world!".to_string())
            .with_model("blaa-small".to_string())
            .with_temperature(0.7)
            .with_max_tokens(100);
        
        let blaa_request = engine.to_blaa_request(&request);
        
        assert_eq!(blaa_request.model, "blaa-small");
        assert_eq!(blaa_request.temperature, Some(0.7));
        assert_eq!(blaa_request.max_tokens, Some(100));
        assert_eq!(blaa_request.messages.len(), 1);
        assert_eq!(blaa_request.messages[0].content, "Hello, world!");
    }
}
