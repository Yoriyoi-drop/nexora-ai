//! Inference Engine Trait

use std::collections::HashMap;
use std::pin::Pin;
use futures::Stream;
use serde_json::Value;
use uuid::Uuid;

use crate::{InferenceRequest, InferenceResponse, Result as InferenceResult};

/// Trait for inference engines
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generate inference response
    async fn generate(&self, request: InferenceRequest) -> InferenceResult<InferenceResponse>;
    
    /// Generate streaming inference response
    async fn generate_stream(
        &self,
        request: InferenceRequest,
    ) -> InferenceResult<Pin<Box<dyn Stream<Item = InferenceResult<InferenceResponse>> + Send>>>;
    
    /// Health check
    async fn health_check(&self) -> InferenceResult<bool>;
    
    /// Get model information
    async fn get_model_info(&self, model_id: &str) -> InferenceResult<HashMap<String, Value>>;
    
    /// List available models
    async fn list_models(&self) -> InferenceResult<Vec<String>>;
    
    /// Check if streaming is supported
    fn supports_streaming(&self) -> bool;
    
    /// Check if batching is supported
    fn supports_batching(&self) -> bool;
    
    /// Get engine type
    fn get_engine_type(&self) -> String;
    
    /// Get engine version
    fn get_engine_version(&self) -> String;
}
