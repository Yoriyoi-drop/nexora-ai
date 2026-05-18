//! Inference Engine Trait

use std::collections::HashMap;
use std::pin::Pin;
use futures::Stream;
use ndarray::Array1;
use serde_json::Value;

use crate::{InferenceRequest, InferenceResponse, Result as InferenceResult};
use nexora_foundation::models::transformer::KVCacheEntry;

/// Trait for model forward pass — abstracts over CausalLM for testing.
pub trait ModelForward: Send + Sync {
    /// Run the model forward for a single token, updating the KV cache.
    fn forward(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> Array1<f32>;

    /// Create a fresh, empty KV cache for this model.
    fn reset_cache(&self) -> Vec<KVCacheEntry>;
}

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
