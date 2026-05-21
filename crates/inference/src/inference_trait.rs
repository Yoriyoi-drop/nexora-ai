//! Inference Engine Trait

use futures::Stream;
use ndarray::Array1;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;

use crate::{InferenceRequest, InferenceResponse, Result as InferenceResult};
use nexora_transformer::KVCacheEntry;

/// Trait for model forward pass — abstracts over CausalLM for testing.
pub trait ModelForward: Send + Sync {
    /// Run the model forward for a single token, updating the KV cache.
    fn forward(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> Array1<f32>;

    /// Run batched forward: process multiple tokens with per-sequence KV caches.
    /// Default implementation falls back to per-sequence forward calls.
    fn forward_batched(
        &self,
        input_ids: &[u32],
        kv_caches: &mut [Vec<KVCacheEntry>],
    ) -> Vec<Array1<f32>> {
        input_ids
            .iter()
            .zip(kv_caches.iter_mut())
            .map(|(&id, cache)| self.forward(&[id], cache))
            .collect()
    }

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

// ── CausalLM implements ModelForward ────────────────────────────────────────
// When gpu feature is enabled, forward_batched delegates to forward_gpu_batched
// for batched GPU execution.

impl ModelForward for nexora_transformer::CausalLM {
    fn forward(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> Array1<f32> {
        self.forward(input_ids, kv_cache)
    }

    fn reset_cache(&self) -> Vec<KVCacheEntry> {
        self.reset_cache()
    }

    fn forward_batched(
        &self,
        input_ids: &[u32],
        kv_caches: &mut [Vec<KVCacheEntry>],
    ) -> Vec<Array1<f32>> {
        #[cfg(feature = "gpu")]
        if nexora_autograd::gpu::GpuContext::is_available() {
            if let Ok(result) = self.forward_gpu_batched(input_ids, kv_caches) {
                return result;
            }
        }
        // CPU fallback: per-sequence via default implementation
        ModelForward::forward_batched(self, input_ids, kv_caches)
    }
}
