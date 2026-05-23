//! Inference Engine Trait

use futures::Stream;
use ndarray::Array1;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{InferenceRequest, InferenceResponse, Result as InferenceResult};
use nexora_transformer::{CpuKVCache, KVCacheProvider};

/// Global counter of GPU forward failures — resettable, observable.
pub static GPU_FORWARD_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Trait for model forward pass — abstracts over CausalLM for testing.
pub trait ModelForward: Send + Sync {
    /// Run the model forward for a single token, updating the KV cache.
    fn forward(&self, input_ids: &[u32], kv_cache: &mut dyn KVCacheProvider) -> Array1<f32>;

    /// Run "batched" forward: process multiple tokens with per-sequence KV caches.
    ///
    /// ⚠️  CURRENT LIMITATION — NOT TRUE BATCHED INFERENCE:
    /// The default implementation processes each input **sequentially** in a
    /// for-loop calling [`forward`] per element. Throughput scales as O(N),
    /// NOT O(1). Downstream implementations backed by GPU/TPU should override
    /// this with a real batched forward path for better throughput.
    fn forward_batched(
        &self,
        input_ids: &[u32],
        kv_caches: &mut [CpuKVCache],
    ) -> Vec<Array1<f32>> {
        if input_ids.len() > 1 {
            tracing::warn!(
                "forward_batched processing {} sequences sequentially — true batch parallelism not yet implemented",
                input_ids.len()
            );
        }
        input_ids
            .iter()
            .zip(kv_caches.iter_mut())
            .map(|(&id, cache)| self.forward(&[id], cache))
            .collect()
    }

    /// Create a fresh, empty KV cache for this model.
    fn reset_cache(&self) -> CpuKVCache;
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
    fn forward(&self, input_ids: &[u32], kv_cache: &mut dyn KVCacheProvider) -> Array1<f32> {
        fn fallible_forward(
            model: &nexora_transformer::CausalLM,
            ids: &[u32],
            cache: &mut dyn KVCacheProvider,
        ) -> Array1<f32> {
            match model.forward(ids, cache) {
                Ok(v) => v,
                Err(e) => {
                    GPU_FORWARD_ERRORS.fetch_add(1, Ordering::Relaxed);
                    tracing::error!("CausalLM forward failed: {e}");
                    let mut zeros = Array1::zeros(model.config.vocab_size);
                    // Inject a non-zero sentinel in the last position so callers
                    // can detect the failure if they check for it.
                    if zeros.len() > 0 {
                        let last = zeros.len() - 1;
                        zeros[last] = -f32::INFINITY;
                    }
                    zeros
                }
            }
        }
        fallible_forward(self, input_ids, kv_cache)
    }

    fn reset_cache(&self) -> CpuKVCache {
        CpuKVCache::new(self.config.num_layers)
    }

    fn forward_batched(
        &self,
        input_ids: &[u32],
        kv_caches: &mut [CpuKVCache],
    ) -> Vec<Array1<f32>> {
        #[cfg(feature = "gpu")]
        if nexora_autograd::gpu::GpuContext::is_available() {
            let mut vec_caches: Vec<Vec<nexora_transformer::KVCacheEntry>> = kv_caches
                .iter_mut()
                .map(|c| std::mem::take(&mut c.entries))
                .collect();
            match self.forward_gpu_batched(input_ids, &mut vec_caches) {
                Ok(result) => {
                    for (i, entries) in vec_caches.into_iter().enumerate() {
                        kv_caches[i].entries = entries;
                    }
                    return result;
                }
                Err(e) => {
                    GPU_FORWARD_ERRORS.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("GPU forward failed: {}, falling back to CPU", e);
                }
            }
            for (i, entries) in vec_caches.into_iter().enumerate() {
                kv_caches[i].entries = entries;
            }
        }

        input_ids
            .iter()
            .zip(kv_caches.iter_mut())
            .map(|(&id, cache)| ModelForward::forward(self, &[id], cache))
            .collect()
    }
}
