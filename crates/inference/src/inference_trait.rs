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

/// Global counter of GPU→CPU fallback events (sampler fallback, forward fallback, etc.).
pub static GPU_CPU_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Global counter of GPU-resident generation path used successfully.
pub static GPU_RESIDENT_SUCCESSES: AtomicU64 = AtomicU64::new(0);

/// Global counter of GPU-resident → fallback events (when GPU-resident path fails).
pub static GPU_RESIDENT_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Global counter of total tokens generated via GPU path.
pub static GPU_TOKENS_GENERATED: AtomicU64 = AtomicU64::new(0);

/// Global counter of total tokens generated via CPU fallback path.
pub static CPU_TOKENS_GENERATED: AtomicU64 = AtomicU64::new(0);

/// Global counter of prefix cache hits.
pub static PREFIX_CACHE_HITS: AtomicU64 = AtomicU64::new(0);

/// Global counter of prefix cache misses.
pub static PREFIX_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Returns current GPU forward error count.
pub fn gpu_forward_error_count() -> u64 {
    GPU_FORWARD_ERRORS.load(Ordering::Relaxed)
}

/// Returns current GPU→CPU fallback count.
pub fn gpu_cpu_fallback_count() -> u64 {
    GPU_CPU_FALLBACKS.load(Ordering::Relaxed)
}

/// Returns current GPU-resident success count.
pub fn gpu_resident_success_count() -> u64 {
    GPU_RESIDENT_SUCCESSES.load(Ordering::Relaxed)
}

/// Returns current GPU-resident fallback count.
pub fn gpu_resident_fallback_count() -> u64 {
    GPU_RESIDENT_FALLBACKS.load(Ordering::Relaxed)
}

/// Returns total tokens generated via GPU path.
pub fn gpu_tokens_generated() -> u64 {
    GPU_TOKENS_GENERATED.load(Ordering::Relaxed)
}

/// Returns total tokens generated via CPU fallback.
pub fn cpu_tokens_generated() -> u64 {
    CPU_TOKENS_GENERATED.load(Ordering::Relaxed)
}

/// Returns current prefix cache hit count.
pub fn prefix_cache_hits() -> u64 {
    PREFIX_CACHE_HITS.load(Ordering::Relaxed)
}

/// Returns current prefix cache miss count.
pub fn prefix_cache_misses() -> u64 {
    PREFIX_CACHE_MISSES.load(Ordering::Relaxed)
}

/// Returns current prefix cache hit ratio (0.0–1.0).
pub fn prefix_cache_hit_ratio() -> f64 {
    let hits = PREFIX_CACHE_HITS.load(Ordering::Relaxed);
    let misses = PREFIX_CACHE_MISSES.load(Ordering::Relaxed);
    let total = hits + misses;
    if total == 0 { 0.0 } else { hits as f64 / total as f64 }
}

/// Returns all observability counters as a snapshot struct.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ObservabilitySnapshot {
    pub gpu_forward_errors: u64,
    pub gpu_cpu_fallbacks: u64,
    pub gpu_resident_successes: u64,
    pub gpu_resident_fallbacks: u64,
    pub gpu_tokens_generated: u64,
    pub cpu_tokens_generated: u64,
    pub gpu_resident_ratio: f64,
    pub gpu_alive: bool,
    pub prefix_cache_hits: u64,
    pub prefix_cache_misses: u64,
    pub prefix_cache_hit_ratio: f64,
}

/// Collect a snapshot of all observability counters.
pub fn observability_snapshot() -> ObservabilitySnapshot {
    let gpu_ok = cfg!(feature = "gpu")
        && nexora_autograd::gpu::GpuContext::is_available();
    let gpu_tokens = GPU_TOKENS_GENERATED.load(Ordering::Relaxed);
    let cpu_tokens = CPU_TOKENS_GENERATED.load(Ordering::Relaxed);
    let total = gpu_tokens + cpu_tokens;
    ObservabilitySnapshot {
        gpu_forward_errors: GPU_FORWARD_ERRORS.load(Ordering::Relaxed),
        gpu_cpu_fallbacks: GPU_CPU_FALLBACKS.load(Ordering::Relaxed),
        gpu_resident_successes: GPU_RESIDENT_SUCCESSES.load(Ordering::Relaxed),
        gpu_resident_fallbacks: GPU_RESIDENT_FALLBACKS.load(Ordering::Relaxed),
        gpu_tokens_generated: gpu_tokens,
        cpu_tokens_generated: cpu_tokens,
        gpu_resident_ratio: if total > 0 { gpu_tokens as f64 / total as f64 } else { 0.0 },
        gpu_alive: gpu_ok,
        prefix_cache_hits: PREFIX_CACHE_HITS.load(Ordering::Relaxed),
        prefix_cache_misses: PREFIX_CACHE_MISSES.load(Ordering::Relaxed),
        prefix_cache_hit_ratio: {
            let h = PREFIX_CACHE_HITS.load(Ordering::Relaxed);
            let m = PREFIX_CACHE_MISSES.load(Ordering::Relaxed);
            let t = h + m;
            if t == 0 { 0.0 } else { h as f64 / t as f64 }
        },
    }
}

/// Resets all observability counters to zero.
pub fn reset_all_observability() {
    GPU_FORWARD_ERRORS.store(0, Ordering::Relaxed);
    GPU_CPU_FALLBACKS.store(0, Ordering::Relaxed);
    GPU_RESIDENT_SUCCESSES.store(0, Ordering::Relaxed);
    GPU_RESIDENT_FALLBACKS.store(0, Ordering::Relaxed);
    GPU_TOKENS_GENERATED.store(0, Ordering::Relaxed);
    CPU_TOKENS_GENERATED.store(0, Ordering::Relaxed);
}

/// Trait for model forward pass — abstracts over CausalLM for testing.
pub trait ModelForward: Send + Sync {
    /// Run the model forward for a single token, updating the KV cache.
    fn forward(&self, input_ids: &[u32], kv_cache: &mut dyn KVCacheProvider) -> Array1<f32>;

    /// Run "batched" forward: process multiple tokens with per-sequence KV caches.
    ///
    /// Default implementation processes each input **sequentially** via
    /// [`forward`]. Throughput scales as O(N). Implementations backed by
    /// GPU/TPU should override this with a true batched forward path.
    fn forward_batched(
        &self,
        input_ids: &[u32],
        kv_caches: &mut [CpuKVCache],
    ) -> Vec<Array1<f32>> {
        input_ids
            .iter()
            .zip(kv_caches.iter_mut())
            .map(|(&id, cache)| self.forward(&[id], cache))
            .collect()
    }

    /// Forward pass with GPU-native sampling — returns sampled token entirely
    /// on GPU without reading back the full logits vector (128KB per token).
    ///
    /// Returns `Some(token_id)` when GPU-resident path succeeds, `None` to
    /// indicate caller should fall back to [`forward`] + CPU sampling.
    ///
    /// Default implementation always returns `None`.
    fn forward_sample_gpu(
        &self,
        _input_ids: &[u32],
        _kv_cache: &mut dyn KVCacheProvider,
        _temperature: f32,
        _top_k: usize,
        _top_p: f32,
        _seed: u64,
    ) -> Option<u32> {
        None
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
                    GPU_CPU_FALLBACKS.fetch_add(1, Ordering::Relaxed);
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
                    GPU_CPU_FALLBACKS.fetch_add(1, Ordering::Relaxed);
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

    fn forward_sample_gpu(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> Option<u32> {
        #[cfg(feature = "gpu")]
        {
            use nexora_transformer::model::sample_token_gpu_keep_gpu;
            let gpu_entries = kv_cache.as_gpu_entries()?;
            let logits_gpu = self
                .forward_gpu_with_cache_keep_gpu(input_ids, gpu_entries)
                .ok()?;
            let token_gpu =
                sample_token_gpu_keep_gpu(&logits_gpu, temperature, top_k, top_p, seed).ok()?;
            let raw = token_gpu.to_cpu_raw_bytes().ok()?;
            GPU_RESIDENT_SUCCESSES.fetch_add(1, Ordering::Relaxed);
            GPU_TOKENS_GENERATED.fetch_add(1, Ordering::Relaxed);
            Some(u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]))
        }
        #[cfg(not(feature = "gpu"))]
        None
    }
}
