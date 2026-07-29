//! Inference Engine Trait

use futures::Stream;
use ndarray::Array1;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

#[cfg(feature = "gpu")]
use nexora_deeplearning::autograd::gpu::gpu_observability as gpu_obs;

use crate::{InferenceRequest, InferenceResponse, Result as InferenceResult};
use nexora_transformer::{CpuKVCache, KVCacheProvider};

/// Global counter of GPU forward failures — resettable, observable.
pub static GPU_FORWARD_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Global counter of GPU→CPU fallback events (sampler fallback, forward fallback, etc.).
pub static GPU_CPU_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Global counter of successful paged (block-based) forward passes.
/// Each success avoids the ~2GB to_flat_cache() allocation.
pub static PAGED_FORWARD_SUCCESSES: AtomicU64 = AtomicU64::new(0);

/// Global counter of paged forward → flat cache fallback events.
pub static PAGED_FORWARD_FALLBACKS: AtomicU64 = AtomicU64::new(0);

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

/// Global counter of GPU device lost events.
pub static GPU_DEVICE_LOST_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Global counter of GPU out-of-memory events.
pub static GPU_OOM_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Global counter of GPU shader compilation errors.
pub static GPU_SHADER_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Global counter of GPU successful recovery events (after device lost/OOM).
pub static GPU_RECOVERED_EVENTS: AtomicU64 = AtomicU64::new(0);

// ── Phase 6b Observability Metrics ──────────────────────────────────────────

/// KV cache pressure: used blocks (updated by paged_cache on alloc/free).
pub static KV_CACHE_USED_BLOCKS: AtomicU64 = AtomicU64::new(0);
/// KV cache pressure: total available blocks.
pub static KV_CACHE_TOTAL_BLOCKS: AtomicU64 = AtomicU64::new(0);
/// KV cache memory usage in bytes (updated by paged_cache on alloc/free).
pub static KV_CACHE_MEMORY_BYTES: AtomicU64 = AtomicU64::new(0);
/// KV cache internal fragmentation ratio (wasted slots / total slots in paged blocks).
pub static KV_CACHE_INTERNAL_FRAG: AtomicU64 = AtomicU64::new(0);
/// KV cache external fragmentation ratio (free blocks / total allocated blocks).
pub static KV_CACHE_EXTERNAL_FRAG: AtomicU64 = AtomicU64::new(0);

/// Current scheduler queue depth (waiting sequences in continuous batching).
pub static SCHEDULER_QUEUE_DEPTH: AtomicI64 = AtomicI64::new(0);

/// Cumulative padding waste × 100 (summed across batches for averaging).
pub static BATCHING_PADDING_WASTE_SUM: AtomicU64 = AtomicU64::new(0);
/// Number of batches accumulated in the waste sum.
pub static BATCHING_COUNT: AtomicU64 = AtomicU64::new(0);

/// Cumulative GPU busy time in nanoseconds.
pub static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
/// Cumulative GPU idle time in nanoseconds.
pub static GPU_IDLE_NS: AtomicU64 = AtomicU64::new(0);

/// Cumulative bytes read from GPU (PCIe readback).
pub static PCIE_READ_BYTES: AtomicU64 = AtomicU64::new(0);
/// Cumulative bytes written to GPU (PCIe upload).
pub static PCIE_WRITE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Memory pool allocation tracking.
pub static POOL_ALLOCS: AtomicU64 = AtomicU64::new(0);
pub static POOL_DEALLOCS: AtomicU64 = AtomicU64::new(0);
pub static POOL_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static POOL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
pub static POOL_BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
pub static POOL_BYTES_REUSED: AtomicU64 = AtomicU64::new(0);

// ── Benchmark / Profiling Accumulators ──────────────────────────────────

/// Cumulative prefill time in microseconds (all steps).
pub static TOTAL_PREFILL_TIME_US: AtomicU64 = AtomicU64::new(0);
/// Cumulative decode time in microseconds (all steps).
pub static TOTAL_DECODE_TIME_US: AtomicU64 = AtomicU64::new(0);
/// Total tokens processed during prefill phases.
pub static TOTAL_PREFILL_TOKENS: AtomicU64 = AtomicU64::new(0);
/// Total tokens generated during decode phases.
pub static TOTAL_DECODE_TOKENS: AtomicU64 = AtomicU64::new(0);
/// Number of step() calls recorded (for averaging).
pub static TOTAL_STEPS: AtomicU64 = AtomicU64::new(0);
/// Peak GPU memory bytes observed (via nvml or process RSS).
pub static PEAK_GPU_MEM_BYTES: AtomicU64 = AtomicU64::new(0);
/// Peak CPU memory bytes observed.
pub static PEAK_CPU_MEM_BYTES: AtomicU64 = AtomicU64::new(0);

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
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

/// Returns KV cache pressure ratio (0.0–1.0), or 0.0 if no blocks configured.
pub fn kv_cache_pressure() -> f64 {
    let total = KV_CACHE_TOTAL_BLOCKS.load(Ordering::Relaxed);
    if total == 0 {
        0.0
    } else {
        let used = KV_CACHE_USED_BLOCKS.load(Ordering::Relaxed);
        used as f64 / total as f64
    }
}

/// Returns average batching efficiency (0.0–1.0), or 1.0 if no batches yet.
pub fn batching_efficiency() -> f64 {
    let count = BATCHING_COUNT.load(Ordering::Relaxed);
    if count == 0 {
        1.0
    } else {
        let waste_sum = BATCHING_PADDING_WASTE_SUM.load(Ordering::Relaxed);
        let avg_waste = waste_sum as f64 / count as f64 / 100.0;
        (1.0 - avg_waste).clamp(0.0, 1.0)
    }
}

/// Returns GPU utilization estimate (0.0–1.0).
pub fn gpu_utilization() -> f64 {
    let busy = GPU_BUSY_NS.load(Ordering::Relaxed);
    let idle = GPU_IDLE_NS.load(Ordering::Relaxed);
    let total = busy + idle;
    if total == 0 {
        0.0
    } else {
        busy as f64 / total as f64
    }
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
    // Phase 6b metrics
    pub kv_cache_pressure: f64,
    pub scheduler_queue_depth: i64,
    pub batching_efficiency_pct: f64,
    pub gpu_utilization_pct: f64,
    pub pcie_read_bytes: u64,
    pub pcie_write_bytes: u64,
    pub pool_allocs: u64,
    pub pool_deallocs: u64,
    pub tokens_per_sec: f64,
    pub kv_internal_frag_ratio: f64,
    pub kv_external_frag_ratio: f64,
}

/// Collect a snapshot of all observability counters.
pub fn observability_snapshot() -> ObservabilitySnapshot {
    let gpu_ok = cfg!(feature = "gpu") && nexora_deeplearning::autograd::gpu::GpuContext::is_available();
    let gpu_tokens = GPU_TOKENS_GENERATED.load(Ordering::Relaxed);
    let cpu_tokens = CPU_TOKENS_GENERATED.load(Ordering::Relaxed);
    let total = gpu_tokens + cpu_tokens;

    let total_blocks = KV_CACHE_TOTAL_BLOCKS.load(Ordering::Relaxed);
    let used_blocks = KV_CACHE_USED_BLOCKS.load(Ordering::Relaxed);

    #[cfg(feature = "gpu")]
    let busy_ns = gpu_obs::GPU_BUSY_NS.load(Ordering::Relaxed);
    #[cfg(not(feature = "gpu"))]
    let busy_ns = 0u64;
    let _idle_ns = 0u64; // not separately tracked
    let gpu_total_ns = busy_ns;

    #[cfg(feature = "gpu")]
    let pcie_read = gpu_obs::PCIE_READ_BYTES.load(Ordering::Relaxed);
    #[cfg(not(feature = "gpu"))]
    let pcie_read = 0u64;
    #[cfg(feature = "gpu")]
    let pcie_write = gpu_obs::PCIE_WRITE_BYTES.load(Ordering::Relaxed);
    #[cfg(not(feature = "gpu"))]
    let pcie_write = 0u64;
    #[cfg(feature = "gpu")]
    let pool_allocs = gpu_obs::POOL_ALLOCS.load(Ordering::Relaxed);
    #[cfg(not(feature = "gpu"))]
    let pool_allocs = 0u64;
    #[cfg(feature = "gpu")]
    let pool_deallocs = gpu_obs::POOL_DEALLOCS.load(Ordering::Relaxed);
    #[cfg(not(feature = "gpu"))]
    let pool_deallocs = 0u64;

    ObservabilitySnapshot {
        gpu_forward_errors: GPU_FORWARD_ERRORS.load(Ordering::Relaxed),
        gpu_cpu_fallbacks: GPU_CPU_FALLBACKS.load(Ordering::Relaxed),
        gpu_resident_successes: GPU_RESIDENT_SUCCESSES.load(Ordering::Relaxed),
        gpu_resident_fallbacks: GPU_RESIDENT_FALLBACKS.load(Ordering::Relaxed),
        gpu_tokens_generated: gpu_tokens,
        cpu_tokens_generated: cpu_tokens,
        gpu_resident_ratio: if total > 0 {
            gpu_tokens as f64 / total as f64
        } else {
            0.0
        },
        gpu_alive: gpu_ok,
        prefix_cache_hits: PREFIX_CACHE_HITS.load(Ordering::Relaxed),
        prefix_cache_misses: PREFIX_CACHE_MISSES.load(Ordering::Relaxed),
        prefix_cache_hit_ratio: {
            let h = PREFIX_CACHE_HITS.load(Ordering::Relaxed);
            let m = PREFIX_CACHE_MISSES.load(Ordering::Relaxed);
            let t = h + m;
            if t == 0 {
                0.0
            } else {
                h as f64 / t as f64
            }
        },
        kv_cache_pressure: if total_blocks > 0 {
            used_blocks as f64 / total_blocks as f64
        } else {
            0.0
        },
        scheduler_queue_depth: SCHEDULER_QUEUE_DEPTH.load(Ordering::Relaxed),
        batching_efficiency_pct: batching_efficiency() * 100.0,
        gpu_utilization_pct: if gpu_total_ns > 0 {
            (busy_ns as f64 / gpu_total_ns as f64) * 100.0
        } else {
            0.0
        },
        pcie_read_bytes: pcie_read,
        pcie_write_bytes: pcie_write,
        pool_allocs,
        pool_deallocs,
        tokens_per_sec: 0.0, // computed by background collector
        kv_internal_frag_ratio: KV_CACHE_INTERNAL_FRAG.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        kv_external_frag_ratio: KV_CACHE_EXTERNAL_FRAG.load(Ordering::Relaxed) as f64 / 1_000_000.0,
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
    fn forward_batched(&self, input_ids: &[u32], kv_caches: &mut [CpuKVCache]) -> Vec<Array1<f32>> {
        input_ids
            .iter()
            .zip(kv_caches.iter_mut())
            .map(|(&id, cache)| self.forward(&[id], cache))
            .collect()
    }

    /// Prefill multiple sequences with their full prompts.
    ///
    /// Each entry in `inputs` holds one sequence's prompt (remaining tokens).
    /// Each entry in `caches` holds that sequence's KV cache.
    /// Returns logits for the **last** token of each sequence.
    ///
    /// Default implementation: **position-by-position** batched prefill.
    /// At each position P, all sequences with `prompt.len() > P` have their
    /// P-th token processed together via [`forward_batched`]. This ensures:
    ///
    /// 1. KV cache is built position-by-position across the batch (better locality)
    /// 2. If [`forward_batched`] has a GPU-accelerated override, each position
    ///    automatically benefits from true batched matmul across all active sequences
    /// 3. Foundation for padded batch prefill where shorter sequences are
    ///    masked out rather than dropped
    ///
    /// GPU-aware implementations should override this to keep intermediate
    /// logits on device, reading back only the final per-sequence results.
    fn forward_prefill_batched(
        &self,
        inputs: &[&[u32]],
        caches: &mut [Box<dyn KVCacheProvider>],
    ) -> Vec<Array1<f32>> {
        let n = inputs.len();
        if n == 0 || n != caches.len() {
            return Vec::new();
        }

        let max_len = inputs.iter().map(|t| t.len()).max().unwrap_or(0);
        if max_len == 0 {
            return vec![Array1::zeros(1); n];
        }

        // Optimized position-by-position prefill using BATCHED forward.
        // At each position P, all sequences with P-th token are processed
        // together via forward_batched() (GPU batched matmul when available).
        // This replaces N individual forward() calls with 1 batched call per position.
        let mut results: Vec<Option<Array1<f32>>> = vec![None; n];
        let mut active: Vec<bool> = vec![true; n];

        for pos in 0..max_len {
            // Collect all tokens at this position across active sequences.
            // Use Vec<(idx, token)> to restore results per sequence later.
            let mut batch_tokens: Vec<u32> = Vec::with_capacity(n);
            let mut batch_indices: Vec<usize> = Vec::with_capacity(n);
            let mut batch_caches: Vec<Box<dyn KVCacheProvider>> = Vec::with_capacity(n);

            for (i, tokens) in inputs.iter().enumerate() {
                if active[i] && pos < tokens.len() {
                    batch_tokens.push(tokens[pos]);
                    batch_indices.push(i);
                    // Take cache ownership temporarily
                    let cache = std::mem::replace(&mut caches[i], Box::new(CpuKVCache::new(0)));
                    batch_caches.push(cache);
                }
            }

            if batch_tokens.is_empty() {
                continue;
            }

            // Convert batch_caches to CpuKVCache for forward_batched
            let mut cpu_caches: Vec<CpuKVCache> = batch_caches
                .into_iter()
                .map(|mut c| {
                    let mut entries = Vec::new();
                    if let Some(cpu) = c.as_cpu_entries() {
                        entries = std::mem::take(cpu);
                    }
                    CpuKVCache { entries }
                })
                .collect();

            // Single batched forward call for all sequences at this position
            let logits_vec = self.forward_batched(&batch_tokens, &mut cpu_caches);

            // Restore caches and store results
            for (batch_idx, &seq_idx) in batch_indices.iter().enumerate() {
                let entries = std::mem::take(&mut cpu_caches[batch_idx].entries);
                let mut cpu_cache = CpuKVCache { entries };
                caches[seq_idx] = Box::new(cpu_cache);
                if batch_idx < logits_vec.len() && pos == inputs[seq_idx].len() - 1 {
                    results[seq_idx] = Some(logits_vec[batch_idx].clone());
                    active[seq_idx] = false;
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Array1::zeros(1)))
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

    /// Batched forward pass with GPU-native sampling — processes one token per
    /// sequence and samples next tokens entirely on GPU. Zero readback of full
    /// logits (128KB/seq). Only token IDs (4 bytes/seq) are returned.
    ///
    /// Returns `Vec<Option<u32>>` — `Some(token_id)` when GPU-resident path
    /// succeeds for that sequence, `None` to indicate caller should fall back
    /// to [`forward`]/[`forward_batched`] + CPU sampling for that sequence.
    ///
    /// Default implementation returns all `None`.
    fn forward_batched_sample_gpu(
        &self,
        _input_ids: &[u32],
        _caches: &mut [Box<dyn KVCacheProvider>],
        _temperatures: &[f32],
        _top_ks: &[usize],
        _top_ps: &[f32],
        _seed: u64,
    ) -> Vec<Option<u32>> {
        vec![None; _input_ids.len()]
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

    fn forward_batched(&self, input_ids: &[u32], kv_caches: &mut [CpuKVCache]) -> Vec<Array1<f32>> {
        #[cfg(feature = "gpu")]
        if nexora_deeplearning::autograd::gpu::GpuContext::is_available() {
            let mut vec_caches: Vec<Vec<nexora_transformer::KVCacheEntry>> = kv_caches
                .iter_mut()
                .map(|c| std::mem::take(&mut c.entries))
                .collect();
            match self.forward_gpu_batched(input_ids, &mut vec_caches) {
                Ok(result) => {
                    for (i, entries) in vec_caches.into_iter().enumerate() {
                        kv_caches[i].entries = entries;
                    }
                    nexora_deeplearning::autograd::gpu::gpu_watchdog::watchdog_ping();
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

    fn forward_prefill_batched(
        &self,
        inputs: &[&[u32]],
        caches: &mut [Box<dyn KVCacheProvider>],
    ) -> Vec<Array1<f32>> {
        #[cfg(feature = "gpu")]
        {
            let use_gpu_path = caches.iter_mut().all(|c| c.as_gpu_entries().is_some());
            if use_gpu_path {
                let n = inputs.len();
                if n == 0 || n != caches.len() {
                    return Vec::new();
                }

                let max_len = inputs.iter().map(|t| t.len()).max().unwrap_or(0);
                if max_len == 0 {
                    return vec![Array1::zeros(1); n];
                }

                // True GPU padded-batch prefill: process ALL remaining tokens
                // across ALL sequences in ONE forward pass.
                // Take GPU entries from ALL sequences at once.
                let mut gpu_entries_owned: Vec<Vec<nexora_transformer::GpuKVCacheEntry>> = caches
                    .iter_mut()
                    .map(|c| std::mem::take(c.as_gpu_entries().unwrap_or(&mut Vec::new())))
                    .collect();

                let result = self.forward_gpu_batched_prefill_full(inputs, &mut gpu_entries_owned);

                // Restore GPU entries to caches
                for (i, entries) in gpu_entries_owned.into_iter().enumerate() {
                    if i < caches.len() {
                        if let Some(dst) = caches[i].as_gpu_entries() {
                            *dst = entries;
                        }
                    }
                }

                match result {
                    Ok(logits) => {
                        nexora_deeplearning::autograd::gpu::gpu_watchdog::watchdog_ping();
                        return logits;
                    }
                    Err(e) => {
                        GPU_FORWARD_ERRORS.fetch_add(1, Ordering::Relaxed);
                        GPU_CPU_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(
                            "GPU padded-batch prefill failed: {}, falling back to position-by-position",
                            e
                        );
                    }
                }
            }
        }
        // Fallback to position-by-position batched prefill (default)
        ModelForward::forward_prefill_batched(self, inputs, caches)
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
            nexora_deeplearning::autograd::gpu::gpu_watchdog::watchdog_ping();
            Some(u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]))
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    fn forward_batched_sample_gpu(
        &self,
        input_ids: &[u32],
        caches: &mut [Box<dyn KVCacheProvider>],
        temperatures: &[f32],
        top_ks: &[usize],
        top_ps: &[f32],
        seed: u64,
    ) -> Vec<Option<u32>> {
        #[cfg(feature = "gpu")]
        {
            let use_gpu_path = caches.iter_mut().all(|c| c.as_gpu_entries().is_some());
            if use_gpu_path {
                let n = input_ids.len();
                if n == 0 || n != caches.len() {
                    return Vec::new();
                }

                // Take GPU entries from all sequences
                let mut gpu_entries: Vec<Vec<nexora_transformer::GpuKVCacheEntry>> = caches
                    .iter_mut()
                    .map(|c| std::mem::take(c.as_gpu_entries().unwrap_or(&mut Vec::new())))
                    .collect();

                // Per-seq seeds: mix base seed with sequence index
                let seeds: Vec<u64> = (0..n)
                    .map(|i| seed.wrapping_add(i as u64))
                    .collect();
                let top_ks_u32: Vec<u32> = top_ks.iter().map(|&k| k as u32).collect();

                let result = self.forward_gpu_batched_sample(
                    input_ids,
                    &mut gpu_entries,
                    temperatures,
                    &top_ks_u32,
                    top_ps,
                    &seeds,
                );

                // Restore GPU entries
                for (i, entries) in gpu_entries.into_iter().enumerate() {
                    if i < caches.len() {
                        if let Some(dst) = caches[i].as_gpu_entries() {
                            *dst = entries;
                        }
                    }
                }

                match result {
                    Ok(tokens) => {
                        GPU_RESIDENT_SUCCESSES.fetch_add(n as u64, Ordering::Relaxed);
                        GPU_TOKENS_GENERATED.fetch_add(n as u64, Ordering::Relaxed);
                        #[cfg(feature = "gpu")]
                        nexora_deeplearning::autograd::gpu::gpu_watchdog::watchdog_ping();
                        return tokens.into_iter().map(Some).collect();
                    }
                    Err(e) => {
                        GPU_FORWARD_ERRORS.fetch_add(1, Ordering::Relaxed);
                        GPU_CPU_FALLBACKS.fetch_add(n as u64, Ordering::Relaxed);
                        tracing::warn!(
                            "GPU batched sample failed: {}, falling back to per-sequence CPU",
                            e
                        );
                    }
                }
            }
        }
        vec![None; input_ids.len()]
    }
}
