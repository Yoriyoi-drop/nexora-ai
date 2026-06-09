use super::admission::SchedulingPolicy;
use crate::degradation::DegradationLevel;
use crate::InferenceResponse;

#[derive(Debug, Clone)]
pub struct ContinuousBatchingConfig {
    /// Max sequences per batch (forward_batched max size).
    pub max_batch_size: usize,
    /// Max total sequences in flight at once.
    pub max_total_sequences: usize,
    /// Minimum prefix length to consider for sharing (0 = disabled).
    pub min_shared_prefix_len: usize,
    /// Whether to enable prefix sharing across sequences.
    pub enable_prefix_sharing: bool,
    /// Use PagedKVCache instead of flat GpuKVCache/CpuKVCache.
    /// Enables block-based allocation for zero fragmentation.
    pub use_paged_cache: bool,
    /// Block size for paged cache (tokens per block). 0 = auto.
    pub paged_block_size: usize,
    /// Use f16 (half-precision) for paged cache K/V storage — 2× memory reduction.
    pub paged_cache_f16: bool,
    /// Use 4-bit quantized for paged cache K/V storage — 8× memory reduction.
    /// When true, overrides paged_cache_f16.
    pub paged_cache_q4: bool,
    /// Max physical blocks for paged cache. 0 = auto.
    pub paged_max_blocks: usize,
    /// Soft memory limit for paged KV cache in bytes (0 = auto from max_blocks).
    pub paged_max_memory_bytes: usize,
    /// Eviction policy (LRU, LFU, TTL).
    pub paged_eviction_policy: crate::paged_cache::EvictionPolicy,
    /// Watermark ratio for eviction (0.0–1.0).
    pub paged_eviction_watermark: f64,
    /// Min age in seconds before a sequence can be evicted.
    pub paged_eviction_min_age_secs: f64,
    /// Number of sequences to evict per cycle.
    pub paged_eviction_batch_size: usize,
    /// Scheduling policy for fairness and starvation avoidance.
    pub scheduling_policy: SchedulingPolicy,
    /// Aging boost per millisecond queued (added to priority).
    /// Only used when `scheduling_policy = PriorityAging`.
    pub aging_boost_per_ms: f64,
    /// Enable dynamic padding: wait up to `padding_wait_ms` for more sequences
    /// before committing to a batch, reducing padding waste.
    pub enable_dynamic_padding: bool,
    /// Max ms to wait for batch to fill before processing with current waste.
    pub padding_wait_ms: u64,
    /// Target padding waste ratio (0.0–1.0). If batch waste exceeds this,
    /// process immediately rather than waiting for more sequences.
    pub target_padding_waste: f64,
    // ── Adaptive Batching ─────────────────────────────────────────────────
    /// Enable adaptive batch sizing: auto-tune max_batch_size based on
    /// measured throughput (tokens/sec). Prevents overload when model is
    /// slow, increases throughput when the system is underutilized.
    pub enable_adaptive_batching: bool,
    /// Target throughput in tokens/sec for adaptive batching to tune toward.
    /// Batch size adjusts: `new_size = clamp(current * throughput / target, min, max)`.
    pub target_tokens_per_sec: f64,
    /// Floor for adaptive batch size — never go below this.
    pub min_adaptive_batch_size: usize,
    /// EWMA smoothing factor for throughput measurement (0.0–1.0).
    /// Higher = more responsive, lower = smoother.
    pub throughput_alpha: f64,
    // ── Load Shedding ─────────────────────────────────────────────────────
    /// Max queued (not yet batched) sequences before rejecting new requests.
    /// 0 = unlimited. Prevents unbounded queue growth under sustained load.
    pub max_queue_depth: usize,
    /// Shed load (reject new requests) when degradation level ≥ this threshold.
    /// Set to `DegradationLevel::Minimal` to stop accepting new work during
    /// severe degradation, preserving capacity for in-flight requests.
    pub shed_at_degradation: DegradationLevel,
}

impl Default for ContinuousBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_total_sequences: 16_384,
            min_shared_prefix_len: 4,
            enable_prefix_sharing: true,
            use_paged_cache: true,
            paged_block_size: 64,
            paged_cache_f16: false,
            paged_cache_q4: true,
            paged_max_blocks: 65_536,
            paged_max_memory_bytes: 0,
            paged_eviction_policy: crate::paged_cache::EvictionPolicy::LRU,
            paged_eviction_watermark: 0.70,
            paged_eviction_min_age_secs: 2.0,
            paged_eviction_batch_size: 8,
            scheduling_policy: SchedulingPolicy::Fifo,
            aging_boost_per_ms: 0.0,
            enable_dynamic_padding: true,
            padding_wait_ms: 20,
            target_padding_waste: 0.5,
            enable_adaptive_batching: true,
            target_tokens_per_sec: 5_000_000.0,
            min_adaptive_batch_size: 2,
            throughput_alpha: 0.5,
            max_queue_depth: 8_192,
            shed_at_degradation: DegradationLevel::Minimal,
        }
    }
}

pub struct StepResult {
    pub completed: Vec<InferenceResponse>,
    pub active_count: usize,
    pub idle: bool,
    /// Number of sequences in the batch (prefill + generation)
    pub batch_size: usize,
    /// Padding waste ratio for this batch (0.0 = perfect fill, 1.0 = all waste).
    /// Higher values indicate inefficient batching.
    pub padding_waste: f64,
    /// Time spent in prefill phase (PHASE 1) in microseconds.
    pub prefill_time_us: u64,
    /// Time spent in generation phase (PHASE 2) in microseconds.
    pub decode_time_us: u64,
    /// Number of tokens prefilled in this step.
    pub prefill_tokens: usize,
    /// Number of tokens decoded in this step.
    pub decode_tokens: usize,
}


