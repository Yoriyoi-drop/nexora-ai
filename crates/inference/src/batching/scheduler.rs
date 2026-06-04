use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array1;
use nexora_tokenizer::BpeTokenizer;
use tracing::{debug, warn};

use super::admission::SchedulingPolicy;
use super::metrics::{ContinuousBatchingConfig, StepResult};
use super::queue::PrefixTrie;
use crate::paged_provider::PagedKVCacheProvider;
use crate::sampler::{Sampler, SamplingConfig};
use crate::sequence_state::{SeqState, Sequence};
use crate::{FinishReason, GeneratedToken, InferenceRequest, InferenceResponse};
#[cfg(feature = "gpu")]
use nexora_transformer::{GpuKVCache, GpuKVCacheEntry};
use nexora_transformer::{CpuKVCache, KVCacheProvider};

pub struct ContinuousBatchingEngine<M> {
    sequences: HashMap<u64, Sequence>,
    kv_caches: HashMap<u64, Box<dyn KVCacheProvider>>,
    model: M,
    next_seq_id: u64,
    max_batch_size: usize,
    max_total_sequences: usize,
    samplers: HashMap<u64, Sampler>,
    use_gpu: bool,
    tokenizer: Option<Arc<std::sync::Mutex<BpeTokenizer>>>,
    pub(crate) config: ContinuousBatchingConfig,
    prefix_trie: Option<PrefixTrie>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    paged_cache_dim_set: bool,
    /// Shared paged cache instance. All PagedKVCacheProviders use the same pool.
    shared_paged: Option<std::sync::Arc<std::sync::Mutex<crate::paged_cache::PagedKVCache>>>,
    // ── Adaptive Batching State ──────────────────────────────────────────────
    /// Exponentially weighted moving average of tokens/sec throughput.
    throughput_ewma: f64,
    /// Current dynamically-tuned batch size.
    pub(crate) adaptive_batch_size: usize,
    /// Last time throughput was sampled.
    last_throughput_update: Instant,
    /// Total steps completed (for EWMA initialization).
    pub(crate) batch_step_count: u64,
    // ── Load Shedding State ───────────────────────────────────────────────────
    /// Whether the engine is currently shedding load (rejecting new requests).
    shed_load: bool,
    /// Number of requests rejected due to load shedding.
    rejected_count: u64,
    /// Recent tokens processed per step (for throughput calculation).
    recent_tokens_per_step: VecDeque<usize>,
}

#[deprecated(note = "renamed to ContinuousBatchingEngine")]
pub type SequentialBatchingEngine<M> = ContinuousBatchingEngine<M>;

impl<M> ContinuousBatchingEngine<M>
where
    M: crate::inference_trait::ModelForward,
{
    pub fn new(model: M, max_batch_size: usize) -> Self {
        Self::with_config(
            model,
            ContinuousBatchingConfig {
                max_batch_size,
                ..Default::default()
            },
        )
    }

    /// Set model architecture dimensions for KV cache creation.
    /// Needed for paged cache and GPU cache; safe to skip for CPU-only GpuKVCache.
    pub fn set_model_dims(
        &mut self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        self.num_layers = num_layers;
        self.num_kv_heads = num_kv_heads;
        self.head_dim = head_dim;
        self.max_seq_len = max_seq_len;
        self.paged_cache_dim_set = true;

        // Initialize shared paged cache if enabled
        if self.config.use_paged_cache && self.shared_paged.is_none() {
            let block_size = if self.config.paged_block_size > 0 {
                self.config.paged_block_size
            } else {
                crate::paged_cache::PagedCacheConfig::suggest_block_size(head_dim, num_kv_heads)
            };
            let max_blocks = if self.config.paged_max_blocks > 0 {
                self.config.paged_max_blocks
            } else {
                (self.max_seq_len / block_size).max(64) * self.config.max_total_sequences
            };
            let max_memory = if self.config.paged_max_memory_bytes > 0 {
                self.config.paged_max_memory_bytes
            } else {
                crate::paged_cache::DEFAULT_MAX_CACHE_MEMORY_BYTES
            };
            let paged_config = crate::paged_cache::PagedCacheConfig {
                block_size,
                max_blocks,
                max_memory_bytes: max_memory,
                eviction_policy: self.config.paged_eviction_policy.clone(),
                eviction_watermark_ratio: self.config.paged_eviction_watermark,
                eviction_min_age_secs: self.config.paged_eviction_min_age_secs,
                eviction_batch_size: self.config.paged_eviction_batch_size,
                num_layers,
                num_kv_heads,
                head_dim,
                max_seq_len,
                f16_storage: self.config.paged_cache_f16,
                q4_storage: self.config.paged_cache_q4,
                enable_memory_tiering: true,
                enable_cold_disk_offload: true,
                ..Default::default()
            };
            self.shared_paged = Some(Arc::new(std::sync::Mutex::new(
                crate::paged_cache::PagedKVCache::new(paged_config),
            )));
        }
    }

    pub fn with_config(model: M, config: ContinuousBatchingConfig) -> Self {
        Self {
            sequences: HashMap::new(),
            kv_caches: HashMap::new(),
            model,
            next_seq_id: 1,
            max_batch_size: if config.enable_adaptive_batching {
                config.max_batch_size
            } else {
                config.max_batch_size
            },
            max_total_sequences: config.max_total_sequences,
            samplers: HashMap::new(),
            use_gpu: {
                #[cfg(feature = "gpu")]
                {
                    nexora_autograd::gpu::GpuContext::is_available()
                }
                #[cfg(not(feature = "gpu"))]
                {
                    false
                }
            },
            tokenizer: None,
            prefix_trie: if config.enable_prefix_sharing {
                Some(PrefixTrie::new())
            } else {
                None
            },
            adaptive_batch_size: config.max_batch_size,
            config,
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            max_seq_len: 0,
            paged_cache_dim_set: false,
            shared_paged: None,
            throughput_ewma: 0.0,
            last_throughput_update: Instant::now(),
            batch_step_count: 0,
            shed_load: false,
            rejected_count: 0,
            recent_tokens_per_step: VecDeque::with_capacity(16),
        }
    }

    pub fn set_use_gpu(&mut self, use_gpu: bool) {
        self.use_gpu = use_gpu;
        for sampler in self.samplers.values_mut() {
            sampler.set_use_gpu(use_gpu);
        }
    }

    pub fn set_tokenizer(&mut self, tokenizer: Arc<std::sync::Mutex<BpeTokenizer>>) {
        self.tokenizer = Some(tokenizer);
    }

    // ── Load Shedding ────────────────────────────────────────────────────────────

    /// Returns `true` when the engine should reject new requests to protect
    /// in-flight sequences from degradation. Considers queue depth and
    /// throughput collapse.
    pub fn should_shed_load(&self) -> bool {
        if self.shed_load {
            return true;
        }
        // Hard queue depth limit
        if self.config.max_queue_depth > 0
            && self.sequences.len() >= self.config.max_queue_depth
        {
            return true;
        }
        // Throughput collapse detection: if EWMA drops below 10% of target
        // and we've processed at least 5 steps (warmup), start shedding.
        if self.config.enable_adaptive_batching
            && self.batch_step_count >= 5
            && self.throughput_ewma > 0.0
            && self.throughput_ewma < self.config.target_tokens_per_sec * 0.1
        {
            return true;
        }
        false
    }

    /// Returns the number of requests rejected by load shedding.
    pub fn rejected_count(&self) -> u64 {
        self.rejected_count
    }

    // ── Adaptive Batch Sizing ─────────────────────────────────────────────────

    /// Update the adaptive batch size based on measured throughput.
    ///
    /// Uses an EWMA (Exponentially Weighted Moving Average) of tokens/sec
    /// to smooth out transient spikes. The batch size is adjusted:
    ///   `new_batch = clamp(current * throughput / target, min, max)`
    ///
    /// Call this after each `step()` with the number of tokens processed.
    fn update_adaptive_batch_size(&mut self, tokens_processed: usize) {
        if !self.config.enable_adaptive_batching {
            return;
        }

        let elapsed = self.last_throughput_update.elapsed();
        self.last_throughput_update = Instant::now();

        let elapsed_secs = elapsed.as_secs_f64().max(1e-6);
        let instant_tps = tokens_processed as f64 / elapsed_secs;

        // Initialize EWMA on first measurement
        if self.batch_step_count == 0 {
            self.throughput_ewma = instant_tps;
        } else {
            self.throughput_ewma = self.config.throughput_alpha * instant_tps
                + (1.0 - self.config.throughput_alpha) * self.throughput_ewma;
        }
        self.batch_step_count += 1;
        self.recent_tokens_per_step.push_back(tokens_processed);
        if self.recent_tokens_per_step.len() > 16 {
            self.recent_tokens_per_step.pop_front();
        }

        // Tune batch size: if throughput < target → shrink batch (reduce pressure)
        // If throughput > target → grow batch (utilize spare capacity)
        let target = self.config.target_tokens_per_sec;
        let ratio = if target > 0.0 {
            (self.throughput_ewma / target).clamp(0.25, 4.0)
        } else {
            1.0
        };

        let current = self.adaptive_batch_size as f64;
        let new_size = (current * ratio).round() as usize;
        let clamped = new_size
            .clamp(self.config.min_adaptive_batch_size, self.config.max_batch_size);

        if clamped != self.adaptive_batch_size {
            debug!(
                "Adaptive batch: {} → {} (tps={:.0}, ewma={:.0}, target={:.0}, ratio={:.2})",
                self.adaptive_batch_size,
                clamped,
                instant_tps,
                self.throughput_ewma,
                target,
                ratio
            );
            self.adaptive_batch_size = clamped;
            self.max_batch_size = clamped;
        }

        // Update load shedding state based on throughput health
        if self.batch_step_count >= 5 {
            let health_ratio = self.throughput_ewma / target.max(1.0);
            self.shed_load = health_ratio < 0.1;
        }
    }

    /// Returns the current adaptive batch size for use in `select_ready_sequences`.
    fn effective_batch_size(&self) -> usize {
        if self.config.enable_adaptive_batching {
            self.adaptive_batch_size
        } else {
            self.max_batch_size
        }
    }

    // ── Prefix Sharing ─────────────────────────────────────────────────────────

    /// Copy shared-prefix K/V from previously-prefilled sequences into prefill
    /// sequences that share a prompt prefix. Updates `prompt_pos` to skip the
    /// prefix, so the prefill forward pass only processes non-shared tokens.
    /// Only operates when GPU is active and the prefix trie is available.
    fn try_gpu_prefix_sharing(&mut self, prefill_ids: &[u64]) {
        if !self.use_gpu {
            return;
        }
        #[cfg(feature = "gpu")]
        {
            let trie = match self.prefix_trie.as_ref() {
                Some(t) => t,
                None => return,
            };
            let ctx = match nexora_autograd::gpu::GpuContext::global() {
                Ok(c) => c,
                _ => return,
            };

            for &sid in prefill_ids {
                // Skip if this sequence already has a partial prefix offset
                let seq = match self.sequences.get(&sid) {
                    Some(s) => s,
                    None => continue,
                };
                if seq.prompt_pos > 0 {
                    continue;
                }

                // Find longest shared prefix with another sequence
                let (prefix_len, src_id) = match trie.find_shared_prefix(&seq.prompt, sid) {
                    Some(m) => m,
                    None => continue,
                };

                // Check if the source sequence has actually been prefilled
                // (has K/V data for at least prefix_len tokens)
                let src_has_data = self
                    .kv_caches
                    .get(&src_id)
                    .map_or(false, |c| c.seq_len(0) >= prefix_len);
                if !src_has_data {
                    continue;
                }

                // Remove both caches from the HashMap to avoid borrow conflicts
                let mut src_dyn = match self.kv_caches.remove(&src_id) {
                    Some(c) => c,
                    None => continue,
                };
                let mut dst_dyn = match self.kv_caches.remove(&sid) {
                    Some(c) => c,
                    None => {
                        self.kv_caches.insert(src_id, src_dyn);
                        continue;
                    }
                };

                // Extract GPU entries from both sides
                let src_entries: &mut Vec<GpuKVCacheEntry> = match src_dyn.as_gpu_entries() {
                    Some(e) if !e.is_empty() => e,
                    _ => {
                        self.kv_caches.insert(src_id, src_dyn);
                        self.kv_caches.insert(sid, dst_dyn);
                        continue;
                    }
                };
                let dst_entries: &mut Vec<
                    nexora_transformer::GpuKVCacheEntry,
                > = match dst_dyn.as_gpu_entries() {
                    Some(e) if !e.is_empty() => e,
                    _ => {
                        self.kv_caches.insert(src_id, src_dyn);
                        self.kv_caches.insert(sid, dst_dyn);
                        continue;
                    }
                };

                // Copy prefix K/V per layer
                let mut ok = true;
                for (e_dst, e_src) in dst_entries.iter_mut().zip(src_entries.iter()) {
                    if e_dst.copy_prefix_from(&ctx, e_src, prefix_len).is_err() {
                        ok = false;
                        break;
                    }
                }

                if ok {
                    if let Some(seq) = self.sequences.get_mut(&sid) {
                        seq.prompt_pos = prefix_len;
                        debug!(
                            "GPU prefix sharing: copied {} tokens from seq {} to seq {}",
                            prefix_len, src_id, sid
                        );
                    }
                }

                self.kv_caches.insert(src_id, src_dyn);
                self.kv_caches.insert(sid, dst_dyn);
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = prefill_ids;
        }
    }

    /// Block-level prefix sharing for paged cache.
    ///
    /// Uses the prefix trie to find sequences with shared prompt prefixes,
    /// then shares physical blocks via `PagedKVCache::share_prefix_in_blocks`.
    /// The shared blocks have `ref_count > 1`, so the first divergent append
    /// triggers a copy-on-write deep copy (zero-copy for the common prefix).
    ///
    /// Only operates when the engine uses a shared paged cache.
    fn try_paged_prefix_sharing(&mut self, prefill_ids: &[u64]) {
        let shared_cache = match self.shared_paged.clone() {
            Some(c) => c,
            None => return,
        };
        let trie = match self.prefix_trie.as_ref() {
            Some(t) => t,
            None => return,
        };

        for &sid in prefill_ids {
            let prompt = match self.sequences.get(&sid) {
                Some(s) => {
                    if s.prompt_pos > 0 {
                        continue;
                    }
                    &s.prompt
                }
                None => continue,
            };

            let (prefix_len, src_id) = match trie.find_shared_prefix(prompt, sid) {
                Some(m) => m,
                None => continue,
            };
            if prefix_len < self.config.min_shared_prefix_len {
                continue;
            }

            let mut guard = match shared_cache.lock() {
                Ok(g) => g,
                _ => continue,
            };

            // Check source has enough KV data
            let src_ok = guard
                .num_tokens(src_id)
                .map_or(false, |n| n >= prefix_len);
            if !src_ok {
                continue;
            }

            let shared = guard.share_prefix_in_blocks(sid, src_id, prefix_len);
            drop(guard);

            if shared > 0 {
                debug!(
                    "Paged prefix sharing: {} blocks shared for seq {} from seq {}",
                    shared, sid, src_id
                );
                if let Some(seq) = self.sequences.get_mut(&sid) {
                    seq.prompt_pos = prefix_len.min(seq.prompt.len());
                }
            }
        }
    }

    fn decode_token(&self, token_id: u32) -> String {
        if let Some(ref tok) = self.tokenizer {
            tok.lock()
                .unwrap_or_else(|e| e.into_inner())
                .decode(&[token_id])
        } else {
            format!("[{}]", token_id)
        }
    }

    /// Extract a CpuKVCache from a Box<dyn KVCacheProvider>.
    /// For CpuKVCache, takes entries directly. For GpuKVCache, reads back K/V from GPU.
    /// Extract CPU entries for CPU fallback forwarding.
    /// For GPU-backed caches, creates an empty cache — entries are populated
    /// on demand by `forward_with_kv` (safe, but loses GPU KV context).
    fn boxed_cache_to_cpu(cache: &mut Box<dyn KVCacheProvider>) -> CpuKVCache {
        if let Some(cpu_entries) = cache.as_cpu_entries() {
            return CpuKVCache {
                entries: std::mem::take(cpu_entries),
            };
        }
        CpuKVCache::new(0)
    }

    pub fn add_request(&mut self, request: InferenceRequest) -> u64 {
        if self.should_shed_load() {
            self.rejected_count += 1;
            warn!(
                "ContinuousBatchingEngine: load shedding active — rejecting request {} (queue={}, rejected={})",
                self.rejected_count,
                self.sequences.len(),
                self.rejected_count
            );
            return 0;
        }
        if self.sequences.len() >= self.max_total_sequences {
            warn!(
                "ContinuousBatchingEngine: max sequences ({}) reached",
                self.max_total_sequences
            );
            return 0;
        }
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let mut seq = Sequence::from_request(seq_id, &request);
        seq.state = SeqState::Prefilling;

        if let Some(ref shared_cache) = self.shared_paged {
            let provider = crate::paged_provider::PagedKVCacheProvider::new_shared(
                seq_id,
                shared_cache.clone(),
                self.num_layers,
            );
            self.kv_caches.insert(seq_id, Box::new(provider));

            // Register in prefix trie for block-level prefix sharing
            if let Some(ref mut trie) = self.prefix_trie {
                trie.insert(seq_id, &request.input_tokens);
            }
        } else if self.use_gpu {
            #[cfg(feature = "gpu")]
            if self.num_layers > 0 {
                let gpu_cache = GpuKVCache::new(
                    self.num_layers,
                    self.num_kv_heads,
                    self.head_dim,
                    self.max_seq_len,
                );
                self.kv_caches.insert(seq_id, Box::new(gpu_cache));
            } else {
                let cache = self.model.reset_cache();
                self.kv_caches.insert(seq_id, Box::new(cache));
            }
            #[cfg(not(feature = "gpu"))]
            {
                let cache = self.model.reset_cache();
                self.kv_caches.insert(seq_id, Box::new(cache));
            }
        } else {
            let cache = self.model.reset_cache();
            self.kv_caches.insert(seq_id, Box::new(cache));
        }

        // Register in prefix trie for shared prefill
        if let Some(ref mut trie) = self.prefix_trie {
            trie.insert(seq_id, &request.input_tokens);
        }

        self.sequences.insert(seq_id, seq);

        let mut sampler = Sampler::new(SamplingConfig {
            method: crate::sampler::SamplingMethod::TemperatureTopKTopP,
            temperature: request.temperature,
            top_k: request.top_k as usize,
            top_p: request.top_p,
            min_prob: 0.0,
            seed: None,
        });
        sampler.set_use_gpu(self.use_gpu);
        self.samplers.insert(seq_id, sampler);

        seq_id
    }

    /// Run a single step, processing up to `max_batch_size` ready sequences.
    ///
    /// Prefill sequences process ALL remaining prompt tokens individually before
    /// transitioning to generation (future: padded batch prefill). Generation
    /// sequences process one token each via [`ModelForward::forward_batched`].
    /// Select ready sequences according to the scheduling policy.
    /// Returns a prioritized list of up to `max_batch_size` sequence IDs.
    pub(crate) fn select_ready_sequences(&self) -> Vec<(u64, bool)> {
        let now = Instant::now();
        let mut candidates: Vec<(u64, bool, f64)> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.is_ready())
            .map(|(&id, s)| {
                let is_prefill = s.has_pending_prompt();
                let priority = match self.config.scheduling_policy {
                    SchedulingPolicy::Fifo => {
                        // Positive elapsed: older = larger value → sorts first (descending)
                        now.duration_since(s.created_at).as_secs_f64()
                    }
                    SchedulingPolicy::PriorityAging => {
                        let base = if is_prefill { 10.0 } else { 5.0 };
                        let age_ms = now.duration_since(s.created_at).as_millis() as f64;
                        let age_boost = age_ms * self.config.aging_boost_per_ms;
                        base + age_boost
                    }
                    SchedulingPolicy::ShortestRemaining => {
                        let remaining = s.max_tokens.saturating_sub(s.generated.len() as u32) as f64;
                        // Negative so smaller remaining = higher priority
                        -remaining
                    }
                    SchedulingPolicy::TokenBucket => {
                        let tokens_generated = s.generated.len() as f64;
                        let age_ms = now.duration_since(s.created_at).as_millis() as f64;
                        tokens_generated / (1.0 + age_ms * self.config.aging_boost_per_ms)
                    }
                };
                (id, is_prefill, priority)
            })
            .collect();

        // Sort by priority descending
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));

        // Take up to effective batch size (may be dynamically tuned)
        let batch_size = self.effective_batch_size();
        candidates
            .into_iter()
            .take(batch_size)
            .map(|(id, is_prefill, _)| (id, is_prefill))
            .collect()
    }

    pub fn step(&mut self) -> StepResult {
        let step_start = Instant::now();
        let mut completed = Vec::with_capacity(self.max_batch_size.min(self.sequences.len()));

        let ready = self.select_ready_sequences();

        if ready.is_empty() {
            let active_count = self.sequences.values().filter(|s| !s.is_finished()).count();
            return StepResult {
                completed,
                active_count,
                idle: active_count == 0,
                batch_size: 0,
                padding_waste: 0.0,
                prefill_time_us: 0,
                decode_time_us: 0,
                prefill_tokens: 0,
                decode_tokens: 0,
            };
        }

        // Separate prefill and generation sequences
        let mut prefill_ids: Vec<u64> = Vec::new();
        let mut gen_ids: Vec<u64> = Vec::new();
        for &(sid, is_prefill) in &ready {
            if is_prefill {
                prefill_ids.push(sid);
            } else {
                gen_ids.push(sid);
            }
        }

        // Compute padding waste: fraction of unused batch capacity
        let total_in_batch = prefill_ids.len() + gen_ids.len();
        let effective_bs = self.effective_batch_size();
        let padding_waste = if effective_bs > 0 {
            1.0 - (total_in_batch as f64 / effective_bs as f64)
        } else {
            0.0
        };

        let mut prefill_time_us: u64 = 0;
        let mut decode_time_us: u64 = 0;
        let mut prefill_tokens: usize = 0;
        let mut decode_tokens: usize = 0;

        // --- PHASE 0: GPU prefix cache sharing ---
        if !prefill_ids.is_empty() {
            self.try_gpu_prefix_sharing(&prefill_ids);
            self.try_paged_prefix_sharing(&prefill_ids);
        }

        // --- PHASE 1: Position-by-position batched prefill ---
        let phase1_start = Instant::now();
        if !prefill_ids.is_empty() {
            // Count remaining prompt tokens for prefill tracking
            let prefill_tokens_before: usize = prefill_ids.iter().filter_map(|sid| {
                self.sequences.get(sid).map(|seq| seq.prompt.len().saturating_sub(seq.prompt_pos))
            }).sum();

            // Extract KV caches for prefill sequences
            let mut prefill_caches: Vec<(u64, Box<dyn KVCacheProvider>)> = prefill_ids
                .iter()
                .filter_map(|&sid| self.kv_caches.remove(&sid).map(|c| (sid, c)))
                .collect();

            let mut prefill_logits: Vec<(u64, Array1<f32>)> = Vec::with_capacity(prefill_ids.len());

            // Drain caches and prompts from prefill_caches
            let mut caches_only: Vec<Box<dyn KVCacheProvider>> =
                prefill_caches.drain(..).map(|(_, c)| c).collect();
            let mut prompt_tokens: Vec<&[u32]> = Vec::with_capacity(caches_only.len());
            for sid in &prefill_ids {
                if let Some(seq) = self.sequences.get(sid) {
                    prompt_tokens.push(&seq.prompt[seq.prompt_pos..]);
                }
            }

            prefill_tokens = prefill_tokens_before;

            let logits_vec = self
                .model
                .forward_prefill_batched(&prompt_tokens, &mut caches_only);

            // Re-pair caches with their IDs (positional, same order as prefill_ids)
            for (i, &sid) in prefill_ids.iter().enumerate() {
                if i < caches_only.len() {
                    let cache =
                        std::mem::replace(&mut caches_only[i], Box::new(CpuKVCache::new(0)));
                    prefill_caches.push((sid, cache));
                }
                if i < logits_vec.len() {
                    prefill_logits.push((sid, logits_vec[i].clone()));
                }
            }

            // Restore all prefill KV caches and update states to Generating
            for (sid, cache) in prefill_caches {
                self.kv_caches.insert(sid, cache);
                // For GPU caches, also sync back to CPU if needed
                if let Some(seq) = self.sequences.get_mut(&sid) {
                    seq.prompt_pos = seq.prompt.len();
                    seq.state = SeqState::Generating;
                }
            }

            // Count tokens processed in prefill
            for sid in &prefill_ids {
                if let Some(seq) = self.sequences.get(sid) {
                    prefill_tokens += seq.prompt.len().saturating_sub(seq.prompt_pos - seq.prompt.len().min(seq.prompt_pos));
                }
            }

            // Generate first output token for each prefill sequence using
            // the logits from the batched prefill forward pass
            for (sid, logits_arr) in prefill_logits {
                let logits_vec: Vec<f32> = logits_arr.into_raw_vec();

                let sampler = match self.samplers.get_mut(&sid) {
                    Some(s) => s,
                    None => continue,
                };
                let token_id = match sampler.sample(&logits_vec) {
                    Ok(idx) => u32::try_from(idx).unwrap_or(u32::MAX),
                    Err(e) => {
                        warn!(
                            "Sampler failed for prefill sequence {}, error: {:?}, falling back to argmax",
                            sid, e
                        );
                        logits_vec
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .map(|(i, _)| u32::try_from(i).unwrap_or(u32::MAX))
                            .unwrap_or(0)
                    }
                };

                let token_text = self.decode_token(token_id);
                let seq = match self.sequences.get_mut(&sid) {
                    Some(s) => s,
                    None => continue,
                };
                let log_prob = logits_vec.get(token_id as usize).copied().unwrap_or(0.0);
                let pos = seq.prompt.len() + seq.generated.len();
                seq.push_token(Arc::new(GeneratedToken::new(
                    token_id, token_text, log_prob, pos,
                )));

                let mut finish_reason: Option<FinishReason> = None;
                if token_id == seq.eos_token_id && !seq.generated.is_empty() {
                    finish_reason = Some(FinishReason::EndOfSequence);
                } else if seq.reached_max_tokens() {
                    finish_reason = Some(FinishReason::MaxTokens);
                }

                if let Some(reason) = finish_reason {
                    seq.finish(reason.clone());
                    if let Some(resp) =
                        self.build_response(sid, reason, step_start.elapsed().as_millis() as u64)
                    {
                        completed.push(resp);
                    }
                }
            }
        }

        // Record prefill timing
        prefill_time_us = phase1_start.elapsed().as_micros() as u64;

        // --- PHASE 2: Batched generation for non-prefill sequences ---
        let phase2_start = Instant::now();
        if !gen_ids.is_empty() {
            let mut gen_boxed_caches: Vec<Box<dyn KVCacheProvider>> =
                Vec::with_capacity(gen_ids.len());
            let mut gen_inputs: Vec<u32> = Vec::with_capacity(gen_ids.len());
            let mut gen_samplers: Vec<(u64, f32, usize, f32)> = Vec::with_capacity(gen_ids.len());

            for &sid in &gen_ids {
                if let Some(cache) = self.kv_caches.remove(&sid) {
                    gen_boxed_caches.push(cache);
                }
                if let Some(s) = self.sequences.get(&sid) {
                    let next_token = s.next_input_token();
                    if next_token.is_none() {
                        warn!("Sequence {} has no input token, using BOS(0)", sid);
                    }
                    gen_inputs.push(next_token.unwrap_or(0));
                    gen_samplers.push((sid, s.temperature, s.top_k as usize, s.top_p));
                }
            }

            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            // Pre-sampled token IDs from GPU path (None = use CPU sampler)
            let mut gpu_tokens: Vec<Option<u32>> = vec![None; gen_ids.len()];

            // Try GPU-native batched sampling first (zero logit readback).
            // Falls back to CPU readback + CPU sampling if GPU path fails or returns None.
            let all_logits: Vec<Array1<f32>> = if gen_ids.len() > 1 && self.use_gpu {
                let temperatures: Vec<f32> = gen_samplers.iter().map(|&(_, t, _, _)| t).collect();
                let top_ks: Vec<usize> = gen_samplers.iter().map(|&(_, _, k, _)| k).collect();
                let top_ps: Vec<f32> = gen_samplers.iter().map(|&(_, _, _, p)| p).collect();
                let gpu_results = self.model.forward_batched_sample_gpu(
                    &gen_inputs,
                    &mut gen_boxed_caches,
                    &temperatures,
                    &top_ks,
                    &top_ps,
                    seed,
                );
                let all_some = gpu_results.iter().all(|r| r.is_some());
                if all_some {
                    for (i, tok) in gpu_results.into_iter().enumerate() {
                        gpu_tokens[i] = tok;
                    }
                    #[cfg(feature = "gpu")]
                    if self.config.use_paged_cache {
                        PagedKVCacheProvider::sync_all_gpu_to_paged_slice(
                            &mut gen_boxed_caches,
                        );
                    }
                    vec![Array1::zeros(gen_ids.len())]
                } else {
                    // Fallback: extract CPU entries for full logit readback
                    let mut caches: Vec<CpuKVCache> = gen_boxed_caches
                        .iter_mut()
                        .map(|c| Self::boxed_cache_to_cpu(c))
                        .collect();
                    self.model.forward_batched(&gen_inputs, &mut caches)
                }
            } else if gen_ids.len() > 1 {
                let mut caches: Vec<CpuKVCache> = gen_boxed_caches
                    .iter_mut()
                    .map(|c| Self::boxed_cache_to_cpu(c))
                    .collect();
                self.model.forward_batched(&gen_inputs, &mut caches)
            } else if gen_ids.len() == 1 && !gen_boxed_caches.is_empty() {
                if self.use_gpu {
                    let cache = &mut *gen_boxed_caches[0];
                    let (_, temperature, top_k, top_p) = gen_samplers[0];
                    if let Some(tok) = self.model.forward_sample_gpu(
                        &gen_inputs,
                        cache,
                        temperature,
                        top_k,
                        top_p,
                        seed.wrapping_add(0),
                    ) {
                        gpu_tokens[0] = Some(tok);
                        #[cfg(feature = "gpu")]
                        if self.config.use_paged_cache {
                            PagedKVCacheProvider::sync_all_gpu_to_paged_slice(
                                &mut gen_boxed_caches,
                            );
                        }
                        vec![Array1::zeros(1)]
                    } else {
                        let mut caches: Vec<CpuKVCache> = gen_boxed_caches
                            .iter_mut()
                            .map(|c| Self::boxed_cache_to_cpu(c))
                            .collect();
                        vec![self.model.forward(&gen_inputs, &mut caches[0])]
                    }
                } else {
                    let mut caches: Vec<CpuKVCache> = gen_boxed_caches
                        .iter_mut()
                        .map(|c| Self::boxed_cache_to_cpu(c))
                        .collect();
                    vec![self.model.forward(&gen_inputs, &mut caches[0])]
                }
            } else {
                warn!("No cache available for generation forward pass — skipping");
                Vec::new()
            };

            // Restore generation caches
            for (i, sid) in gen_ids.iter().enumerate() {
                if i < gen_boxed_caches.len() {
                    let cache =
                        std::mem::replace(&mut gen_boxed_caches[i], Box::new(CpuKVCache::new(0)));
                    self.kv_caches.insert(*sid, cache);
                }
            }

            // Process each generation result
            for (seq_idx, (&seq_id, logits_arr)) in
                gen_ids.iter().zip(all_logits.iter()).enumerate()
            {
                let logits_slice = match logits_arr.as_slice() {
                    Some(s) => s,
                    None => {
                        warn!("Non-contiguous logits array for sequence {}, skipping", seq_id);
                        continue;
                    }
                };

                let token_id = if let Some(gpu_tok) = gpu_tokens[seq_idx] {
                    gpu_tok
                } else {
                    let sampler = match self.samplers.get_mut(&seq_id) {
                        Some(s) => s,
                        None => {
                            warn!("Sampler not found for sequence {}, skipping", seq_id);
                            continue;
                        }
                    };
                    match sampler.sample(logits_slice) {
                        Ok(idx) => u32::try_from(idx).unwrap_or(u32::MAX),
                        Err(e) => {
                            warn!(
                                "Sampler failed for sequence {}, error: {:?}, falling back to argmax",
                                seq_id, e
                            );
                            logits_slice
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                .map(|(i, _)| u32::try_from(i).unwrap_or(u32::MAX))
                                .unwrap_or(0)
                        }
                    }
                };

                let token_text = self.decode_token(token_id);

                let seq = match self.sequences.get_mut(&seq_id) {
                    Some(s) => s,
                    None => {
                        warn!("Sequence {} not found, skipping", seq_id);
                        continue;
                    }
                };

                let log_prob = if gpu_tokens[seq_idx].is_some() {
                    0.0
                } else {
                    logits_slice
                        .get(token_id as usize)
                        .copied()
                        .unwrap_or_else(|| {
                            warn!(
                                "token_id {} out of range for logits of length {}",
                                token_id,
                                logits_slice.len()
                            );
                            0.0
                        })
                };
                let pos = seq.prompt.len() + seq.generated.len();
                seq.push_token(Arc::new(GeneratedToken::new(
                    token_id, token_text, log_prob, pos,
                )));

                let mut finish_reason: Option<FinishReason> = None;
                if token_id == seq.eos_token_id && !seq.generated.is_empty() {
                    finish_reason = Some(FinishReason::EndOfSequence);
                } else if seq.reached_max_tokens() {
                    finish_reason = Some(FinishReason::MaxTokens);
                }

                if let Some(reason) = finish_reason {
                    seq.finish(reason.clone());
                    if let Some(resp) =
                        self.build_response(seq_id, reason, step_start.elapsed().as_millis() as u64)
                    {
                        completed.push(resp);
                    }
                }
            }
        }

        let active_count = self.sequences.values().filter(|s| !s.is_finished()).count();
        let waiting_count = self.sequences.len();

        // Count total tokens processed in this step for throughput tracking
        let tokens_in_step = {
            let mut count = 0usize;
            for sid in prefill_ids.iter().chain(gen_ids.iter()) {
                if let Some(seq) = self.sequences.get(sid) {
                    // Prefill: all remaining prompt tokens were processed.
                    // Generation: 1 new token was generated.
                    count += seq.total_tokens().saturating_sub(seq.prompt.len().max(seq.prompt_pos));
                }
            }
            count.max(1)
        };

        // Record decode timing and count decoded tokens
        decode_time_us = phase2_start.elapsed().as_micros() as u64;
        decode_tokens = gen_ids.len();

        // Update adaptive batching and load shedding after each step
        self.update_adaptive_batch_size(tokens_in_step);

        // Update cumulative benchmark statics
        use crate::inference_trait::*;
        TOTAL_PREFILL_TIME_US.fetch_add(prefill_time_us, std::sync::atomic::Ordering::Relaxed);
        TOTAL_DECODE_TIME_US.fetch_add(decode_time_us, std::sync::atomic::Ordering::Relaxed);
        TOTAL_PREFILL_TOKENS.fetch_add(prefill_tokens as u64, std::sync::atomic::Ordering::Relaxed);
        TOTAL_DECODE_TOKENS.fetch_add(decode_tokens as u64, std::sync::atomic::Ordering::Relaxed);
        TOTAL_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Update global observability atomics
        crate::inference_trait::SCHEDULER_QUEUE_DEPTH
            .store(waiting_count as i64, std::sync::atomic::Ordering::Relaxed);
        crate::inference_trait::BATCHING_PADDING_WASTE_SUM
            .fetch_add((padding_waste * 100.0) as u64, std::sync::atomic::Ordering::Relaxed);
        crate::inference_trait::BATCHING_COUNT
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Track load shedding metrics
        if self.shed_load {
            crate::inference_trait::SCHEDULER_QUEUE_DEPTH
                .store(-(self.rejected_count as i64).max(1), std::sync::atomic::Ordering::Relaxed);
        }

        StepResult {
            completed,
            active_count,
            idle: ready.is_empty() && active_count == 0,
            batch_size: total_in_batch,
            padding_waste,
            prefill_time_us,
            decode_time_us,
            prefill_tokens,
            decode_tokens,
        }
    }

    fn build_response(
        &self,
        seq_id: u64,
        reason: FinishReason,
        elapsed_ms: u64,
    ) -> Option<InferenceResponse> {
        let seq = self.sequences.get(&seq_id)?;
        let text: String = seq
            .generated
            .iter()
            .map(|t| t.token_text.to_string())
            .collect();
        Some(InferenceResponse {
            request_id: seq.request_id,
            tokens: seq.generated.iter().map(|t| (**t).clone()).collect(),
            text,
            finish_reason: reason,
            total_tokens: seq.total_tokens(),
            inference_time_ms: elapsed_ms,
            metadata: HashMap::new(),
        })
    }

    pub fn active_count(&self) -> usize {
        self.sequences.values().filter(|s| !s.is_finished()).count()
    }

    pub fn drain_completed(&mut self) -> Vec<InferenceResponse> {
        let drain_start = Instant::now();
        let to_remove: Vec<u64> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.is_finished())
            .map(|(id, _)| *id)
            .collect();
        let mut results = Vec::with_capacity(to_remove.len());

        for seq_id in to_remove {
            self.samplers.remove(&seq_id);
            self.kv_caches.remove(&seq_id);
            if let Some(seq) = self.sequences.remove(&seq_id) {
                let reason = match &seq.state {
                    SeqState::Finished(r) => r.clone(),
                    _ => continue,
                };
                let total_tokens = seq.total_tokens();
                let text: String = seq
                    .generated
                    .iter()
                    .map(|t| t.token_text.to_string())
                    .collect();
                results.push(InferenceResponse {
                    request_id: seq.request_id,
                    tokens: seq.generated.iter().map(|t| (**t).clone()).collect(),
                    text,
                    finish_reason: reason,
                    total_tokens,
                    inference_time_ms: drain_start.elapsed().as_millis() as u64,
                    metadata: HashMap::new(),
                });
            }
        }

        results
    }

    pub fn get_sequence(&self, seq_id: u64) -> Option<&Sequence> {
        self.sequences.get(&seq_id)
    }

    pub fn get_sequence_mut(&mut self, seq_id: u64) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id)
    }
}
