use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array1;
use nexora_tokenizer::BpeTokenizer;
use tracing::{debug, warn};

use crate::sampler::{Sampler, SamplingConfig};
use crate::sequence_state::{SeqState, Sequence};
use crate::{FinishReason, GeneratedToken, InferenceRequest, InferenceResponse};
#[cfg(feature = "gpu")]
use nexora_transformer::{GpuKVCache, GpuKVCacheEntry};
use nexora_transformer::{CpuKVCache, KVCacheProvider};

/// Configuration for continuous batching engine.
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
    /// Max physical blocks for paged cache. 0 = auto.
    pub paged_max_blocks: usize,
}

impl Default for ContinuousBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_total_sequences: 1024,
            min_shared_prefix_len: 4,
            enable_prefix_sharing: true,
            use_paged_cache: false,
            paged_block_size: 0,
            paged_max_blocks: 0,
        }
    }
}

/// A simple prefix tree (trie) for detecting shared prefixes across sequences.
/// When `enable_prefix_sharing` is on, sequences with matching prefixes share
/// the initial KV cache computation — the common prefix is processed once
/// and the result is copied to each sequence's KV cache.
#[derive(Debug, Default)]
struct PrefixTrieNode {
    children: HashMap<u32, PrefixTrieNode>,
    /// Sequence IDs that share this prefix
    seq_ids: Vec<u64>,
}

#[derive(Debug)]
struct PrefixTrie {
    root: PrefixTrieNode,
}

impl PrefixTrie {
    fn new() -> Self {
        Self {
            root: PrefixTrieNode::default(),
        }
    }

    /// Insert a sequence's prompt into the trie (full walk).
    fn insert(&mut self, seq_id: u64, prompt: &[u32]) {
        let mut node = &mut self.root;
        for &token in prompt {
            node = node.children.entry(token).or_default();
            node.seq_ids.push(seq_id);
        }
    }

    /// Find the longest prefix of `prompt` shared with another sequence.
    /// Returns `(prefix_len, other_seq_id)` where `other_seq_id` is another
    /// sequence with a matching prefix at that depth. Excludes `exclude_seq_id`.
    /// Returns `None` when no shared prefix of any length exists.
    /// Handles partial matches: if prompt diverges at token P+1, the match
    /// at depth P is still returned.
    fn find_shared_prefix(
        &self,
        prompt: &[u32],
        exclude_seq_id: u64,
    ) -> Option<(usize, u64)> {
        let mut node = &self.root;
        let mut prefix_len = 0usize;
        let mut result: Option<(usize, u64)> = None;
        for &token in prompt {
            match node.children.get(&token) {
                Some(child) => {
                    node = child;
                    prefix_len += 1;
                    if let Some(&other) = node
                        .seq_ids
                        .iter()
                        .find(|&&id| id != exclude_seq_id)
                    {
                        result = Some((prefix_len, other));
                    }
                }
                None => break,
            }
        }
        result
    }
}

/// Result of a single sequential batching step.
#[derive(Debug)]
pub struct StepResult {
    pub completed: Vec<InferenceResponse>,
    pub active_count: usize,
    pub idle: bool,
}

/// A continuous batching engine that manages multiple sequences.
///
/// Each step collects all ready sequences and processes them through the model's
/// batched forward pass (`forward_batched`), then samples the next token for each.
/// Prefill sequences process ALL remaining prompt tokens at once (full prefill)
/// before moving to the generation phase.
/// Sequences dynamically enter (via `add_request`) and leave (via completion or error).
///
/// Generation phase provides true batched matmul via [`ModelForward::forward_batched`].
/// Prefill uses position-by-position batched processing: at each position P,
/// ALL sequences still in prefill have their P-th remaining token processed
/// together via [`ModelForward::forward_prefill_batched`].
///
/// To use, set `InferenceConfig::use_continuous_batching = true`.
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
    config: ContinuousBatchingConfig,
    prefix_trie: Option<PrefixTrie>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    paged_cache_dim_set: bool,
    /// Shared paged cache instance. All PagedKVCacheProviders use the same pool.
    shared_paged: Option<std::sync::Arc<std::sync::Mutex<crate::paged_cache::PagedKVCache>>>,
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
            let paged_config = crate::paged_cache::PagedCacheConfig {
                block_size,
                max_blocks,
                num_layers,
                num_kv_heads,
                head_dim,
                max_seq_len,
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
            max_batch_size: config.max_batch_size,
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
            config,
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            max_seq_len: 0,
            paged_cache_dim_set: false,
            shared_paged: None,
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

    /// Copy shared-prefix K/V from previously-prefilled sequences into prefill
    /// sequences that share a prompt prefix. Updates `prompt_pos` to skip the
    /// prefix, so the prefill forward pass only processes non-shared tokens.
    /// Only operates when GPU is active and the prefix trie is available.
    fn try_gpu_prefix_sharing(&mut self, prefill_ids: &[u64]) {
        if !self.use_gpu || self.prefix_trie.is_none() {
            return;
        }
        #[cfg(feature = "gpu")]
        {
            let trie = self.prefix_trie.as_ref().unwrap();
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
                    s.prompt.clone()
                }
                None => continue,
            };

            let (prefix_len, src_id) = match trie.find_shared_prefix(&prompt, sid) {
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
    /// Used to provide CPU caches to `forward_batched`.
    fn boxed_cache_to_cpu(cache: &mut Box<dyn KVCacheProvider>) -> CpuKVCache {
        if let Some(cpu_entries) = cache.as_cpu_entries() {
            CpuKVCache {
                entries: std::mem::take(cpu_entries),
            }
        } else {
            CpuKVCache::new(0)
        }
    }

    pub fn add_request(&mut self, request: InferenceRequest) -> u64 {
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
    pub fn step(&mut self) -> StepResult {
        let step_start = Instant::now();
        let mut completed = Vec::with_capacity(self.max_batch_size.min(self.sequences.len()));

        let ready_ids: Vec<u64> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.is_ready())
            .take(self.max_batch_size)
            .map(|(id, _)| *id)
            .collect();

        if ready_ids.is_empty() {
            let active_count = self.sequences.values().filter(|s| !s.is_finished()).count();
            return StepResult {
                completed,
                active_count,
                idle: active_count == 0,
            };
        }

        // Separate prefill and generation sequences
        let mut prefill_ids: Vec<u64> = Vec::new();
        let mut gen_ids: Vec<u64> = Vec::new();
        for &sid in &ready_ids {
            if let Some(s) = self.sequences.get(&sid) {
                if s.has_pending_prompt() {
                    prefill_ids.push(sid);
                } else {
                    gen_ids.push(sid);
                }
            }
        }

        // --- PHASE 0: GPU prefix cache sharing ---
        // Before executing prefill, copy K/V from previously-prefilled sequences
        // that share a prompt prefix, so we can skip the common prefix computation.
        if !prefill_ids.is_empty() {
            self.try_gpu_prefix_sharing(&prefill_ids);
            self.try_paged_prefix_sharing(&prefill_ids);
        }

        // --- PHASE 1: Position-by-position batched prefill ---
        // All prefill sequences' remaining prompt tokens are processed via
        // [`forward_prefill_batched`], which processes position-by-position:
        // at each position P, ALL sequences still in prefill have their P-th
        // remaining token processed together. This replaces the old per-sequence
        // prefill loop and provides the foundation for true GPU batched prefill.
        if !prefill_ids.is_empty() {
            // Extract KV caches for prefill sequences
            let mut prefill_caches: Vec<(u64, Box<dyn KVCacheProvider>)> = prefill_ids
                .iter()
                .filter_map(|&sid| self.kv_caches.remove(&sid).map(|c| (sid, c)))
                .collect();

            let mut prefill_logits: Vec<(u64, Array1<f32>)> = Vec::with_capacity(prefill_ids.len());

            // Drain caches and prompts from prefill_caches
            let mut caches_only: Vec<Box<dyn KVCacheProvider>> =
                prefill_caches.drain(..).map(|(_, c)| c).collect();
            let mut prompt_tokens: Vec<Vec<u32>> = Vec::with_capacity(caches_only.len());
            for sid in &prefill_ids {
                if let Some(seq) = self.sequences.get(sid) {
                    prompt_tokens.push(seq.prompt[seq.prompt_pos..].to_vec());
                }
            }

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

        // --- PHASE 2: Batched generation for non-prefill sequences ---
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
                let logits_vec: Vec<f32> = logits_arr.clone().into_raw_vec();

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
                    match sampler.sample(&logits_vec) {
                        Ok(idx) => u32::try_from(idx).unwrap_or(u32::MAX),
                        Err(e) => {
                            warn!(
                                "Sampler failed for sequence {}, error: {:?}, falling back to argmax",
                                seq_id, e
                            );
                            logits_vec
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
                    logits_vec
                        .get(token_id as usize)
                        .copied()
                        .unwrap_or_else(|| {
                            warn!(
                                "token_id {} out of range for logits of length {}",
                                token_id,
                                logits_vec.len()
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
        StepResult {
            completed,
            active_count,
            idle: ready_ids.is_empty() && active_count == 0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_trait::ModelForward;
    use ndarray::Array1;
    use nexora_transformer::KVCacheProvider;

    struct MockModel {
        vocab_size: usize,
    }

    impl ModelForward for MockModel {
        fn forward(&self, input_ids: &[u32], _kv_cache: &mut dyn KVCacheProvider) -> Array1<f32> {
            let mut logits = Array1::zeros(self.vocab_size);
            if let Some(&tid) = input_ids.last() {
                let idx = (tid as usize).min(self.vocab_size - 1);
                logits[idx] = 10.0;
            } else {
                logits[0] = 10.0;
            }
            logits
        }

        fn reset_cache(&self) -> CpuKVCache {
            CpuKVCache::new(0)
        }
    }

    fn test_request(input_tokens: Vec<u32>, max_tokens: u32) -> InferenceRequest {
        InferenceRequest {
            input_tokens,
            max_tokens,
            ..Default::default()
        }
    }

    #[test]
    fn test_add_sequence_increases_active_count() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = ContinuousBatchingEngine::new(model, 4);
        let seq_id = engine.add_request(test_request(vec![1, 2, 3], 10));
        assert!(engine.get_sequence(seq_id).is_some());
        assert_eq!(engine.active_count(), 1);
    }

    #[test]
    fn test_step_prefill_then_generate() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = ContinuousBatchingEngine::new(model, 4);
        engine.add_request(test_request(vec![5, 10], 20));

        let r = engine.step();
        assert_eq!(r.active_count, 1);
        assert!(r.completed.is_empty());

        let r = engine.step();
        assert_eq!(r.active_count, 1);
        assert!(r.completed.is_empty());

        let seq = engine.get_sequence(1).unwrap();
        assert!(!seq.generated.is_empty());
    }

    #[test]
    fn test_sequence_completes_at_max_tokens() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = ContinuousBatchingEngine::new(model, 4);
        engine.add_request(test_request(vec![1, 2], 4));

        let r = engine.step();
        assert!(r.completed.is_empty());
        let r = engine.step();
        assert!(r.completed.is_empty());
        let r = engine.step();
        assert!(!r.completed.is_empty());
        assert_eq!(r.completed[0].finish_reason, FinishReason::MaxTokens);
    }

    #[test]
    fn test_multiple_sequences() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = ContinuousBatchingEngine::new(model, 4);
        engine.add_request(test_request(vec![1, 2], 5));
        engine.add_request(test_request(vec![3, 4, 5], 6));
        assert_eq!(engine.active_count(), 2);

        let mut steps = 0;
        while engine.active_count() > 0 && steps < 30 {
            let r = engine.step();
            steps += 1;
            for resp in &r.completed {
                assert!(matches!(
                    resp.finish_reason,
                    FinishReason::MaxTokens | FinishReason::EndOfSequence
                ));
            }
        }
        assert!(steps < 30, "Should complete within 30 steps (was {steps})");
    }

    #[test]
    fn test_drain_completed() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = ContinuousBatchingEngine::new(model, 4);
        engine.add_request(test_request(vec![1], 2));
        for _ in 0..10 {
            let r = engine.step();
            if !r.completed.is_empty() {
                break;
            }
        }
        let drained = engine.drain_completed();
        assert!(!drained.is_empty());
        assert_eq!(engine.active_count(), 0);
    }

    #[test]
    fn test_idle_step_empty_no_panic() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = ContinuousBatchingEngine::new(model, 4);
        let r = engine.step();
        assert!(r.idle);
        assert!(r.completed.is_empty());
        assert_eq!(r.active_count, 0);
    }
}
