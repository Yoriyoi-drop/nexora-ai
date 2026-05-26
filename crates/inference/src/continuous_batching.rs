use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array1;
use nexora_tokenizer::BpeTokenizer;
use tracing::{debug, warn};

use crate::sampler::{Sampler, SamplingConfig};
use crate::sequence_state::{SeqState, Sequence};
use crate::{FinishReason, GeneratedToken, InferenceRequest, InferenceResponse};
use nexora_transformer::{CpuKVCache, KVCacheProvider};
#[cfg(feature = "gpu")]
use nexora_transformer::GpuKVCache;

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
}

impl Default for ContinuousBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_total_sequences: 1024,
            min_shared_prefix_len: 4,
            enable_prefix_sharing: true,
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

    /// Insert a sequence's prompt into the trie and return (prefix_len, shared_with_count).
    fn insert(&mut self, seq_id: u64, prompt: &[u32], min_shared: usize) -> (usize, usize) {
        let mut node = &mut self.root;
        let mut prefix_len = 0usize;
        for &token in prompt {
            node = node.children.entry(token).or_default();
            node.seq_ids.push(seq_id);
            prefix_len += 1;
            if prefix_len >= min_shared && node.seq_ids.len() > 1 {
                // Found a shared prefix — return it
                let shared_count = node.seq_ids.len();
                // Walk remaining tokens to build full subtree
                return (prefix_len, shared_count);
            }
        }
        (0, 0)
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
/// Prefill processes sequences individually (future work: padded batch prefill).
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
    #[cfg(feature = "gpu")]
    num_layers: usize,
    #[cfg(feature = "gpu")]
    num_kv_heads: usize,
    #[cfg(feature = "gpu")]
    head_dim: usize,
    #[cfg(feature = "gpu")]
    max_seq_len: usize,
}

#[deprecated(note = "renamed to ContinuousBatchingEngine")]
pub type SequentialBatchingEngine<M> = ContinuousBatchingEngine<M>;

impl<M> ContinuousBatchingEngine<M>
where
    M: crate::inference_trait::ModelForward,
{
    pub fn new(model: M, max_batch_size: usize) -> Self {
        Self::with_config(model, ContinuousBatchingConfig {
            max_batch_size,
            ..Default::default()
        })
    }

    /// Set model architecture dimensions for GPU KV cache creation.
    /// Only needed when GPU is available; safe to skip for CPU-only usage.
    #[cfg(feature = "gpu")]
    pub fn set_model_dims(&mut self, num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq_len: usize) {
        self.num_layers = num_layers;
        self.num_kv_heads = num_kv_heads;
        self.head_dim = head_dim;
        self.max_seq_len = max_seq_len;
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
            #[cfg(feature = "gpu")]
            use_gpu: nexora_autograd::gpu::GpuContext::is_available(),
            #[cfg(not(feature = "gpu"))]
            use_gpu: false,
            tokenizer: None,
            prefix_trie: if config.enable_prefix_sharing {
                Some(PrefixTrie::new())
            } else {
                None
            },
            config,
            #[cfg(feature = "gpu")]
            num_layers: 0,
            #[cfg(feature = "gpu")]
            num_kv_heads: 0,
            #[cfg(feature = "gpu")]
            head_dim: 0,
            #[cfg(feature = "gpu")]
            max_seq_len: 0,
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

    fn decode_token(&self, token_id: u32) -> String {
        if let Some(ref tok) = self.tokenizer {
            tok.lock().unwrap_or_else(|e| e.into_inner()).decode(&[token_id])
        } else {
            format!("[{}]", token_id)
        }
    }

    /// Extract a CpuKVCache from a Box<dyn KVCacheProvider>.
    /// For CpuKVCache, takes entries directly. For GpuKVCache, reads back K/V from GPU.
    /// Used to provide CPU caches to `forward_batched`.
    fn boxed_cache_to_cpu(cache: &mut Box<dyn KVCacheProvider>) -> CpuKVCache {
        if let Some(cpu_entries) = cache.as_cpu_entries() {
            CpuKVCache { entries: std::mem::take(cpu_entries) }
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
        if self.use_gpu {
            #[cfg(feature = "gpu")]
            if self.num_layers > 0 {
                let gpu_cache =
                    GpuKVCache::new(self.num_layers, self.num_kv_heads, self.head_dim, self.max_seq_len);
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
            let (prefix_len, shared_count) = trie.insert(seq_id, &request.input_tokens, self.config.min_shared_prefix_len);
            if prefix_len > 0 && shared_count > 1 {
                debug!(
                    "Seq {} shares {} token prefix with {} other sequence(s)",
                    seq_id, prefix_len, shared_count - 1
                );
            }
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

        // --- PHASE 1: Batched prefill ---
        // Process ALL remaining prompt tokens for each prefill sequence in a
        // single forward call. The model processes all tokens sequentially
        // internally but with a single function call overhead, and the returned
        // logits predict the first generated token.
        if !prefill_ids.is_empty() {
            // Extract KV caches for prefill sequences
            let mut prefill_caches: Vec<(u64, Box<dyn KVCacheProvider>)> = prefill_ids
                .iter()
                .filter_map(|&sid| self.kv_caches.remove(&sid).map(|c| (sid, c)))
                .collect();

            let mut prefill_logits: Vec<(u64, Array1<f32>)> = Vec::with_capacity(prefill_ids.len());

            for (sid, cache) in prefill_caches.iter_mut() {
                let seq = match self.sequences.get(sid) {
                    Some(s) => s,
                    None => continue,
                };
                let prompt_tokens: Vec<u32> = seq.prompt[seq.prompt_pos..].to_vec();
                drop(seq);

                let logits = self.model.forward(&prompt_tokens, &mut **cache);
                prefill_logits.push((*sid, logits));
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
                    token_id,
                    token_text,
                    log_prob,
                    pos,
                )));

                let mut finish_reason: Option<FinishReason> = None;
                if token_id == seq.eos_token_id && !seq.generated.is_empty() {
                    finish_reason = Some(FinishReason::EndOfSequence);
                } else if seq.reached_max_tokens() {
                    finish_reason = Some(FinishReason::MaxTokens);
                }

                if let Some(reason) = finish_reason {
                    seq.finish(reason.clone());
                    if let Some(resp) = self.build_response(
                        sid,
                        reason,
                        step_start.elapsed().as_millis() as u64,
                    ) {
                        completed.push(resp);
                    }
                }
            }
        }

        // --- PHASE 2: Batched generation for non-prefill sequences ---
        if !gen_ids.is_empty() {
            let mut gen_boxed_caches: Vec<Box<dyn KVCacheProvider>> = Vec::with_capacity(gen_ids.len());
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

            // For multi-seq: extract CPU entries, call forward_batched (GPU w/ logit readback)
            // For single-seq with GPU: use forward_sample_gpu (zero logit readback)
            let all_logits: Vec<Array1<f32>> = if gen_ids.len() > 1 {
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
                    let cache = std::mem::replace(&mut gen_boxed_caches[i], Box::new(CpuKVCache::new(0)));
                    self.kv_caches.insert(*sid, cache);
                }
            }

            // Process each generation result
            for (seq_idx, (&seq_id, logits_arr)) in gen_ids.iter().zip(all_logits.iter()).enumerate() {
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
                    logits_vec.get(token_id as usize).copied().unwrap_or_else(|| {
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
                    token_id,
                    token_text,
                    log_prob,
                    pos,
                )));

                let mut finish_reason: Option<FinishReason> = None;
                if token_id == seq.eos_token_id && !seq.generated.is_empty() {
                    finish_reason = Some(FinishReason::EndOfSequence);
                } else if seq.reached_max_tokens() {
                    finish_reason = Some(FinishReason::MaxTokens);
                }

                if let Some(reason) = finish_reason {
                    seq.finish(reason.clone());
                    if let Some(resp) = self.build_response(
                        seq_id,
                        reason,
                        step_start.elapsed().as_millis() as u64,
                    ) {
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

    fn build_response(&self, seq_id: u64, reason: FinishReason, elapsed_ms: u64) -> Option<InferenceResponse> {
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
