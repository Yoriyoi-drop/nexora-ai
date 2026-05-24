use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array1;
use tracing::warn;

use crate::sampler::{Sampler, SamplingConfig};
use crate::sequence_state::{SeqState, Sequence};
use crate::{FinishReason, GeneratedToken, InferenceRequest, InferenceResponse};
use nexora_transformer::CpuKVCache;

/// Result of a single sequential batching step.
#[derive(Debug)]
pub struct StepResult {
    pub completed: Vec<InferenceResponse>,
    pub active_count: usize,
    pub idle: bool,
}

/// A batched inference engine that manages multiple sequences.
///
/// Each step collects all ready sequences and processes them through the model's
/// batched forward pass (`forward_batched`), then samples the next token for each.
/// Prefill sequences process ALL remaining prompt tokens at once (full batched prefill)
/// before moving to the generation phase.
/// Sequences dynamically enter (via `add_request`) and leave (via completion or error).
///
/// This provides true batched inference at both the request level and the
/// computation level (single kernel launch via `forward_batched`).
///
/// Variable-length sequences are handled by processing all remaining prompt tokens
/// for prefill sequences before proceeding to batched generation for all sequences.
pub struct SequentialBatchingEngine<M> {
    sequences: HashMap<u64, Sequence>,
    kv_caches: HashMap<u64, CpuKVCache>,
    model: M,
    next_seq_id: u64,
    max_batch_size: usize,
    max_total_sequences: usize,
    samplers: HashMap<u64, Sampler>,
    use_gpu: bool,
}

impl<M> SequentialBatchingEngine<M>
where
    M: crate::inference_trait::ModelForward,
{
    pub fn new(model: M, max_batch_size: usize) -> Self {
        Self {
            sequences: HashMap::new(),
            kv_caches: HashMap::new(),
            model,
            next_seq_id: 1,
            max_batch_size,
            max_total_sequences: 1024,
            samplers: HashMap::new(),
            #[cfg(feature = "gpu")]
            use_gpu: nexora_autograd::gpu::GpuContext::is_available(),
            #[cfg(not(feature = "gpu"))]
            use_gpu: false,
        }
    }

    pub fn set_use_gpu(&mut self, use_gpu: bool) {
        self.use_gpu = use_gpu;
        for sampler in self.samplers.values_mut() {
            sampler.set_use_gpu(use_gpu);
        }
    }

    pub fn add_request(&mut self, request: InferenceRequest) -> u64 {
        if self.sequences.len() >= self.max_total_sequences {
            warn!(
                "SequentialBatchingEngine: max sequences ({}) reached",
                self.max_total_sequences
            );
            return 0;
        }
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let mut seq = Sequence::from_request(seq_id, &request);
        seq.state = SeqState::Prefilling;
        let cache = self.model.reset_cache();
        self.kv_caches.insert(seq_id, cache);
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
    /// Prefill sequences process ALL remaining prompt tokens at once (full batched
    /// prefill) before transitioning to generation. Generation sequences process
    /// one token each via `forward_batched`. Variable-length sequences are handled
    /// by fully prefilling each sequence before batched generation.
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

        // --- PHASE 1: Full batched prefill ---
        // Process ALL remaining prompt tokens for each prefill sequence.
        // For each token position up to max remaining length, call forward_batched
        // with the tokens at that position across all sequences that still have
        // tokens remaining. This gives true batched prefill with masking.
        if !prefill_ids.is_empty() {
            // Determine max remaining prompt length across all prefill sequences
            let max_remaining: usize = prefill_ids
                .iter()
                .filter_map(|id| self.sequences.get(id))
                .map(|s| s.prompt.len() - s.prompt_pos)
                .max()
                .unwrap_or(0);

            // Extract KV caches for prefill sequences
            let mut prefill_caches: Vec<(u64, CpuKVCache)> = prefill_ids
                .iter()
                .filter_map(|&sid| self.kv_caches.remove(&sid).map(|c| (sid, c)))
                .collect();

            // Process one token position at a time across all sequences
            for pos_offset in 0..max_remaining {
                let mut batch_ids: Vec<u64> = Vec::new();
                let mut batch_inputs: Vec<u32> = Vec::new();
                let mut batch_cache_indices: Vec<usize> = Vec::new();

                for (ci, (sid, cache)) in prefill_caches.iter_mut().enumerate() {
                    let seq = match self.sequences.get(sid) {
                        Some(s) => s,
                        None => continue,
                    };
                    let prompt_idx = seq.prompt_pos + pos_offset;
                    if prompt_idx < seq.prompt.len() {
                        batch_ids.push(*sid);
                        batch_inputs.push(seq.prompt[prompt_idx]);
                        batch_cache_indices.push(ci);
                    }
                }

                if batch_ids.is_empty() {
                    continue;
                }

                // Gather caches for this batch position
                let mut batch_caches: Vec<CpuKVCache> = batch_cache_indices
                    .iter()
                    .map(|&ci| std::mem::replace(&mut prefill_caches[ci].1, CpuKVCache::new(0)))
                    .collect();

                // Batched forward at this position
                let _all_logits: Vec<Array1<f32>> = if batch_ids.len() > 1 {
                    self.model.forward_batched(&batch_inputs, &mut batch_caches)
                } else if let Some(cache) = batch_caches.first_mut() {
                    vec![self.model.forward(&batch_inputs, cache)]
                } else {
                    Vec::new()
                };

                // Restore caches
                for (bi, &ci) in batch_cache_indices.iter().enumerate() {
                    if bi < batch_caches.len() {
                        prefill_caches[ci].1 = std::mem::replace(
                            &mut batch_caches[bi],
                            CpuKVCache::new(0),
                        );
                    }
                }
            }

            // Restore all prefill KV caches and update states to Generating
            for (sid, cache) in prefill_caches {
                self.kv_caches.insert(sid, cache);
                if let Some(seq) = self.sequences.get_mut(&sid) {
                    seq.prompt_pos = seq.prompt.len();
                    seq.state = SeqState::Generating;
                }
            }

            // After full prefill, generate first output token for each prefill sequence
            // by doing a single forward pass with the last prompt token
            for &sid in &prefill_ids {
                if let Some(cache) = self.kv_caches.get_mut(&sid) {
                    if let Some(seq) = self.sequences.get(&sid) {
                        let last_prompt = seq.prompt.last().copied().unwrap_or(0);
                        let logits_arr = self.model.forward(&[last_prompt], cache);
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

                        let seq = match self.sequences.get_mut(&sid) {
                            Some(s) => s,
                            None => continue,
                        };
                        let log_prob = logits_vec.get(token_id as usize).copied().unwrap_or(0.0);
                        let pos = seq.prompt.len() + seq.generated.len();
                        seq.push_token(Arc::new(GeneratedToken::new(
                            token_id,
                            String::new(),
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
            }
        }

        // --- PHASE 2: Batched generation for non-prefill sequences ---
        if !gen_ids.is_empty() {
            let mut gen_caches: Vec<CpuKVCache> = Vec::with_capacity(gen_ids.len());
            let mut gen_inputs: Vec<u32> = Vec::with_capacity(gen_ids.len());

            for &sid in &gen_ids {
                if let Some(cache) = self.kv_caches.remove(&sid) {
                    gen_caches.push(cache);
                }
                if let Some(s) = self.sequences.get(&sid) {
                    let next_token = s.next_input_token();
                    if next_token.is_none() {
                        warn!("Sequence {} has no input token, using BOS(0)", sid);
                    }
                    gen_inputs.push(next_token.unwrap_or(0));
                }
            }

            let all_logits: Vec<Array1<f32>> = if gen_ids.len() > 1 {
                self.model.forward_batched(&gen_inputs, &mut gen_caches)
            } else if let Some(cache) = gen_caches.first_mut() {
                vec![self.model.forward(&gen_inputs, cache)]
            } else {
                warn!("No cache available for generation forward pass — skipping");
                Vec::new()
            };

            // Restore generation caches
            for (i, sid) in gen_ids.iter().enumerate() {
                if i < gen_caches.len() {
                    let cache = std::mem::replace(&mut gen_caches[i], CpuKVCache::new(0));
                    self.kv_caches.insert(*sid, cache);
                }
            }

            // Process each generation result
            for (i, (&seq_id, logits_arr)) in gen_ids.iter().zip(all_logits.iter()).enumerate() {
                let logits_vec: Vec<f32> = logits_arr.clone().into_raw_vec();

                let sampler = match self.samplers.get_mut(&seq_id) {
                    Some(s) => s,
                    None => {
                        warn!("Sampler not found for sequence {}, skipping", seq_id);
                        continue;
                    }
                };
                let token_id = match sampler.sample(&logits_vec) {
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
                };

                let seq = match self.sequences.get_mut(&seq_id) {
                    Some(s) => s,
                    None => {
                        warn!("Sequence {} not found, skipping", seq_id);
                        continue;
                    }
                };

                let log_prob = logits_vec.get(token_id as usize).copied().unwrap_or_else(|| {
                    warn!(
                        "token_id {} out of range for logits of length {}",
                        token_id,
                        logits_vec.len()
                    );
                    0.0
                });
                let pos = seq.prompt.len() + seq.generated.len();
                seq.push_token(Arc::new(GeneratedToken::new(
                    token_id,
                    String::new(),
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
            .map(|t| (&*t.token_text).to_string())
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
                    .map(|t| (&*t.token_text).to_string())
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
        let mut engine = SequentialBatchingEngine::new(model, 4);
        let seq_id = engine.add_request(test_request(vec![1, 2, 3], 10));
        assert!(engine.get_sequence(seq_id).is_some());
        assert_eq!(engine.active_count(), 1);
    }

    #[test]
    fn test_step_prefill_then_generate() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = SequentialBatchingEngine::new(model, 4);
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
        let mut engine = SequentialBatchingEngine::new(model, 4);
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
        let mut engine = SequentialBatchingEngine::new(model, 4);
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
        let mut engine = SequentialBatchingEngine::new(model, 4);
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
        let mut engine = SequentialBatchingEngine::new(model, 4);
        let r = engine.step();
        assert!(r.idle);
        assert!(r.completed.is_empty());
        assert_eq!(r.active_count, 0);
    }
}
