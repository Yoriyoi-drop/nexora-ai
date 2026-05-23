use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array1;
use tracing::warn;

use crate::sampler::{Sampler, SamplingConfig};
use crate::sequence_state::{SeqState, Sequence};
use crate::{FinishReason, GeneratedToken, InferenceRequest, InferenceResponse};
use nexora_transformer::CpuKVCache;

/// Result of a single continuous batching step.
#[derive(Debug)]
pub struct StepResult {
    pub completed: Vec<InferenceResponse>,
    pub active_count: usize,
    pub idle: bool,
}

/// A continuous batching engine that manages multiple sequences.
///
/// Each step collects all ready sequences and runs a batched forward pass
/// through the model (via `ModelForward::forward_batched`), then samples
/// the next token for each. Sequences dynamically enter (via `add_request`)
/// and leave (via completion or error).
///
/// NOTE: The current implementation is a hybrid. When multiple sequences are
/// ready, they are processed via `forward_batched`. For single sequences,
/// the single-sequence `forward` is used. True batched computation across
/// all sequences simultaneously requires model-level support.
pub struct ContinuousBatchingEngine<M> {
    sequences: HashMap<u64, Sequence>,
    kv_caches: HashMap<u64, CpuKVCache>,
    model: M,
    next_seq_id: u64,
    max_batch_size: usize,
    max_total_sequences: usize,
    samplers: HashMap<u64, Sampler>,
    use_gpu: bool,
}

impl<M> ContinuousBatchingEngine<M>
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
                "ContinuousBatchingEngine: max sequences ({}) reached",
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

    /// Run a single step, processing up to `max_batch_size` ready sequences
    /// in a batched forward pass.
    ///
    /// NOTE: Currently processes sequences one-by-one via `forward_batched`.
    /// True batched forward (all sequences in a single kernel launch) is not
    /// yet implemented.
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

        // Extract caches and inputs for batched forward
        let mut caches: Vec<CpuKVCache> = Vec::with_capacity(ready_ids.len());
        let mut inputs: Vec<u32> = Vec::with_capacity(ready_ids.len());
        let mut prefill_flags: Vec<bool> = Vec::with_capacity(ready_ids.len());

        for &sid in &ready_ids {
            if let Some(cache) = self.kv_caches.remove(&sid) {
                caches.push(cache);
            }
            if let Some(s) = self.sequences.get(&sid) {
                inputs.push(s.next_input_token().unwrap_or(0));
                prefill_flags.push(s.has_pending_prompt());
            }
        }

        // Batched forward pass
        let all_logits: Vec<Array1<f32>> = if inputs.len() > 1 {
            self.model.forward_batched(&inputs, &mut caches)
        } else {
            match caches.first_mut() {
                Some(cache) => {
                    vec![self.model.forward(&inputs, cache)]
                }
                None => Vec::new(),
            }
        };

        // Restore caches
        for (i, sid) in ready_ids.iter().enumerate() {
            if i < caches.len() {
                let cache = std::mem::replace(&mut caches[i], CpuKVCache::new(0));
                self.kv_caches.insert(*sid, cache);
            }
        }

        // Process each result
        for (i, (&seq_id, logits_arr)) in ready_ids.iter().zip(all_logits.iter()).enumerate() {
            let logits_vec: Vec<f32> = logits_arr.clone().into_raw_vec();
            let was_prefill = prefill_flags[i];

            let sampler = match self.samplers.get_mut(&seq_id) {
                Some(s) => s,
                None => {
                    warn!("Sampler not found for sequence {}, skipping", seq_id);
                    continue;
                }
            };
            let token_id = match sampler.sample(&logits_vec) {
                Ok(idx) => idx as u32,
                Err(e) => {
                    warn!(
                        "Sampler failed for sequence {}, error: {:?}, falling back to argmax",
                        seq_id, e
                    );
                    logits_vec
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(i, _)| i as u32)
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
            if was_prefill {
                seq.advance_prompt();
                if seq.has_pending_prompt() {
                    continue;
                }
                seq.state = SeqState::Generating;
            }

            let log_prob = logits_vec.get(token_id as usize).copied().unwrap_or_else(|| {
                warn!(
                    "token_id {} out of range for logits of length {}",
                    token_id,
                    logits_vec.len()
                );
                0.0
            });
            let pos = seq.prompt.len() + seq.generated.len();
            seq.push_token(Arc::new(GeneratedToken::new(token_id, String::new(), log_prob, pos)));

            let mut finish_reason: Option<FinishReason> = None;
            if token_id == seq.eos_token_id && !seq.generated.is_empty() {
                finish_reason = Some(FinishReason::EndOfSequence);
            } else if seq.reached_max_tokens() {
                finish_reason = Some(FinishReason::MaxTokens);
            }

            if let Some(reason) = finish_reason {
                seq.finish(reason.clone());
                if let Some(resp) = self.build_response(seq_id, reason, step_start.elapsed().as_millis() as u64) {
                    completed.push(resp);
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
            request_id: uuid::Uuid::new_v4(),
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
                    request_id: uuid::Uuid::new_v4(),
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
