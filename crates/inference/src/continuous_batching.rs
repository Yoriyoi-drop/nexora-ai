use std::collections::HashMap;

use tracing::warn;

use crate::sampler::{Sampler, SamplingConfig};
use crate::sequence_state::{SeqState, Sequence};
use crate::{FinishReason, GeneratedToken, InferenceRequest, InferenceResponse};
use nexora_foundation::models::transformer::KVCacheEntry;

/// Result of a single continuous batching step.
#[derive(Debug)]
pub struct StepResult {
    /// Sequences that completed in this step.
    pub completed: Vec<InferenceResponse>,
    /// Number of sequences still active.
    pub active_count: usize,
    /// Whether the scheduler is idle (no active sequences).
    pub idle: bool,
}

/// A continuous batching engine that manages multiple sequences at different
/// stages of their lifecycle (prefill, generate, complete) and processes them
/// one step at a time.
///
/// Each step picks the next ready sequence and runs a single token through
/// the model. Sequences dynamically enter (via `add_request`) and leave
/// (via completion or error) the pool.
pub struct ContinuousBatchingEngine<M> {
    /// Active sequences, keyed by sequence ID
    sequences: HashMap<u64, Sequence>,
    /// KV cache per sequence: seq_id → Vec<KVCacheEntry>
    kv_caches: HashMap<u64, Vec<KVCacheEntry>>,
    /// The underlying model
    model: M,
    /// Next available sequence ID
    next_seq_id: u64,
    /// Maximum sequences to process per step
    max_batch_size: usize,
    /// Maximum total sequences in the pool
    max_total_sequences: usize,
    /// Per-sequence sampler
    samplers: HashMap<u64, Sampler>,
}

impl<M> ContinuousBatchingEngine<M>
where
    M: crate::inference_trait::ModelForward,
{
    /// Create a new continuous batching engine.
    pub fn new(model: M, max_batch_size: usize) -> Self {
        Self {
            sequences: HashMap::new(),
            kv_caches: HashMap::new(),
            model,
            next_seq_id: 1,
            max_batch_size,
            max_total_sequences: 1024,
            samplers: HashMap::new(),
        }
    }

    /// Add a new request to the batching pool.
    /// Returns the assigned sequence ID, or 0 if the pool is full.
    pub fn add_request(&mut self, request: InferenceRequest) -> u64 {
        if self.sequences.len() >= self.max_total_sequences {
            warn!("ContinuousBatchingEngine: max sequences ({}) reached", self.max_total_sequences);
            return 0;
        }
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let mut seq = Sequence::from_request(seq_id, &request);
        seq.state = SeqState::Prefilling;
        let cache = self.model.reset_cache();
        self.kv_caches.insert(seq_id, cache);
        self.sequences.insert(seq_id, seq);

        let sampler = Sampler::new(SamplingConfig {
            method: crate::sampler::SamplingMethod::TemperatureTopKTopP,
            temperature: request.temperature,
            top_k: request.top_k as usize,
            top_p: request.top_p,
            min_prob: 0.0,
            seed: None,
        });
        self.samplers.insert(seq_id, sampler);

        seq_id
    }

    /// Run a single step of the continuous batching loop.
    /// Processes up to `max_batch_size` ready sequences through the model,
    /// sampling the next token for each.
    pub fn step(&mut self) -> StepResult {
        let mut completed = Vec::new();
        let mut processed = 0_usize;

        let ready_ids: Vec<u64> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.is_ready())
            .take(self.max_batch_size)
            .map(|(id, _)| *id)
            .collect();

        for seq_id in ready_ids {
            let ready = self
                .sequences
                .get(&seq_id)
                .map(|s| s.is_ready())
                .unwrap_or(false);
            if !ready {
                continue;
            }

            let result = self.process_sequence(seq_id);
            if let Some(response) = result {
                completed.push(response);
            }
            processed += 1;
        }

        let active_count = self
            .sequences
            .values()
            .filter(|s| !s.is_finished())
            .count();

        StepResult {
            completed,
            active_count,
            idle: processed == 0 && active_count == 0,
        }
    }

    /// Process one sequence for a single token step (prefill or decode).
    fn process_sequence(&mut self, seq_id: u64) -> Option<InferenceResponse> {
        let seq = self.sequences.get(&seq_id)?;
        let input_token = seq.next_input_token()?;
        let was_prefilling = seq.has_pending_prompt();
        drop(seq);

        let cache = self.kv_caches.get_mut(&seq_id)?;
        let logits = self.model.forward(&[input_token], cache);

        let logits_slice: &[f32] = logits.as_slice().unwrap_or(&[]);
        let sampler = self.samplers.get_mut(&seq_id)?;
        let token_id = match sampler.sample(logits_slice) {
            Ok(idx) => idx as u32,
            Err(e) => {
                warn!("Sampler failed for sequence {}, error: {:?}, falling back to argmax", seq_id, e);
                let max_idx = logits_slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                max_idx
            }
        };

        let seq = self.sequences.get_mut(&seq_id)?;

        if was_prefilling {
            seq.advance_prompt();
            if seq.has_pending_prompt() {
                // More prompt tokens remaining — continue prefill next step
                return None;
            }
            // Prefill just completed — transition to generating
            seq.state = SeqState::Generating;
            // Fall through to push the sampled token as first generated token
        }

        let log_prob = logits_slice.get(token_id as usize).copied().unwrap_or(0.0);
        let pos = seq.prompt.len() + seq.generated.len();
        seq.push_token(GeneratedToken::new(token_id, String::new(), log_prob, pos));

        let mut finish_reason: Option<FinishReason> = None;
        if token_id == seq.eos_token_id && !seq.generated.is_empty() {
            finish_reason = Some(FinishReason::EndOfSequence);
        } else if seq.reached_max_tokens() {
            finish_reason = Some(FinishReason::MaxTokens);
        }

        if let Some(reason) = finish_reason {
            seq.finish(reason.clone());
            self.build_response(seq_id, reason)
        } else {
            None
        }
    }

    fn build_response(&self, seq_id: u64, reason: FinishReason) -> Option<InferenceResponse> {
        let seq = self.sequences.get(&seq_id)?;
                let text: String = seq.generated.iter().map(|t| (&*t.token_text).to_string()).collect();
        Some(InferenceResponse {
            request_id: uuid::Uuid::new_v4(),
            tokens: seq.generated.clone(),
            text,
            finish_reason: reason,
            total_tokens: seq.total_tokens(),
            inference_time_ms: 0,
            metadata: HashMap::new(),
        })
    }

    /// Number of active (non-finished) sequences.
    pub fn active_count(&self) -> usize {
        self.sequences
            .values()
            .filter(|s| !s.is_finished())
            .count()
    }

    /// Remove all completed sequences, returning their responses.
    pub fn drain_completed(&mut self) -> Vec<InferenceResponse> {
        let mut results = Vec::new();
        let to_remove: Vec<u64> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.is_finished())
            .map(|(id, _)| *id)
            .collect();

        for seq_id in to_remove {
            self.samplers.remove(&seq_id);
            self.kv_caches.remove(&seq_id);
            if let Some(seq) = self.sequences.remove(&seq_id) {
                let reason = match &seq.state {
                    SeqState::Finished(r) => r.clone(),
                    _ => continue,
                };
                let total_tokens = seq.total_tokens();
        let text: String = seq.generated.iter().map(|t| (&*t.token_text).to_string()).collect();
                results.push(InferenceResponse {
                    request_id: uuid::Uuid::new_v4(),
                    tokens: seq.generated,
                    text,
                    finish_reason: reason,
                    total_tokens,
                    inference_time_ms: 0,
                    metadata: HashMap::new(),
                });
            }
        }

        results
    }

    /// Get a reference to a specific sequence.
    pub fn get_sequence(&self, seq_id: u64) -> Option<&Sequence> {
        self.sequences.get(&seq_id)
    }

    /// Get mutable sequence reference.
    pub fn get_sequence_mut(&mut self, seq_id: u64) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_trait::ModelForward;
    use ndarray::Array1;

    /// Mock model: returns logits where the input token ID has highest value.
    struct MockModel {
        vocab_size: usize,
    }

    impl ModelForward for MockModel {
        fn forward(
            &self,
            input_ids: &[u32],
            _kv_cache: &mut Vec<KVCacheEntry>,
        ) -> Array1<f32> {
            let mut logits = Array1::zeros(self.vocab_size);
            if let Some(&tid) = input_ids.last() {
                let idx = (tid as usize).min(self.vocab_size - 1);
                logits[idx] = 10.0;
            } else {
                logits[0] = 10.0;
            }
            logits
        }

        fn reset_cache(&self) -> Vec<KVCacheEntry> {
            Vec::new()
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

        // Step 1: prefill first prompt token
        let r = engine.step();
        assert_eq!(r.active_count, 1);
        assert!(r.completed.is_empty());

        // Step 2: prefill second prompt token + first generated token
        let r = engine.step();
        assert_eq!(r.active_count, 1);
        assert!(r.completed.is_empty());

        let seq = engine.get_sequence(1).unwrap();
        assert!(!seq.generated.is_empty(), "should have at least 1 generated token after prefill");
    }

    #[test]
    fn test_sequence_completes_at_max_tokens() {
        let model = MockModel { vocab_size: 100 };
        let mut engine = ContinuousBatchingEngine::new(model, 4);

        // Max = 4 total: 2 prompt + 2 generate
        engine.add_request(test_request(vec![1, 2], 4));

        // Step 1: prefill first prompt token
        let r = engine.step();
        assert!(r.completed.is_empty());

        // Step 2: prefill second prompt token + first generated token (total=3)
        let r = engine.step();
        assert!(r.completed.is_empty());

        // Step 3: generate second token (total=4 = max) → completes
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
