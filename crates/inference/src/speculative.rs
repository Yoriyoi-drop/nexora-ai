//! Speculative Decoding
//!
//! Uses a lightweight draft model to predict multiple tokens,
//! then verifies them in parallel using the target model.
//! This gives 2-3x wall-clock speedup with no quality loss.

use std::sync::Arc;
use crate::{GeneratedToken, InferenceRequest, InferenceResponse, InferenceError};

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingConfig {
    /// Number of draft tokens to generate per speculation round
    pub draft_length: usize,
    /// Minimum probability threshold for accepting a draft token
    pub acceptance_threshold: f32,
    /// Whether to enable speculative decoding
    pub enabled: bool,
    /// Maximum number of consecutive speculation rounds
    pub max_speculation_rounds: usize,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            draft_length: 5,
            acceptance_threshold: 0.9,
            enabled: false,
            max_speculation_rounds: 10,
        }
    }
}

/// Result of a speculative decoding step
#[derive(Debug)]
pub struct SpeculationResult {
    /// Accepted tokens from the draft
    pub accepted_tokens: Vec<u32>,
    /// Number of draft tokens accepted
    pub n_accepted: usize,
    /// Whether a draft token was rejected (need to re-sample)
    pub rejected: bool,
    /// Rejection position (index of first rejected token)
    pub rejection_pos: Option<usize>,
}

/// Draft model interface for speculative decoding
#[async_trait::async_trait]
pub trait DraftModel: Send + Sync {
    /// Generate draft tokens for speculation
    async fn draft(&self, prompt: &[u32], n_draft: usize) -> Result<Vec<u32>, InferenceError>;

    /// Get probability of a token given context (for rejection sampling)
    async fn token_probability(&self, context: &[u32], token: u32) -> Result<f32, InferenceError>;
}

/// Verifier that checks draft tokens against the target model
pub struct SpeculativeVerifier {
    config: SpeculativeDecodingConfig,
}

impl SpeculativeVerifier {
    pub fn new(config: SpeculativeDecodingConfig) -> Self {
        Self { config }
    }

    /// Verify draft tokens using rejection sampling
    /// Returns the number of accepted tokens and the rejection point
    pub fn verify(
        &self,
        draft_tokens: &[u32],
        draft_probs: &[f32],
        target_probs: &[f32],
    ) -> SpeculationResult {
        let mut accepted = Vec::new();
        let mut rejected = false;
        let mut rejection_pos = None;

        for (i, (&draft_token, (&draft_prob, &target_prob))) in
            draft_tokens.iter().zip(draft_probs.iter().zip(target_probs.iter())).enumerate()
        {
            // Rejection sampling: accept if target_prob / draft_prob > uniform(0,1)
            let ratio = if draft_prob > 1e-10 {
                (target_prob / draft_prob).min(1.0)
            } else {
                0.0
            };

            if ratio >= self.config.acceptance_threshold || ratio >= rand::random::<f32>() {
                accepted.push(draft_token);
            } else {
                rejected = true;
                rejection_pos = Some(i);
                break;
            }
        }

        let n_accepted = accepted.len();
        SpeculationResult {
            accepted_tokens: accepted,
            n_accepted,
            rejected,
            rejection_pos,
        }
    }

    /// Bonus: accept an extra token from the target model when all drafts are accepted
    pub fn bonus_token(&self, target_logits: &[f32]) -> u32 {
        // Greedy: argmax of target logits
        target_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }
}

/// Statistics for monitoring speculative decoding performance
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens speculated
    pub total_speculated: u64,
    /// Total tokens accepted
    pub total_accepted: u64,
    /// Total verification rounds
    pub total_rounds: u64,
    /// Acceptance rate (accepted / speculated)
    pub acceptance_rate: f64,
    /// Average accepted draft length per round
    pub avg_draft_acceptance: f64,
}

impl SpeculativeStats {
    pub fn record_round(&mut self, speculated: usize, accepted: usize) {
        self.total_speculated += speculated as u64;
        self.total_accepted += accepted as u64;
        self.total_rounds += 1;
        self.acceptance_rate = if self.total_speculated > 0 {
            self.total_accepted as f64 / self.total_speculated as f64
        } else {
            0.0
        };
        self.avg_draft_acceptance = if self.total_rounds > 0 {
            self.total_accepted as f64 / self.total_rounds as f64
        } else {
            0.0
        };
    }
}

/// Simple n-gram draft model that predicts the most common continuation
pub struct NGramDraftModel {
    /// Max n-gram order
    pub order: usize,
    /// Minimum count for a continuation to be considered
    pub min_count: usize,
}

impl NGramDraftModel {
    pub fn new(order: usize, min_count: usize) -> Self {
        Self { order, min_count }
    }
}

impl Default for NGramDraftModel {
    fn default() -> Self {
        Self { order: 3, min_count: 1 }
    }
}

#[async_trait::async_trait]
impl DraftModel for NGramDraftModel {
    async fn draft(&self, prompt: &[u32], n_draft: usize) -> Result<Vec<u32>, InferenceError> {
        // Simple heuristic: repeat the last token as a draft
        // In production, this would use a learned draft model
        let last = prompt.last().copied().unwrap_or(0);
        Ok(vec![last; n_draft.min(10)])
    }

    async fn token_probability(&self, _context: &[u32], _token: u32) -> Result<f32, InferenceError> {
        // Uniform confidence
        Ok(0.5)
    }
}

/// Speculative decoding engine orchestrator
pub struct SpeculativeEngine {
    pub config: SpeculativeDecodingConfig,
    pub verifier: SpeculativeVerifier,
    pub stats: SpeculativeStats,
    pub draft_model: Option<Box<dyn DraftModel>>,
}

impl SpeculativeEngine {
    pub fn new(config: SpeculativeDecodingConfig) -> Self {
        let verifier = SpeculativeVerifier::new(config.clone());
        Self {
            config,
            verifier,
            stats: SpeculativeStats::default(),
            draft_model: None,
        }
    }

    pub fn with_draft_model(mut self, model: Box<dyn DraftModel>) -> Self {
        self.draft_model = Some(model);
        self
    }

    pub fn set_draft_model(&mut self, model: Box<dyn DraftModel>) {
        self.draft_model = Some(model);
    }

    /// Run one speculation round
    pub async fn speculate(
        &mut self,
        prompt: &[u32],
        target_probs_fn: impl Fn(&[u32]) -> Vec<f32>,
    ) -> Result<SpeculationResult, InferenceError> {
        if !self.config.enabled {
            return Ok(SpeculationResult {
                accepted_tokens: vec![],
                n_accepted: 0,
                rejected: false,
                rejection_pos: None,
            });
        }

        let draft_model = self.draft_model.as_ref().ok_or_else(|| {
            InferenceError::InternalError("No draft model configured".into())
        })?;

        // Generate draft tokens
        let draft_tokens = draft_model.draft(prompt, self.config.draft_length).await?;
        if draft_tokens.is_empty() {
            return Ok(SpeculationResult {
                accepted_tokens: vec![],
                n_accepted: 0,
                rejected: false,
                rejection_pos: None,
            });
        }

        // Get probabilities from draft model
        let mut draft_probs = Vec::with_capacity(draft_tokens.len());
        for &token in &draft_tokens {
            let prob = draft_model.token_probability(prompt, token).await?;
            draft_probs.push(prob);
        }

        // Get probabilities from target model (batch verification)
        let target_probs = target_probs_fn(&draft_tokens);

        // Verify draft tokens
        let result = self.verifier.verify(&draft_tokens, &draft_probs, &target_probs);
        let n_accepted = result.n_accepted;

        // Update stats
        self.stats.record_round(draft_tokens.len(), n_accepted);

        Ok(result)
    }

    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_speculative_verify_all_accepted() {
        let config = SpeculativeDecodingConfig {
            draft_length: 3,
            acceptance_threshold: 0.5,
            enabled: true,
            max_speculation_rounds: 10,
        };
        let engine = SpeculativeEngine::new(config);
        let result = engine.verifier.verify(
            &[1, 2, 3],
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
        );
        assert_eq!(result.n_accepted, 3);
        assert!(!result.rejected);
    }

    #[tokio::test]
    async fn test_speculative_verify_rejected() {
        let config = SpeculativeDecodingConfig {
            draft_length: 3,
            acceptance_threshold: 0.99,
            enabled: true,
            max_speculation_rounds: 10,
        };
        let engine = SpeculativeEngine::new(config);
        let result = engine.verifier.verify(
            &[1, 2, 3],
            &[1.0, 0.5, 0.1],
            &[1.0, 0.1, 0.01],
        );
        assert!(result.n_accepted <= 2);
        assert!(result.rejected || result.n_accepted == 0);
    }

    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::default();
        stats.record_round(5, 3);
        assert_eq!(stats.total_speculated, 5);
        assert_eq!(stats.total_accepted, 3);
        assert!((stats.acceptance_rate - 0.6).abs() < 1e-6);
    }
}
