//! Speculative Decoding
//!
//! Uses a lightweight draft model to predict multiple tokens,
//! then verifies them in parallel using the target model.
//! This gives 2-3x wall-clock speedup with no quality loss.

use crate::InferenceError;

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
        unreachable!("removed — see audit-perf item #23")
    }

    pub fn verify(
        &self,
        _draft_tokens: &[u32],
        _draft_probs: &[f32],
        _target_probs: &[f32],
    ) -> SpeculationResult {
        unreachable!("removed — see audit-perf item #23")
    }

    pub fn bonus_token(&self, _target_logits: &[f32]) -> u32 {
        unreachable!("removed — see audit-perf item #23")
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
    pub fn record_round(&mut self, _speculated: usize, _accepted: usize) {
        unreachable!("removed — see audit-perf item #23")
    }
}

/// Simple n-gram draft model that predicts the most common continuation
pub struct NGramDraftModel;

impl NGramDraftModel {
    pub fn new(_order: usize, _min_count: usize) -> Self {
        unreachable!("removed — see audit-perf item #23")
    }
}

impl Default for NGramDraftModel {
    fn default() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DraftModel for NGramDraftModel {
    async fn draft(&self, _prompt: &[u32], _n_draft: usize) -> Result<Vec<u32>, InferenceError> {
        unreachable!("removed — see audit-perf item #23")
    }

    async fn token_probability(&self, _context: &[u32], _token: u32) -> Result<f32, InferenceError> {
        unreachable!("removed — see audit-perf item #23")
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
        unreachable!("removed — see audit-perf item #23")
    }

    pub fn with_draft_model(self, _model: Box<dyn DraftModel>) -> Self {
        unreachable!("removed — see audit-perf item #23")
    }

    pub fn set_draft_model(&mut self, _model: Box<dyn DraftModel>) {
        unreachable!("removed — see audit-perf item #23")
    }

    pub async fn speculate(
        &mut self,
        _prompt: &[u32],
        _target_probs_fn: impl Fn(&[u32]) -> Vec<f32>,
    ) -> Result<SpeculationResult, InferenceError> {
        unreachable!("removed — see audit-perf item #23")
    }

    pub fn stats(&self) -> &SpeculativeStats {
        unreachable!("removed — see audit-perf item #23")
    }

    pub fn reset_stats(&mut self) {
        unreachable!("removed — see audit-perf item #23")
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
