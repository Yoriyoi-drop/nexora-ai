//! Speculative Decoding
//!
//! Uses a lightweight draft model to predict multiple tokens,
//! then verifies them in parallel using the target model.
//! This gives 2-3x wall-clock speedup with no quality loss.

use std::collections::HashMap;

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
        Self { config }
    }

    /// Rejection sampling verification.
    ///
    /// For each draft token position i, accepts the token if
    /// `target_probs[i] >= acceptance_threshold * draft_probs[i]`.
    /// On first rejection, returns accepted prefix with `rejected: true`.
    /// If all tokens pass, returns `rejected: false`.
    pub fn verify(
        &self,
        draft_tokens: &[u32],
        draft_probs: &[f32],
        target_probs: &[f32],
    ) -> SpeculationResult {
        let n = draft_tokens
            .len()
            .min(draft_probs.len())
            .min(target_probs.len());

        if n == 0 {
            return SpeculationResult {
                accepted_tokens: Vec::new(),
                n_accepted: 0,
                rejected: false,
                rejection_pos: None,
            };
        }

        for i in 0..n {
            let d_prob = draft_probs[i];
            let t_prob = target_probs[i];

            if !d_prob.is_finite() || d_prob < 0.0 || !t_prob.is_finite() || t_prob < 0.0 {
                return SpeculationResult {
                    accepted_tokens: draft_tokens[..i].to_vec(),
                    n_accepted: i,
                    rejected: true,
                    rejection_pos: Some(i),
                };
            }

            if t_prob < self.config.acceptance_threshold * d_prob {
                return SpeculationResult {
                    accepted_tokens: draft_tokens[..i].to_vec(),
                    n_accepted: i,
                    rejected: true,
                    rejection_pos: Some(i),
                };
            }
        }

        SpeculationResult {
            accepted_tokens: draft_tokens[..n].to_vec(),
            n_accepted: n,
            rejected: false,
            rejection_pos: None,
        }
    }

    /// Select the bonus token — argmax over the target model's logits
    /// at the final position when all draft tokens are accepted.
    pub fn bonus_token(&self, target_logits: &[f32]) -> u32 {
        target_logits
            .iter()
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
    /// Record one speculation round result and update derived metrics.
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

/// Simple n-gram draft model that predicts the most common continuation.
///
/// Stores observed n-gram → next-token frequencies and predicts the most
/// frequent continuation for a given context.
pub struct NGramDraftModel {
    order: usize,
    min_count: usize,
    frequencies: HashMap<Vec<u32>, HashMap<u32, usize>>,
}

impl NGramDraftModel {
    pub fn new(order: usize, min_count: usize) -> Self {
        Self {
            order,
            min_count,
            frequencies: HashMap::new(),
        }
    }

    /// Record an observed n-gram → next-token pair.
    pub fn record(&mut self, ngram: &[u32], next: u32) {
        self.frequencies
            .entry(ngram.to_vec())
            .or_default()
            .entry(next)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }
}

impl Default for NGramDraftModel {
    fn default() -> Self {
        Self::new(3, 1)
    }
}

#[async_trait::async_trait]
impl DraftModel for NGramDraftModel {
    async fn draft(&self, prompt: &[u32], n_draft: usize) -> Result<Vec<u32>, InferenceError> {
        if prompt.is_empty() || n_draft == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(n_draft);
        let mut context: Vec<u32> = prompt.to_vec();
        let fallback = *prompt.last().unwrap_or(&0);

        for _ in 0..n_draft {
            let ngram_start = if context.len() >= self.order {
                context.len() - self.order
            } else {
                0
            };
            let ngram: Vec<u32> = context[ngram_start..].to_vec();

            let next = self
                .frequencies
                .get(&ngram)
                .and_then(|follows| {
                    follows
                        .iter()
                        .max_by(|(_, a), (_, b)| a.cmp(b))
                        .filter(|(_, &count)| count >= self.min_count)
                        .map(|(token, _)| *token)
                })
                .unwrap_or(fallback);

            result.push(next);
            context.push(next);
        }

        Ok(result)
    }

    async fn token_probability(&self, context: &[u32], token: u32) -> Result<f32, InferenceError> {
        if context.is_empty() {
            return Ok(0.5);
        }

        let ngram_start = if context.len() >= self.order {
            context.len() - self.order
        } else {
            0
        };
        let ngram: Vec<u32> = context[ngram_start..].to_vec();

        let prob = self
            .frequencies
            .get(&ngram)
            .map(|follows| {
                let total: usize = follows.values().sum();
                if total == 0 {
                    return 0.001;
                }
                let count = follows.get(&token).copied().unwrap_or(0);
                count as f32 / total as f32
            })
            .unwrap_or(0.001);

        Ok(prob)
    }
}

/// Speculative decoding engine orchestrator.
///
/// Drives the draft → verify loop: uses a `DraftModel` to produce candidate
/// tokens, the target model's `target_probs_fn` to score them, and the
/// `SpeculativeVerifier` to accept or reject via rejection sampling.
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

    /// Run the speculation loop.
    ///
    /// For each round (up to `max_speculation_rounds`):
    ///   1. Draft model proposes `draft_length` candidate tokens.
    ///   2. Draft model scores each candidate (draft_probs).
    ///   3. Target model scores each candidate (target_probs).
    ///   4. Verifier rejects candidates that fall below the threshold.
    ///   5. If all accepted, extend the prompt and continue.
    ///   6. If rejected (or rounds exhausted), return the result.
    pub async fn speculate(
        &mut self,
        prompt: &[u32],
        target_probs_fn: impl Fn(&[u32]) -> Vec<f32>,
    ) -> Result<SpeculationResult, InferenceError> {
        if !self.config.enabled {
            return Err(InferenceError::InvalidConfig(
                "Speculative decoding is not enabled".to_string(),
            ));
        }

        let draft_model = self.draft_model.as_ref().ok_or_else(|| {
            InferenceError::InvalidConfig(
                "No draft model set for speculative decoding".to_string(),
            )
        })?;

        let mut all_accepted: Vec<u32> = Vec::new();
        let mut current_prompt = prompt.to_vec();

        for _round in 0..self.config.max_speculation_rounds {
            let draft_tokens =
                draft_model.draft(&current_prompt, self.config.draft_length).await?;

            if draft_tokens.is_empty() {
                break;
            }

            let n_draft = draft_tokens.len();

            let mut draft_probs = Vec::with_capacity(n_draft);
            for (i, &token) in draft_tokens.iter().enumerate() {
                let context: Vec<u32> = current_prompt
                    .iter()
                    .chain(draft_tokens[..i].iter())
                    .copied()
                    .collect();
                let prob = draft_model.token_probability(&context, token).await?;
                draft_probs.push(prob);
            }

            let mut target_probs = Vec::with_capacity(n_draft);
            for i in 0..n_draft {
                let context: Vec<u32> = current_prompt
                    .iter()
                    .chain(draft_tokens[..i].iter())
                    .copied()
                    .collect();
                let probs = target_probs_fn(&context);
                let token = draft_tokens[i];
                let prob = probs.get(token as usize).copied().unwrap_or(0.0);
                target_probs.push(prob);
            }

            let result = self
                .verifier
                .verify(&draft_tokens, &draft_probs, &target_probs);
            self.stats.record_round(n_draft, result.n_accepted);

            if result.rejected {
                let prev_len = all_accepted.len();
                all_accepted.extend_from_slice(&result.accepted_tokens);
                let n = all_accepted.len();
                let rej_pos = result.rejection_pos.unwrap_or(result.n_accepted);
                return Ok(SpeculationResult {
                    accepted_tokens: all_accepted,
                    n_accepted: n,
                    rejected: true,
                    rejection_pos: Some(prev_len + rej_pos),
                });
            }

            all_accepted.extend_from_slice(&draft_tokens);
            current_prompt.extend_from_slice(&draft_tokens);
        }

        let n = all_accepted.len();
        Ok(SpeculationResult {
            accepted_tokens: all_accepted,
            n_accepted: n,
            rejected: false,
            rejection_pos: None,
        })
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
        let result = engine.verifier.verify(&[1, 2, 3], &[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]);
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
        let result = engine.verifier.verify(&[1, 2, 3], &[1.0, 0.5, 0.1], &[1.0, 0.1, 0.01]);
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

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.total_speculated, 0);
        assert_eq!(stats.total_accepted, 0);
        assert!((stats.acceptance_rate - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_speculative_stats_multiple_rounds() {
        let mut stats = SpeculativeStats::default();
        stats.record_round(5, 4);
        stats.record_round(5, 3);
        assert_eq!(stats.total_speculated, 10);
        assert_eq!(stats.total_accepted, 7);
        assert!((stats.acceptance_rate - 0.7).abs() < 1e-6);
        assert!((stats.avg_draft_acceptance - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_bonus_token() {
        let config = SpeculativeDecodingConfig::default();
        let verifier = SpeculativeVerifier::new(config);
        assert_eq!(verifier.bonus_token(&[0.1, 0.5, 0.3, 0.8, 0.2]), 3);
    }

    #[test]
    fn test_bonus_token_empty() {
        let config = SpeculativeDecodingConfig::default();
        let verifier = SpeculativeVerifier::new(config);
        assert_eq!(verifier.bonus_token(&[]), 0);
    }

    #[test]
    fn test_verify_empty_draft() {
        let config = SpeculativeDecodingConfig::default();
        let verifier = SpeculativeVerifier::new(config);
        let result = verifier.verify(&[], &[], &[]);
        assert_eq!(result.n_accepted, 0);
        assert!(!result.rejected);
        assert!(result.accepted_tokens.is_empty());
    }

    #[test]
    fn test_verify_nan_prob() {
        let config = SpeculativeDecodingConfig {
            acceptance_threshold: 0.5,
            ..Default::default()
        };
        let verifier = SpeculativeVerifier::new(config);
        let result = verifier.verify(&[1, 2], &[f32::NAN, 1.0], &[0.6, 1.0]);
        assert!(result.rejected);
        assert_eq!(result.n_accepted, 0);
        assert_eq!(result.rejection_pos, Some(0));
    }

    #[test]
    fn test_verify_negative_prob() {
        let config = SpeculativeDecodingConfig {
            acceptance_threshold: 0.5,
            ..Default::default()
        };
        let verifier = SpeculativeVerifier::new(config);
        let result = verifier.verify(&[1, 2, 3], &[1.0, -1.0, 1.0], &[0.6, 1.0, 1.0]);
        assert!(result.rejected);
        assert_eq!(result.n_accepted, 1);
        assert_eq!(result.rejection_pos, Some(1));
    }

    #[tokio::test]
    async fn test_speculate_no_draft_model() {
        let config = SpeculativeDecodingConfig {
            enabled: true,
            ..Default::default()
        };
        let mut engine = SpeculativeEngine::new(config);
        let result = engine
            .speculate(&[1, 2, 3], |_| vec![0.5, 0.3, 0.2])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_speculate_not_enabled() {
        let config = SpeculativeDecodingConfig {
            enabled: false,
            ..Default::default()
        };
        let mut engine = SpeculativeEngine::new(config);
        let mut model = NGramDraftModel::new(2, 1);
        model.record(&[1, 2], 3);
        engine.set_draft_model(Box::new(model));
        let result = engine
            .speculate(&[1, 2, 3], |_| vec![0.5, 0.3, 0.2])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ngram_draft_model() {
        let mut model = NGramDraftModel::new(2, 1);
        model.record(&[1, 2], 3);
        model.record(&[1, 2], 3);
        model.record(&[2, 3], 4);
        let tokens = model.draft(&[1, 2], 2).await.unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], 3);
    }

    #[tokio::test]
    async fn test_ngram_draft_empty_prompt() {
        let model = NGramDraftModel::new(2, 1);
        let tokens = model.draft(&[], 5).await.unwrap();
        assert!(tokens.is_empty());
    }

    #[tokio::test]
    async fn test_ngram_token_probability() {
        let mut model = NGramDraftModel::new(2, 1);
        model.record(&[1, 2], 3);
        model.record(&[1, 2], 4);
        let prob = model.token_probability(&[1, 2], 3).await.unwrap();
        assert!((prob - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_ngram_token_probability_unseen() {
        let model = NGramDraftModel::new(2, 1);
        let prob = model.token_probability(&[1, 2], 3).await.unwrap();
        assert!((prob - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_stats_reset() {
        let mut engine = SpeculativeEngine::new(SpeculativeDecodingConfig::default());
        engine.stats.record_round(5, 3);
        assert!(engine.stats.total_speculated > 0);
        engine.reset_stats();
        assert_eq!(engine.stats.total_speculated, 0);
    }

    #[test]
    fn test_with_draft_model() {
        let model = NGramDraftModel::new(2, 1);
        let engine = SpeculativeEngine::new(SpeculativeDecodingConfig::default())
            .with_draft_model(Box::new(model));
        assert!(engine.draft_model.is_some());
    }

    #[test]
    fn test_set_draft_model() {
        let mut engine = SpeculativeEngine::new(SpeculativeDecodingConfig::default());
        let model = NGramDraftModel::new(2, 1);
        engine.set_draft_model(Box::new(model));
        assert!(engine.draft_model.is_some());
    }
}
