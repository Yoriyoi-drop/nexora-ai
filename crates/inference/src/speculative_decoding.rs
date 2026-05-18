//! Speculative Decoding
//!
//! SOTA decoding strategy: draft model generate K token → target model verify dalam 1 forward pass.
//! Speedup 2-3x tanpa quality loss.
//!
//! Cara kerja:
//! 1. Draft model M_q (kecil) → generate K token draft secara autoregressive
//! 2. Target model M_p (besar) → verifikasi semua K token dalam SATU forward pass
//! 3. Accept/reject berdasarkan rejection sampling untuk menjaga distribusi identik

use rand::Rng;

use crate::{Result, GeneratedToken};
use crate::decoding::{self, DecodingConfig, DecodingContext, DecodingStrategy};
use crate::sampler::Sampler;

/// Configuration untuk speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingConfig {
    /// Number of draft tokens to generate (K)
    pub num_draft_tokens: usize,
    /// Minimum acceptance rate threshold
    pub min_acceptance_rate: f32,
    /// Enable dynamic draft length adjustment
    pub dynamic_draft_length: bool,
    /// Target acceptance rate (for dynamic adjustment)
    pub target_acceptance_rate: f32,
    /// Use rejection sampling (guarantees identical distribution)
    pub use_rejection_sampling: bool,
    /// Enable bonus token (always add 1 from target model)
    pub enable_bonus_token: bool,
    /// Draft model scaling factor (larger = more aggressive)
    pub draft_scale_factor: f32,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 5,
            min_acceptance_rate: 0.5,
            dynamic_draft_length: true,
            target_acceptance_rate: 0.8,
            use_rejection_sampling: true,
            enable_bonus_token: true,
            draft_scale_factor: 1.0,
        }
    }
}

/// Result dari speculative decoding step
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Accepted tokens (dari draft + bonus)
    pub accepted_tokens: Vec<GeneratedToken>,
    /// Total tokens generated this step
    pub total_tokens: usize,
    /// Draft tokens generated
    pub draft_tokens: usize,
    /// Draft tokens accepted
    pub accepted_draft: usize,
    /// Acceptance rate
    pub acceptance_rate: f32,
    /// Whether bonus token was added
    pub bonus_added: bool,
}

/// Speculative decoding engine
pub struct SpeculativeDecoder<D: DecodingStrategy> {
    /// Config
    config: SpeculativeDecodingConfig,
    /// Target model decoding strategy (M_p)
    target_strategy: D,
    /// Draft model decoding strategy (M_q)
    draft_strategy: D,
    /// Sampler untuk rejection sampling
    sampler: Sampler,
    /// Current acceptance rate (moving average)
    current_acceptance_rate: f32,
    /// Current draft length (dynamic)
    current_draft_length: usize,
}

impl<D: DecodingStrategy> SpeculativeDecoder<D> {
    pub fn new(
        config: SpeculativeDecodingConfig,
        target_strategy: D,
        draft_strategy: D,
    ) -> Self {
        let initial_draft = config.num_draft_tokens;
        Self {
            config,
            target_strategy,
            draft_strategy,
            sampler: Sampler::new(
                crate::sampler::SamplingConfig::default()
            ),
            current_acceptance_rate: 0.8,
            current_draft_length: initial_draft,
        }
    }

    /// Execute speculative decoding step
    /// 1. Draft: generate K token dengan draft model
    /// 2. Verify: target model verifikasi dalam 1 forward pass
    /// 3. Accept/reject: rejection sampling untuk menjaga distribusi
    pub fn step(
        &mut self,
        draft_logits_fn: &impl Fn(&[u32]) -> Result<Vec<f32>>,
        target_logits_fn: &impl Fn(&[u32]) -> Result<Vec<Vec<f32>>>,
        input_ids: &[u32],
        decoding_config: &DecodingConfig,
        context: &DecodingContext,
    ) -> Result<SpeculativeResult> {
        let k = self.current_draft_length;

        // Phase 1: Draft generation (autoregressive, model kecil)
        let (draft_tokens, draft_logits) = self.generate_draft_tokens(
            draft_logits_fn,
            input_ids,
            k,
            decoding_config,
            context,
        )?;

        let draft_ids: Vec<u32> = draft_tokens.iter().map(|t| t.token_id).collect();

        // Phase 2: Verification (target model, 1 forward pass for all K tokens)
        let mut full_input = input_ids.to_vec();
        full_input.extend_from_slice(&draft_ids);

        let target_logits = target_logits_fn(&full_input)?;
        // target_logits: Vec<Vec<f32>> — logits untuk setiap posisi setelah input

        // Phase 3: Rejection sampling per token position
        let mut accepted = Vec::with_capacity(k + 1);
        let mut rng = rand::thread_rng();

        for (i, draft_token) in draft_tokens.iter().enumerate() {
            if i >= target_logits.len() {
                break;
            }

            let target_logit = &target_logits[i];
            let draft_logit = if i < draft_logits.len() { &draft_logits[i] } else { target_logit };

            if self.config.use_rejection_sampling {
                // Rejection sampling: accept if p_target / p_draft > uniform(0,1)
                let p_target = self.compute_token_probability(target_logit, draft_token.token_id);
                let p_draft = self.compute_token_probability(draft_logit, draft_token.token_id);
                let ratio = if p_draft > 1e-10 { p_target / p_draft } else { 0.0 };
                let u: f32 = rng.gen();

                if ratio >= u {
                    accepted.push(draft_token.clone());
                } else {
                    // Resample dari adjusted distribution
                    let adjusted = self.compute_adjusted_distribution(target_logit, draft_logit);
                    let sum: f32 = adjusted.iter().sum();
                    let probs: Vec<f32> = if sum > 0.0 {
                        adjusted.iter().map(|v| v / sum).collect()
                    } else {
                        vec![1.0 / adjusted.len() as f32; adjusted.len()]
                    };
                    let resampled_id = self.sampler.sample(&probs)?;
                    let log_prob = target_logit.get(resampled_id).copied().unwrap_or(0.0);
                    let token = GeneratedToken::new(
                        resampled_id as u32,
                        decoding::alloc_token_text(resampled_id),
                        log_prob,
                        context.step + accepted.len(),
                    );
                    accepted.push(token);
                    break; // Stop after first rejection
                }
            } else {
                // Greedy acceptance: accept if argmax matches
                let target_argmax = self.argmax(target_logit);
                if target_argmax == draft_token.token_id {
                    accepted.push(draft_token.clone());
                } else {
                    let token = GeneratedToken::new(
                        target_argmax,
                        decoding::alloc_token_text(target_argmax as usize),
                        target_logit[target_argmax as usize],
                        context.step + accepted.len(),
                    );
                    accepted.push(token);
                    break;
                }
            }
        }

        let accepted_draft = accepted.len();
        let acceptance_rate = if k > 0 {
            accepted_draft as f32 / k as f32
        } else {
            1.0
        };

        // Phase 4: Bonus token (always from target model)
        let mut bonus_added = false;
        if self.config.enable_bonus_token && accepted.len() > 0 {
            let last_pos = accepted.len().min(target_logits.len().saturating_sub(1));
            if last_pos < target_logits.len() {
                let bonus_logit = &target_logits[last_pos];
                let bonus_selection = self.target_strategy.select_token(
                    bonus_logit,
                    decoding_config,
                    context,
                )?;
                let bonus_token = GeneratedToken::new(
                    bonus_selection.token_id,
                    decoding::alloc_token_text(bonus_selection.token_id as usize),
                    bonus_selection.log_prob,
                    context.step + accepted.len(),
                );
                accepted.push(bonus_token);
                bonus_added = true;
            }
        }

        // Update dynamic draft length
        if self.config.dynamic_draft_length {
            self.update_draft_length(acceptance_rate);
        }
        self.current_acceptance_rate = self.current_acceptance_rate * 0.9 + acceptance_rate * 0.1;

        Ok(SpeculativeResult {
            total_tokens: accepted.len(),
            draft_tokens: k,
            accepted_draft,
            acceptance_rate,
            bonus_added,
            accepted_tokens: accepted,
        })
    }

    /// Generate K draft tokens menggunakan draft model (autoregressive)
    /// Returns (draft_tokens, draft_logits_per_step)
    fn generate_draft_tokens(
        &self,
        draft_logits_fn: &impl Fn(&[u32]) -> Result<Vec<f32>>,
        input_ids: &[u32],
        k: usize,
        decoding_config: &DecodingConfig,
        context: &DecodingContext,
    ) -> Result<(Vec<GeneratedToken>, Vec<Vec<f32>>)> {
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits = Vec::with_capacity(k);
        let mut current_ids = input_ids.to_vec();

        for i in 0..k {
            let logits = draft_logits_fn(&current_ids)?;
            let selection = self.draft_strategy.select_token(
                &logits,
                decoding_config,
                context,
            )?;

            let token = GeneratedToken::new(
                selection.token_id,
                decoding::alloc_token_text(selection.token_id as usize),
                selection.log_prob * self.config.draft_scale_factor,
                context.step + i,
            );

            draft_tokens.push(token);
            draft_logits.push(logits);
            current_ids.push(selection.token_id);
        }

        Ok((draft_tokens, draft_logits))
    }

    /// Compute probability of a specific token from logits
    fn compute_token_probability(&self, logits: &[f32], token_id: u32) -> f32 {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        let p = (logits[token_id as usize] - max_logit).exp();
        if sum > 1e-10 { p / sum } else { 0.0 }
    }

    /// Compute adjusted distribution for resampling after rejection
    fn compute_adjusted_distribution(&self, target_logits: &[f32], draft_logits: &[f32]) -> Vec<f32> {
        let max_t = target_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_d = draft_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        target_logits.iter().zip(draft_logits.iter()).map(|(&t, &d)| {
            let p_target = (t - max_t).exp();
            let p_draft = (d - max_d).exp();
            // Adjusted: max(0, p_target - p_draft) — ensures correct distribution
            (p_target - p_draft).max(0.0)
        }).collect()
    }

    fn argmax(&self, logits: &[f32]) -> u32 {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    /// Adjust draft length based on acceptance rate
    fn update_draft_length(&mut self, acceptance_rate: f32) {
        let target = self.config.target_acceptance_rate;
        let current = self.current_draft_length as f32;

        if acceptance_rate > target + 0.1 {
            // Acceptance rate tinggi → generate lebih banyak draft tokens
            self.current_draft_length = (current * 1.2).ceil() as usize;
        } else if acceptance_rate < target - 0.2 {
            // Acceptance rate rendah → kurangi draft tokens
            self.current_draft_length = (current * 0.8).max(2.0) as usize;
        }

        self.current_draft_length = self.current_draft_length
            .clamp(2, self.config.num_draft_tokens * 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoding::TokenSelection;
    use std::collections::HashMap;

    /// Mock decoding strategy untuk testing
    struct MockStrategy;

    impl DecodingStrategy for MockStrategy {
        fn name(&self) -> &str { "mock" }
        fn select_token(&self, logits: &[f32], _config: &DecodingConfig, _context: &DecodingContext) -> Result<TokenSelection> {
            let argmax = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, &v)| (idx as u32, v))
                .unwrap_or((0, 0.0));
            Ok(TokenSelection {
                token_id: argmax.0,
                token_text: "[token_".to_string() + &argmax.0.to_string() + "]",
                log_prob: argmax.1,
                selection_prob: argmax.1,
                metadata: HashMap::new(),
            })
        }
        fn validate_config(&self, _config: &DecodingConfig) -> Result<()> { Ok(()) }
    }

    #[test]
    fn test_draft_length_dynamic_adjustment() {
        let config = SpeculativeDecodingConfig::default();
        let target = MockStrategy;
        let draft = MockStrategy;
        let mut decoder = SpeculativeDecoder::new(config, target, draft);

        assert_eq!(decoder.current_draft_length, 5);
        decoder.update_draft_length(0.95); // High acceptance
        assert!(decoder.current_draft_length >= 5);
    }
}
