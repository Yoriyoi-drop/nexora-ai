//! Beam Search
//!
//! Beam search decoding strategy untuk inference.
//! Fully wired and usable — call `BeamSearchEngine::search()` for a complete beam search run,
//! or use `initialize()` / `expand_beam()` / `should_stop()` for manual control.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

/// Persistent, append-only token list using shared tails.
/// Adding a token is O(1) — no full Vec clone.
/// This implementation uses Arc-based sharing to avoid O(n²) clone complexity.
/// The to_vec() path is O(n) and only used for final output; intermediate
/// beam expansion never clones the full token vector, only shares Arc references.
#[derive(Debug, Clone)]
pub struct PersistentTokens {
    prefix: Option<Arc<PersistentTokens>>,
    token: Arc<GeneratedToken>,
    len: usize,
}

impl PersistentTokens {
    pub fn empty() -> Option<Arc<PersistentTokens>> {
        Some(Arc::new(PersistentTokens {
            prefix: None,
            token: Arc::new(GeneratedToken {
                token_id: 0,
                token_text: Arc::from(""),
                log_prob: 0.0,
                position: 0,
                metadata: std::collections::HashMap::new(),
            }),
            len: 0,
        }))
    }

    pub fn append(
        prefix: Option<Arc<PersistentTokens>>,
        token: Arc<GeneratedToken>,
    ) -> Arc<PersistentTokens> {
        let new_len = prefix.as_ref().map(|p| p.len).unwrap_or(0) + 1;
        Arc::new(PersistentTokens {
            prefix: prefix,
            token,
            len: new_len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> PersistentTokensIter {
        PersistentTokensIter {
            current: Some(Arc::new(self.clone())),
        }
    }

    pub fn to_vec(&self) -> Vec<Arc<GeneratedToken>> {
        let mut result = Vec::with_capacity(self.len);
        let mut current = Some(Arc::new(self.clone()));
        while let Some(node) = current.take() {
            result.push(Arc::clone(&node.token));
            current = node.prefix.clone();
        }
        result.reverse();
        result
    }
}

pub struct PersistentTokensIter {
    current: Option<Arc<PersistentTokens>>,
}

impl Iterator for PersistentTokensIter {
    type Item = Arc<GeneratedToken>;
    fn next(&mut self) -> Option<Self::Item> {
        let node = self.current.take()?;
        let token = Arc::clone(&node.token);
        self.current = node.prefix.clone();
        Some(token)
    }
}

use crate::decoding;
use crate::{GeneratedToken, InferenceError, Result};

/// Configuration untuk beam search
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Beam size
    pub beam_size: usize,
    /// Length penalty coefficient
    pub length_penalty: f32,
    /// Early stopping criteria
    pub early_stopping: bool,
    /// Minimum beam size for early stopping
    pub min_beam_size: usize,
    /// Divergence penalty
    pub divergence_penalty: f32,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Enable beam pruning
    pub enable_pruning: bool,
    /// Maximum beam candidates
    pub max_candidates: usize,
    /// Maximum total stored hypotheses (including converged/diverged)
    pub max_hypotheses: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        // beam_size=4 is the standard starting point — large enough to explore
        // diverse hypotheses but small enough to keep compute ~4× greedy decoding.
        // Beam size 1 is equivalent to greedy decoding (no search), so 4 is the
        // minimum recommended for meaningful beam search.
        Self {
            beam_size: 4,
            length_penalty: 1.0,
            early_stopping: true,
            min_beam_size: 2,
            divergence_penalty: 0.1,
            convergence_threshold: 0.01,
            enable_pruning: true,
            max_candidates: 100,
            max_hypotheses: 200,
        }
    }
}

/// Beam hypothesis
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Unique hypothesis ID
    pub id: Uuid,
    /// Generated tokens (persistent, O(1) append)
    pub tokens: Option<Arc<PersistentTokens>>,
    /// Cached token count
    pub token_count: usize,
    /// Cumulative log probability
    pub cumulative_log_prob: f32,
    /// Normalized score
    pub score: f32,
    /// Length penalty coefficient
    pub length_penalty: f32,
    /// Is this hypothesis finished?
    pub finished: bool,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Generation metadata (Arc for O(1) clone)
    pub metadata: Arc<HashMap<String, serde_json::Value>>,
}

impl BeamHypothesis {
    /// Create new hypothesis
    pub fn new(length_penalty: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            tokens: None,
            token_count: 0,
            cumulative_log_prob: 0.0,
            score: 0.0,
            length_penalty,
            finished: false,
            finish_reason: None,
            metadata: Arc::new(HashMap::new()),
        }
    }

    /// Create hypothesis from parent — shares metadata Arc (O(1))
    fn from_parent(
        parent: &BeamHypothesis,
        tokens: Option<Arc<PersistentTokens>>,
        token_count: usize,
        cumulative_log_prob: f32,
        score: f32,
    ) -> Self {
        Self {
            id: parent.id,
            tokens,
            token_count,
            cumulative_log_prob,
            score,
            length_penalty: parent.length_penalty,
            finished: parent.finished,
            finish_reason: parent.finish_reason.clone(),
            metadata: Arc::clone(&parent.metadata),
        }
    }

    /// Add token to hypothesis (O(1) — no full Vec clone)
    pub fn add_token(&mut self, token: Arc<GeneratedToken>) {
        self.cumulative_log_prob += token.log_prob;
        let prev = self.tokens.take();
        self.tokens = Some(PersistentTokens::append(prev, token));
        self.token_count += 1;
        self.update_score();
    }

    /// Mark as finished
    pub fn finish(&mut self, reason: String) {
        self.finished = true;
        self.finish_reason = Some(reason);
        self.update_score();
    }

    /// Get generated text
    pub fn get_text(&self) -> String {
        let tokens = self.collect_tokens();
        tokens
            .iter()
            .map(|t| (*t.token_text).to_string())
            .collect::<String>()
    }

    /// Get token count
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Collect tokens into a Vec (O(n), called only for final output)
    pub fn collect_tokens(&self) -> Vec<Arc<GeneratedToken>> {
        match &self.tokens {
            Some(chain) => chain.to_vec(),
            None => Vec::new(),
        }
    }

    /// Iterate over tokens
    pub fn iter_tokens(&self) -> PersistentTokensIter {
        match &self.tokens {
            Some(chain) => chain.iter(),
            None => PersistentTokensIter { current: None },
        }
    }

    /// Update normalized score
    fn update_score(&mut self) {
        let length = self.token_count as f32;
        if length > 0.0 {
            let lp = ((length + 5.0) / (6.0)).powf(self.length_penalty);
            self.score = self.cumulative_log_prob / lp;
        } else {
            self.score = self.cumulative_log_prob;
        }
    }

    /// Check if hypothesis is valid
    pub fn is_valid(&self) -> bool {
        self.token_count > 0 && self.cumulative_log_prob.is_finite()
    }
}

/// Beam search state
#[derive(Debug)]
pub struct BeamSearchState {
    /// Current beam hypotheses
    pub hypotheses: Vec<BeamHypothesis>,
    /// Generation step
    pub step: usize,
    /// Converged hypotheses
    pub converged_hypotheses: Vec<BeamHypothesis>,
    /// Diverged hypotheses
    pub diverged_hypotheses: Vec<BeamHypothesis>,
    /// Search metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Beam search engine
pub struct BeamSearchEngine {
    config: BeamSearchConfig,
}

impl BeamSearchEngine {
    /// Create new beam search engine
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    /// Initialize beam search
    pub fn initialize(&self, initial_logits: &[f32]) -> Result<BeamSearchState> {
        debug!(
            "Initializing beam search with beam size {}",
            self.config.beam_size
        );

        // Create initial hypotheses
        let mut hypotheses = Vec::with_capacity(self.config.beam_size);

        let mut indexed_logits: Vec<(usize, f32)> = initial_logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l))
            .collect();
        let k = self.config.beam_size.min(indexed_logits.len());
        indexed_logits
            .select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (_i, &(token_id, logit)) in indexed_logits
            .iter()
            .take(self.config.beam_size)
            .enumerate()
        {
            let token = GeneratedToken::new(
                token_id as u32,
                decoding::alloc_token_text(token_id),
                logit,
                0,
            );

            let mut hypothesis = BeamHypothesis::new(self.config.length_penalty);
            hypothesis.add_token(Arc::new(token));

            hypotheses.push(hypothesis);
        }

        let state = BeamSearchState {
            hypotheses,
            step: 1,
            converged_hypotheses: Vec::with_capacity(self.config.beam_size),
            diverged_hypotheses: Vec::with_capacity(self.config.beam_size / 2),
            metadata: HashMap::new(),
        };

        info!(
            "Beam search initialized with {} hypotheses",
            state.hypotheses.len()
        );
        Ok(state)
    }

    /// Expand beam hypotheses
    pub fn expand_beam(
        &self,
        state: &mut BeamSearchState,
        logits_batch: &[&[f32]],
    ) -> Result<()> {
        debug!(
            "Expanding beam at step {}, {} hypotheses",
            state.step,
            state.hypotheses.len()
        );

        if state.hypotheses.len() != logits_batch.len() {
            return Err(InferenceError::DecodingError(format!(
                "Logits batch size ({}) doesn't match hypotheses count ({})",
                logits_batch.len(),
                state.hypotheses.len()
            )));
        }

        let mut new_candidates = Vec::with_capacity(state.hypotheses.len());

        for (hypothesis_idx, hypothesis) in state.hypotheses.iter().enumerate() {
            if hypothesis.finished {
                // Keep finished hypotheses as-is
                new_candidates.push(hypothesis.clone());
                continue;
            }

            let logits = &logits_batch[hypothesis_idx];

            // Select top candidates for this hypothesis
            let candidates_per_hypothesis =
                std::cmp::max(self.config.beam_size / state.hypotheses.len(), 1);

            let mut indexed_logits: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
            let k = candidates_per_hypothesis.min(indexed_logits.len());
            indexed_logits
                .select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            for &(token_id, logit) in indexed_logits.iter().take(candidates_per_hypothesis) {
                let token = Arc::new(GeneratedToken::new(
                    token_id as u32,
                    decoding::alloc_token_text(token_id),
                    logit,
                    state.step,
                ));

                // O(1) append using persistent tokens — no full Vec clone
                let new_tokens =
                    PersistentTokens::append(hypothesis.tokens.clone(), Arc::clone(&token));
                let new_token_count = hypothesis.token_count + 1;
                let new_cumulative_log_prob = hypothesis.cumulative_log_prob + logit;
                let new_len = new_token_count as f32;
                let lp = ((new_len + 5.0) / (6.0)).powf(hypothesis.length_penalty);
                let new_score = new_cumulative_log_prob / lp;
                let new_hypothesis = BeamHypothesis::from_parent(
                    hypothesis,
                    Some(new_tokens),
                    new_token_count,
                    new_cumulative_log_prob,
                    new_score,
                );

                new_candidates.push(new_hypothesis);
            }
        }

        // Apply pruning if enabled
        if self.config.enable_pruning {
            new_candidates = self.prune_candidates(new_candidates)?;
        }

        // Select best candidates for next beam
        let beam_size = self.config.beam_size.min(new_candidates.len());
        new_candidates.select_nth_unstable_by(beam_size, |a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        });

        state.hypotheses = new_candidates.into_iter().take(beam_size).collect();

        // Check for convergence
        self.check_convergence(state)?;

        // Bound total stored hypotheses to prevent unbounded memory growth
        self.prune_hypothesis_store(state);

        state.step += 1;

        debug!("Beam expanded to {} hypotheses", state.hypotheses.len());
        Ok(())
    }

    /// Check if search should stop
    pub fn should_stop(&self, state: &BeamSearchState, max_steps: usize) -> bool {
        // Check step limit
        if state.step >= max_steps {
            return true;
        }

        // Check if all hypotheses are finished
        if state.hypotheses.iter().all(|h| h.finished) {
            return true;
        }

        // Check early stopping criteria
        if self.config.early_stopping {
            if state.hypotheses.len() <= self.config.min_beam_size {
                return true;
            }

            // Check if top hypotheses are very similar (converged)
            if self.are_hypotheses_converged(&state.hypotheses) {
                return true;
            }
        }

        false
    }

    /// Get best hypothesis
    pub fn get_best_hypothesis<'a>(
        &self,
        state: &'a BeamSearchState,
    ) -> Option<&'a BeamHypothesis> {
        state
            .hypotheses
            .iter()
            .chain(state.converged_hypotheses.iter())
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
    }

    /// Get top N hypotheses
    pub fn get_top_hypotheses<'a>(
        &self,
        state: &'a BeamSearchState,
        n: usize,
    ) -> Vec<&'a BeamHypothesis> {
        let mut all_hypotheses: Vec<&BeamHypothesis> = state
            .hypotheses
            .iter()
            .chain(state.converged_hypotheses.iter())
            .collect();

        all_hypotheses.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        all_hypotheses.into_iter().take(n).collect()
    }

    /// Prune candidates based on diversity and quality
    fn prune_candidates(&self, candidates: Vec<BeamHypothesis>) -> Result<Vec<BeamHypothesis>> {
        if candidates.len() <= self.config.max_candidates {
            return Ok(candidates);
        }

        let mut pruned = Vec::with_capacity(candidates.len().min(self.config.max_candidates));
        let mut diversity_tracker: HashMap<u32, f32> = HashMap::with_capacity(candidates.len());

        for candidate in candidates {
            // Check diversity using first token ID as key (avoids String allocation for dedup)
            let first_token_id = candidate
                .tokens
                .as_ref()
                .and_then(|pt| pt.iter().last().map(|t| t.token_id))
                .unwrap_or(0);
            let diversity_score = self.calculate_diversity_score(&candidate, &diversity_tracker);

            // Apply divergence penalty
            let adjusted_score =
                candidate.score - (self.config.divergence_penalty * diversity_score);

            // Keep if score is good enough or not too similar to existing candidates
            if adjusted_score > -10.0 || diversity_score < 0.8 {
                pruned.push(candidate);
                diversity_tracker.insert(first_token_id, diversity_score);
            }

            if pruned.len() >= self.config.max_candidates {
                break;
            }
        }

        Ok(pruned)
    }

    /// Prune hypothesis store to bounded capacity
    fn prune_hypothesis_store(&self, state: &mut BeamSearchState) {
        let max = self.config.max_hypotheses;
        let to_prune = state.hypotheses.len()
            + state.converged_hypotheses.len()
            + state.diverged_hypotheses.len();

        if to_prune <= max {
            return;
        }

        // Sort converged by score, keep highest scorers
        state
            .converged_hypotheses
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        let budget = max.saturating_sub(state.hypotheses.len() + state.diverged_hypotheses.len());
        state.converged_hypotheses.truncate(budget);
    }

    /// Check for convergence among hypotheses (token-ID based, no string allocation)
    fn check_convergence(&self, state: &mut BeamSearchState) -> Result<()> {
        if state.hypotheses.len() < 2 {
            return Ok(());
        }

        let mut converged = Vec::with_capacity(state.hypotheses.len() / 2);
        let mut remaining: Vec<BeamHypothesis> = Vec::with_capacity(state.hypotheses.len());

        let hyp_len = state.hypotheses.len();
        let mut used = vec![false; hyp_len];

        // Pre-compute HashSet for each hypothesis to avoid O(n²) allocation
        let mut token_sets: Vec<HashSet<u32>> = Vec::with_capacity(hyp_len);
        for hyp in &state.hypotheses {
            token_sets.push(hyp.iter_tokens().map(|t| t.token_id).collect());
        }

        for i in 0..hyp_len {
            if used[i] {
                continue;
            }
            used[i] = true;

            let mut group = vec![i];
            for j in (i + 1)..hyp_len {
                if !used[j] {
                    let similarity =
                        Self::token_sets_similarity(&token_sets[i], &token_sets[j]);
                    if similarity > (1.0 - self.config.convergence_threshold) {
                        group.push(j);
                        used[j] = true;
                    }
                }
            }

            group.sort_by(|&a, &b| {
                state.hypotheses[b]
                    .score
                    .partial_cmp(&state.hypotheses[a].score)
                    .unwrap_or(Ordering::Equal)
            });

            remaining.push(state.hypotheses[group[0]].clone());
            for &idx in group.iter().skip(1) {
                let mut converged_hypothesis = state.hypotheses[idx].clone();
                converged_hypothesis.finish("converged".to_string());
                converged.push(converged_hypothesis);
            }
        }

        state.hypotheses = remaining;
        state.converged_hypotheses.extend(converged);

        Ok(())
    }

    /// Check if hypotheses are converged (token-ID based)
    fn are_hypotheses_converged(&self, hypotheses: &[BeamHypothesis]) -> bool {
        if hypotheses.len() < 2 {
            return true;
        }

        let first_set: HashSet<u32> = hypotheses[0].iter_tokens().map(|t| t.token_id).collect();
        for hypothesis in hypotheses.iter().skip(1) {
            let other_set: HashSet<u32> = hypothesis.iter_tokens().map(|t| t.token_id).collect();
            let similarity = Self::token_sets_similarity(&first_set, &other_set);
            if similarity < (1.0 - self.config.convergence_threshold) {
                return false;
            }
        }

        true
    }

    /// Jaccard similarity on pre-computed token ID sets
    fn token_sets_similarity(a_ids: &HashSet<u32>, b_ids: &HashSet<u32>) -> f32 {
        if a_ids.is_empty() || b_ids.is_empty() {
            return 0.0;
        }
        let intersection = a_ids.intersection(b_ids).count();
        let union = a_ids.union(b_ids).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Calculate diversity score using token IDs
    fn calculate_diversity_score(
        &self,
        hypothesis: &BeamHypothesis,
        _existing: &HashMap<u32, f32>,
    ) -> f32 {
        if _existing.is_empty() {
            return 0.0;
        }
        let count = hypothesis.token_count;
        if count == 0 {
            return 0.0;
        }
        let unique: HashSet<u32> = hypothesis.iter_tokens().map(|t| t.token_id).collect();
        1.0 - (unique.len() as f32 / count as f32)
    }

    /// Run a complete beam search, returning the best hypothesis.
    ///
    /// `initial_logits` — logits for the first token position (vocab-sized).
    /// `logits_provider` — subsequent logits for each remaining step.
    /// `max_steps` — maximum number of decoding steps.
    pub fn search(
        &self,
        initial_logits: &[f32],
        logits_provider: &[Vec<f32>],
        max_steps: usize,
    ) -> Result<Option<BeamHypothesis>> {
        let mut state = self.initialize(initial_logits)?;
        for step_logits in logits_provider.iter().take(max_steps - 1) {
            let batched_logits: Vec<&[f32]> = (0..state.hypotheses.len())
                .map(|_| step_logits.as_slice())
                .collect();
            self.expand_beam(&mut state, &batched_logits)?;
            if self.should_stop(&state, max_steps) {
                break;
            }
        }
        Ok(self.get_best_hypothesis(&state).cloned())
    }
}

impl Default for BeamSearchEngine {
    fn default() -> Self {
        Self::new(BeamSearchConfig::default())
    }
}

/// Helper function to create beam search configuration
pub fn create_beam_search_config(
    beam_size: usize,
    length_penalty: f32,
    early_stopping: bool,
) -> BeamSearchConfig {
    BeamSearchConfig {
        beam_size,
        length_penalty,
        early_stopping,
        ..Default::default()
    }
}

/// Helper function to run a complete beam search.
///
/// Convenience wrapper around [`BeamSearchEngine::search`].
pub fn run_beam_search(
    engine: &BeamSearchEngine,
    initial_logits: &[f32],
    subsequent_logits: &[Vec<f32>],
    max_steps: usize,
) -> Result<Option<BeamHypothesis>> {
    engine.search(initial_logits, subsequent_logits, max_steps)
}
