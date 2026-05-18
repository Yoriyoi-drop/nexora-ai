//! Beam Search
//! 
//! Beam search decoding strategy untuk inference.

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::Arc;
use uuid::Uuid;
use tracing::{debug, info};

use crate::{Result, InferenceError, GeneratedToken};
use crate::decoding;

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
    /// Generated tokens
    pub tokens: Arc<Vec<GeneratedToken>>,
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
    /// Generation metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl BeamHypothesis {
    /// Create new hypothesis
    pub fn new(length_penalty: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            tokens: Arc::new(Vec::new()),
            cumulative_log_prob: 0.0,
            score: 0.0,
            length_penalty,
            finished: false,
            finish_reason: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Add token to hypothesis
    pub fn add_token(&mut self, token: GeneratedToken) {
        self.cumulative_log_prob += token.log_prob;
        let mut new_tokens = (*self.tokens).clone();
        new_tokens.push(token);
        self.tokens = Arc::new(new_tokens);
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
        self.tokens.iter().map(|t| (*t.token_text).to_string()).collect::<String>()
    }
    
    /// Get token count
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
    
    /// Update normalized score
    fn update_score(&mut self) {
        let length = self.tokens.len() as f32;
        if length > 0.0 {
            // Length penalty: (length + 5) / (5 + 1) ^ length_penalty
            let lp = ((length + 5.0) / (6.0)).powf(self.length_penalty);
            self.score = self.cumulative_log_prob / lp;
        } else {
            self.score = self.cumulative_log_prob;
        }
    }
    
    /// Check if hypothesis is valid
    pub fn is_valid(&self) -> bool {
        !self.tokens.is_empty() && self.cumulative_log_prob.is_finite()
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
        debug!("Initializing beam search with beam size {}", self.config.beam_size);
        
        // Create initial hypotheses
        let mut hypotheses = Vec::with_capacity(self.config.beam_size);
        
        let mut indexed_logits: Vec<(usize, f32)> = initial_logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        let k = self.config.beam_size.min(indexed_logits.len());
        indexed_logits.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        
        for (_i, &(token_id, logit)) in indexed_logits.iter().take(self.config.beam_size).enumerate() {
            let token = GeneratedToken::new(
                token_id as u32,
                decoding::alloc_token_text(token_id),
                logit,
                0,
            );
            
            let mut hypothesis = BeamHypothesis::new(self.config.length_penalty);
            hypothesis.add_token(token);
            
            hypotheses.push(hypothesis);
        }
        
        let state = BeamSearchState {
            hypotheses,
            step: 1,
            converged_hypotheses: Vec::with_capacity(self.config.beam_size),
            diverged_hypotheses: Vec::with_capacity(self.config.beam_size / 2),
            metadata: HashMap::new(),
        };
        
        info!("Beam search initialized with {} hypotheses", state.hypotheses.len());
        Ok(state)
    }
    
    /// Expand beam hypotheses
    pub fn expand_beam(&self, state: &mut BeamSearchState, logits_batch: &[Vec<f32>]) -> Result<()> {
        debug!("Expanding beam at step {}, {} hypotheses", state.step, state.hypotheses.len());
        
        if state.hypotheses.len() != logits_batch.len() {
            return Err(InferenceError::DecodingError(
                format!("Logits batch size ({}) doesn't match hypotheses count ({})", 
                       logits_batch.len(), state.hypotheses.len())
            ));
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
            let candidates_per_hypothesis = std::cmp::max(
                self.config.beam_size / state.hypotheses.len(),
                1
            );
            
            let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
            let k = candidates_per_hypothesis.min(indexed_logits.len());
            indexed_logits.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            
            for &(token_id, logit) in indexed_logits.iter().take(candidates_per_hypothesis) {
                let token = GeneratedToken::new(
                    token_id as u32,
                    decoding::alloc_token_text(token_id),
                    logit,
                    state.step,
                );
                
                let mut new_tokens = (*hypothesis.tokens).clone();
                new_tokens.push(token);
                let new_cumulative_log_prob = hypothesis.cumulative_log_prob + logit;
                let new_len = new_tokens.len() as f32;
                let lp = ((new_len + 5.0) / (6.0)).powf(hypothesis.length_penalty);
                let new_score = new_cumulative_log_prob / lp;
                let new_hypothesis = BeamHypothesis {
                    id: hypothesis.id,
                    tokens: Arc::new(new_tokens),
                    cumulative_log_prob: new_cumulative_log_prob,
                    score: new_score,
                    length_penalty: hypothesis.length_penalty,
                    finished: hypothesis.finished,
                    finish_reason: hypothesis.finish_reason.clone(),
                    metadata: hypothesis.metadata.clone(),
                };
                
                new_candidates.push(new_hypothesis);
            }
        }
        
        // Apply pruning if enabled
        if self.config.enable_pruning {
            new_candidates = self.prune_candidates(new_candidates)?;
        }
        
        // Select best candidates for next beam
        let beam_size = self.config.beam_size.min(new_candidates.len());
        new_candidates.select_nth_unstable_by(beam_size, |a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        
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
    pub fn get_best_hypothesis<'a>(&self, state: &'a BeamSearchState) -> Option<&'a BeamHypothesis> {
        state.hypotheses.iter()
            .chain(state.converged_hypotheses.iter())
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
    }
    
    /// Get top N hypotheses
    pub fn get_top_hypotheses<'a>(&self, state: &'a BeamSearchState, n: usize) -> Vec<&'a BeamHypothesis> {
        let mut all_hypotheses: Vec<&BeamHypothesis> = state.hypotheses.iter()
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
            let key = candidate.tokens.first().map(|t| t.token_id).unwrap_or(0);
            let diversity_score = self.calculate_diversity_score(&candidate.tokens, &diversity_tracker);
            
            // Apply divergence penalty
            let adjusted_score = candidate.score - (self.config.divergence_penalty * diversity_score);
            
            // Keep if score is good enough or not too similar to existing candidates
            if adjusted_score > -10.0 || diversity_score < 0.8 {
                pruned.push(candidate);
                diversity_tracker.insert(key, diversity_score);
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
        state.converged_hypotheses.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        });

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
        
        for i in 0..hyp_len {
            if used[i] {
                continue;
            }
            used[i] = true;
            
            let mut group = vec![i];
            for j in (i + 1)..hyp_len {
                if !used[j] {
                    let similarity = Self::token_id_similarity(
                        &state.hypotheses[i].tokens,
                        &state.hypotheses[j].tokens,
                    );
                    if similarity > (1.0 - self.config.convergence_threshold) {
                        group.push(j);
                        used[j] = true;
                    }
                }
            }
            
            group.sort_by(|&a, &b| state.hypotheses[b].score.partial_cmp(&state.hypotheses[a].score).unwrap_or(Ordering::Equal));
            
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
        
        let first_tokens = &hypotheses[0].tokens;
        for hypothesis in hypotheses.iter().skip(1) {
            let similarity = Self::token_id_similarity(first_tokens, &hypothesis.tokens);
            if similarity < (1.0 - self.config.convergence_threshold) {
                return false;
            }
        }
        
        true
    }
    
    /// Jaccard similarity on token IDs (avoids String allocation)
    fn token_id_similarity(a: &Arc<Vec<GeneratedToken>>, b: &Arc<Vec<GeneratedToken>>) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let a_ids: HashSet<u32> = a.iter().map(|t| t.token_id).collect();
        let b_ids: HashSet<u32> = b.iter().map(|t| t.token_id).collect();
        let intersection = a_ids.intersection(&b_ids).count();
        let union = a_ids.union(&b_ids).count();
        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }
    
    /// Calculate diversity score using token IDs (avoids String allocation)
    fn calculate_diversity_score(&self, tokens: &Arc<Vec<GeneratedToken>>, _existing: &HashMap<u32, f32>) -> f32 {
        if _existing.is_empty() {
            return 0.0;
        }
        // Simplified diversity: ratio of unique token IDs
        if tokens.is_empty() {
            return 0.0;
        }
        let unique: HashSet<u32> = tokens.iter().map(|t| t.token_id).collect();
        1.0 - (unique.len() as f32 / tokens.len() as f32)
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

/// Helper function to run complete beam search
pub fn run_beam_search(
    engine: &BeamSearchEngine,
    initial_logits: &[f32],
    subsequent_logits: &[Vec<f32>],
    max_steps: usize,
) -> Result<Option<BeamHypothesis>> {
    let mut state = engine.initialize(initial_logits)?;
    
    for step_logits in subsequent_logits.iter().take(max_steps - 1) {
        engine.expand_beam(&mut state, &[step_logits.clone()])?;
        
        if engine.should_stop(&state, max_steps) {
            break;
        }
    }
    
    Ok(engine.get_best_hypothesis(&state).cloned())
}
