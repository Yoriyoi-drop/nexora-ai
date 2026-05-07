//! Stop Conditions
//! 
//! Stop conditions untuk inference generation.

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

use crate::{Result, InferenceError, GeneratedToken};

/// Stop condition untuk inference
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Maximum tokens reached
    MaxTokens(u32),
    /// Stop sequence encountered
    StopSequence(String),
    /// End of sequence token
    EndOfSequence(u32),
    /// Time limit reached (seconds)
    TimeLimit(u64),
    /// Repetition detected
    Repetition {
        /// N-gram size untuk check
        ngram_size: usize,
        /// Maximum repetitions allowed
        max_repetitions: u32,
    },
    /// Probability threshold
    ProbabilityThreshold(f32),
    /// Custom condition
    Custom(Arc<dyn CustomStopCondition>),
}

/// Custom stop condition trait
pub trait CustomStopCondition: Send + Sync + std::fmt::Debug {
    /// Check if generation should stop
    fn should_stop(&self, tokens: &[GeneratedToken], context: &StopContext) -> bool;
    
    /// Condition name
    fn name(&self) -> &str;
}

/// Context untuk stop condition checking
#[derive(Debug, Clone)]
pub struct StopContext {
    /// Current token count
    pub token_count: usize,
    /// Generation start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Current time
    pub current_time: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Stop conditions manager
pub struct StopConditions {
    /// Active stop conditions
    conditions: Arc<RwLock<Vec<StopCondition>>>,
    /// Stop sequence cache
    stop_sequences: Arc<RwLock<HashSet<String>>>,
    /// End of sequence tokens
    eos_tokens: Arc<RwLock<HashSet<u32>>>,
    /// Statistics
    stats: Arc<RwLock<StopStats>>,
}

/// Stop statistics
#[derive(Debug, Clone, Default)]
pub struct StopStats {
    /// Total checks performed
    pub total_checks: u64,
    /// Conditions triggered
    pub conditions_triggered: std::collections::HashMap<String, u64>,
    /// Average check time (microseconds)
    pub avg_check_time_us: f64,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl StopConditions {
    /// Create new stop conditions manager
    pub fn new() -> Self {
        Self {
            conditions: Arc::new(RwLock::new(Vec::new())),
            stop_sequences: Arc::new(RwLock::new(HashSet::new())),
            eos_tokens: Arc::new(RwLock::new(HashSet::new())),
            stats: Arc::new(RwLock::new(StopStats::default())),
        }
    }
    
    /// Add stop condition
    pub async fn add_condition(&self, condition: StopCondition) {
        let mut conditions = self.conditions.write().await;
        
        // Update caches based on condition type
        match &condition {
            StopCondition::StopSequence(seq) => {
                let mut stop_sequences = self.stop_sequences.write().await;
                stop_sequences.insert(seq.clone());
            }
            StopCondition::EndOfSequence(token) => {
                let mut eos_tokens = self.eos_tokens.write().await;
                eos_tokens.insert(*token);
            }
            _ => {}
        }
        
        conditions.push(condition);
    }
    
    /// Remove stop condition
    pub async fn remove_condition(&self, index: usize) -> Result<()> {
        let mut conditions = self.conditions.write().await;
        
        if index < conditions.len() {
            let removed = conditions.remove(index);
            
            // Update caches
            match removed {
                StopCondition::StopSequence(seq) => {
                    let mut stop_sequences = self.stop_sequences.write().await;
                    stop_sequences.remove(&seq);
                }
                StopCondition::EndOfSequence(token) => {
                    let mut eos_tokens = self.eos_tokens.write().await;
                    eos_tokens.remove(&token);
                }
                _ => {}
            }
            
            Ok(())
        } else {
            Err(InferenceError::InternalError("Invalid condition index".to_string()))
        }
    }
    
    /// Clear all conditions
    pub async fn clear(&self) {
        let mut conditions = self.conditions.write().await;
        conditions.clear();
        
        let mut stop_sequences = self.stop_sequences.write().await;
        stop_sequences.clear();
        
        let mut eos_tokens = self.eos_tokens.write().await;
        eos_tokens.clear();
    }
    
    /// Check if generation should stop
    pub async fn should_stop(&self, tokens: &[GeneratedToken], context: &StopContext) -> Option<String> {
        let start_time = std::time::Instant::now();
        
        let conditions = self.conditions.read().await;
        let stop_sequences = self.stop_sequences.read().await;
        let eos_tokens = self.eos_tokens.read().await;
        
        for condition in conditions.iter() {
            if let Some(reason) = self.check_condition(condition, tokens, context, &stop_sequences, &eos_tokens) {
                // Update statistics
                {
                    let mut stats = self.stats.write().await;
                    stats.total_checks += 1;
                    let condition_name = self.get_condition_name(condition);
                    *stats.conditions_triggered.entry(condition_name).or_insert(0) += 1;
                    
                    let check_time_us = start_time.elapsed().as_micros() as f64;
                    stats.avg_check_time_us = 
                        (stats.avg_check_time_us * (stats.total_checks - 1) as f64 + check_time_us) / 
                        stats.total_checks as f64;
                    stats.last_updated = chrono::Utc::now();
                }
                
                debug!("Stop condition triggered: {}", reason);
                return Some(reason);
            }
        }
        
        // Update statistics (no condition triggered)
        {
            let mut stats = self.stats.write().await;
            stats.total_checks += 1;
            stats.last_updated = chrono::Utc::now();
        }
        
        None
    }
    
    /// Check individual condition
    fn check_condition(
        &self,
        condition: &StopCondition,
        tokens: &[GeneratedToken],
        context: &StopContext,
        _stop_sequences: &HashSet<String>,
        _eos_tokens: &HashSet<u32>,
    ) -> Option<String> {
        match condition {
            StopCondition::MaxTokens(max_tokens) => {
                if context.token_count >= *max_tokens as usize {
                    Some(format!("Maximum tokens ({}) reached", max_tokens))
                } else {
                    None
                }
            }
            
            StopCondition::StopSequence(seq) => {
                if self.check_stop_sequence(tokens, seq) {
                    Some(format!("Stop sequence '{}' encountered", seq))
                } else {
                    None
                }
            }
            
            StopCondition::EndOfSequence(eos_token) => {
                if tokens.last().map(|t| t.token_id == *eos_token).unwrap_or(false) {
                    Some("End of sequence token encountered".to_string())
                } else {
                    None
                }
            }
            
            StopCondition::TimeLimit(max_seconds) => {
                let elapsed = (context.current_time - context.start_time).num_seconds() as u64;
                if elapsed >= *max_seconds {
                    Some(format!("Time limit ({}) seconds reached", max_seconds))
                } else {
                    None
                }
            }
            
            StopCondition::Repetition { ngram_size, max_repetitions } => {
                if self.check_repetition(tokens, *ngram_size, *max_repetitions) {
                    Some(format!("Repetition detected ({}-gram repeated {} times)", ngram_size, max_repetitions))
                } else {
                    None
                }
            }
            
            StopCondition::ProbabilityThreshold(threshold) => {
                if let Some(last_token) = tokens.last() {
                    if last_token.log_prob < *threshold {
                        Some(format!("Probability threshold ({}) exceeded", threshold))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            
            StopCondition::Custom(custom) => {
                if custom.should_stop(tokens, context) {
                    Some(format!("Custom condition '{}' triggered", custom.name()))
                } else {
                    None
                }
            }
        }
    }
    
    /// Check if stop sequence is encountered
    fn check_stop_sequence(&self, tokens: &[GeneratedToken], sequence: &str) -> bool {
        if tokens.is_empty() || sequence.is_empty() {
            return false;
        }
        
        let generated_text: String = tokens.iter().map(|t| t.token_text.clone()).collect();
        generated_text.contains(sequence)
    }
    
    /// Check for repetition
    fn check_repetition(&self, tokens: &[GeneratedToken], ngram_size: usize, max_repetitions: u32) -> bool {
        if tokens.len() < ngram_size * max_repetitions as usize {
            return false;
        }
        
        // Extract n-grams from tokens
        let ngrams: Vec<String> = tokens.windows(ngram_size)
            .map(|window| window.iter().map(|t| t.token_text.clone()).collect::<String>())
            .collect();
        
        // Check for consecutive repetitions
        for i in 0..=(ngrams.len() - max_repetitions as usize) {
            let first_ngram = &ngrams[i];
            let mut consecutive_count = 1;
            
            for j in (i + 1)..ngrams.len() {
                if ngrams[j] == *first_ngram {
                    consecutive_count += 1;
                } else {
                    break;
                }
            }
            
            if consecutive_count >= max_repetitions {
                return true;
            }
        }
        
        false
    }
    
    /// Get condition name for statistics
    fn get_condition_name(&self, condition: &StopCondition) -> String {
        match condition {
            StopCondition::MaxTokens(_) => "max_tokens".to_string(),
            StopCondition::StopSequence(_) => "stop_sequence".to_string(),
            StopCondition::EndOfSequence(_) => "end_of_sequence".to_string(),
            StopCondition::TimeLimit(_) => "time_limit".to_string(),
            StopCondition::Repetition { .. } => "repetition".to_string(),
            StopCondition::ProbabilityThreshold(_) => "probability_threshold".to_string(),
            StopCondition::Custom(custom) => custom.name().to_string(),
        }
    }
    
    /// Get statistics
    pub async fn get_stats(&self) -> StopStats {
        self.stats.read().await.clone()
    }
    
    /// Get active conditions count
    pub async fn get_conditions_count(&self) -> usize {
        self.conditions.read().await.len()
    }
}

/// Predefined custom stop conditions
pub mod custom_conditions {
    use super::*;
    
    /// Length-based stop condition
    #[derive(Debug)]
    pub struct LengthStopCondition {
        min_length: usize,
        max_length: usize,
    }
    
    impl LengthStopCondition {
        pub fn new(min_length: usize, max_length: usize) -> Self {
            Self { min_length, max_length }
        }
    }
    
    impl CustomStopCondition for LengthStopCondition {
        fn should_stop(&self, tokens: &[GeneratedToken], _context: &StopContext) -> bool {
            let total_length: usize = tokens.iter().map(|t| t.token_text.len()).sum();
            total_length >= self.max_length
        }
        
        fn name(&self) -> &str {
            "length_stop"
        }
    }
    
    /// Coherence-based stop condition
    #[derive(Debug)]
    pub struct CoherenceStopCondition {
        coherence_threshold: f32,
    }
    
    impl CoherenceStopCondition {
        pub fn new(threshold: f32) -> Self {
            Self { coherence_threshold: threshold }
        }
    }
    
    impl CustomStopCondition for CoherenceStopCondition {
        fn should_stop(&self, tokens: &[GeneratedToken], _context: &StopContext) -> bool {
            if tokens.len() < 2 {
                return false;
            }
            
            // Simple coherence check based on token probability variance
            let probs: Vec<f32> = tokens.iter().map(|t| t.log_prob.exp()).collect();
            let mean_prob = probs.iter().sum::<f32>() / probs.len() as f32;
            let variance: f32 = probs.iter()
                .map(|p| (p - mean_prob).powi(2))
                .sum::<f32>() / probs.len() as f32;
            
            // Low variance might indicate incoherence
            variance < self.coherence_threshold
        }
        
        fn name(&self) -> &str {
            "coherence_stop"
        }
    }
    
    /// Diversity-based stop condition
    #[derive(Debug)]
    pub struct DiversityStopCondition {
        diversity_threshold: f32,
        window_size: usize,
    }
    
    impl DiversityStopCondition {
        pub fn new(threshold: f32, window_size: usize) -> Self {
            Self { 
                diversity_threshold: threshold,
                window_size,
            }
        }
    }
    
    impl CustomStopCondition for DiversityStopCondition {
        fn should_stop(&self, tokens: &[GeneratedToken], _context: &StopContext) -> bool {
            if tokens.len() < self.window_size {
                return false;
            }
            
            // Calculate token diversity in the last window
            let window = &tokens[tokens.len() - self.window_size..];
            let unique_tokens: std::collections::HashSet<_> = window.iter()
                .map(|t| &t.token_text)
                .collect();
            
            let diversity = unique_tokens.len() as f32 / window.len() as f32;
            diversity < self.diversity_threshold
        }
        
        fn name(&self) -> &str {
            "diversity_stop"
        }
    }
}

/// Utility functions for creating common stop conditions
pub mod builders {
    use super::*;
    
    /// Create stop conditions for typical use cases
    pub async fn create_standard_conditions(
        max_tokens: u32,
        stop_sequences: Vec<String>,
        eos_token: Option<u32>,
    ) -> StopConditions {
        let conditions = StopConditions::new();
        
        // Add max tokens condition
        conditions.add_condition(StopCondition::MaxTokens(max_tokens)).await;
        
        // Add stop sequences
        for seq in stop_sequences {
            conditions.add_condition(StopCondition::StopSequence(seq)).await;
        }
        
        // Add EOS token if provided
        if let Some(eos) = eos_token {
            conditions.add_condition(StopCondition::EndOfSequence(eos)).await;
        }
        
        conditions
    }
    
    /// Create conditions for creative writing
    pub async fn create_creative_conditions(max_tokens: u32) -> StopConditions {
        let conditions = StopConditions::new();
        
        conditions.add_condition(StopCondition::MaxTokens(max_tokens)).await;
        
        // Add repetition detection
        conditions.add_condition(StopCondition::Repetition {
            ngram_size: 3,
            max_repetitions: 2,
        }).await;
        
        // Add probability threshold
        conditions.add_condition(StopCondition::ProbabilityThreshold(-5.0)).await;
        
        conditions
    }
    
    /// Create conditions for code generation
    pub async fn create_code_conditions(max_tokens: u32) -> StopConditions {
        let conditions = StopConditions::new();
        
        conditions.add_condition(StopCondition::MaxTokens(max_tokens)).await;
        
        // Add common code stop sequences
        let code_stops = vec![
            "```".to_string(),
            "def ".to_string(),
            "class ".to_string(),
            "function ".to_string(),
        ];
        
        for stop in code_stops {
            conditions.add_condition(StopCondition::StopSequence(stop)).await;
        }
        
        conditions
    }
    
    /// Create conditions with time limit
    pub async fn create_time_limited_conditions(max_tokens: u32, time_limit_seconds: u64) -> StopConditions {
        let conditions = StopConditions::new();
        
        conditions.add_condition(StopCondition::MaxTokens(max_tokens)).await;
        conditions.add_condition(StopCondition::TimeLimit(time_limit_seconds)).await;
        
        conditions
    }
}

impl Default for StopConditions {
    fn default() -> Self {
        Self::new()
    }
}
