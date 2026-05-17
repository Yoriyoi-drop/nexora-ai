//! Decoding Strategies
//! 
//! Token selection strategies untuk inference.

use std::collections::HashMap;
use tracing::{debug, warn};

use crate::{Result, InferenceError, GeneratedToken};

/// Configuration untuk decoding
#[derive(Debug, Clone)]
pub struct DecodingConfig {
    /// Temperature untuk sampling
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: u32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Minimum probability threshold
    pub min_prob: f32,
    /// Enable logit filtering
    pub enable_logit_filter: bool,
    /// Logit bias adjustments
    pub logit_bias: HashMap<u32, f32>,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            min_prob: 1e-5,
            enable_logit_filter: false,
            logit_bias: HashMap::new(),
        }
    }
}

/// Decoding strategy trait
pub trait DecodingStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;
    
    /// Select next token based on logits
    fn select_token(&self, logits: &[f32], config: &DecodingConfig, context: &DecodingContext) -> Result<TokenSelection>;
    
    /// Validate configuration
    fn validate_config(&self, config: &DecodingConfig) -> Result<()>;
}

/// Context untuk decoding
#[derive(Debug, Clone)]
pub struct DecodingContext {
    /// Generated tokens so far
    pub generated_tokens: Vec<GeneratedToken>,
    /// Token frequencies in current generation
    pub token_frequencies: HashMap<u32, usize>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Special tokens to avoid
    pub forbidden_tokens: Vec<u32>,
    /// Required tokens (if any)
    pub required_tokens: Vec<u32>,
    /// Generation step
    pub step: usize,
    /// Context metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Token selection result
#[derive(Debug, Clone)]
pub struct TokenSelection {
    /// Selected token ID
    pub token_id: u32,
    /// Token text
    pub token_text: String,
    /// Log probability
    pub log_prob: f32,
    /// Selection probability
    pub selection_prob: f32,
    /// Selection metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Greedy decoding strategy
pub struct GreedyDecoding;

impl DecodingStrategy for GreedyDecoding {
    fn name(&self) -> &str {
        "greedy"
    }
    
    fn select_token(&self, logits: &[f32], config: &DecodingConfig, context: &DecodingContext) -> Result<TokenSelection> {
        debug!("Greedy decoding: selecting token from {} logits", logits.len());
        
        // Apply penalties and biases
        let adjusted_logits = self.adjust_logits(logits, config, context)?;
        
        // Find maximum logit
        let (max_index, &max_logit) = adjusted_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| InferenceError::DecodingError("Empty logits".to_string()))?;
        
        // Check if token is forbidden
        if context.forbidden_tokens.contains(&(max_index as u32)) {
            // Find next best token
            let (next_index, &next_logit) = adjusted_logits
                .iter()
                .enumerate()
                .filter(|(i, _)| !context.forbidden_tokens.contains(&(*i as u32)))
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| InferenceError::DecodingError("No valid tokens available".to_string()))?;
            
            let log_prob = self.compute_log_prob(next_logit, config);
            let selection_prob = log_prob.exp();
            
            return Ok(TokenSelection {
                token_id: next_index as u32,
                token_text: alloc_token_text(next_index),
                log_prob,
                selection_prob,
                metadata: HashMap::new(),
            });
        }
        
        let log_prob = self.compute_log_prob(max_logit, config);
        let selection_prob = log_prob.exp();
        
        Ok(TokenSelection {
            token_id: max_index as u32,
            token_text: alloc_token_text(max_index),
            log_prob,
            selection_prob,
            metadata: HashMap::new(),
        })
    }
    
    fn validate_config(&self, config: &DecodingConfig) -> Result<()> {
        if config.temperature != 1.0 {
            warn!("Greedy decoding ignores temperature parameter");
        }
        if config.top_p != 1.0 {
            warn!("Greedy decoding ignores top_p parameter");
        }
        if config.top_k != 50 {
            warn!("Greedy decoding ignores top_k parameter");
        }
        Ok(())
    }
}

impl GreedyDecoding {
    fn adjust_logits(&self, logits: &[f32], config: &DecodingConfig, context: &DecodingContext) -> Result<Vec<f32>> {
        let mut adjusted = Vec::with_capacity(logits.len());
        adjusted.extend_from_slice(logits);
        
        if config.presence_penalty != 0.0 {
            for token_id in context.token_frequencies.keys() {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit -= config.presence_penalty;
                }
            }
        }
        
        // Apply frequency penalty
        if config.frequency_penalty != 0.0 {
            for (token_id, &frequency) in &context.token_frequencies {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit -= config.frequency_penalty * frequency as f32;
                }
            }
        }
        
        // Apply repetition penalty
        if config.repetition_penalty != 1.0 {
            for token_id in context.token_frequencies.keys() {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit /= config.repetition_penalty;
                }
            }
        }
        
        // Apply logit bias
        if config.enable_logit_filter {
            for (token_id, &bias) in &config.logit_bias {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit += bias;
                }
            }
        }
        
        Ok(adjusted)
    }
    
    fn compute_log_prob(&self, logit: f32, _config: &DecodingConfig) -> f32 {
        // For greedy decoding, log_prob is just the logit
        logit
    }
}

/// Temperature sampling strategy
pub struct TemperatureSampling;

impl DecodingStrategy for TemperatureSampling {
    fn name(&self) -> &str {
        "temperature"
    }
    
    fn select_token(&self, logits: &[f32], config: &DecodingConfig, context: &DecodingContext) -> Result<TokenSelection> {
        debug!("Temperature sampling: temp={:.2}, top_p={:.2}, top_k={}", 
               config.temperature, config.top_p, config.top_k);
        
        // Adjust logits
        let adjusted_logits = self.adjust_logits(logits, config, context)?;
        
        // Apply temperature
        let scaled_logits: Vec<f32> = adjusted_logits
            .iter()
            .map(|&logit| logit / config.temperature)
            .collect();
        
        // Compute probabilities
        let probs: Vec<f32> = self.compute_softmax(&scaled_logits)?;
        
        // Apply top-k filtering
        let filtered_probs = if config.top_k > 0 && config.top_k < probs.len() as u32 {
            self.apply_top_k(&probs, config.top_k)?
        } else {
            probs
        };
        
        // Apply top-p filtering
        let final_probs = if config.top_p < 1.0 {
            self.apply_top_p(&filtered_probs, config.top_p)?
        } else {
            filtered_probs
        };
        
        // Sample token
        let token_id = self.sample_token(&final_probs)?;
        
        let log_prob = adjusted_logits[token_id].ln();
        let selection_prob = final_probs[token_id];
        
        Ok(TokenSelection {
            token_id: token_id as u32,
            token_text: alloc_token_text(token_id),
            log_prob,
            selection_prob,
            metadata: HashMap::new(),
        })
    }
    
    fn validate_config(&self, config: &DecodingConfig) -> Result<()> {
        if config.temperature <= 0.0 {
            return Err(InferenceError::DecodingError("Temperature must be positive".to_string()));
        }
        if config.top_p <= 0.0 || config.top_p > 1.0 {
            return Err(InferenceError::DecodingError("top_p must be in (0, 1]".to_string()));
        }
        Ok(())
    }
}

impl TemperatureSampling {
    fn compute_softmax(&self, logits: &[f32]) -> Result<Vec<f32>> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs = Vec::with_capacity(logits.len());
        let mut sum = 0.0;
        for &l in logits {
            let e = (l - max_logit).exp();
            sum += e;
            probs.push(e);
        }
        if sum == 0.0 {
            return Err(InferenceError::DecodingError("Softmax sum is zero".to_string()));
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        Ok(probs)
    }
    
    fn apply_top_k(&self, probs: &[f32], top_k: u32) -> Result<Vec<f32>> {
        let k = std::cmp::min(top_k as usize, probs.len());
        let n = probs.len();
        
        let mut indexed_probs: Vec<(usize, f32)> = Vec::with_capacity(n);
        for (i, &p) in probs.iter().enumerate() {
            indexed_probs.push((i, p));
        }
        indexed_probs.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let threshold = indexed_probs[k.saturating_sub(1)].1;
        let mut filtered = vec![0.0; n];
        for &(idx, prob) in indexed_probs.iter().take(k) {
            filtered[idx] = prob;
        }
        
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            for prob in filtered.iter_mut() {
                *prob /= sum;
            }
        }
        Ok(filtered)
    }
    
    fn apply_top_p(&self, probs: &[f32], top_p: f32) -> Result<Vec<f32>> {
        let n = probs.len();
        let mut indexed_probs: Vec<(usize, f32)> = Vec::with_capacity(n);
        for (i, &p) in probs.iter().enumerate() {
            indexed_probs.push((i, p));
        }
        indexed_probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut cumulative_sum = 0.0;
        let mut cutoff_index = 0;
        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_sum += prob;
            if cumulative_sum >= top_p {
                cutoff_index = i + 1;
                break;
            }
        }
        
        let mut filtered = vec![0.0; n];
        for (idx, prob) in indexed_probs.iter().take(cutoff_index) {
            filtered[*idx] = *prob;
        }
        
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            for prob in filtered.iter_mut() {
                *prob /= sum;
            }
        }
        Ok(filtered)
    }
    
    fn sample_token(&self, probs: &[f32]) -> Result<usize> {
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let random_val: f32 = rng.gen();
        
        let mut cumulative_sum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative_sum += prob;
            if random_val <= cumulative_sum {
                return Ok(i);
            }
        }
        
        // Fallback to last token (shouldn't happen if probabilities sum to 1)
        Ok(probs.len() - 1)
    }
    
    fn adjust_logits(&self, logits: &[f32], config: &DecodingConfig, context: &DecodingContext) -> Result<Vec<f32>> {
        let mut adjusted = logits.to_vec();
        
        // Apply penalties
        if config.presence_penalty != 0.0 {
            for token_id in context.token_frequencies.keys() {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit -= config.presence_penalty;
                }
            }
        }
        
        if config.frequency_penalty != 0.0 {
            for (token_id, &frequency) in &context.token_frequencies {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit -= config.frequency_penalty * frequency as f32;
                }
            }
        }
        
        if config.repetition_penalty != 1.0 {
            for token_id in context.token_frequencies.keys() {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit /= config.repetition_penalty;
                }
            }
        }
        
        // Apply logit bias
        if config.enable_logit_filter {
            for (token_id, &bias) in &config.logit_bias {
                if let Some(logit) = adjusted.get_mut(*token_id as usize) {
                    *logit += bias;
                }
            }
        }
        
        Ok(adjusted)
    }
}

/// Nucleus sampling (top-p only) strategy
pub struct NucleusSampling;

impl DecodingStrategy for NucleusSampling {
    fn name(&self) -> &str {
        "nucleus"
    }
    
    fn select_token(&self, logits: &[f32], config: &DecodingConfig, context: &DecodingContext) -> Result<TokenSelection> {
        debug!("Nucleus sampling: top_p={:.2}", config.top_p);
        
        TemperatureSampling.select_token(logits, config, context)
    }
    
    fn validate_config(&self, config: &DecodingConfig) -> Result<()> {
        if config.top_p <= 0.0 || config.top_p > 1.0 {
            return Err(InferenceError::DecodingError("top_p must be in (0, 1]".to_string()));
        }
        Ok(())
    }
}

/// Top-k sampling strategy
pub struct TopKSampling;

impl DecodingStrategy for TopKSampling {
    fn name(&self) -> &str {
        "top_k"
    }
    
    fn select_token(&self, logits: &[f32], config: &DecodingConfig, context: &DecodingContext) -> Result<TokenSelection> {
        debug!("Top-k sampling: top_k={}", config.top_k);
        
        TemperatureSampling.select_token(logits, config, context)
    }
    
    fn validate_config(&self, config: &DecodingConfig) -> Result<()> {
        if config.top_k == 0 {
            return Err(InferenceError::DecodingError("top_k must be greater than 0".to_string()));
        }
        Ok(())
    }
}

/// Create default decoding strategies
pub fn create_default_strategies() -> HashMap<String, Box<dyn DecodingStrategy>> {
    let mut strategies: HashMap<String, Box<dyn DecodingStrategy>> = HashMap::with_capacity(4);
    
    strategies.insert("greedy".to_string(), Box::new(GreedyDecoding) as Box<dyn DecodingStrategy>);
    strategies.insert("temperature".to_string(), Box::new(TemperatureSampling) as Box<dyn DecodingStrategy>);
    strategies.insert("nucleus".to_string(), Box::new(NucleusSampling) as Box<dyn DecodingStrategy>);
    strategies.insert("top_k".to_string(), Box::new(TopKSampling) as Box<dyn DecodingStrategy>);
    
    strategies
}

impl DecodingContext {
    /// Create new decoding context
    pub fn new(vocab_size: usize) -> Self {
        Self {
            generated_tokens: Vec::new(),
            token_frequencies: HashMap::new(),
            vocab_size,
            forbidden_tokens: Vec::new(),
            required_tokens: Vec::new(),
            step: 0,
            metadata: HashMap::new(),
        }
    }
    
    /// Add generated token
    pub fn add_token(&mut self, token: GeneratedToken) {
        let _token_id = token.token_id as usize;
        *self.token_frequencies.entry(token.token_id).or_insert(0) += 1;
        self.generated_tokens.push(token);
        self.step += 1;
    }
    
    /// Get token frequency
    pub fn get_token_frequency(&self, token_id: u32) -> usize {
        self.token_frequencies.get(&token_id).copied().unwrap_or(0)
    }
    
    /// Add forbidden token
    pub fn add_forbidden_token(&mut self, token_id: u32) {
        self.forbidden_tokens.push(token_id);
    }
    
    /// Add required token
    pub fn add_required_token(&mut self, token_id: u32) {
        self.required_tokens.push(token_id);
    }
    
    /// Check if token is forbidden
    pub fn is_forbidden(&self, token_id: u32) -> bool {
        self.forbidden_tokens.contains(&token_id)
    }
    
    /// Check if token is required
    pub fn is_required(&self, token_id: u32) -> bool {
        self.required_tokens.contains(&token_id)
    }
}

pub(crate) fn alloc_token_text(token_id: usize) -> String {
    let mut buf = String::with_capacity(12);
    buf.push_str("[t");
    let mut n = token_id;
    if n == 0 {
        buf.push('0');
    } else {
        let mut digits = [0u8; 10];
        let mut i = digits.len();
        while n > 0 {
            i -= 1;
            digits[i] = (n % 10) as u8 + b'0';
            n /= 10;
        }
        buf.push_str(unsafe { core::str::from_utf8_unchecked(&digits[i..]) });
    }
    buf.push(']');
    buf
}
