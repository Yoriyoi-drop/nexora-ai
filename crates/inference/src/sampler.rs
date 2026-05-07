//! Sampler
//! 
//! Sampling methods untuk token selection.

use std::collections::HashMap;
use tracing::debug;

use crate::{Result, InferenceError};

/// Sampling method
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    /// Uniform sampling
    Uniform,
    /// Weighted sampling
    Weighted,
    /// Multinomial sampling
    Multinomial,
    /// Categorical sampling
    Categorical,
}

/// Configuration untuk sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling method
    pub method: SamplingMethod,
    /// Temperature parameter
    pub temperature: f32,
    /// Minimum probability
    pub min_prob: f32,
    /// Maximum samples to consider
    pub max_samples: usize,
    /// Enable probability filtering
    pub enable_filter: bool,
    /// Filter threshold
    pub filter_threshold: f32,
    /// Random seed (for reproducibility)
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            method: SamplingMethod::Multinomial,
            temperature: 1.0,
            min_prob: 1e-5,
            max_samples: 1000,
            enable_filter: true,
            filter_threshold: 1e-6,
            seed: None,
        }
    }
}

/// Sampler untuk token selection
pub struct Sampler {
    config: SamplingConfig,
    rng_state: Option<u64>,
}

impl Sampler {
    /// Create new sampler
    pub fn new(config: SamplingConfig) -> Self {
        let rng_state = config.seed;
        Self {
            config,
            rng_state,
        }
    }
    
    /// Sample from probabilities
    pub fn sample(&mut self, probs: &[f32]) -> Result<usize> {
        debug!("Sampling from {} probabilities with method {:?}", probs.len(), self.config.method);
        
        if probs.is_empty() {
            return Err(InferenceError::DecodingError("Empty probability distribution".to_string()));
        }
        
        // Validate probabilities
        self.validate_probabilities(probs)?;
        
        // Apply temperature and filtering
        let adjusted_probs = self.adjust_probabilities(probs)?;
        
        // Sample based on method
        let result = match self.config.method {
            SamplingMethod::Uniform => self.sample_uniform(&adjusted_probs),
            SamplingMethod::Weighted => self.sample_weighted(&adjusted_probs),
            SamplingMethod::Multinomial => self.sample_multinomial(&adjusted_probs),
            SamplingMethod::Categorical => self.sample_categorical(&adjusted_probs),
        };
        
        debug!("Sampled token index: {:?}", result);
        result
    }
    
    /// Sample multiple tokens
    pub fn sample_multiple(&mut self, probs: &[f32], count: usize) -> Result<Vec<usize>> {
        debug!("Sampling {} tokens from {} probabilities", count, probs.len());
        
        if count == 0 {
            return Ok(Vec::new());
        }
        
        if count > probs.len() {
            return Err(InferenceError::DecodingError(
                format!("Cannot sample {} tokens from {} probabilities", count, probs.len())
            ));
        }
        
        let mut results = Vec::new();
        let mut remaining_probs = probs.to_vec();
        
        for _ in 0..count {
            let index = self.sample(&remaining_probs)?;
            results.push(index);
            
            // Remove sampled probability to avoid duplicates
            remaining_probs[index] = 0.0;
            
            // Renormalize remaining probabilities
            let sum: f32 = remaining_probs.iter().sum();
            if sum > 0.0 {
                for prob in remaining_probs.iter_mut() {
                    *prob /= sum;
                }
            }
        }
        
        Ok(results)
    }
    
    /// Get sampling statistics
    pub fn get_stats(&self) -> SamplingStats {
        SamplingStats {
            method: self.config.method.clone(),
            temperature: self.config.temperature,
            min_prob: self.config.min_prob,
            max_samples: self.config.max_samples,
            enable_filter: self.config.enable_filter,
            filter_threshold: self.config.filter_threshold,
        }
    }
    
    /// Reset random seed
    pub fn reset_seed(&mut self, seed: u64) {
        self.rng_state = Some(seed);
        self.config.seed = Some(seed);
    }
    
    /// Validate probability distribution
    fn validate_probabilities(&self, probs: &[f32]) -> Result<()> {
        // Check for negative probabilities
        if probs.iter().any(|&p| p < 0.0) {
            return Err(InferenceError::DecodingError("Negative probabilities found".to_string()));
        }
        
        // Check if sum is reasonable
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return Err(InferenceError::DecodingError("Probability sum is zero or negative".to_string()));
        }
        
        // Check for NaN or infinite values
        if probs.iter().any(|&p| !p.is_finite()) {
            return Err(InferenceError::DecodingError("Invalid probability values (NaN or infinite)".to_string()));
        }
        
        Ok(())
    }
    
    /// Adjust probabilities based on configuration
    fn adjust_probabilities(&self, probs: &[f32]) -> Result<Vec<f32>> {
        let mut adjusted = probs.to_vec();
        
        // Apply temperature
        if self.config.temperature != 1.0 {
            for prob in adjusted.iter_mut() {
                *prob = (*prob).powf(1.0 / self.config.temperature);
            }
        }
        
        // Apply minimum probability
        for prob in adjusted.iter_mut() {
            *prob = prob.max(self.config.min_prob);
        }
        
        // Filter small probabilities if enabled
        if self.config.enable_filter {
            for prob in adjusted.iter_mut() {
                if *prob < self.config.filter_threshold {
                    *prob = 0.0;
                }
            }
        }
        
        // Renormalize
        let sum: f32 = adjusted.iter().sum();
        if sum > 0.0 {
            for prob in adjusted.iter_mut() {
                *prob /= sum;
            }
        } else {
            // If all probabilities were filtered out, use uniform distribution
            let uniform_prob = 1.0 / adjusted.len() as f32;
            for prob in adjusted.iter_mut() {
                *prob = uniform_prob;
            }
        }
        
        Ok(adjusted)
    }
    
    /// Uniform sampling
    fn sample_uniform(&mut self, probs: &[f32]) -> Result<usize> {
        use rand::Rng;
        
        let mut rng = self.get_rng();
        let random_val: f32 = rng.gen();
        
        let mut cumulative_sum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative_sum += prob;
            if random_val <= cumulative_sum {
                return Ok(i);
            }
        }
        
        // Fallback to last index
        Ok(probs.len() - 1)
    }
    
    /// Weighted sampling
    fn sample_weighted(&mut self, probs: &[f32]) -> Result<usize> {
        // Weighted sampling is similar to uniform but with explicit weights
        self.sample_uniform(probs)
    }
    
    /// Multinomial sampling
    fn sample_multinomial(&mut self, probs: &[f32]) -> Result<usize> {
        use rand::Rng;
        
        let mut rng = self.get_rng();
        let random_val: f32 = rng.gen();
        
        // Use cumulative distribution
        let mut cumulative_sum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative_sum += prob;
            if random_val <= cumulative_sum {
                return Ok(i);
            }
        }
        
        // Handle floating point precision issues
        Ok(probs.len().saturating_sub(1))
    }
    
    /// Categorical sampling
    fn sample_categorical(&mut self, probs: &[f32]) -> Result<usize> {
        // Categorical sampling is similar to multinomial but with explicit categories
        self.sample_multinomial(probs)
    }
    
    /// Get random number generator
    fn get_rng(&mut self) -> Box<dyn rand::RngCore> {
        use rand::SeedableRng;
        
        if let Some(seed) = self.rng_state {
            // Use seeded RNG for reproducibility
            let rng = rand::rngs::StdRng::seed_from_u64(seed);
            // Update seed for next call
            self.rng_state = Some(seed.wrapping_add(1));
            Box::new(rng)
        } else {
            // Use thread RNG
            Box::new(rand::thread_rng())
        }
    }
}

/// Sampling statistics
#[derive(Debug, Clone)]
pub struct SamplingStats {
    pub method: SamplingMethod,
    pub temperature: f32,
    pub min_prob: f32,
    pub max_samples: usize,
    pub enable_filter: bool,
    pub filter_threshold: f32,
}

/// Advanced sampler with additional features
pub struct AdvancedSampler {
    base_sampler: Sampler,
    history: Vec<usize>,
    max_history: usize,
    repetition_penalty: f32,
}

impl AdvancedSampler {
    /// Create new advanced sampler
    pub fn new(config: SamplingConfig, max_history: usize, repetition_penalty: f32) -> Self {
        Self {
            base_sampler: Sampler::new(config),
            history: Vec::new(),
            max_history,
            repetition_penalty,
        }
    }
    
    /// Sample with repetition penalty
    pub fn sample_with_penalty(&mut self, probs: &[f32]) -> Result<usize> {
        // Apply repetition penalty
        let mut adjusted_probs = probs.to_vec();
        
        for &past_token in &self.history {
            if let Some(prob) = adjusted_probs.get_mut(past_token) {
                *prob *= (1.0 - self.repetition_penalty).max(0.0);
            }
        }
        
        // Renormalize
        let sum: f32 = adjusted_probs.iter().sum();
        if sum > 0.0 {
            for prob in adjusted_probs.iter_mut() {
                *prob /= sum;
            }
        }
        
        // Sample
        let result = self.base_sampler.sample(&adjusted_probs)?;
        
        // Update history
        self.history.push(result);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        
        Ok(result)
    }
    
    /// Get repetition statistics
    pub fn get_repetition_stats(&self) -> RepetitionStats {
        let mut frequency_map = HashMap::new();
        
        for &token in &self.history {
            *frequency_map.entry(token).or_insert(0) += 1;
        }
        
        let max_frequency = frequency_map.values().max().copied().unwrap_or(0);
        let unique_tokens = frequency_map.len();
        
        RepetitionStats {
            total_tokens: self.history.len(),
            unique_tokens,
            max_frequency,
            repetition_rate: if self.history.len() > 0 {
                1.0 - (unique_tokens as f32 / self.history.len() as f32)
            } else {
                0.0
            },
        }
    }
    
    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
    
    /// Set repetition penalty
    pub fn set_repetition_penalty(&mut self, penalty: f32) {
        self.repetition_penalty = penalty.clamp(0.0, 1.0);
    }
}

/// Repetition statistics
#[derive(Debug, Clone)]
pub struct RepetitionStats {
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub max_frequency: usize,
    pub repetition_rate: f32,
}

/// Helper functions for creating common sampling configurations
pub mod configs {
    use super::*;
    
    /// Conservative sampling (low temperature, high filtering)
    pub fn conservative() -> SamplingConfig {
        SamplingConfig {
            method: SamplingMethod::Multinomial,
            temperature: 0.7,
            min_prob: 1e-4,
            enable_filter: true,
            filter_threshold: 1e-5,
            ..Default::default()
        }
    }
    
    /// Creative sampling (high temperature, low filtering)
    pub fn creative() -> SamplingConfig {
        SamplingConfig {
            method: SamplingMethod::Multinomial,
            temperature: 1.5,
            min_prob: 1e-6,
            enable_filter: false,
            ..Default::default()
        }
    }
    
    /// Balanced sampling
    pub fn balanced() -> SamplingConfig {
        SamplingConfig::default()
    }
    
    /// Deterministic sampling (temperature = 0)
    pub fn deterministic() -> SamplingConfig {
        SamplingConfig {
            method: SamplingMethod::Uniform,
            temperature: 0.1,
            min_prob: 1e-3,
            enable_filter: true,
            filter_threshold: 1e-4,
            ..Default::default()
        }
    }
}

/// Utility functions for probability manipulation
pub mod utils {
    /// Apply temperature to logits
    pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        if temperature <= 0.0 {
            return logits.to_vec();
        }
        
        logits.iter().map(|&logit| logit / temperature).collect()
    }
    
    /// Compute softmax
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = logits.iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect();
        
        let sum: f32 = exp_values.iter().sum();
        if sum == 0.0 {
            return vec![1.0 / logits.len() as f32; logits.len()];
        }
        
        exp_values.iter().map(|&val| val / sum).collect()
    }
    
    /// Apply top-k filtering
    pub fn top_k_filter(probs: &[f32], k: usize) -> Vec<f32> {
        if k >= probs.len() {
            return probs.to_vec();
        }
        
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut filtered = vec![0.0; probs.len()];
        for (idx, prob) in indexed_probs.iter().take(k) {
            filtered[*idx] = *prob;
        }
        
        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            for prob in filtered.iter_mut() {
                *prob /= sum;
            }
        }
        
        filtered
    }
    
    /// Apply top-p (nucleus) filtering
    pub fn top_p_filter(probs: &[f32], p: f32) -> Vec<f32> {
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut cumulative_sum = 0.0;
        let mut cutoff_index = 0;
        
        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_sum += prob;
            if cumulative_sum >= p {
                cutoff_index = i + 1;
                break;
            }
        }
        
        let mut filtered = vec![0.0; probs.len()];
        for (idx, prob) in indexed_probs.iter().take(cutoff_index) {
            filtered[*idx] = *prob;
        }
        
        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            for prob in filtered.iter_mut() {
                *prob /= sum;
            }
        }
        
        filtered
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(SamplingConfig::default())
    }
}
