//! Utility functions for SPARO (Structured Preference Alignment and Reward Optimization)

use std::collections::HashMap;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Utility functions for alignment algorithms
pub struct AlignmentUtils;

impl AlignmentUtils {
    /// Calculate KL divergence between two distributions
    pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
        if p.len() != q.len() {
            return f32::INFINITY;
        }

        p.iter().zip(q.iter())
            .map(|(pi, qi)| {
                if *pi > 0.0 && *qi > 0.0 {
                    pi * (pi / qi).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Calculate entropy of a distribution
    pub fn entropy(dist: &[f32]) -> f32 {
        dist.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Normalize probability distribution
    pub fn normalize_distribution(dist: &mut [f32]) {
        let sum: f32 = dist.iter().sum();
        if sum > 0.0 {
            for value in dist.iter_mut() {
                *value /= sum;
            }
        }
    }

    /// Calculate cosine similarity between vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Utility functions for reward modeling
pub struct RewardUtils;

impl RewardUtils {
    /// Calculate reward statistics
    pub fn reward_stats(rewards: &[f32]) -> HashMap<String, f32> {
        if rewards.is_empty() {
            return HashMap::new();
        }

        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rewards.len() as f32;
        let std_dev = variance.sqrt();
        
        let mut stats = HashMap::new();
        stats.insert("mean".to_string(), mean);
        stats.insert("std_dev".to_string(), std_dev);
        stats.insert("min".to_string(), rewards.iter().cloned().fold(f32::INFINITY, f32::min));
        stats.insert("max".to_string(), rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        
        stats
    }

    /// Apply reward shaping
    pub fn shape_reward(base_reward: f32, shaping_factor: f32, potential: f32) -> f32 {
        base_reward + shaping_factor * potential
    }

    /// Calculate advantage values
    pub fn calculate_advantages(rewards: &[f32], values: &[f32], gamma: f32) -> Vec<f32> {
        let mut advantages = Vec::with_capacity(rewards.len());
        
        for i in 0..rewards.len() {
            let mut advantage = 0.0;
            for j in i..rewards.len() {
                let discount = gamma.powi((j - i) as i32);
                advantage += discount * (rewards[j] - values[j]);
            }
            advantages.push(advantage);
        }
        
        advantages
    }
}

/// Configuration utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub gamma: f32,
    pub epsilon: f32,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 32,
            max_epochs: 100,
            gamma: 0.99,
            epsilon: 1e-8,
        }
    }
}

impl AlignmentConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("Batch size must be non-zero".to_string());
        }
        if self.max_epochs == 0 {
            return Err("Max epochs must be non-zero".to_string());
        }
        if ! (0.0..=1.0).contains(&self.gamma) {
            return Err("Gamma must be between 0 and 1".to_string());
        }
        Ok(())
    }
}
