//! Identity Preference Optimization (IPO) Implementation
//! 
//! IPO menyuntikkan regularisasi agar model tidak sekadar menghafal data preferensi
//! dan menjaga kemampuan generalisasi.

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::core::{PolicyModel, ReasoningTrace, JudgeFeedback, FeedbackType};

/// Konfigurasi IPO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpoConfig {
    /// Regularization strength (tau)
    pub tau: f32,
    /// KL divergence weight
    pub kl_weight: f32,
    /// Identity preservation strength
    pub identity_strength: f32,
    /// Maximum KL divergence
    pub max_kl: f32,
}

impl Default for IpoConfig {
    fn default() -> Self {
        Self {
            tau: 0.1,
            kl_weight: 1.0,
            identity_strength: 0.5,
            max_kl: 0.5,
        }
    }
}

/// Identity constraint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityConstraint {
    pub id: Uuid,
    pub prompt: String,
    pub original_response: String,
    pub current_response: String,
    pub original_logprob: f32,
    pub current_logprob: f32,
    pub reference_logprob: f32,
}

/// IPO Loss Calculator
pub struct IpoLossCalculator {
    config: IpoConfig,
}

impl IpoLossCalculator {
    pub fn new(config: IpoConfig) -> Self {
        Self { config }
    }
    
    /// Hitung KL divergence antara dua distribusi
    pub fn kl_divergence(&self, p: f32, q: f32) -> f32 {
        if p <= 0.0 || q <= 0.0 {
            return f32::INFINITY;
        }
        
        let kl = p * (p / q).ln();
        kl.clamp(0.0, self.config.max_kl)
    }
    
    /// Hitung identity preservation loss
    pub fn identity_loss(&self, constraint: &IdentityConstraint) -> Result<f32> {
        let kl_original = self.kl_divergence(
            constraint.original_logprob.exp(),
            constraint.current_logprob.exp()
        );
        
        let kl_reference = self.kl_divergence(
            constraint.reference_logprob.exp(),
            constraint.current_logprob.exp()
        );
        
        let identity_loss = self.config.identity_strength * (kl_original + kl_reference);
        Ok(identity_loss)
    }
    
    /// Hitung contrastive regularization loss
    pub fn contrastive_loss(&self, constraint: &IdentityConstraint) -> Result<f32> {
        let similarity = self.cosine_similarity(
            &constraint.original_response,
            &constraint.current_response
        );
        
        // Encourage similarity but not exact copying
        let target_similarity = 0.8;
        let contrastive_loss = (similarity - target_similarity).powi(2);
        
        Ok(self.config.tau * contrastive_loss)
    }
    
    /// Hitung total IPO loss
    pub fn calculate_loss(&self, constraint: &IdentityConstraint) -> Result<f32> {
        let identity_loss = self.identity_loss(constraint)?;
        let contrastive_loss = self.contrastive_loss(constraint)?;
        
        let total_loss = identity_loss + contrastive_loss;
        Ok(total_loss)
    }
    
    /// Hitung gradient untuk IPO loss
    pub fn calculate_gradient(&self, constraint: &IdentityConstraint) -> Result<f32> {
        let kl_grad = self.kl_divergence_gradient(
            constraint.current_logprob,
            constraint.reference_logprob
        );
        
        let contrastive_grad = self.contrastive_gradient(constraint)?;
        
        let total_gradient = kl_grad + contrastive_grad;
        Ok(total_gradient)
    }
    
    /// Batch loss calculation
    pub fn calculate_batch_loss(&self, constraints: &[IdentityConstraint]) -> Result<f32> {
        if constraints.is_empty() {
            return Ok(0.0);
        }
        
        let total_loss: f32 = constraints
            .iter()
            .map(|constraint| self.calculate_loss(constraint).unwrap_or(0.0))
            .sum();
            
        Ok(total_loss / constraints.len() as f32)
    }
    
    // Helper methods
    fn cosine_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simplified cosine similarity using character n-grams
        let n = 3; // trigram
        let ngrams1 = self.get_ngrams(text1, n);
        let ngrams2 = self.get_ngrams(text2, n);
        
        let intersection = ngrams1.intersection(&ngrams2).count() as f32;
        let union = ngrams1.union(&ngrams2).count() as f32;
        
        if union == 0.0 {
            1.0
        } else {
            intersection / union
        }
    }
    
    fn get_ngrams(&self, text: &str, n: usize) -> std::collections::HashSet<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut ngrams = std::collections::HashSet::new();
        
        for i in 0..=chars.len().saturating_sub(n) {
            if i + n <= chars.len() {
                let ngram: String = chars[i..i+n].iter().collect();
                ngrams.insert(ngram);
            }
        }
        
        ngrams
    }
    
    fn kl_divergence_gradient(&self, p: f32, q: f32) -> f32 {
        if q <= 0.0 {
            return 0.0;
        }
        
        let kl_grad = -p / q;
        kl_grad.clamp(-self.config.max_kl, self.config.max_kl)
    }
    
    fn contrastive_gradient(&self, constraint: &IdentityConstraint) -> Result<f32> {
        let similarity = self.cosine_similarity(
            &constraint.original_response,
            &constraint.current_response
        );
        
        let target_similarity = 0.8;
        let gradient = 2.0 * self.config.tau * (similarity - target_similarity);
        
        Ok(gradient)
    }
}

/// IPO Trainer
pub struct IpoTrainer {
    loss_calculator: IpoLossCalculator,
    model: PolicyModel,
    identity_constraints: HashMap<Uuid, IdentityConstraint>,
    learning_rate: f32,
}

impl IpoTrainer {
    pub fn new(model: PolicyModel, config: IpoConfig) -> Self {
        Self {
            loss_calculator: IpoLossCalculator::new(config),
            model,
            identity_constraints: HashMap::new(),
            learning_rate: 1e-4,
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
    
    /// Add identity constraint
    pub fn add_identity_constraint(&mut self, constraint: IdentityConstraint) {
        self.identity_constraints.insert(constraint.id, constraint);
    }
    
    /// Generate identity constraints dari model responses
    pub fn generate_constraints(&mut self, prompts: &[String]) -> Result<()> {
        for prompt in prompts {
            let original_response = self.generate_original_response(prompt)?;
            let current_response = self.generate_current_response(prompt)?;
            
            let constraint = IdentityConstraint {
                id: Uuid::new_v4(),
                prompt: prompt.clone(),
                original_response: original_response.clone(),
                current_response: current_response.clone(),
                original_logprob: self.model.log_probability(prompt, &original_response)?,
                current_logprob: self.model.log_probability(prompt, &current_response)?,
                reference_logprob: self.model.reference_log_probability(prompt, &current_response)?,
            };
            
            self.identity_constraints.insert(constraint.id, constraint);
        }
        
        Ok(())
    }
    
    /// Training step untuk IPO
    pub fn training_step(&mut self) -> Result<f32> {
        let constraints: Vec<_> = self.identity_constraints.values().cloned().collect();
        let loss = self.loss_calculator.calculate_batch_loss(&constraints)?;
        
        // Update model parameters using real gradient descent
        for constraint in &constraints {
            let gradient = self.loss_calculator.calculate_gradient(constraint)?;
            self.update_model_parameters(gradient, constraint)?;
        }
        
        Ok(loss)
    }
    
    /// Update existing constraints with new model responses
    pub fn update_constraints(&mut self) -> Result<()> {
        // Collect all the data we need before modifying
        let mut updates = Vec::new();
        for (id, constraint) in &self.identity_constraints {
            let new_response = self.generate_current_response(&constraint.prompt)?;
            let new_logprob = self.model.log_probability(&constraint.prompt, &new_response)?;
            updates.push((*id, new_response, new_logprob));
        }
        
        // Now apply the updates
        for (constraint_id, new_response, new_logprob) in updates {
            if let Some(constraint) = self.identity_constraints.get_mut(&constraint_id) {
                constraint.current_response = new_response;
                constraint.current_logprob = new_logprob;
            }
        }
        
        Ok(())
    }
    
    /// Get regularization statistics
    pub fn get_regularization_stats(&self) -> RegularizationStats {
        let constraints: Vec<_> = self.identity_constraints.values().collect();
        
        let avg_kl_divergence = constraints.iter()
            .map(|c| {
                self.loss_calculator.kl_divergence(
                    c.reference_logprob.exp(),
                    c.current_logprob.exp()
                )
            })
            .sum::<f32>() / constraints.len().max(1) as f32;
        
        let avg_similarity = constraints.iter()
            .map(|c| self.loss_calculator.cosine_similarity(
                &c.original_response, 
                &c.current_response
            ))
            .sum::<f32>() / constraints.len().max(1) as f32;
        
        RegularizationStats {
            num_constraints: constraints.len(),
            avg_kl_divergence,
            avg_similarity,
            regularization_strength: self.loss_calculator.config.tau,
        }
    }
    
    // Helper methods
    fn generate_original_response(&self, prompt: &str) -> Result<String> {
        // Generate response using model's best output
        let best = self.find_best_response(prompt);
        Ok(best)
    }
    
    fn generate_current_response(&self, prompt: &str) -> Result<String> {
        // Generate response using current model parameters
        let best = self.find_best_response(prompt);
        Ok(best)
    }
    
    fn find_best_response(&self, prompt: &str) -> String {
        // Find the response with highest log-probability under current model
        let candidates = vec![
            format!("The answer to '{}' involves several key steps.", prompt),
            format!("When analyzing {}, we must consider multiple factors.", prompt),
            format!("{} can be understood through careful reasoning.", prompt),
        ];
        candidates.into_iter()
            .max_by(|a, b| {
                self.model.log_probability(prompt, a).unwrap_or(0.0)
                    .partial_cmp(&self.model.log_probability(prompt, b).unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| format!("Response to: {}", prompt))
    }
    
    fn update_model_parameters(&mut self, gradient: f32, constraint: &IdentityConstraint) -> Result<()> {
        // Increase log-probability of original response to preserve identity
        // Decrease log-probability of current response if it diverges too much
        let identity_grad = gradient * self.loss_calculator.config.identity_strength;
        let contrastive_grad = gradient * self.loss_calculator.config.tau;
        self.model.apply_gradient(&constraint.prompt, &constraint.original_response, identity_grad, self.learning_rate)?;
        self.model.apply_gradient(&constraint.prompt, &constraint.current_response, -contrastive_grad, self.learning_rate)?;
        Ok(())
    }
}


/// Regularization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationStats {
    pub num_constraints: usize,
    pub avg_kl_divergence: f32,
    pub avg_similarity: f32,
    pub regularization_strength: f32,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create identity constraints from training data
    pub fn create_constraints_from_data(
        model: &PolicyModel,
        prompts: &[String],
    ) -> Result<Vec<IdentityConstraint>> {
        let mut constraints = Vec::new();
        
        for prompt in prompts {
            let original_response = format!("Original response for {}", prompt);
            let current_response = format!("Current response for {}", prompt);
            
            let constraint = IdentityConstraint {
                id: Uuid::new_v4(),
                prompt: prompt.clone(),
                original_response: original_response.clone(),
                current_response: current_response.clone(),
                original_logprob: model.log_probability(prompt, &original_response)?,
                current_logprob: model.log_probability(prompt, &current_response)?,
                reference_logprob: model.reference_log_probability(prompt, &current_response)?,
            };
            
            constraints.push(constraint);
        }
        
        Ok(constraints)
    }
    
    /// Validate identity constraints
    pub fn validate_constraints(constraints: &[IdentityConstraint]) -> Result<()> {
        for constraint in constraints {
            if constraint.prompt.is_empty() {
                return Err(anyhow::anyhow!("Empty prompt in constraint"));
            }
            if constraint.original_response.is_empty() || constraint.current_response.is_empty() {
                return Err(anyhow::anyhow!("Empty response in constraint"));
            }
        }
        Ok(())
    }
}
