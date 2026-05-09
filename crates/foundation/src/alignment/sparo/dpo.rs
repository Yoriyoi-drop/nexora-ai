//! Direct Preference Optimization (DPO) Implementation
//! 
//! DPO menghilangkan reward model terpisah dan mengubah pelatihan preferensi
//! menjadi klasifikasi biner langsung.

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::core::{PolicyModel, ReasoningTrace, JudgeFeedback, FeedbackType};

/// Konfigurasi DPO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig {
    /// Temperature untuk softmax
    pub beta: f32,
    /// Regularization strength
    pub regularization_strength: f32,
    /// Label smoothing
    pub label_smoothing: f32,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            regularization_strength: 0.01,
            label_smoothing: 0.0,
        }
    }
}

/// Data preferensi berpasangan untuk DPO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    pub id: Uuid,
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
    pub chosen_logprob: f32,
    pub rejected_logprob: f32,
    pub reference_chosen_logprob: f32,
    pub reference_rejected_logprob: f32,
}

/// DPO Loss Calculator
pub struct DpoLossCalculator {
    config: DpoConfig,
}

impl DpoLossCalculator {
    pub fn new(config: DpoConfig) -> Self {
        Self { config }
    }
    
    /// Hitung DPO loss untuk satu preferensi pair
    pub fn calculate_loss(&self, pair: &PreferencePair) -> Result<f32> {
        let pi_lograt = pair.chosen_logprob - pair.rejected_logprob;
        let ref_lograt = pair.reference_chosen_logprob - pair.reference_rejected_logprob;
        
        let ratio = pi_lograt - ref_lograt;
        let sigmoid_input = self.config.beta * ratio;
        
        // Avoid numerical issues
        let sigmoid_input_clamped = sigmoid_input.clamp(-20.0, 20.0);
        let loss = -sigmoid_input_clamped.ln_1p();
        
        // Add regularization
        let regularization = self.config.regularization_strength * (pi_lograt * pi_lograt);
        
        Ok(loss + regularization)
    }
    
    /// Hitung gradient untuk DPO loss
    pub fn calculate_gradient(&self, pair: &PreferencePair) -> Result<(f32, f32)> {
        let pi_lograt = pair.chosen_logprob - pair.rejected_logprob;
        let ref_lograt = pair.reference_chosen_logprob - pair.reference_rejected_logprob;
        
        let ratio = pi_lograt - ref_lograt;
        let sigmoid_input = self.config.beta * ratio;
        let sigmoid_input_clamped = sigmoid_input.clamp(-20.0, 20.0);
        let sigmoid = 1.0 / (1.0 + (-sigmoid_input_clamped).exp());
        
        let grad_chosen = -self.config.beta * (1.0 - sigmoid);
        let grad_rejected = self.config.beta * sigmoid;
        
        // Add regularization gradient
        let reg_grad = 2.0 * self.config.regularization_strength;
        
        Ok((grad_chosen + reg_grad, grad_rejected + reg_grad))
    }
    
    /// Batch loss calculation
    pub fn calculate_batch_loss(&self, pairs: &[PreferencePair]) -> Result<f32> {
        if pairs.is_empty() {
            return Ok(0.0);
        }
        
        let total_loss: f32 = pairs
            .iter()
            .map(|pair| self.calculate_loss(pair).unwrap_or(0.0))
            .sum();
            
        Ok(total_loss / pairs.len() as f32)
    }
}

/// DPO Trainer
pub struct DpoTrainer {
    loss_calculator: DpoLossCalculator,
    model: PolicyModel,
}

impl DpoTrainer {
    pub fn new(model: PolicyModel, config: DpoConfig) -> Self {
        Self {
            loss_calculator: DpoLossCalculator::new(config),
            model,
        }
    }
    
    /// Extract preference pairs dari feedback
    pub fn extract_preference_pairs(
        &self,
        traces: &[ReasoningTrace],
        feedback: &[JudgeFeedback],
    ) -> Result<Vec<PreferencePair>> {
        let mut pairs = Vec::new();
        
        for fb in feedback {
            if let FeedbackType::Pairwise { preferred, rejected, confidence: _ } = &fb.feedback_type {
                // Cari traces yang mengandung preferred dan rejected steps
                if let (Some(pref_trace), Some(rej_trace)) = self.find_traces_for_steps(traces, preferred, rejected) {
                    let pair = PreferencePair {
                        id: Uuid::new_v4(),
                        prompt: pref_trace.prompt.clone(),
                        chosen: self.extract_step_content(&pref_trace, preferred)?,
                        rejected: self.extract_step_content(&rej_trace, rejected)?,
                        chosen_logprob: self.model.log_probability(&pref_trace.prompt, 
                            &self.extract_step_content(&pref_trace, preferred)?)?,
                        rejected_logprob: self.model.log_probability(&rej_trace.prompt,
                            &self.extract_step_content(&rej_trace, rejected)?)?,
                        reference_chosen_logprob: self.model.reference_log_probability(&pref_trace.prompt,
                            &self.extract_step_content(&pref_trace, preferred)?)?,
                        reference_rejected_logprob: self.model.reference_log_probability(&rej_trace.prompt,
                            &self.extract_step_content(&rej_trace, rejected)?)?,
                    };
                    pairs.push(pair);
                }
            }
        }
        
        Ok(pairs)
    }
    
    /// Training step untuk DPO
    pub fn training_step(&mut self, pairs: &[PreferencePair]) -> Result<f32> {
        let loss = self.loss_calculator.calculate_batch_loss(pairs)?;
        
        // Update model parameters (simplified implementation)
        for pair in pairs {
            let (grad_chosen, grad_rejected) = self.loss_calculator.calculate_gradient(pair)?;
            self.update_model_parameters(grad_chosen, grad_rejected)?;
        }
        
        Ok(loss)
    }
    
    // Helper methods
    fn find_traces_for_steps<'a>(
        &self,
        traces: &'a [ReasoningTrace],
        preferred_id: &Uuid,
        rejected_id: &Uuid,
    ) -> (Option<&'a ReasoningTrace>, Option<&'a ReasoningTrace>) {
        let pref_trace = traces.iter().find(|t| 
            t.steps.iter().any(|s| s.id == *preferred_id));
        let rej_trace = traces.iter().find(|t| 
            t.steps.iter().any(|s| s.id == *rejected_id));
        (pref_trace, rej_trace)
    }
    
    fn extract_step_content(&self, trace: &ReasoningTrace, step_id: &Uuid) -> Result<String> {
        trace.steps
            .iter()
            .find(|s| s.id == *step_id)
            .map(|s| s.content.clone())
            .ok_or_else(|| anyhow::anyhow!("Step not found in trace"))
    }
    
    fn update_model_parameters(&mut self, grad_chosen: f32, grad_rejected: f32) -> Result<()> {
        // Simplified parameter update - will be expanded
        // In real implementation, this would update actual neural network parameters
        Ok(())
    }
}


/// Utility functions
pub mod utils {
    use super::*;
    
    /// Convert feedback ke preference pairs
    pub fn feedback_to_pairs(
        traces: &[ReasoningTrace],
        feedback: &[JudgeFeedback],
    ) -> Result<Vec<PreferencePair>> {
        let mut pairs = Vec::new();
        
        for fb in feedback {
            if let FeedbackType::Pairwise { preferred, rejected, confidence } = &fb.feedback_type {
                // Implementation similar to DpoTrainer::extract_preference_pairs
                // This is a simplified version
                let pair = PreferencePair {
                    id: Uuid::new_v4(),
                    prompt: "sample_prompt".to_string(),
                    chosen: "chosen_response".to_string(),
                    rejected: "rejected_response".to_string(),
                    chosen_logprob: -1.0,
                    rejected_logprob: -2.0,
                    reference_chosen_logprob: -1.1,
                    reference_rejected_logprob: -2.1,
                };
                pairs.push(pair);
            }
        }
        
        Ok(pairs)
    }
    
    /// Validate preference pairs
    pub fn validate_pairs(pairs: &[PreferencePair]) -> Result<()> {
        for pair in pairs {
            if pair.chosen.is_empty() || pair.rejected.is_empty() {
                return Err(anyhow::anyhow!("Empty chosen or rejected response"));
            }
        }
        Ok(())
    }
}
