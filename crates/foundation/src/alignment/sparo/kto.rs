//! Kahneman-Tversky Optimization (KTO) Implementation
//! 
//! KTO menerima label "baik/buruk" secara independen tanpa data berpasangan,
//! terinspirasi oleh Prospect Theory dari Kahneman & Tversky.

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::core::{PolicyModel, ReasoningTrace, JudgeFeedback, FeedbackType};

/// Konfigurasi KTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KtoConfig {
    /// Reference point untuk prospect theory
    pub reference_point: f32,
    /// Loss aversion coefficient (lambda)
    pub loss_aversion: f32,
    /// Probability weighting parameter (alpha)
    pub probability_weighting: f32,
    /// Regularization strength
    pub regularization_strength: f32,
}

impl Default for KtoConfig {
    fn default() -> Self {
        Self {
            reference_point: 0.0,
            loss_aversion: 2.25,  // Classic value from prospect theory
            probability_weighting: 0.88,
            regularization_strength: 0.01,
        }
    }
}

/// Data label independen untuk KTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentLabel {
    pub id: Uuid,
    pub prompt: String,
    pub response: String,
    pub is_good: bool,
    pub confidence: f32,
    pub log_probability: f32,
    pub reference_log_probability: f32,
}

/// Prospect Theory Calculator
pub struct ProspectTheoryCalculator {
    config: KtoConfig,
}

impl ProspectTheoryCalculator {
    pub fn new(config: KtoConfig) -> Self {
        Self { config }
    }
    
    /// Hitung value function berdasarkan prospect theory
    pub fn value_function(&self, x: f32) -> f32 {
        if x >= self.config.reference_point {
            // Gains: concave function
            (x - self.config.reference_point).powf(self.config.probability_weighting)
        } else {
            // Losses: convex function with loss aversion
            -self.config.loss_aversion * 
            (self.config.reference_point - x).powf(self.config.probability_weighting)
        }
    }
    
    /// Hitung probability weighting function
    pub fn probability_weighting(&self, p: f32) -> f32 {
        let p_clamped = p.clamp(0.001, 0.999); // Avoid extreme values
        (p_clamped.powf(self.config.probability_weighting) / 
         (p_clamped.powf(self.config.probability_weighting) + 
          (1.0 - p_clamped).powf(self.config.probability_weighting)).powf(1.0 / self.config.probability_weighting))
    }
    
    /// Hitung prospect value
    pub fn prospect_value(&self, outcomes: &[f32], probabilities: &[f32]) -> f32 {
        let mut prospect = 0.0;
        for (outcome, prob) in outcomes.iter().zip(probabilities.iter()) {
            prospect += self.value_function(*outcome) * self.probability_weighting(*prob);
        }
        prospect
    }
}

/// KTO Loss Calculator
pub struct KtoLossCalculator {
    config: KtoConfig,
    prospect_calc: ProspectTheoryCalculator,
}

impl KtoLossCalculator {
    pub fn new(config: KtoConfig) -> Self {
        Self {
            prospect_calc: ProspectTheoryCalculator::new(config.clone()),
            config,
        }
    }
    
    /// Hitung KTO loss untuk satu label independen
    pub fn calculate_loss(&self, label: &IndependentLabel) -> Result<f32> {
        let log_ratio = label.log_probability - label.reference_log_probability;
        
        // Apply prospect theory
        let outcomes = if label.is_good {
            vec![log_ratio, 0.0] // Good outcome vs neutral
        } else {
            vec![0.0, log_ratio] // Neutral vs bad outcome
        };
        
        let probabilities = vec![label.confidence, 1.0 - label.confidence];
        let prospect = self.prospect_calc.prospect_value(&outcomes, &probabilities);
        
        // Convert to loss (negative prospect)
        let loss = -prospect;
        
        // Add regularization
        let regularization = self.config.regularization_strength * (log_ratio * log_ratio);
        
        Ok(loss + regularization)
    }
    
    /// Hitung gradient untuk KTO loss
    pub fn calculate_gradient(&self, label: &IndependentLabel) -> Result<f32> {
        let log_ratio = label.log_probability - label.reference_log_probability;
        
        // Gradient of prospect value with respect to log probability
        let value_grad = if log_ratio >= self.config.reference_point {
            self.config.probability_weighting * 
            (log_ratio - self.config.reference_point).powf(self.config.probability_weighting - 1.0)
        } else {
            -self.config.loss_aversion * self.config.probability_weighting *
            (self.config.reference_point - log_ratio).powf(self.config.probability_weighting - 1.0)
        };
        
        let weighted_prob = self.prospect_calc.probability_weighting(label.confidence);
        let gradient = -value_grad * weighted_prob;
        
        // Add regularization gradient
        let reg_grad = 2.0 * self.config.regularization_strength * log_ratio;
        
        Ok(gradient + reg_grad)
    }
    
    /// Batch loss calculation
    pub fn calculate_batch_loss(&self, labels: &[IndependentLabel]) -> Result<f32> {
        if labels.is_empty() {
            return Ok(0.0);
        }
        
        let total_loss: f32 = labels
            .iter()
            .map(|label| self.calculate_loss(label).unwrap_or(0.0))
            .sum();
            
        Ok(total_loss / labels.len() as f32)
    }
    
    /// Calculate class balance loss untuk mengatasi imbalance
    pub fn calculate_balance_loss(&self, labels: &[IndependentLabel]) -> Result<f32> {
        let good_count = labels.iter().filter(|l| l.is_good).count() as f32;
        let bad_count = labels.iter().filter(|l| !l.is_good).count() as f32;
        let total = labels.len() as f32;
        
        if total == 0.0 {
            return Ok(0.0);
        }
        
        let good_ratio = good_count / total;
        let bad_ratio = bad_count / total;
        
        // Entropy-based balance loss
        let entropy = -(good_ratio * good_ratio.ln() + bad_ratio * bad_ratio.ln());
        let max_entropy = 2.0_f32.ln(); // Maximum entropy for binary classification
        
        let imbalance_loss = (max_entropy - entropy) / max_entropy;
        
        Ok(imbalance_loss * 0.1) // Scale down the balance loss
    }
}

/// KTO Trainer
pub struct KtoTrainer {
    loss_calculator: KtoLossCalculator,
    model: PolicyModel,
    learning_rate: f32,
}

impl KtoTrainer {
    pub fn new(model: PolicyModel, config: KtoConfig) -> Self {
        Self {
            loss_calculator: KtoLossCalculator::new(config),
            model,
            learning_rate: 1e-4,
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
    
    /// Extract independent labels dari feedback
    pub fn extract_independent_labels(
        &self,
        traces: &[ReasoningTrace],
        feedback: &[JudgeFeedback],
    ) -> Result<Vec<IndependentLabel>> {
        let mut labels = Vec::new();
        
        for fb in feedback {
            if let FeedbackType::Independent { step_id, is_good, confidence } = &fb.feedback_type {
                // Cari trace yang mengandung step ini
                if let Some(trace) = traces.iter().find(|t| 
                    t.steps.iter().any(|s| s.id == *step_id)) {
                    
                    let step_content = trace.steps
                        .iter()
                        .find(|s| s.id == *step_id)
                        .map(|s| s.content.clone())
                        .ok_or_else(|| anyhow::anyhow!("Step not found in trace"))?;
                    
                    let label = IndependentLabel {
                        id: Uuid::new_v4(),
                        prompt: trace.prompt.clone(),
                        response: step_content.clone(),
                        is_good: *is_good,
                        confidence: *confidence,
                        log_probability: self.model.log_probability(&trace.prompt, 
                            &step_content)?,
                        reference_log_probability: self.model.reference_log_probability(&trace.prompt,
                            &step_content)?,
                    };
                    labels.push(label);
                }
            }
        }
        
        Ok(labels)
    }
    
    /// Training step untuk KTO
    pub fn training_step(&mut self, labels: &[IndependentLabel]) -> Result<f32> {
        let main_loss = self.loss_calculator.calculate_batch_loss(labels)?;
        let balance_loss = self.loss_calculator.calculate_balance_loss(labels)?;
        let total_loss = main_loss + balance_loss;
        
        // Update model parameters using real gradient descent
        for label in labels {
            let gradient = self.loss_calculator.calculate_gradient(label)?;
            self.update_model_parameters(gradient, label)?;
        }
        
        Ok(total_loss)
    }
    
    /// Update model parameters using real gradient descent
    fn update_model_parameters(&mut self, gradient: f32, label: &IndependentLabel) -> Result<()> {
        // Adjust model parameters to decrease KTO loss
        // If label.is_good, increase log-probability; if bad, decrease it
        self.model.apply_gradient(&label.prompt, &label.response, gradient, self.learning_rate)?;
        Ok(())
    }
}


/// Utility functions
pub mod utils {
    use super::*;
    
    /// Convert feedback ke independent labels
    pub fn feedback_to_labels(
        traces: &[ReasoningTrace],
        feedback: &[JudgeFeedback],
    ) -> Result<Vec<IndependentLabel>> {
        let mut labels = Vec::new();
        
        for fb in feedback {
            if let FeedbackType::Independent { step_id, is_good, confidence } = &fb.feedback_type {
                // Simplified implementation
                let label = IndependentLabel {
                    id: Uuid::new_v4(),
                    prompt: "sample_prompt".to_string(),
                    response: "sample_response".to_string(),
                    is_good: *is_good,
                    confidence: *confidence,
                    log_probability: -1.0,
                    reference_log_probability: -1.1,
                };
                labels.push(label);
            }
        }
        
        Ok(labels)
    }
    
    /// Analyze label distribution
    pub fn analyze_distribution(labels: &[IndependentLabel]) -> DistributionStats {
        let total = labels.len();
        let good_count = labels.iter().filter(|l| l.is_good).count();
        let bad_count = total - good_count;
        
        let avg_confidence_good = labels.iter()
            .filter(|l| l.is_good)
            .map(|l| l.confidence)
            .sum::<f32>() / good_count.max(1) as f32;
            
        let avg_confidence_bad = labels.iter()
            .filter(|l| !l.is_good)
            .map(|l| l.confidence)
            .sum::<f32>() / bad_count.max(1) as f32;
        
        DistributionStats {
            total,
            good_count,
            bad_count,
            good_ratio: good_count as f32 / total as f32,
            bad_ratio: bad_count as f32 / total as f32,
            avg_confidence_good,
            avg_confidence_bad,
        }
    }
}

/// Statistics for label distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    pub total: usize,
    pub good_count: usize,
    pub bad_count: usize,
    pub good_ratio: f32,
    pub bad_ratio: f32,
    pub avg_confidence_good: f32,
    pub avg_confidence_bad: f32,
}
