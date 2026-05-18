//! Core components untuk SPARO framework

use anyhow::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Konfigurasi utama SPARO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparoConfig {
    /// Bobot untuk komponen DPO
    pub alpha: f32,
    /// Bobot untuk komponen KTO  
    pub beta: f32,
    /// Bobot untuk komponen IPO
    pub gamma: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Threshold untuk konvergensi
    pub convergence_threshold: f32,
}

impl Default for SparoConfig {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            beta: 0.3,
            gamma: 0.3,
            learning_rate: 1e-4,
            batch_size: 32,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
        }
    }
}

/// Representasi satu langkah penalaran
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub id: Uuid,
    pub content: String,
    pub step_number: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Trace lengkap dari penalaran
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    pub id: Uuid,
    pub prompt: String,
    pub steps: Vec<ReasoningStep>,
    pub final_answer: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Jenis feedback dari AI judge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Preferensi berpasangan (untuk DPO)
    Pairwise {
        preferred: Uuid,
        rejected: Uuid,
        confidence: f32,
    },
    /// Label independen (untuk KTO)
    Independent {
        step_id: Uuid,
        is_good: bool,
        confidence: f32,
    },
}

/// Feedback dari AI judge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeFeedback {
    pub id: Uuid,
    pub trace_id: Uuid,
    pub feedback_type: FeedbackType,
    pub reasoning: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Model state untuk tracking pelatihan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub iteration: usize,
    pub loss_history: Vec<f32>,
    pub current_loss: f32,
    pub best_loss: f32,
    pub converged: bool,
}

/// Loss components untuk SPARO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparoLoss {
    pub total_loss: f32,
    pub dpo_loss: f32,
    pub kto_loss: f32,
    pub ipo_loss: f32,
}

/// Policy model untuk representasi model language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyModel {
    pub model_id: Uuid,
    pub parameters: Array2<f32>,
    pub reference_params: Array2<f32>,
    pub is_teacher: bool,
}

impl PolicyModel {
    pub fn new(model_id: Uuid, param_dim: (usize, usize)) -> Self {
        let parameters = Array2::zeros(param_dim);
        let reference_params = parameters.clone();
        
        Self {
            model_id,
            parameters,
            reference_params,
            is_teacher: false,
        }
    }
    
    pub fn set_as_teacher(&mut self) {
        self.is_teacher = true;
    }
    
    pub fn log_probability(&self, input: &str, output: &str) -> Result<f32> {
        if output.is_empty() {
            return Ok(f32::NEG_INFINITY);
        }
        let (rows, cols) = self.parameters.dim();
        if rows == 0 || cols == 0 {
            return Ok(-(output.len() as f32));
        }

        let input_bytes = input.as_bytes();
        let output_bytes = output.as_bytes();
        let mut score = 0.0f32;

        for (i, &ob) in output_bytes.iter().enumerate() {
            let i_feat = if input_bytes.is_empty() {
                1.0
            } else {
                let idx = (i * input_bytes.len() / output_bytes.len()).min(input_bytes.len() - 1);
                input_bytes[idx] as f32 / 255.0
            };
            for j in 0..cols.min(output_bytes.len()) {
                let w = self.parameters[[i.min(rows - 1) % rows, j]];
                score += i_feat * w * (output_bytes[j] as f32 / 255.0);
            }
        }

        let per_token = score / output.len() as f32;
        Ok(per_token.clamp(-20.0, 0.0))
    }
    
    pub fn reference_log_probability(&self, input: &str, output: &str) -> Result<f32> {
        self.log_probability(input, output)
    }

    /// Apply gradient descent update to model parameters.
    /// Uses real gradient of log_probability w.r.t. each parameter.
    /// d(log_prob)/d(w[i,j]) = input_feat[i] * output_feat[j] / (255 * 255 * output_len)
    pub fn apply_gradient(&mut self, input: &str, output: &str, grad_loss: f32, learning_rate: f32) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }
        let (rows, cols) = self.parameters.dim();
        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let input_bytes = input.as_bytes();
        let output_bytes = output.as_bytes();
        let output_len = output.len().max(1) as f32;

        for i in 0..rows.min(output_bytes.len()) {
            let i_feat = if input_bytes.is_empty() {
                1.0
            } else {
                let idx = (i * input_bytes.len() / output_bytes.len().max(1)).min(input_bytes.len() - 1);
                input_bytes[idx] as f32 / 255.0
            };
            for j in 0..cols.min(output_bytes.len()) {
                let out_feat = output_bytes[j] as f32 / 255.0;
                let d_logprob = i_feat * out_feat / output_len;
                let gradient = grad_loss * d_logprob;
                self.parameters[[i, j]] -= learning_rate * gradient;
            }
        }
        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
}

/// Metrics untuk monitoring pelatihan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub iteration: usize,
    pub loss: SparoLoss,
    pub accuracy: f32,
    pub throughput: f32,
    pub convergence_rate: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TrainingMetrics {
    pub fn new(iteration: usize, loss: SparoLoss) -> Self {
        Self {
            iteration,
            loss,
            accuracy: 0.0,
            throughput: 0.0,
            convergence_rate: 0.0,
            timestamp: chrono::Utc::now(),
        }
    }
}
