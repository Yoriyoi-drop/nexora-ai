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
        // TODO: Replace with actual language model log-probability computation.
        // Placeholder: character-level n-gram overlap scoring between input and output.
        if output.is_empty() {
            return Ok(f32::NEG_INFINITY);
        }
        let in_ngrams: std::collections::HashSet<u32> = input.as_bytes()
            .windows(3)
            .map(|w| u32::from_ne_bytes([w[0], w.get(1).copied().unwrap_or(0), w.get(2).copied().unwrap_or(0), 0]))
            .collect();
        let out_ngrams: std::collections::HashSet<u32> = output.as_bytes()
            .windows(3)
            .map(|w| u32::from_ne_bytes([w[0], w.get(1).copied().unwrap_or(0), w.get(2).copied().unwrap_or(0), 0]))
            .collect();
        if out_ngrams.is_empty() {
            return Ok(-(output.len() as f32));
        }
        let overlap = out_ngrams.iter().filter(|ng| in_ngrams.contains(ng)).count();
        let score = (overlap as f32 / out_ngrams.len() as f32).clamp(1e-10, 1.0);
        Ok(score.ln())
    }
    
    pub fn reference_log_probability(&self, input: &str, output: &str) -> Result<f32> {
        self.log_probability(input, output)
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
