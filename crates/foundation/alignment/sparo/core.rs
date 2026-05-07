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
#[derive(Debug, Clone)]
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
        // Implementasi sederhana - akan diperluas
        let combined = format!("{} {}", input, output);
        let hash = format!("{:x}", md5::compute(combined.as_bytes()));
        let hash_value = u64::from_str_radix(&hash[..8], 16).unwrap_or(0);
        let prob = (hash_value as f32 / u64::MAX as f32).ln();
        Ok(prob)
    }
}

/// Training batch untuk SPARO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingBatch {
    pub id: Uuid,
    pub traces: Vec<ReasoningTrace>,
    pub feedback: Vec<JudgeFeedback>,
    pub iteration: usize,
}

impl TrainingBatch {
    pub fn new(traces: Vec<ReasoningTrace>, feedback: Vec<JudgeFeedback>, iteration: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            traces,
            feedback,
            iteration,
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty() || self.feedback.is_empty()
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
