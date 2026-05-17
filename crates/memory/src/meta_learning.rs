//! Meta-Learning Tensor
//!
//! M(t) = ∂T/∂t
//!
//! Sistem belajar bagaimana belajar:
//! - coupling mana yang efektif
//! - memory pathway mana yang buruk
//! - adaptasi learning rate secara dinamis
//!
//! MNEMIS bukan static learner, tetapi adaptive cognitive ecology.

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningSample {
    pub time: f64,
    pub learning_speed: f64,
    pub coupling_efficiency: f64,
    pub pathway_quality: f64,
    pub plasticity: f64,
}

/// Meta-learning tensor M(t) = ∂T/∂t
/// Melacak perubahan coupling tensor terhadap waktu
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningTensor {
    /// ∂T/∂t: how coupling changes over time
    tensor: Vec<Vec<Vec<f64>>>,
    /// Accumulated meta-knowledge
    meta_knowledge: Vec<Vec<Vec<f64>>>,
    /// Base learning rate untuk setiap dimensi
    base_learning_rates: Vec<Vec<Vec<f64>>>,
    /// Decay factor for older meta-learning signals
    epsilon: f64,
    history: VecDeque<MetaLearningSample>,
    max_history: usize,
    /// Running statistics
    total_updates: u64,
    avg_coupling_change: f64,
}

impl MetaLearningTensor {
    /// Create new meta-learning tensor with given dimensions
    pub fn new(dim1: usize, dim2: usize, dim3: usize) -> Self {
        Self {
            tensor: vec![vec![vec![0.0; dim3]; dim2]; dim1],
            meta_knowledge: vec![vec![vec![0.0; dim3]; dim2]; dim1],
            base_learning_rates: vec![vec![vec![0.01; dim3]; dim2]; dim1],
            epsilon: 0.01,
            history: VecDeque::new(),
            max_history: 1000,
            total_updates: 0,
            avg_coupling_change: 0.0,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Update meta-learning tensor:
    /// M(t+1) = (1-ε)·M(t) + ε·∂T/∂t
    pub fn update(&mut self, coupling_delta: &[Vec<Vec<f64>>]) {
        assert_eq!(coupling_delta.len(), self.tensor.len());
        assert_eq!(coupling_delta[0].len(), self.tensor[0].len());
        assert_eq!(coupling_delta[0][0].len(), self.tensor[0][0].len());

        let dim1 = self.tensor.len();
        let dim2 = self.tensor[0].len();
        let dim3 = self.tensor[0][0].len();

        let mut total_change = 0.0;
        let mut count = 0;

        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    let delta = coupling_delta[i][j][k];

                    // Update meta-learning tensor
                    self.tensor[i][j][k] = (1.0 - self.epsilon) * self.tensor[i][j][k]
                        + self.epsilon * delta;

                    // Accumulate meta-knowledge
                    self.meta_knowledge[i][j][k] += delta.abs();

                    // Adapt learning rate berdasarkan meta-knowledge
                    let knowledge = self.meta_knowledge[i][j][k];
                    self.base_learning_rates[i][j][k] = Self::adaptive_lr(knowledge);

                    total_change += delta.abs();
                    count += 1;
                }
            }
        }

        self.total_updates += 1;
        let n = count as f64;
        self.avg_coupling_change = if n > 0.0 { total_change / n } else { 0.0 };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let sample = MetaLearningSample {
            time: now,
            learning_speed: self.avg_learning_speed(),
            coupling_efficiency: self.coupling_efficiency(),
            pathway_quality: self.pathway_quality(),
            plasticity: self.plasticity(),
        };

        self.history.push_back(sample);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Adaptive learning rate based on meta-knowledge
    fn adaptive_lr(knowledge: f64) -> f64 {
        // High knowledge → lower LR (converged)
        // Low knowledge → higher LR (exploring)
        let base = 0.01;
        let decay = 1.0 / (1.0 + knowledge * 10.0);
        base * decay.max(0.001)
    }

    /// Get adapted learning rate for a specific coupling
    pub fn learning_rate(&self, i: usize, j: usize, k: usize) -> f64 {
        self.base_learning_rates[i][j][k]
    }

    /// Which pathways are ineffective (low coupling change)
    pub fn ineffective_pathways(&self, threshold: f64) -> Vec<(usize, usize, usize)> {
        let dims = (self.tensor.len(), self.tensor[0].len(), self.tensor[0][0].len());
        let mut pathways = Vec::with_capacity(dims.0 * dims.1 * dims.2);

        for i in 0..dims.0 {
            for j in 0..dims.1 {
                for k in 0..dims.2 {
                    if self.meta_knowledge[i][j][k] < threshold {
                        pathways.push((i, j, k));
                    }
                }
            }
        }
        pathways
    }

    /// Average learning speed across all couplings
    pub fn avg_learning_speed(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for layer in &self.base_learning_rates {
            for row in layer {
                for &lr in row {
                    sum += lr;
                    count += 1;
                }
            }
        }
        if count > 0 { sum / count as f64 } else { 0.0 }
    }

    /// Coupling efficiency: seberapa efektif coupling berubah
    pub fn coupling_efficiency(&self) -> f64 {
        let total_knowledge: f64 = self.meta_knowledge.iter()
            .flat_map(|layer| layer.iter())
            .flat_map(|row| row.iter())
            .sum();

        let n = (self.tensor.len() * self.tensor[0].len() * self.tensor[0][0].len()) as f64;
        if n > 0.0 { (total_knowledge / n).min(1.0) } else { 0.0 }
    }

    /// Pathway quality: rata-rata meta-knowledge
    pub fn pathway_quality(&self) -> f64 {
        self.coupling_efficiency()
    }

    /// Plasticity: seberapa mudah sistem berubah
    pub fn plasticity(&self) -> f64 {
        self.avg_learning_speed() * 100.0
    }

    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    pub fn tensor(&self) -> &[Vec<Vec<f64>>] {
        &self.tensor
    }

    pub fn meta_knowledge(&self) -> &[Vec<Vec<f64>>] {
        &self.meta_knowledge
    }

    pub fn recent_history(&self, n: usize) -> Vec<&MetaLearningSample> {
        self.history.iter().rev().take(n).collect()
    }

    pub fn reset(&mut self) {
        let dims = (self.tensor.len(), self.tensor[0].len(), self.tensor[0][0].len());
        self.tensor = vec![vec![vec![0.0; dims.2]; dims.1]; dims.0];
        self.meta_knowledge = vec![vec![vec![0.0; dims.2]; dims.1]; dims.0];
        self.base_learning_rates = vec![vec![vec![0.01; dims.2]; dims.1]; dims.0];
        self.total_updates = 0;
        self.avg_coupling_change = 0.0;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learning_creation() {
        let mt = MetaLearningTensor::new(3, 3, 3);
        assert_eq!(mt.tensor().len(), 3);
        assert_eq!(mt.total_updates(), 0);
    }

    #[test]
    fn test_meta_learning_update() {
        let mut mt = MetaLearningTensor::new(2, 2, 2);
        let delta = vec![
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![vec![0.5, 0.6], vec![0.7, 0.8]],
        ];
        mt.update(&delta);
        assert_eq!(mt.total_updates(), 1);
        assert!(mt.avg_coupling_change > 0.0);
    }

    #[test]
    fn test_adaptive_lr() {
        let lr_high = MetaLearningTensor::adaptive_lr(100.0);
        let lr_low = MetaLearningTensor::adaptive_lr(0.0);
        assert!(lr_low > lr_high);
        assert!(lr_low <= 0.01);
    }

    #[test]
    fn test_ineffective_pathways() {
        let mut mt = MetaLearningTensor::new(2, 2, 2);
        // Update with small delta for some pathways
        let delta = vec![
            vec![vec![0.0, 0.0], vec![0.5, 0.5]],
            vec![vec![0.0, 0.0], vec![0.5, 0.5]],
        ];
        mt.update(&delta);

        let ineffective = mt.ineffective_pathways(0.01);
        assert!(!ineffective.is_empty());
    }

    #[test]
    fn test_plasticity() {
        let mt = MetaLearningTensor::new(2, 2, 2);
        let plasticity = mt.plasticity();
        // Initial plasticity: 0.01 * 100 = 1.0
        assert!((plasticity - 1.0).abs() < 1e-10);
    }
}
