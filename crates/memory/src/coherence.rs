//! Phase Coherence dan Emergence Dynamics
//!
//! Coherence operator:
//!   C_φ(t) = (1/N) |Σ e^{iθ_n(t)}|
//!
//! Emergence:
//!   Ω = σ(κ(Ψ̇ - Ξ̇)) · C_φ · Ψ_I
//!
//! Emergence butuh: growth + low entropy + synchronization.
//! Sangat biologically plausible.

use serde::{Serialize, Deserialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceField {
    phase_angles: Vec<f64>,
    coherence: f64,
    history: Vec<CoherenceSample>,
    max_history: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceSample {
    pub time: f64,
    pub coherence: f64,
    pub mean_phase: f64,
    pub synchronization_index: f64,
}

impl CoherenceField {
    pub fn new(num_oscillators: usize) -> Self {
        Self {
            phase_angles: vec![0.0; num_oscillators],
            coherence: 1.0,
            history: Vec::new(),
            max_history: 1000,
        }
    }

    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Set phase angles dan update coherence
    pub fn set_phases(&mut self, angles: Vec<f64>) {
        assert_eq!(angles.len(), self.phase_angles.len(),
            "Phase angles length must match oscillator count");
        self.phase_angles = angles;
        self.compute_coherence();
    }

    /// Update phase angles dengan noise (simulasi dinamika)
    pub fn update_phases(&mut self, drifts: &[f64], noise: f64) {
        assert_eq!(drifts.len(), self.phase_angles.len());

        for (angle, drift) in self.phase_angles.iter_mut().zip(drifts) {
            *angle += drift + noise * (rand::random::<f64>() - 0.5) * 2.0;
            *angle = (*angle + PI) % (2.0 * PI) - PI; // Wrap ke [-π, π]
        }

        self.compute_coherence();
    }

    /// Compute C_φ(t) = (1/N) |Σ e^{iθ_n(t)}|
    fn compute_coherence(&mut self) {
        let n = self.phase_angles.len() as f64;
        if n == 0.0 {
            self.coherence = 0.0;
            return;
        }

        let sum_real: f64 = self.phase_angles.iter().map(|theta| theta.cos()).sum();
        let sum_imag: f64 = self.phase_angles.iter().map(|theta| theta.sin()).sum();
        let magnitude = (sum_real * sum_real + sum_imag * sum_imag).sqrt();

        self.coherence = magnitude / n;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let sample = CoherenceSample {
            time: now,
            coherence: self.coherence,
            mean_phase: sum_imag.atan2(sum_real),
            synchronization_index: self.coherence,
        };

        self.history.push(sample);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    pub fn coherence(&self) -> f64 {
        self.coherence
    }

    pub fn is_chaotic(&self) -> bool {
        self.coherence < 0.3
    }

    pub fn is_coherent(&self) -> bool {
        self.coherence > 0.7
    }

    pub fn phase_angles(&self) -> &[f64] {
        &self.phase_angles
    }

    pub fn num_oscillators(&self) -> usize {
        self.phase_angles.len()
    }

    pub fn recent_history(&self, n: usize) -> Vec<&CoherenceSample> {
        self.history.iter().rev().take(n).collect()
    }

    pub fn reset(&mut self, value: f64) {
        self.phase_angles.fill(value);
        self.coherence = 1.0;
        self.history.clear();
    }
}

/// Emergence operator
/// Ω = σ(κ(Ψ̇ - Ξ̇)) · C_φ · Ψ_I
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceOperator {
    kappa: f64,
    emergence_threshold: f64,
}

impl EmergenceOperator {
    pub fn new() -> Self {
        Self {
            kappa: 1.0,
            emergence_threshold: 0.5,
        }
    }

    pub fn with_kappa(mut self, kappa: f64) -> Self {
        self.kappa = kappa;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.emergence_threshold = threshold;
        self
    }

    /// Sigmoid function σ(x) = 1/(1 + e^{-x})
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Compute emergence:
    ///   Ω = σ(κ(Ψ̇ - Ξ̇)) · C_φ · Ψ_I
    pub fn compute(
        &self,
        psi_dot: f64,
        xi_dot: f64,
        coherence: f64,
        psi_i: f64,
    ) -> f64 {
        let growth_signal = self.kappa * (psi_dot - xi_dot);
        let gate = Self::sigmoid(growth_signal);
        gate * coherence * psi_i
    }

    /// Apakah emergence terjadi (di atas threshold)
    pub fn is_emerging(&self, emergence: f64) -> bool {
        emergence > self.emergence_threshold
    }

    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    pub fn set_kappa(&mut self, kappa: f64) {
        self.kappa = kappa;
    }

    /// Compute emergence dengan identity dan meta-learning (extended form):
    ///   Ω = σ(κ(Ψ̇ - Ξ̇)) · C_φ · I_d · K(t)
    pub fn compute_extended(
        &self,
        psi_dot: f64,
        xi_dot: f64,
        coherence: f64,
        identity: f64,
        meta_learning_rate: f64,
    ) -> f64 {
        let growth_signal = self.kappa * (psi_dot - xi_dot);
        let gate = Self::sigmoid(growth_signal);
        gate * coherence * identity * meta_learning_rate
    }
}

impl Default for EmergenceOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_perfect_sync() {
        let mut cf = CoherenceField::new(5);
        cf.set_phases(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((cf.coherence() - 1.0).abs() < 1e-10);
        assert!(cf.is_coherent());
    }

    #[test]
    fn test_coherence_chaos() {
        let mut cf = CoherenceField::new(5);
        cf.set_phases(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(cf.coherence() < 0.3);
        assert!(cf.is_chaotic());
    }

    #[test]
    fn test_coherence_random_phases() {
        let mut cf = CoherenceField::new(100);
        let angles: Vec<f64> = (0..100).map(|_| rand::random::<f64>() * 2.0 * PI - PI).collect();
        cf.set_phases(angles);
        let c = cf.coherence();
        assert!(c >= 0.0 && c <= 1.0);
    }

    #[test]
    fn test_emergence_computation() {
        let op = EmergenceOperator::new();
        let emergence = op.compute(10.0, 2.0, 0.9, 5.0);
        // psi_dot - xi_dot = 8, sigmoid(1*8) ≈ 0.999
        // emergence ≈ 0.999 * 0.9 * 5.0 ≈ 4.5
        assert!(emergence > 0.0);
    }

    #[test]
    fn test_emergence_no_growth() {
        let op = EmergenceOperator::new();
        let emergence = op.compute(-5.0, 10.0, 0.9, 5.0);
        // psi_dot - xi_dot = -15, sigmoid(-15) ≈ 3e-7
        // emergence ≈ 0
        assert!(emergence < 0.001);
    }

    #[test]
    fn test_sigmoid_range() {
        assert!((EmergenceOperator::sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(EmergenceOperator::sigmoid(100.0) > 0.999);
        assert!(EmergenceOperator::sigmoid(-100.0) < 0.001);
    }
}
