//! Cognitive Energy Conservation Law
//!
//! d(Ψ_I + Ξ)/dt = Π(t) - Δ(t)
//!
//! Intelligence tidak muncul gratis — sistem obey thermodynamic-like constraints.
//! Ada tradeoff energi antara information field intensity dan cognitive entropy.

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySample {
    pub time: f64,
    pub psi_i: f64,
    pub xi: f64,
    pub pi: f64,
    pub delta: f64,
    pub total: f64,
    pub conservation_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConservation {
    psi_i: f64,
    xi: f64,
    pi: f64,
    delta: f64,
    history: VecDeque<EnergySample>,
    max_history: usize,
    threshold: f64,
}

impl EnergyConservation {
    pub fn new() -> Self {
        Self {
            psi_i: 0.0,
            xi: 0.0,
            pi: 0.0,
            delta: 0.0,
            history: VecDeque::new(),
            max_history: 1000,
            threshold: 0.01,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Update state: d(Ψ_I + Ξ)/dt = Π - Δ
    /// Sebagian energi masuk ke field (Ψ_I), sebagian ke entropy (Ξ)
    pub fn update(&mut self, pi: f64, delta: f64, dt: f64) {
        self.pi = pi;
        self.delta = delta;

        let net_flow = pi - delta;
        let d_total = net_flow * dt;

        // Distribusi energi: 60% ke field, 40% ke entropy
        let psi_delta = d_total * 0.6;
        let xi_delta = d_total * 0.4;

        self.psi_i = (self.psi_i + psi_delta).max(0.0);
        self.xi = (self.xi + xi_delta).max(0.0);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let sample = EnergySample {
            time: now,
            psi_i: self.psi_i,
            xi: self.xi,
            pi,
            delta,
            total: self.psi_i + self.xi,
            conservation_error: self.check_conservation(),
        };

        self.history.push_back(sample);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Ψ_I = total conserved - entropy
    pub fn field_intensity(&self) -> f64 {
        self.psi_i
    }

    pub fn entropy(&self) -> f64 {
        self.xi
    }

    pub fn total_energy(&self) -> f64 {
        self.psi_i + self.xi
    }

    /// Conservation error: seberapa jauh dari hukum konservasi
    pub fn check_conservation(&self) -> f64 {
        let total = self.psi_i + self.xi;
        if total < 1e-10 {
            return 0.0;
        }
        (self.pi - self.delta - total).abs()
    }

    /// Apakah sistem dalam kondisi konservasi yang baik
    pub fn is_conserved(&self) -> bool {
        self.check_conservation() < self.threshold
    }

    pub fn recent_history(&self, n: usize) -> Vec<&EnergySample> {
        self.history.iter().rev().take(n).collect()
    }

    /// Energy dissipation ratio
    pub fn dissipation_ratio(&self) -> f64 {
        if self.pi.abs() < 1e-10 {
            return 0.0;
        }
        self.delta / self.pi
    }

    /// Thermodynamic efficiency η = (Π - Δ) / Π
    pub fn efficiency(&self) -> f64 {
        if self.pi.abs() < 1e-10 {
            return 0.0;
        }
        (self.pi - self.delta) / self.pi
    }

    pub fn reset(&mut self) {
        self.psi_i = 0.0;
        self.xi = 0.0;
        self.pi = 0.0;
        self.delta = 0.0;
        self.history.clear();
    }
}

impl Default for EnergyConservation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_conservation_initial_state() {
        let ec = EnergyConservation::new();
        assert_eq!(ec.psi_i, 0.0);
        assert_eq!(ec.xi, 0.0);
        assert_eq!(ec.total_energy(), 0.0);
    }

    #[test]
    fn test_energy_conservation_update() {
        let mut ec = EnergyConservation::new();
        ec.update(10.0, 3.0, 0.1);

        // d_total = (10 - 3) * 0.1 = 0.7
        // psi_i += 0.7 * 0.6 = 0.42
        // xi += 0.7 * 0.4 = 0.28
        assert!((ec.psi_i - 0.42).abs() < 1e-10);
        assert!((ec.xi - 0.28).abs() < 1e-10);
        assert!((ec.total_energy() - 0.70).abs() < 1e-10);
    }

    #[test]
    fn test_energy_never_negative() {
        let mut ec = EnergyConservation::new();
        ec.update(-5.0, 10.0, 1.0);
        assert!(ec.psi_i >= 0.0);
        assert!(ec.xi >= 0.0);
    }

    #[test]
    fn test_efficiency() {
        let mut ec = EnergyConservation::new();
        ec.update(10.0, 2.0, 1.0);
        let eff = ec.efficiency();
        assert!((eff - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_dissipation_ratio() {
        let mut ec = EnergyConservation::new();
        ec.update(10.0, 4.0, 1.0);
        assert!((ec.dissipation_ratio() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut ec = EnergyConservation::new();
        ec.update(10.0, 3.0, 0.1);
        assert!(ec.psi_i > 0.0);
        ec.reset();
        assert_eq!(ec.psi_i, 0.0);
        assert_eq!(ec.xi, 0.0);
    }
}
