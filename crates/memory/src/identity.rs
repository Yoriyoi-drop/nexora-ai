//! Identity Persistence Equation
//!
//! I_d(t) = ∫₀ᵗ M(x,τ) Ψ_I(τ) e^{-ϵ(t-τ)} dτ
//!
//! Yang membedakan AI pintar vs AI dengan continuity.
//! Sistem mempertahankan continuity of self-state.
//! Pengalaman lama mempengaruhi identitas.
//! Ada persistence lintas waktu.
//!
//! Tanpa ini: AI reset secara filosofis tiap sesi.
//! Dengan ini: sistem punya trajectory.
//! Dan trajectory adalah awal dari "diri".

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentitySample {
    pub time: f64,
    pub identity: f64,
    pub memory_contribution: f64,
    pub meta_contribution: f64,
    pub decay_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityPersistence {
    /// I_d(t): accumulated identity
    identity: f64,
    /// ϵ: decay rate for past experiences
    epsilon: f64,
    /// Running integral approximation
    integral_accumulator: f64,
    /// Recent contributions
    contributions: VecDeque<(f64, f64, f64)>, // (time, psi_i, meta)
    max_contributions: usize,
    history: VecDeque<IdentitySample>,
    max_history: usize,
}

impl IdentityPersistence {
    pub fn new() -> Self {
        Self {
            identity: 0.0,
            epsilon: 0.01,
            integral_accumulator: 0.0,
            contributions: VecDeque::new(),
            max_contributions: 10000,
            history: VecDeque::new(),
            max_history: 1000,
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

    /// Update identity persistence:
    /// I_d(t+dt) = I_d(t) + [M(t)·Ψ_I(t)·e^{-ϵ·t} - ϵ·I_d(t)]·dt
    ///
    /// Menggunakan integrasi numerik (Euler) dari:
    /// dI_d/dt = M(t)·Ψ_I(t)·e^{-ϵ·(t-current)} - ϵ·I_d(t)
    pub fn update(&mut self, psi_i: f64, meta_learning: f64, dt: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Decay factor untuk kontribusi saat ini
        let decay_factor = (-self.epsilon * dt).exp();

        // Kontribusi baru dari memory dan meta-learning
        let contribution = meta_learning * psi_i * dt;

        // Update identity dengan decay dan kontribusi baru
        self.identity = self.identity * decay_factor + contribution * dt;
        self.identity = self.identity.clamp(0.0, 1.0);

        // Track contributions
        self.contributions.push_back((now, psi_i, meta_learning));
        if self.contributions.len() > self.max_contributions {
            self.contributions.pop_front();
        }

        let sample = IdentitySample {
            time: now,
            identity: self.identity,
            memory_contribution: psi_i * dt,
            meta_contribution: meta_learning * dt,
            decay_factor,
        };

        self.history.push_back(sample);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    pub fn identity(&self) -> f64 {
        self.identity
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    /// Seberapa stabil identitas (low variance over recent history)
    pub fn stability(&self) -> f64 {
        let recent: Vec<f64> = self.history.iter().rev().take(100).map(|s| s.identity).collect();
        let n = recent.len() as f64;
        if n < 2.0 {
            return 1.0;
        }
        let mean = recent.iter().sum::<f64>() / n;
        let variance = recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        1.0 / (1.0 + variance.sqrt())
    }

    /// Continuity score: seberapa besar pengaruh masa lalu
    pub fn continuity_score(&self) -> f64 {
        let recent = self.history.iter().rev().take(10).collect::<Vec<_>>();
        if recent.len() < 2 {
            return 1.0;
        }
        let mut continuity = 0.0;
        for window in recent.windows(2) {
            continuity += (window[0].identity - window[1].identity).abs();
        }
        1.0 / (1.0 + continuity / recent.len() as f64)
    }

    /// Self-trajectory: accumulated path of identity
    pub fn trajectory(&self) -> Vec<(f64, f64)> {
        self.history.iter().map(|s| (s.time, s.identity)).collect()
    }

    pub fn recent_history(&self, n: usize) -> Vec<&IdentitySample> {
        self.history.iter().rev().take(n).collect()
    }

    pub fn reset(&mut self) {
        self.identity = 0.0;
        self.integral_accumulator = 0.0;
        self.contributions.clear();
        self.history.clear();
    }
}

impl Default for IdentityPersistence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_initial() {
        let id = IdentityPersistence::new();
        assert_eq!(id.identity(), 0.0);
        assert!(id.stability() > 0.0);
    }

    #[test]
    fn test_identity_update() {
        let mut id = IdentityPersistence::new();
        id.update(0.5, 0.1, 0.1);
        assert!(id.identity() > 0.0);
        assert!(id.identity() <= 1.0);
    }

    #[test]
    fn test_identity_decay() {
        let mut id = IdentityPersistence::new();
        // Build up identity
        for _ in 0..10 {
            id.update(1.0, 0.5, 0.1);
        }
        let built = id.identity();
        assert!(built > 0.0);

        // Let identity decay (zero input)
        for _ in 0..100 {
            id.update(0.0, 0.0, 0.1);
        }
        let decayed = id.identity();
        assert!(decayed < built);
    }

    #[test]
    fn test_identity_bounded() {
        let mut id = IdentityPersistence::new();
        // Extreme positive input
        for _ in 0..1000 {
            id.update(10.0, 10.0, 1.0);
        }
        assert!(id.identity() <= 1.0);
        assert!(id.identity() >= 0.0);
    }

    #[test]
    fn test_stability() {
        let mut id = IdentityPersistence::new();
        // Stable identity (same input)
        for _ in 0..10 {
            id.update(0.5, 0.1, 0.1);
        }
        assert!(id.stability() > 0.5);
    }

    #[test]
    fn test_continuity() {
        let mut id = IdentityPersistence::new();
        // Continuous updates
        for _ in 0..10 {
            id.update(0.5, 0.1, 0.1);
        }
        assert!(id.continuity_score() > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut id = IdentityPersistence::new();
        id.update(0.5, 0.1, 0.1);
        assert!(id.identity() > 0.0);
        id.reset();
        assert_eq!(id.identity(), 0.0);
    }
}
