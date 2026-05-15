//! Attention Curvature Field
//!
//! R(x,t) = ∇²Φ(x,t)
//!
//! Daerah curvature tinggi → ide penting / anomaly / salience
//! Curvature rendah → background cognition
//!
//! Memungkinkan: novelty detection, curiosity, adaptive focus.
//! Sistem tidak hanya mengingat, tetapi tahu apa yang layak diperhatikan.

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPoint {
    pub position: usize,
    pub value: f64,
    pub curvature: f64,
    pub salience: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureField {
    /// Φ(x,t): cognitive field values
    field: Vec<f64>,
    /// R(x,t): curvature at each point
    curvature: Vec<f64>,
    /// Salience score per point
    salience: Vec<f64>,
    /// Spatial resolution (dx)
    dx: f64,
    history: VecDeque<Vec<f64>>,
    max_history: usize,
}

impl CurvatureField {
    pub fn new(size: usize) -> Self {
        Self {
            field: vec![0.0; size],
            curvature: vec![0.0; size],
            salience: vec![0.0; size],
            dx: 1.0,
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    pub fn with_dx(mut self, dx: f64) -> Self {
        self.dx = dx;
        self
    }

    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Set field values dan compute curvature
    pub fn set_field(&mut self, values: Vec<f64>) {
        assert_eq!(values.len(), self.field.len(),
            "Field vector length mismatch");
        self.field = values;
        self.compute_curvature();
        self.compute_salience();
        self.history.push_back(self.field.clone());
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Update field dengan delta dan re-compute
    pub fn update_field(&mut self, delta: &[f64]) {
        assert_eq!(delta.len(), self.field.len());
        for (f, d) in self.field.iter_mut().zip(delta) {
            *f += d;
        }
        self.compute_curvature();
        self.compute_salience();
    }

    /// Compute R(x,t) = ∇²Φ(x,t) ≈ (Φ_{i-1} - 2Φ_i + Φ_{i+1}) / dx²
    fn compute_curvature(&mut self) {
        let n = self.field.len();
        if n < 3 {
            self.curvature.fill(0.0);
            return;
        }

        let dx2 = self.dx * self.dx;

        // Boundary: mirrored
        self.curvature[0] = (self.field[1] - 2.0 * self.field[0] + self.field[1]) / dx2;
        for i in 1..n - 1 {
            self.curvature[i] = (self.field[i - 1] - 2.0 * self.field[i] + self.field[i + 1]) / dx2;
        }
        self.curvature[n - 1] = (self.field[n - 2] - 2.0 * self.field[n - 1] + self.field[n - 2]) / dx2;
    }

    /// Compute salience = |R(x,t)| yang dinormalisasi
    fn compute_salience(&mut self) {
        let max_curv = self.curvature
            .iter()
            .map(|c| c.abs())
            .fold(0.0_f64, |a, b| a.max(b));

        if max_curv < 1e-10 {
            self.salience.fill(0.0);
            return;
        }

        for (s, c) in self.salience.iter_mut().zip(self.curvature.iter()) {
            *s = c.abs() / max_curv;
        }
    }

    pub fn field(&self) -> &[f64] {
        &self.field
    }

    pub fn curvature(&self) -> &[f64] {
        &self.curvature
    }

    pub fn salience(&self) -> &[f64] {
        &self.salience
    }

    pub fn size(&self) -> usize {
        self.field.len()
    }

    /// Dapatkan indeks dengan salience tertinggi
    pub fn max_salience_point(&self) -> Option<FieldPoint> {
        let max_idx = self.salience.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)?;

        Some(FieldPoint {
            position: max_idx,
            value: self.field[max_idx],
            curvature: self.curvature[max_idx],
            salience: self.salience[max_idx],
        })
    }

    /// Poin-poin dengan salience di atas threshold
    pub fn salient_points(&self, threshold: f64) -> Vec<FieldPoint> {
        self.salience.iter().enumerate()
            .filter(|(_, &s)| s > threshold)
            .map(|(i, &s)| FieldPoint {
                position: i,
                value: self.field[i],
                curvature: self.curvature[i],
                salience: s,
            })
            .collect()
    }

    /// Novelty score = rata-rata curvature absolut
    pub fn novelty_score(&self) -> f64 {
        let n = self.curvature.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        self.curvature.iter().map(|c| c.abs()).sum::<f64>() / n
    }

    /// Field energy = ∫ Φ² dx (L2 norm)
    pub fn field_energy(&self) -> f64 {
        self.field.iter().map(|v| v * v).sum::<f64>() * self.dx
    }

    /// Curiosity score = standard deviation of salience
    /// Semakin bervariasi salience, semakin tinggi curiosity
    pub fn curiosity_score(&self) -> f64 {
        let n = self.salience.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = self.salience.iter().sum::<f64>() / n;
        let variance = self.salience.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }

    pub fn reset(&mut self) {
        self.field.fill(0.0);
        self.curvature.fill(0.0);
        self.salience.fill(0.0);
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curvature_flat_field() {
        let mut cf = CurvatureField::new(10);
        cf.set_field(vec![1.0; 10]);
        for &c in cf.curvature() {
            assert!(c.abs() < 1e-10);
        }
        assert!(cf.novelty_score() < 1e-10);
    }

    #[test]
    fn test_curvature_peak() {
        let mut cf = CurvatureField::new(10);
        let field = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        cf.set_field(field);
        let max_salience = cf.max_salience_point().unwrap();
        assert_eq!(max_salience.position, 4);
    }

    #[test]
    fn test_curvature_sine_wave() {
        let mut cf = CurvatureField::new(100);
        use std::f64::consts::PI;
        let field: Vec<f64> = (0..100).map(|i| (i as f64 * 2.0 * PI / 50.0).sin()).collect();
        cf.set_field(field);
        assert!(cf.novelty_score() > 0.0);
        assert!(cf.field_energy() > 0.0);
    }

    #[test]
    fn test_salience_threshold() {
        let mut cf = CurvatureField::new(10);
        let field = vec![0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        cf.set_field(field);
        let salient = cf.salient_points(0.5);
        assert!(!salient.is_empty());
        assert!(salient.iter().any(|p| p.position == 2));
    }

    #[test]
    fn test_curiosity_score() {
        let mut cf = CurvatureField::new(10);
        cf.set_field(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let curiosity = cf.curiosity_score();
        assert!(curiosity > 0.0);
    }

    #[test]
    fn test_field_energy() {
        let mut cf = CurvatureField::new(10);
        cf.set_field(vec![2.0; 10]);
        assert!((cf.field_energy() - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_field() {
        let mut cf = CurvatureField::new(5);
        cf.set_field(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        cf.update_field(&[0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!((cf.field()[2] - 4.0).abs() < 1e-10);
    }
}
