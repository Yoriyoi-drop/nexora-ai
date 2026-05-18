//! EMA (Exponential Moving Average) Stabilizer untuk VOGP+
//!
//! Implementasi EMA untuk stabilisasi gradien historis yang membuat
//! smoothness penalty menjadi scale-invariant dan stabil lintas arsitektur.

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// EMA Stabilizer untuk gradien historis
/// 
/// Formula: EMA_t = β * EMA_{t-1} + (1-β) * ĝ_t
/// 
/// Dimana:
/// - β: decay rate (biasanya 0.99)
/// - ĝ_t: gradien norm saat ini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMAStabilizer {
    /// EMA decay rate (β)
    beta: f32,
    /// Current EMA value
    current_ema: f32,
    /// History untuk analisis (opsional)
    history: VecDeque<f32>,
    /// Maximum history length
    max_history: usize,
    /// Total update count
    update_count: usize,
    /// Initial value untuk warmup
    initial_value: f32,
    /// Warmup period
    warmup_steps: usize,
}

impl EMAStabilizer {
    /// Buat EMA Stabilizer baru
    pub fn new(beta: f32) -> Self {
        assert!(beta > 0.0 && beta < 1.0, "Beta harus dalam (0, 1)");
        
        Self {
            beta,
            current_ema: 0.0,
            history: VecDeque::with_capacity(1000),
            max_history: 1000,
            update_count: 0,
            initial_value: 1.0,  // Reasonable starting point
            warmup_steps: 10,
        }
    }

    /// Buat EMA dengan konfigurasi kustom
    pub fn with_config(beta: f32, initial_value: f32, warmup_steps: usize, max_history: usize) -> Self {
        assert!(beta > 0.0 && beta < 1.0, "Beta harus dalam (0, 1)");
        
        Self {
            beta,
            current_ema: initial_value,
            history: VecDeque::with_capacity(max_history),
            max_history,
            update_count: 0,
            initial_value,
            warmup_steps,
        }
    }

    /// Update EMA dengan nilai baru
    pub fn update(&mut self, value: f32) {
        self.update_count += 1;
        
        // Warmup period: gunakan simple average
        if self.update_count <= self.warmup_steps {
            self.current_ema = (self.current_ema * (self.update_count - 1) as f32 + value) 
                              / self.update_count as f32;
        } else {
            // Standard EMA update
            self.current_ema = self.beta * self.current_ema + (1.0 - self.beta) * value;
        }
        
        // Update history
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(self.current_ema);
    }

    /// Get current EMA value
    pub fn get_current(&self) -> f32 {
        self.current_ema
    }

    /// Get current EMA sebagai gradient norm
    pub fn get_current_norm(&self) -> f32 {
        self.current_ema.max(1e-8)  // Prevent division by zero
    }

    /// Update beta parameter runtime
    pub fn update_beta(&mut self, new_beta: f32) {
        assert!(new_beta > 0.0 && new_beta < 1.0, "Beta harus dalam (0, 1)");
        self.beta = new_beta;
    }

    /// Reset EMA ke initial state
    pub fn reset(&mut self) {
        self.current_ema = self.initial_value;
        self.history.clear();
        self.update_count = 0;
    }

    /// Get variance dari EMA history
    pub fn get_variance(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        
        let mean = self.history.iter().sum::<f32>() / self.history.len() as f32;
        let variance = self.history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (self.history.len() - 1) as f32;
        
        variance
    }

    /// Get trend (positive = increasing, negative = decreasing)
    pub fn get_trend(&self, window_size: usize) -> f32 {
        if self.history.len() < window_size + 1 {
            return 0.0;
        }
        
        let recent: Vec<f32> = self.history.iter()
            .rev()
            .take(window_size)
            .cloned()
            .collect();
        
        let older: Vec<f32> = self.history.iter()
            .rev()
            .skip(window_size)
            .take(window_size)
            .cloned()
            .collect();
        
        if older.is_empty() {
            return 0.0;
        }
        
        let recent_mean = recent.iter().sum::<f32>() / recent.len() as f32;
        let older_mean = older.iter().sum::<f32>() / older.len() as f32;
        
        recent_mean - older_mean
    }

    /// Check jika EMA stabil (low variance)
    pub fn is_stable(&self, threshold: f32) -> bool {
        self.get_variance() < threshold
    }

    /// Get statistics untuk debugging
    pub fn get_statistics(&self) -> EMAStatistics {
        EMAStatistics {
            current_ema: self.current_ema,
            variance: self.get_variance(),
            trend: self.get_trend(10),
            update_count: self.update_count,
            is_stable: self.is_stable(0.01),
            history_length: self.history.len(),
        }
    }

    /// Apply EMA untuk array nilai (batch processing)
    pub fn update_batch(&mut self, values: &[f32]) {
        for &value in values {
            self.update(value);
        }
    }

    /// Predict next EMA value
    pub fn predict_next(&self) -> f32 {
        if self.update_count == 0 {
            return self.initial_value;
        }
        
        // Simple prediction: assume next value = current EMA
        self.current_ema
    }
}

/// Statistics dari EMA untuk monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMAStatistics {
    pub current_ema: f32,
    pub variance: f32,
    pub trend: f32,
    pub update_count: usize,
    pub is_stable: bool,
    pub history_length: usize,
}

/// Advanced EMA dengan adaptive decay
#[derive(Debug, Clone)]
pub struct AdaptiveEMAStabilizer {
    base_ema: EMAStabilizer,
    /// Adaptive beta based on volatility
    adaptive_beta: f32,
    /// Volatility threshold untuk beta adjustment
    volatility_threshold: f32,
    /// Beta adjustment factor
    beta_adjustment: f32,
    /// Minimum beta
    min_beta: f32,
    /// Maximum beta
    max_beta: f32,
}

impl AdaptiveEMAStabilizer {
    /// Buat adaptive EMA stabilizer
    pub fn new(base_beta: f32) -> Self {
        Self {
            base_ema: EMAStabilizer::new(base_beta),
            adaptive_beta: base_beta,
            volatility_threshold: 0.1,
            beta_adjustment: 0.1,
            min_beta: 0.9,
            max_beta: 0.999,
        }
    }

    /// Update dengan adaptive beta adjustment
    pub fn update(&mut self, value: f32) {
        // Calculate volatility
        let volatility = self.calculate_volatility();
        
        // Adjust beta based on volatility
        if volatility > self.volatility_threshold {
            // High volatility: increase beta (more smoothing)
            self.adaptive_beta = (self.adaptive_beta + self.beta_adjustment).min(self.max_beta);
        } else {
            // Low volatility: decrease beta (more responsive)
            self.adaptive_beta = (self.adaptive_beta - self.beta_adjustment).max(self.min_beta);
        }
        
        // Update base EMA with adaptive beta
        self.base_ema.update_beta(self.adaptive_beta);
        self.base_ema.update(value);
    }

    /// Calculate current volatility
    fn calculate_volatility(&self) -> f32 {
        let stats = self.base_ema.get_statistics();
        stats.variance.sqrt()
    }

    /// Get current adaptive beta
    pub fn get_adaptive_beta(&self) -> f32 {
        self.adaptive_beta
    }

    /// Get current EMA value
    pub fn get_current(&self) -> f32 {
        self.base_ema.get_current()
    }

    /// Reset ke initial state
    pub fn reset(&mut self) {
        self.base_ema.reset();
        self.adaptive_beta = self.base_ema.get_statistics().current_ema;
    }
}

/// Multi-dimensional EMA untuk vektor gradien
#[derive(Debug, Clone)]
pub struct MultiDimensionalEMA {
    /// EMA untuk setiap dimensi
    emas: Vec<EMAStabilizer>,
    /// Global EMA untuk aggregation
    global_ema: EMAStabilizer,
}

impl MultiDimensionalEMA {
    /// Buat multi-dimensional EMA
    pub fn new(dimensions: usize, beta: f32) -> Self {
        let emas = (0..dimensions).map(|_| EMAStabilizer::new(beta)).collect();
        let global_ema = EMAStabilizer::new(beta);
        
        Self { emas, global_ema }
    }

    /// Update semua dimensi
    pub fn update(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.emas.len(), "Dimension mismatch");
        
        let mut sum = 0.0;
        for (ema, &value) in self.emas.iter_mut().zip(values.iter()) {
            ema.update(value);
            sum += value;
        }
        
        // Update global EMA dengan mean
        let mean_value = sum / values.len() as f32;
        self.global_ema.update(mean_value);
    }

    /// Get EMA values untuk semua dimensi
    pub fn get_values(&self) -> Vec<f32> {
        self.emas.iter().map(|ema| ema.get_current()).collect()
    }

    /// Get global EMA
    pub fn get_global(&self) -> f32 {
        self.global_ema.get_current()
    }

    /// Get variance per dimensi
    pub fn get_variances(&self) -> Vec<f32> {
        self.emas.iter().map(|ema| ema.get_variance()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let mut ema = EMAStabilizer::new(0.9);
        
        ema.update(1.0);
        assert!(ema.get_current() > 0.0);
        
        ema.update(2.0);
        assert!(ema.get_current() > 1.0);
    }

    #[test]
    fn test_ema_reset() {
        let mut ema = EMAStabilizer::new(0.9);
        
        ema.update(1.0);
        ema.update(2.0);
        ema.reset();
        
        assert_eq!(ema.get_current(), 1.0); // initial_value
        assert_eq!(ema.update_count, 0);
    }

    #[test]
    fn test_ema_variance() {
        let mut ema = EMAStabilizer::new(0.9);
        
        for i in 0..100 {
            ema.update(i as f32);
        }
        
        let variance = ema.get_variance();
        assert!(variance > 0.0);
    }

    #[test]
    fn test_adaptive_ema() {
        let mut adaptive_ema = AdaptiveEMAStabilizer::new(0.95);
        
        // High volatility updates
        for i in 0..50 {
            adaptive_ema.update((i % 10) as f32);
        }
        
        let beta = adaptive_ema.get_adaptive_beta();
        assert!(beta > 0.9 && beta < 1.0);
    }

    #[test]
    fn test_multidimensional_ema() {
        let mut multi_ema = MultiDimensionalEMA::new(3, 0.9);
        
        multi_ema.update(&[1.0, 2.0, 3.0]);
        multi_ema.update(&[2.0, 3.0, 4.0]);
        
        let values = multi_ema.get_values();
        assert_eq!(values.len(), 3);
        
        let global = multi_ema.get_global();
        assert!(global > 0.0);
    }
}
