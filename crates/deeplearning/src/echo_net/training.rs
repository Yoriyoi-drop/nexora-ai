//! Training Stabilization for ECHO-Net Ω
//!
//! Implementasi Polar Gradient Normalization (PGN) dan Resonance Energy Clipping (REC)
//! untuk stabilisasi training model holographic.
//!
//! PGN - Polar Gradient Normalization:
//! - Memisahkan gradient magnitude dan phase
//! - Update phase dengan tanh activation
//! - Mencegah exploding phase dan chaotic oscillation
//!
//! REC - Resonance Energy Clipping:
//! - Membatasi energi resonansi
//! - Mencegah resonance explosion
//! - Normalisasi energi: R ← R / (1 + ||R||)

use crate::DLResult;
use crate::echo_net::ComplexTensor;
use ndarray::ArrayD;

/// Gradient statistics for monitoring
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    pub magnitude_mean: f32,
    pub magnitude_std: f32,
    pub phase_mean: f32,
    pub phase_std: f32,
    pub max_magnitude: f32,
    pub max_phase: f32,
    pub gradient_norm: f32,
}

impl Default for GradientStatistics {
    fn default() -> Self {
        Self {
            magnitude_mean: 0.0,
            magnitude_std: 0.0,
            phase_mean: 0.0,
            phase_std: 0.0,
            max_magnitude: 0.0,
            max_phase: 0.0,
            gradient_norm: 0.0,
        }
    }
}

/// Resonance energy statistics
#[derive(Debug, Clone)]
pub struct ResonanceEnergyStats {
    pub current_energy: f32,
    pub average_energy: f32,
    pub max_energy: f32,
    pub energy_variance: f32,
    pub clipping_count: u64,
    pub clipping_ratio: f32,
}

impl Default for ResonanceEnergyStats {
    fn default() -> Self {
        Self {
            current_energy: 0.0,
            average_energy: 0.0,
            max_energy: 0.0,
            energy_variance: 0.0,
            clipping_count: 0,
            clipping_ratio: 0.0,
        }
    }
}

/// Polar Gradient Normalization (PGN)
#[derive(Debug, Clone)]
pub struct PolarGradientNormalization {
    // PGN parameters
    phase_lr: f32,
    magnitude_lr: f32,
    max_phase_update: f32,
    momentum_factor: f32,
    
    // Gradient tracking
    phase_gradients: ArrayD<f32>,
    magnitude_gradients: ArrayD<f32>,
    phase_momentum: ArrayD<f32>,
    magnitude_momentum: ArrayD<f32>,
    
    // Statistics
    gradient_history: Vec<GradientStatistics>,
    average_phase_update: f32,
    average_magnitude_update: f32,
    
    // Adaptive parameters
    adaptive_lr: bool,
    lr_decay_rate: f32,
    min_lr: f32,
    max_lr: f32,
    
    // Stabilization
    phase_clipping: bool,
    magnitude_clipping: bool,
    max_phase_value: f32,
    max_magnitude_value: f32,
}

impl PolarGradientNormalization {
    /// Create new Polar Gradient Normalization
    pub fn new(
        parameter_shape: Vec<usize>,
        phase_lr: f32,
        magnitude_lr: f32,
        max_phase_update: f32,
    ) -> DLResult<Self> {
        let phase_gradients = ArrayD::zeros(parameter_shape.clone());
        let magnitude_gradients = ArrayD::zeros(parameter_shape.clone());
        let phase_momentum = ArrayD::zeros(parameter_shape.clone());
        let magnitude_momentum = ArrayD::zeros(parameter_shape);
        
        Ok(Self {
            phase_lr,
            magnitude_lr,
            max_phase_update,
            momentum_factor: 0.9,
            phase_gradients,
            magnitude_gradients,
            phase_momentum,
            magnitude_momentum,
            gradient_history: Vec::new(),
            average_phase_update: 0.0,
            average_magnitude_update: 0.0,
            adaptive_lr: true,
            lr_decay_rate: 0.995,
            min_lr: 1e-6,
            max_lr: 1e-2,
            phase_clipping: true,
            magnitude_clipping: true,
            max_phase_value: std::f32::consts::PI,
            max_magnitude_value: 10.0,
        })
    }
    
    /// Apply PGN to complex gradients
    pub fn apply_pgn(&mut self, complex_gradients: &ComplexTensor, learning_rate: f32) -> DLResult<ComplexTensor> {
        // Separate magnitude and phase gradients
        let (magnitude_grad, phase_grad) = self.separate_gradients(complex_gradients)?;
        
        // Store gradients for statistics
        self.phase_gradients = phase_grad.clone();
        self.magnitude_gradients = magnitude_grad.clone();
        
        // Apply momentum
        self.apply_momentum(&magnitude_grad, &phase_grad)?;
        
        // Apply phase normalization with tanh
        let normalized_phase = self.normalize_phase_gradients(&phase_grad, learning_rate)?;
        
        // Apply magnitude normalization
        let normalized_magnitude = self.normalize_magnitude_gradients(&magnitude_grad, learning_rate)?;
        
        // Recombine into complex gradients
        let normalized_complex = self.recombine_gradients(&normalized_magnitude, &normalized_phase)?;
        
        // Update statistics
        self.update_gradient_statistics(&magnitude_grad, &phase_grad)?;
        
        // Update adaptive learning rates
        if self.adaptive_lr {
            self.update_adaptive_lr()?;
        }
        
        Ok(normalized_complex)
    }
    
    /// Separate complex gradients into magnitude and phase components
    fn separate_gradients(&self, complex_gradients: &ComplexTensor) -> DLResult<(ArrayD<f32>, ArrayD<f32>)> {
        let magnitude = complex_gradients.amplitude().mapv(|x| x);
        let phase = complex_gradients.phase().mapv(|x| x);
        
        Ok((magnitude, phase))
    }
    
    /// Apply momentum to gradients
    fn apply_momentum(&mut self, magnitude_grad: &ArrayD<f32>, phase_grad: &ArrayD<f32>) -> DLResult<()> {
        // Update momentum
        self.magnitude_momentum = &self.magnitude_momentum * self.momentum_factor + magnitude_grad;
        self.phase_momentum = &self.phase_momentum * self.momentum_factor + phase_grad;
        
        Ok(())
    }
    
    /// Normalize phase gradients with tanh activation
    fn normalize_phase_gradients(&self, phase_grad: &ArrayD<f32>, learning_rate: f32) -> DLResult<ArrayD<f32>> {
        let effective_lr = self.phase_lr * learning_rate;
        
        // Apply tanh activation to phase gradients
        let mut normalized = phase_grad.mapv(|g| g.tanh() * effective_lr);
        
        // Apply phase clipping if enabled
        if self.phase_clipping {
            normalized.mapv_inplace(|g| g.clamp(-self.max_phase_update, self.max_phase_update));
        }
        
        // Apply momentum
        let momentum_grad = &self.phase_momentum * effective_lr;
        normalized = normalized + momentum_grad;
        
        Ok(normalized)
    }
    
    /// Normalize magnitude gradients
    fn normalize_magnitude_gradients(&self, magnitude_grad: &ArrayD<f32>, learning_rate: f32) -> DLResult<ArrayD<f32>> {
        let effective_lr = self.magnitude_lr * learning_rate;
        
        // Apply gradient scaling
        let mut normalized = magnitude_grad.mapv(|g| g * effective_lr);
        
        // Apply magnitude clipping if enabled
        if self.magnitude_clipping {
            normalized.mapv_inplace(|g| g.clamp(-self.max_magnitude_value, self.max_magnitude_value));
        }
        
        // Apply momentum
        let momentum_grad = &self.magnitude_momentum * effective_lr;
        normalized = normalized + momentum_grad;
        
        Ok(normalized)
    }
    
    /// Recombine normalized gradients into complex tensor
    fn recombine_gradients(&self, magnitude: &ArrayD<f32>, phase: &ArrayD<f32>) -> DLResult<ComplexTensor> {
        ComplexTensor::from_polar(magnitude, phase)
    }
    
    /// Update gradient statistics
    fn update_gradient_statistics(&mut self, magnitude_grad: &ArrayD<f32>, phase_grad: &ArrayD<f32>) -> DLResult<()> {
        // Calculate statistics
        let magnitude_mean = magnitude_grad.iter().sum::<f32>() / magnitude_grad.len() as f32;
        let magnitude_var = magnitude_grad.iter()
            .map(|&g| (g - magnitude_mean).powi(2))
            .sum::<f32>() / magnitude_grad.len() as f32;
        let magnitude_std = magnitude_var.sqrt();
        
        let phase_mean = phase_grad.iter().sum::<f32>() / phase_grad.len() as f32;
        let phase_var = phase_grad.iter()
            .map(|&g| (g - phase_mean).powi(2))
            .sum::<f32>() / phase_grad.len() as f32;
        let phase_std = phase_var.sqrt();
        
        let max_magnitude = magnitude_grad.iter().fold(0.0f32, |acc, &g| acc.max(g.abs()));
        let max_phase = phase_grad.iter().fold(0.0f32, |acc, &g| acc.max(g.abs()));
        
        let gradient_norm = (magnitude_grad.iter().map(|&g| g * g).sum::<f32>() + 
                            phase_grad.iter().map(|&g| g * g).sum::<f32>()).sqrt();
        
        let stats = GradientStatistics {
            magnitude_mean,
            magnitude_std,
            phase_mean,
            phase_std,
            max_magnitude,
            max_phase,
            gradient_norm,
        };
        
        self.gradient_history.push(stats);
        
        // Keep only recent history
        if self.gradient_history.len() > 1000 {
            self.gradient_history.remove(0);
        }
        
        // Update running averages
        self.average_phase_update = self.average_phase_update * 0.9 + phase_mean.abs() * 0.1;
        self.average_magnitude_update = self.average_magnitude_update * 0.9 + magnitude_mean.abs() * 0.1;
        
        Ok(())
    }
    
    /// Update adaptive learning rates
    fn update_adaptive_lr(&mut self) -> DLResult<()> {
        if self.gradient_history.len() < 10 {
            return Ok(());
        }
        
        // Calculate recent gradient statistics
        let recent_stats: Vec<&GradientStatistics> = self.gradient_history
            .iter()
            .rev()
            .take(10)
            .collect();
        
        let avg_magnitude = recent_stats.iter().map(|s| s.magnitude_mean.abs()).sum::<f32>() / 10.0;
        let avg_phase = recent_stats.iter().map(|s| s.phase_mean.abs()).sum::<f32>() / 10.0;
        
        // Adjust learning rates based on gradient magnitudes
        if avg_magnitude > 0.1 {
            self.magnitude_lr *= self.lr_decay_rate;
        } else if avg_magnitude < 0.01 {
            self.magnitude_lr = (self.magnitude_lr / self.lr_decay_rate).min(self.max_lr);
        }
        
        if avg_phase > 0.5 {
            self.phase_lr *= self.lr_decay_rate;
        } else if avg_phase < 0.1 {
            self.phase_lr = (self.phase_lr / self.lr_decay_rate).min(self.max_lr);
        }
        
        // Clamp learning rates
        self.magnitude_lr = self.magnitude_lr.clamp(self.min_lr, self.max_lr);
        self.phase_lr = self.phase_lr.clamp(self.min_lr, self.max_lr);
        
        Ok(())
    }
    
    /// Get gradient statistics
    pub fn get_gradient_statistics(&self) -> Option<&GradientStatistics> {
        self.gradient_history.last()
    }
    
    /// Set learning rates
    pub fn set_learning_rates(&mut self, phase_lr: f32, magnitude_lr: f32) {
        self.phase_lr = phase_lr.clamp(self.min_lr, self.max_lr);
        self.magnitude_lr = magnitude_lr.clamp(self.min_lr, self.max_lr);
    }
    
    /// Reset PGN state
    pub fn reset(&mut self) -> DLResult<()> {
        self.phase_gradients.fill(0.0);
        self.magnitude_gradients.fill(0.0);
        self.phase_momentum.fill(0.0);
        self.magnitude_momentum.fill(0.0);
        self.gradient_history.clear();
        self.average_phase_update = 0.0;
        self.average_magnitude_update = 0.0;
        
        Ok(())
    }
}

/// Resonance Energy Clipping (REC)
#[derive(Debug, Clone)]
pub struct ResonanceEnergyClipping {
    // REC parameters
    energy_clip_threshold: f32,
    energy_decay_factor: f32,
    adaptive_threshold: bool,
    
    // Energy tracking
    energy_history: Vec<f32>,
    current_energy: f32,
    max_energy_seen: f32,
    
    // Statistics
    energy_stats: ResonanceEnergyStats,
    clipping_events: Vec<ClippingEvent>,
    
    // Adaptive parameters
    threshold_adjustment_rate: f32,
    min_threshold: f32,
    max_threshold: f32,
    
    // Normalization
    normalization_method: NormalizationMethod,
    soft_clipping: bool,
    clipping_alpha: f32,
}

#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    L2,
    L1,
    Softmax,
    Tanh,
}

impl Default for NormalizationMethod {
    fn default() -> Self {
        NormalizationMethod::L2
    }
}

/// Clipping event record
#[derive(Debug, Clone)]
pub struct ClippingEvent {
    pub timestamp: u64,
    pub original_energy: f32,
    pub clipped_energy: f32,
    pub clipping_ratio: f32,
    pub threshold_used: f32,
}

impl ResonanceEnergyClipping {
    /// Create new Resonance Energy Clipping
    pub fn new(
        energy_clip_threshold: f32,
        energy_decay_factor: f32,
        adaptive_threshold: bool,
    ) -> DLResult<Self> {
        Ok(Self {
            energy_clip_threshold,
            energy_decay_factor,
            adaptive_threshold,
            energy_history: Vec::new(),
            current_energy: 0.0,
            max_energy_seen: 0.0,
            energy_stats: ResonanceEnergyStats::default(),
            clipping_events: Vec::new(),
            threshold_adjustment_rate: 0.01,
            min_threshold: 0.1,
            max_threshold: 100.0,
            normalization_method: NormalizationMethod::L2,
            soft_clipping: true,
            clipping_alpha: 2.0,
        })
    }
    
    /// Apply REC to resonance data
    pub fn apply_rec(&mut self, resonance_data: &mut ArrayD<f32>) -> DLResult<ClippingEvent> {
        // Calculate current energy
        let original_energy = self.calculate_energy(resonance_data);
        self.current_energy = original_energy;
        
        // Update energy history
        self.energy_history.push(original_energy);
        if self.energy_history.len() > 1000 {
            self.energy_history.remove(0);
        }
        
        // Update max energy seen
        self.max_energy_seen = self.max_energy_seen.max(original_energy);
        
        // Determine if clipping is needed
        let effective_threshold = if self.adaptive_threshold {
            self.calculate_adaptive_threshold()?
        } else {
            self.energy_clip_threshold
        };
        
        let mut clipped_energy = original_energy;
        let mut clipping_ratio = 0.0;
        
        if original_energy > effective_threshold {
            // Apply energy clipping
            clipping_ratio = 1.0 - effective_threshold / original_energy;
            
            if self.soft_clipping {
                self.apply_soft_clipping(resonance_data, effective_threshold)?;
            } else {
                self.apply_hard_clipping(resonance_data, effective_threshold)?;
            }
            
            // Recalculate energy after clipping
            clipped_energy = self.calculate_energy(resonance_data);
            
            // Record clipping event
            let event = ClippingEvent {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_secs(),
                original_energy,
                clipped_energy,
                clipping_ratio,
                threshold_used: effective_threshold,
            };
            
            self.clipping_events.push(event);
            
            // Keep only recent events
            if self.clipping_events.len() > 1000 {
                self.clipping_events.remove(0);
            }
            
            // Update statistics
            self.energy_stats.clipping_count += 1;
        }
        
        // Update energy statistics
        self.update_energy_statistics(original_energy, clipped_energy, clipping_ratio)?;
        
        // Adjust adaptive threshold
        if self.adaptive_threshold {
            self.adjust_adaptive_threshold()?;
        }
        
        // Return clipping event
        Ok(ClippingEvent {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_secs(),
            original_energy,
            clipped_energy,
            clipping_ratio,
            threshold_used: effective_threshold,
        })
    }
    
    /// Calculate energy of resonance data
    fn calculate_energy(&self, data: &ArrayD<f32>) -> f32 {
        match self.normalization_method {
            NormalizationMethod::L2 => {
                data.iter().map(|&x| x * x).sum::<f32>().sqrt()
            }
            NormalizationMethod::L1 => {
                data.iter().map(|&x| x.abs()).sum()
            }
            NormalizationMethod::Softmax => {
                let max_val = data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let exp_sum: f32 = data.iter().map(|&x| (x - max_val).exp()).sum();
                exp_sum.ln()
            }
            NormalizationMethod::Tanh => {
                data.iter().map(|&x| x.tanh().abs()).sum()
            }
        }
    }
    
    /// Calculate adaptive threshold based on energy history
    fn calculate_adaptive_threshold(&self) -> DLResult<f32> {
        if self.energy_history.len() < 10 {
            return Ok(self.energy_clip_threshold);
        }
        
        // Calculate statistics from recent history
        let recent_energies: Vec<f32> = self.energy_history
            .iter()
            .rev()
            .take(50)
            .cloned()
            .collect();
        
        let mean_energy = recent_energies.iter().sum::<f32>() / recent_energies.len() as f32;
        let std_energy = (recent_energies.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>() / recent_energies.len() as f32).sqrt();
        
        // Set threshold at mean + 2*std
        let adaptive_threshold = mean_energy + 2.0 * std_energy;
        
        Ok(adaptive_threshold.clamp(self.min_threshold, self.max_threshold))
    }
    
    /// Apply soft clipping to resonance data
    fn apply_soft_clipping(&self, data: &mut ArrayD<f32>, threshold: f32) -> DLResult<()> {
        let current_energy = self.calculate_energy(data);
        
        if current_energy <= threshold {
            return Ok(());
        }
        
        // Apply soft clipping using tanh
        let scale_factor = threshold / current_energy;
        data.mapv_inplace(|x| x.tanh() * scale_factor * current_energy);
        
        Ok(())
    }
    
    /// Apply hard clipping to resonance data
    fn apply_hard_clipping(&self, data: &mut ArrayD<f32>, threshold: f32) -> DLResult<()> {
        let current_energy = self.calculate_energy(data);
        
        if current_energy <= threshold {
            return Ok(());
        }
        
        // Apply hard clipping with normalization
        let scale_factor = threshold / current_energy;
        data.mapv_inplace(|x| x * scale_factor);
        
        Ok(())
    }
    
    /// Update energy statistics
    fn update_energy_statistics(&mut self, original_energy: f32, clipped_energy: f32, clipping_ratio: f32) -> DLResult<()> {
        self.energy_stats.current_energy = clipped_energy;
        
        // Update average energy
        if self.energy_stats.average_energy == 0.0 {
            self.energy_stats.average_energy = clipped_energy;
        } else {
            self.energy_stats.average_energy = self.energy_stats.average_energy * 0.99 + clipped_energy * 0.01;
        }
        
        // Update max energy
        self.energy_stats.max_energy = self.energy_stats.max_energy.max(clipped_energy);
        
        // Update energy variance
        if !self.energy_history.is_empty() {
            let mean = self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32;
            let variance = self.energy_history.iter()
                .map(|&e| (e - mean).powi(2))
                .sum::<f32>() / self.energy_history.len() as f32;
            self.energy_stats.energy_variance = variance;
        }
        
        // Update clipping ratio
        if clipping_ratio > 0.0 {
            self.energy_stats.clipping_ratio = self.energy_stats.clipping_ratio * 0.99 + clipping_ratio * 0.01;
        }
        
        Ok(())
    }
    
    /// Adjust adaptive threshold based on clipping frequency
    fn adjust_adaptive_threshold(&mut self) -> DLResult<()> {
        if self.clipping_events.len() < 100 {
            return Ok(());
        }
        
        // Calculate recent clipping frequency
        let recent_events: Vec<&ClippingEvent> = self.clipping_events
            .iter()
            .rev()
            .take(100)
            .collect();
        
        let clipping_frequency = recent_events.iter()
            .filter(|e| e.clipping_ratio > 0.0)
            .count() as f32 / recent_events.len() as f32;
        
        // Adjust threshold based on clipping frequency
        if clipping_frequency > 0.2 {
            // Too much clipping, increase threshold
            self.energy_clip_threshold *= 1.0 + self.threshold_adjustment_rate;
        } else if clipping_frequency < 0.05 {
            // Too little clipping, decrease threshold
            self.energy_clip_threshold *= 1.0 - self.threshold_adjustment_rate;
        }
        
        // Clamp threshold
        self.energy_clip_threshold = self.energy_clip_threshold.clamp(self.min_threshold, self.max_threshold);
        
        Ok(())
    }
    
    /// Get energy statistics
    pub fn get_energy_statistics(&self) -> &ResonanceEnergyStats {
        &self.energy_stats
    }
    
    /// Get recent clipping events
    pub fn get_recent_clipping_events(&self, count: usize) -> &[ClippingEvent] {
        let start = if self.clipping_events.len() > count {
            self.clipping_events.len() - count
        } else {
            0
        };
        
        &self.clipping_events[start..]
    }
    
    /// Set energy clip threshold
    pub fn set_energy_clip_threshold(&mut self, threshold: f32) {
        self.energy_clip_threshold = threshold.clamp(self.min_threshold, self.max_threshold);
    }
    
    /// Set normalization method
    pub fn set_normalization_method(&mut self, method: NormalizationMethod) {
        self.normalization_method = method;
    }
    
    /// Enable/disable soft clipping
    pub fn set_soft_clipping(&mut self, soft_clipping: bool) {
        self.soft_clipping = soft_clipping;
    }
    
    /// Reset REC state
    pub fn reset(&mut self) -> DLResult<()> {
        self.energy_history.clear();
        self.current_energy = 0.0;
        self.max_energy_seen = 0.0;
        self.energy_stats = ResonanceEnergyStats::default();
        self.clipping_events.clear();
        
        Ok(())
    }
}

/// Combined Training Stabilizer
#[derive(Debug, Clone)]
pub struct TrainingStabilizer {
    pub pgn: PolarGradientNormalization,
    pub rec: ResonanceEnergyClipping,
    
    // Combined statistics
    training_steps: u64,
    stabilization_events: Vec<StabilizationEvent>,
    
    // Performance metrics
    average_gradient_norm: f32,
    average_energy: f32,
    stabilization_efficiency: f32,
}

/// Stabilization event record
#[derive(Debug, Clone)]
pub struct StabilizationEvent {
    pub timestamp: u64,
    pub training_step: u64,
    pub gradient_norm: f32,
    pub resonance_energy: f32,
    pub pgn_applied: bool,
    pub rec_applied: bool,
    pub clipping_ratio: f32,
}

impl TrainingStabilizer {
    /// Create new Training Stabilizer
    pub fn new(
        parameter_shape: Vec<usize>,
        pgn_config: (f32, f32, f32), // (phase_lr, magnitude_lr, max_phase_update)
        rec_config: (f32, f32, bool), // (energy_threshold, decay_factor, adaptive)
    ) -> DLResult<Self> {
        let pgn = PolarGradientNormalization::new(
            parameter_shape.clone(),
            pgn_config.0,
            pgn_config.1,
            pgn_config.2,
        )?;
        
        let rec = ResonanceEnergyClipping::new(
            rec_config.0,
            rec_config.1,
            rec_config.2,
        )?;
        
        Ok(Self {
            pgn,
            rec,
            training_steps: 0,
            stabilization_events: Vec::new(),
            average_gradient_norm: 0.0,
            average_energy: 0.0,
            stabilization_efficiency: 0.0,
        })
    }
    
    /// Apply complete stabilization
    pub fn stabilize(&mut self, 
                     complex_gradients: &ComplexTensor, 
                     resonance_data: &mut ArrayD<f32>,
                     learning_rate: f32) -> DLResult<(ComplexTensor, ClippingEvent)> {
        self.training_steps += 1;
        
        // Apply PGN to gradients
        let normalized_gradients = self.pgn.apply_pgn(complex_gradients, learning_rate)?;
        
        // Apply REC to resonance data
        let clipping_event = self.rec.apply_rec(resonance_data)?;
        
        // Record stabilization event
        let gradient_stats = self.pgn.get_gradient_statistics().expect("gradient statistics available");
        let event = StabilizationEvent {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_secs(),
            training_step: self.training_steps,
            gradient_norm: gradient_stats.gradient_norm,
            resonance_energy: clipping_event.clipped_energy,
            pgn_applied: true,
            rec_applied: clipping_event.clipping_ratio > 0.0,
            clipping_ratio: clipping_event.clipping_ratio,
        };
        
        self.stabilization_events.push(event);
        
        // Keep only recent events
        if self.stabilization_events.len() > 1000 {
            self.stabilization_events.remove(0);
        }
        
        // Update performance metrics
        self.update_performance_metrics()?;
        
        Ok((normalized_gradients, clipping_event))
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self) -> DLResult<()> {
        if let Some(gradient_stats) = self.pgn.get_gradient_statistics() {
            self.average_gradient_norm = self.average_gradient_norm * 0.99 + gradient_stats.gradient_norm * 0.01;
        }
        
        self.average_energy = self.average_energy * 0.99 + self.rec.current_energy * 0.01;
        
        // Calculate stabilization efficiency
        if !self.stabilization_events.is_empty() {
            let recent_events: Vec<&StabilizationEvent> = self.stabilization_events
                .iter()
                .rev()
                .take(100)
                .collect();
            
            let pgn_applied_ratio = recent_events.iter()
                .filter(|e| e.pgn_applied)
                .count() as f32 / recent_events.len() as f32;
            
            let rec_applied_ratio = recent_events.iter()
                .filter(|e| e.rec_applied)
                .count() as f32 / recent_events.len() as f32;
            
            self.stabilization_efficiency = (pgn_applied_ratio + rec_applied_ratio) / 2.0;
        }
        
        Ok(())
    }
    
    /// Get combined statistics
    pub fn get_combined_statistics(&self) -> CombinedStabilizationStats {
        CombinedStabilizationStats {
            training_steps: self.training_steps,
            average_gradient_norm: self.average_gradient_norm,
            average_energy: self.average_energy,
            stabilization_efficiency: self.stabilization_efficiency,
            pgn_stats: self.pgn.get_gradient_statistics().cloned().unwrap_or_default(),
            rec_stats: self.rec.get_energy_statistics().clone(),
        }
    }
    
    /// Reset all stabilization state
    pub fn reset(&mut self) -> DLResult<()> {
        self.pgn.reset()?;
        self.rec.reset()?;
        self.training_steps = 0;
        self.stabilization_events.clear();
        self.average_gradient_norm = 0.0;
        self.average_energy = 0.0;
        self.stabilization_efficiency = 0.0;
        
        Ok(())
    }
}

/// Combined stabilization statistics
#[derive(Debug, Clone)]
pub struct CombinedStabilizationStats {
    pub training_steps: u64,
    pub average_gradient_norm: f32,
    pub average_energy: f32,
    pub stabilization_efficiency: f32,
    pub pgn_stats: GradientStatistics,
    pub rec_stats: ResonanceEnergyStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    
    #[test]
    fn test_pgn_creation() {
        let pgn = PolarGradientNormalization::new(vec![10, 10], 0.01, 0.001, 0.1).unwrap();
        assert_eq!(pgn.phase_lr, 0.01);
        assert_eq!(pgn.magnitude_lr, 0.001);
        assert_eq!(pgn.max_phase_update, 0.1);
    }
    
    #[test]
    fn test_rec_creation() {
        let rec = ResonanceEnergyClipping::new(10.0, 0.99, true).unwrap();
        assert_eq!(rec.energy_clip_threshold, 10.0);
        assert_eq!(rec.energy_decay_factor, 0.99);
        assert!(rec.adaptive_threshold);
    }
    
    #[test]
    fn test_energy_calculation() {
        let rec = ResonanceEnergyClipping::new(10.0, 0.99, false).unwrap();
        
        let data = ArrayD::from_shape_vec(vec![2], vec![3.0, 4.0]).unwrap(); // Should have energy 5
        let energy = rec.calculate_energy(&data);
        
        assert!((energy - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_training_stabilizer_creation() {
        let stabilizer = TrainingStabilizer::new(
            vec![10, 10],
            (0.01, 0.001, 0.1),
            (10.0, 0.99, true),
        ).unwrap();
        
        assert_eq!(stabilizer.training_steps, 0);
        assert_eq!(stabilizer.average_gradient_norm, 0.0);
        assert_eq!(stabilizer.average_energy, 0.0);
    }
    
    #[test]
    fn test_clipping_event_creation() {
        let event = ClippingEvent {
            timestamp: 123456789,
            original_energy: 15.0,
            clipped_energy: 10.0,
            clipping_ratio: 0.333,
            threshold_used: 10.0,
        };
        
        assert_eq!(event.original_energy, 15.0);
        assert_eq!(event.clipped_energy, 10.0);
        assert!((event.clipping_ratio - 0.333).abs() < 1e-3);
    }
}
