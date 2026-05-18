use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use super::{Tensor, Adam};
use super::mixed_precision::{LossScaler, DType};

// ─── Optimizer State ─────────────────────────────────────────────────────────

/// Serializable optimizer state for checkpoint resume
#[derive(Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub step: usize,
}

impl OptimizerState {
    pub fn from_adam(adam: &Adam) -> Self {
        Self {
            m: adam.m.iter().map(|arr| arr.iter().copied().collect()).collect(),
            v: adam.v.iter().map(|arr| arr.iter().copied().collect()).collect(),
            step: adam.step,
        }
    }

    pub fn apply_to_adam(&self, adam: &mut Adam, shapes: &[Vec<usize>]) {
        for (i, shape) in shapes.iter().enumerate() {
            if i < self.m.len() {
                let m_arr = ArrayD::from_shape_vec(shape.clone(), self.m[i].clone())
                    .expect("shape mismatch restoring Adam m");
                let v_arr = ArrayD::from_shape_vec(shape.clone(), self.v[i].clone())
                    .expect("shape mismatch restoring Adam v");
                if i < adam.m.len() {
                    adam.m[i] = m_arr;
                }
                if i < adam.v.len() {
                    adam.v[i] = v_arr;
                }
            }
        }
        adam.step = self.step;
    }
}

// ─── Checkpoint ──────────────────────────────────────────────────────────────

/// Full training state for save/load
#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub step: usize,
    pub epoch: usize,
    pub best_val_loss: Option<f64>,
    pub loss_scaler_scale: Option<f32>,
    pub model_params: Vec<Vec<f32>>,
    pub model_shapes: Vec<Vec<usize>>,
    pub optimizer_state: Option<OptimizerState>,
}

impl Checkpoint {
    /// Save checkpoint to disk
    pub fn save(
        path: impl AsRef<Path>,
        params: &[Tensor],
        adam: &Adam,
        step: usize,
        epoch: usize,
        best_val_loss: Option<f64>,
        loss_scaler: Option<&LossScaler>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ckpt = Self {
            step,
            epoch,
            best_val_loss,
            loss_scaler_scale: loss_scaler.map(|ls| ls.scale()),
            model_params: params.iter().map(|p| p.data().iter().copied().collect()).collect(),
            model_shapes: params.iter().map(|p| p.shape()).collect(),
            optimizer_state: Some(OptimizerState::from_adam(adam)),
        };
        let json = serde_json::to_string_pretty(&ckpt)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load checkpoint from disk
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let ckpt: Self = serde_json::from_str(&json)?;
        Ok(ckpt)
    }

    /// Apply saved weights to model parameters
    pub fn restore_params(&self, params: &[Tensor]) {
        for (i, p) in params.iter().enumerate() {
            if i < self.model_params.len() && i < self.model_shapes.len() {
                let arr = ArrayD::from_shape_vec(
                    self.model_shapes[i].clone(),
                    self.model_params[i].clone(),
                ).expect("shape mismatch restoring params");
                p.set_data(arr);
            }
        }
    }

    /// Restore optimizer state
    pub fn restore_optimizer(&self, adam: &mut Adam) {
        if let Some(ref opt_state) = self.optimizer_state {
            let shapes: Vec<Vec<usize>> = adam.parameters.iter().map(|p| p.shape()).collect();
            opt_state.apply_to_adam(adam, &shapes);
        }
    }
}

// ─── Metrics Tracker ─────────────────────────────────────────────────────────

/// Training metrics collected over time
#[derive(Clone, Debug)]
pub struct TrainingMetrics {
    pub steps: Vec<usize>,
    pub losses: Vec<f64>,
    pub learning_rates: Vec<f32>,
    pub grad_norms: Vec<f32>,
    pub throughputs: Vec<f32>, // tokens/second
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            losses: Vec::new(),
            learning_rates: Vec::new(),
            grad_norms: Vec::new(),
            throughputs: Vec::new(),
        }
    }

    pub fn record(&mut self, step: usize, loss: f64, lr: f32, grad_norm: f32, throughput: f32) {
        self.steps.push(step);
        self.losses.push(loss);
        self.learning_rates.push(lr);
        self.grad_norms.push(grad_norm);
        self.throughputs.push(throughput);
    }

    pub fn last_loss(&self) -> Option<f64> {
        self.losses.last().copied()
    }

    pub fn best_loss(&self) -> Option<f64> {
        self.losses.iter().copied().fold(None, |best, x| {
            Some(best.map_or(x, |b| b.min(x)))
        })
    }

    pub fn avg_loss(&self, window: usize) -> Option<f64> {
        let n = self.losses.len();
        if n == 0 {
            return None;
        }
        let start = n.saturating_sub(window);
        let sum: f64 = self.losses[start..].iter().sum();
        Some(sum / (n - start) as f64)
    }

    pub fn smoothed_loss(&self, alpha: f64) -> Option<f64> {
        self.losses.iter().copied().reduce(|a, b| alpha * b + (1.0 - alpha) * a)
    }

    pub fn avg_throughput(&self) -> Option<f32> {
        let n = self.throughputs.len();
        if n == 0 {
            return None;
        }
        Some(self.throughputs.iter().sum::<f32>() / n as f32)
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ─── Training Loop Config ────────────────────────────────────────────────────

/// High-level training configuration
#[derive(Clone, Debug)]
pub struct TrainingLoopConfig {
    pub epochs: usize,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub save_interval: usize,
    pub early_stop_patience: usize,
    pub max_steps: usize,
    pub checkpoint_dir: Option<PathBuf>,
    pub resume_from: Option<PathBuf>,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            log_interval: 10,
            eval_interval: 100,
            save_interval: 1000,
            early_stop_patience: 5,
            max_steps: usize::MAX,
            checkpoint_dir: None,
            resume_from: None,
        }
    }
}

// ─── Training Loop ───────────────────────────────────────────────────────────

/// Core training loop. Generic over model forward + loss computation.
pub struct TrainingLoop {
    pub config: TrainingLoopConfig,
    pub metrics: TrainingMetrics,
    pub step: usize,
    pub epoch: usize,
    pub best_val_loss: Option<f64>,
    pub patience_counter: usize,
    pub total_tokens: usize,
    pub start_time: Option<Instant>,
    pub stop_flag: Arc<AtomicBool>,
}

impl TrainingLoop {
    pub fn new(config: TrainingLoopConfig) -> Self {
        Self {
            config,
            metrics: TrainingMetrics::new(),
            step: 0,
            epoch: 0,
            best_val_loss: None,
            patience_counter: 0,
            total_tokens: 0,
            start_time: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Load state from checkpoint
    pub fn resume(&mut self, ckpt: &Checkpoint) {
        self.step = ckpt.step;
        self.epoch = ckpt.epoch;
        self.best_val_loss = ckpt.best_val_loss;
    }

    /// Create a shared stop signal
    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stop_flag.clone()
    }

    /// Request graceful stop
    pub fn request_stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    /// Call at start of training
    pub fn on_start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Call each optimizer step.
    /// Returns true if training should continue, false if stopped.
    pub fn on_step(
        &mut self,
        loss: f64,
        lr: f32,
        grad_norm: f32,
        tokens_in_batch: usize,
    ) -> bool {
        self.step += 1;
        self.total_tokens += tokens_in_batch;

        // Throughput
        let elapsed = self.start_time.map(|t| t.elapsed().as_secs_f32()).unwrap_or(1.0);
        let throughput = self.total_tokens as f32 / elapsed;

        self.metrics.record(self.step, loss, lr, grad_norm, throughput);

        // Logging
        if self.step % self.config.log_interval == 0 {
            let avg = self.metrics.avg_loss(self.config.log_interval).unwrap_or(loss);
            println!(
                "[Step {}] loss={:.4} (avg={:.4}) lr={:.2e} grad_norm={:.4} tok/s={:.0}",
                self.step, loss, avg, lr, grad_norm, throughput
            );
        }

        // Save checkpoint
        if self.step % self.config.save_interval == 0 {
            if let Some(ref dir) = self.config.checkpoint_dir {
                let path = dir.join(format!("checkpoint-{}.json", self.step));
                println!("  Saving checkpoint to {:?}", path);
                // Note: actual save needs model params + optimizer reference
                // which are passed separately
            }
        }

        // Stop check
        if self.step >= self.config.max_steps || self.should_stop() {
            return false;
        }

        true
    }

    /// Update validation loss, returns true if improved
    pub fn on_eval(&mut self, val_loss: f64) -> bool {
        match self.best_val_loss {
            Some(best) if val_loss < best => {
                self.best_val_loss = Some(val_loss);
                self.patience_counter = 0;
                println!("  Validation loss improved to {:.4}", val_loss);
                true
            }
            None => {
                self.best_val_loss = Some(val_loss);
                self.patience_counter = 0;
                println!("  Initial validation loss: {:.4}", val_loss);
                true
            }
            Some(_) => {
                self.patience_counter += 1;
                if self.patience_counter >= self.config.early_stop_patience {
                    println!("  Early stopping triggered (patience={})", self.config.early_stop_patience);
                }
                false
            }
        }
    }

    /// Early stopping triggered
    pub fn should_early_stop(&self) -> bool {
        self.patience_counter >= self.config.early_stop_patience
    }

    /// Call at end of epoch
    pub fn on_epoch_end(&mut self) {
        self.epoch += 1;
        let avg = self.metrics.avg_loss(self.config.log_interval).unwrap_or(0.0);
        println!("=== Epoch {} complete | avg_loss={:.4} | best_val={:?} ===", self.epoch, avg, self.best_val_loss);
    }

    /// Check external stop signal
    pub fn should_stop(&self) -> bool {
        self.stop_flag.load(Ordering::SeqCst)
    }

    /// Reset stop signal
    pub fn reset_stop(&self) {
        self.stop_flag.store(false, Ordering::SeqCst);
    }

    /// Training duration in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
    }

    /// Summary report
    pub fn report(&self) {
        println!("═══════════════════════════════════════");
        println!("  Training complete");
        println!("  Steps: {}", self.step);
        println!("  Epochs: {}", self.epoch);
        println!("  Total tokens: {}", self.total_tokens);
        println!("  Duration: {:.1}s", self.elapsed_secs());
        println!("  Avg throughput: {:.0} tok/s", self.metrics.avg_throughput().unwrap_or(0.0));
        println!("  Best loss: {:?}", self.best_val_loss);
        println!("  Final loss: {:?}", self.metrics.last_loss().unwrap_or(0.0));
        println!("═══════════════════════════════════════");
    }
}

/// Compute L2 norm of gradients across all parameters
pub fn compute_grad_norm(params: &[Tensor]) -> f32 {
    let mut sum_sq = 0.0f32;
    for p in params {
        if let Some(g) = p.grad() {
            sum_sq += g.iter().map(|x| x * x).sum::<f32>();
        }
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Tensor, TensorOps, Linear, Module};

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let w = Tensor::randn(&[4, 2], true);
        let b = Tensor::randn(&[2], true);
        let params = vec![w.clone(), b.clone()];

        let adam = Adam::new(params.clone(), 0.001);
        let ckpt = Checkpoint {
            step: 100,
            epoch: 2,
            best_val_loss: Some(0.5),
            loss_scaler_scale: Some(65536.0),
            model_params: params.iter().map(|p| p.data().iter().copied().collect()).collect(),
            model_shapes: params.iter().map(|p| p.shape()).collect(),
            optimizer_state: Some(OptimizerState::from_adam(&adam)),
        };

        let json = serde_json::to_string_pretty(&ckpt).unwrap();
        let loaded: Checkpoint = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.epoch, 2);
        assert_eq!(loaded.best_val_loss, Some(0.5));
        assert_eq!(loaded.loss_scaler_scale, Some(65536.0));
        assert!(loaded.optimizer_state.is_some());
    }

    #[test]
    fn test_metrics_tracker() {
        let mut m = TrainingMetrics::new();
        assert!(m.is_empty());

        m.record(1, 0.5, 0.001, 0.1, 1000.0);
        m.record(2, 0.4, 0.0009, 0.08, 1100.0);
        m.record(3, 0.3, 0.0008, 0.06, 1200.0);

        assert!(!m.is_empty());
        assert!((m.last_loss().unwrap() - 0.3).abs() < 1e-6);
        assert!((m.best_loss().unwrap() - 0.3).abs() < 1e-6);
        assert!((m.avg_loss(2).unwrap() - 0.35).abs() < 1e-6);
        // EWMA: alpha=0.5 => 0.5*0.5 + 0.5*0.4 = 0.45; then 0.5*0.3 + 0.5*0.45 = 0.375
        assert!((m.smoothed_loss(0.5).unwrap() - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_training_loop_basics() {
        let config = TrainingLoopConfig {
            log_interval: 1,
            max_steps: 10,
            ..Default::default()
        };
        let mut loop_ = TrainingLoop::new(config);
        loop_.on_start();

        for _ in 0..5 {
            let cont = loop_.on_step(0.5, 0.001, 0.1, 32);
            assert!(cont);
        }

        assert_eq!(loop_.step, 5);
        assert_eq!(loop_.total_tokens, 160);
        assert!(!loop_.should_stop());
    }

    #[test]
    fn test_training_loop_stop_signal() {
        let mut loop_ = TrainingLoop::new(TrainingLoopConfig::default());
        loop_.on_start();
        loop_.request_stop();
        assert!(loop_.should_stop());
        loop_.reset_stop();
        assert!(!loop_.should_stop());
    }

    #[test]
    fn test_training_loop_max_steps() {
        let config = TrainingLoopConfig {
            max_steps: 3,
            log_interval: 1,
            ..Default::default()
        };
        let mut loop_ = TrainingLoop::new(config);
        loop_.on_start();

        assert!(loop_.on_step(0.5, 0.001, 0.1, 32));
        assert!(loop_.on_step(0.4, 0.001, 0.1, 32));
        let cont = loop_.on_step(0.3, 0.001, 0.1, 32);
        assert!(!cont); // max_steps reached
    }

    #[test]
    fn test_training_loop_early_stop() {
        let config = TrainingLoopConfig {
            early_stop_patience: 2,
            ..Default::default()
        };
        let mut loop_ = TrainingLoop::new(config);
        loop_.on_eval(1.0);
        loop_.on_eval(1.0);
        assert!(!loop_.should_early_stop());
        loop_.on_eval(1.0);
        assert!(loop_.should_early_stop());
        assert_eq!(loop_.patience_counter, 2);
    }

    #[test]
    fn test_training_loop_eval_improvement() {
        let mut loop_ = TrainingLoop::new(TrainingLoopConfig::default());
        assert!(loop_.on_eval(0.5)); // first eval, improves from None
        assert!(loop_.on_eval(0.3)); // improves
        assert!(!loop_.on_eval(0.4)); // worsened
        assert!(!loop_.on_eval(0.35)); // still worse than best (0.3)
    }

    #[test]
    fn test_compute_grad_norm() {
        let w = Tensor::randn(&[4, 2], true);
        let x = Tensor::randn(&[2, 4], false);
        let y = x.matmul(&w).sum();
        y.backward();

        let norm = compute_grad_norm(&[w.clone()]);
        assert!(norm > 0.0);
        assert!(!norm.is_nan());
    }

    #[test]
    fn test_resume_from_checkpoint() {
        let config = TrainingLoopConfig::default();
        let mut loop_ = TrainingLoop::new(config);

        let ckpt = Checkpoint {
            step: 50,
            epoch: 2,
            best_val_loss: Some(0.3),
            loss_scaler_scale: None,
            model_params: vec![],
            model_shapes: vec![],
            optimizer_state: None,
        };
        loop_.resume(&ckpt);
        assert_eq!(loop_.step, 50);
        assert_eq!(loop_.epoch, 2);
        assert_eq!(loop_.best_val_loss, Some(0.3));
    }
}
