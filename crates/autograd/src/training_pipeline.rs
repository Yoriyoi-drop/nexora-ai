use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use super::mixed_precision::LossScaler;
use super::{Adam, Tensor};

#[cfg(feature = "gpu")]
use super::gpu::{GpuContext, GpuTensor};

// ─── Optimizer State ─────────────────────────────────────────────────────────

/// Serializable optimizer state for checkpoint resume (CPU)
#[derive(Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub step: usize,
}

/// GPU-resident optimizer state for persistent GPU residency
#[cfg(feature = "gpu")]
pub struct GpuOptimizerState {
    pub m: Vec<GpuTensor>,
    pub v: Vec<GpuTensor>,
    pub step: usize,
}

#[cfg(feature = "gpu")]
impl GpuOptimizerState {
    /// Create GPU-resident optimizer state from CPU state
    pub fn from_cpu(
        cpu_state: &OptimizerState,
        ctx: &GpuContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let m = cpu_state
            .m
            .iter()
            .map(|arr| {
                let arr_d = ArrayD::from_shape_vec(vec![arr.len()], arr.clone())?;
                GpuTensor::from_cpu(&arr_d).map_err(|e| Box::from(e) as Box<dyn std::error::Error>)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let v = cpu_state
            .v
            .iter()
            .map(|arr| {
                let arr_d = ArrayD::from_shape_vec(vec![arr.len()], arr.clone())?;
                GpuTensor::from_cpu(&arr_d).map_err(|e| Box::from(e) as Box<dyn std::error::Error>)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            m,
            v,
            step: cpu_state.step,
        })
    }

    /// Convert to CPU state for checkpointing
    /// Panics on GPU readback failure — checkpoint cannot proceed without valid data.
    pub fn to_cpu(&self) -> OptimizerState {
        fn gpu_tensor_to_vec(t: &GpuTensor) -> Vec<f32> {
            match t.to_cpu() {
                Ok(cpu) => cpu.as_slice().map(|s| s.to_vec()).unwrap_or_else(|| {
                    tracing::warn!(
                        "GpuOptimizerState::to_cpu: non-contiguous tensor, falling back to iter"
                    );
                    cpu.iter().copied().collect()
                }),
                Err(e) => {
                    tracing::error!(
                        "FATAL: GpuOptimizerState::to_cpu GPU readback failed: {e}. \
                         Checkpoint save cannot continue."
                    );
                    panic!("GpuOptimizerState::to_cpu GPU readback failed: {e}");
                }
            }
        }
        OptimizerState {
            m: self.m.iter().map(|t| gpu_tensor_to_vec(t)).collect(),
            v: self.v.iter().map(|t| gpu_tensor_to_vec(t)).collect(),
            step: self.step,
        }
    }

    /// Create GPU-resident optimizer state directly from Adam optimizer
    pub fn from_adam_gpu(
        adam: &Adam,
        ctx: &GpuContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let m = adam
            .m
            .iter()
            .map(|arr| {
                GpuTensor::from_cpu(arr).map_err(|e| Box::from(e) as Box<dyn std::error::Error>)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let v = adam
            .v
            .iter()
            .map(|arr| {
                GpuTensor::from_cpu(arr).map_err(|e| Box::from(e) as Box<dyn std::error::Error>)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            m,
            v,
            step: adam.step,
        })
    }

    /// Apply GPU-resident state to Adam optimizer (for CPU fallback)
    pub fn apply_to_adam(&self, adam: &mut Adam) {
        let cpu_state = self.to_cpu();
        let shapes: Vec<Vec<usize>> = adam.parameters.iter().map(|p| p.shape()).collect();
        cpu_state.apply_to_adam(adam, &shapes);
    }
}

impl OptimizerState {
    pub fn from_adam(adam: &Adam) -> Self {
        Self {
            m: adam
                .m
                .iter()
                .map(|arr| arr.iter().copied().collect())
                .collect(),
            v: adam
                .v
                .iter()
                .map(|arr| arr.iter().copied().collect())
                .collect(),
            step: adam.step,
        }
    }

    /// Apply optimizer state to Adam parameters.
    /// Logs error and skips mismatched entries instead of panicking.
    pub fn apply_to_adam(&self, adam: &mut Adam, shapes: &[Vec<usize>]) {
        for (i, shape) in shapes.iter().enumerate() {
            if i < self.m.len() {
                let m_arr = match ArrayD::from_shape_vec(shape.clone(), self.m[i].clone()) {
                    Ok(arr) => arr,
                    Err(e) => {
                        tracing::error!(
                            "Adam m shape mismatch at index {i}: expected shape {shape:?} but \
                             data length {} doesn't match product {}: {e}. Skipping.",
                            self.m[i].len(),
                            shape.iter().product::<usize>()
                        );
                        continue;
                    }
                };
                let v_arr = match ArrayD::from_shape_vec(shape.clone(), self.v[i].clone()) {
                    Ok(arr) => arr,
                    Err(e) => {
                        tracing::error!(
                            "Adam v shape mismatch at index {i}: expected shape {shape:?} but \
                             data length {} doesn't match product {}: {e}. Skipping.",
                            self.v[i].len(),
                            shape.iter().product::<usize>()
                        );
                        continue;
                    }
                };
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

/// Full training state for save/load (CPU)
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

/// GPU-resident checkpoint for persistent GPU residency
#[cfg(feature = "gpu")]
pub struct GpuCheckpoint {
    pub step: usize,
    pub epoch: usize,
    pub best_val_loss: Option<f64>,
    pub loss_scaler_scale: Option<f32>,
    pub model_params: Vec<GpuTensor>,
    pub model_shapes: Vec<Vec<usize>>,
    pub optimizer_state: Option<GpuOptimizerState>,
}

#[cfg(feature = "gpu")]
impl GpuCheckpoint {
    /// Create GPU-resident checkpoint from CPU checkpoint
    pub fn from_cpu(
        cpu_ckpt: &Checkpoint,
        ctx: &GpuContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model_params = cpu_ckpt
            .model_params
            .iter()
            .zip(&cpu_ckpt.model_shapes)
            .map(|(params, shape)| {
                let arr_d = ArrayD::from_shape_vec(shape.clone(), params.clone())?;
                GpuTensor::from_cpu(&arr_d).map_err(|e| Box::from(e) as Box<dyn std::error::Error>)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let optimizer_state = cpu_ckpt
            .optimizer_state
            .as_ref()
            .map(|opt| GpuOptimizerState::from_cpu(opt, ctx))
            .transpose()?;

        Ok(Self {
            step: cpu_ckpt.step,
            epoch: cpu_ckpt.epoch,
            best_val_loss: cpu_ckpt.best_val_loss,
            loss_scaler_scale: cpu_ckpt.loss_scaler_scale,
            model_params,
            model_shapes: cpu_ckpt.model_shapes.clone(),
            optimizer_state,
        })
    }

    /// Convert to CPU checkpoint for serialization
    /// Panics on GPU readback failure — checkpoint cannot proceed without valid data.
    pub fn to_cpu(&self) -> Checkpoint {
        fn gpu_tensor_to_vec(t: &GpuTensor) -> Vec<f32> {
            match t.to_cpu() {
                Ok(cpu) => cpu.as_slice().map(|s| s.to_vec()).unwrap_or_else(|| {
                    tracing::warn!(
                        "CheckpointGpu::to_cpu: non-contiguous tensor, falling back to iter"
                    );
                    cpu.iter().copied().collect()
                }),
                Err(e) => {
                    tracing::error!(
                        "FATAL: CheckpointGpu::to_cpu GPU readback failed: {e}. \
                         Checkpoint save cannot continue."
                    );
                    panic!("CheckpointGpu::to_cpu GPU readback failed: {e}");
                }
            }
        }
        Checkpoint {
            step: self.step,
            epoch: self.epoch,
            best_val_loss: self.best_val_loss,
            loss_scaler_scale: self.loss_scaler_scale,
            model_params: self
                .model_params
                .iter()
                .map(|t| gpu_tensor_to_vec(t))
                .collect(),
            model_shapes: self.model_shapes.clone(),
            optimizer_state: self.optimizer_state.as_ref().map(|opt| opt.to_cpu()),
        }
    }

    /// Save GPU-resident checkpoint to disk (converts to CPU first)
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let cpu_ckpt = self.to_cpu();
        let json = serde_json::to_string_pretty(&cpu_ckpt)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load checkpoint from disk and convert to GPU-resident
    pub fn load(
        path: impl AsRef<Path>,
        ctx: &GpuContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let cpu_ckpt: Checkpoint = serde_json::from_str(&json)?;
        Self::from_cpu(&cpu_ckpt, ctx)
    }

    /// Apply saved GPU weights to model parameters
    pub fn restore_params(&self, params: &[Tensor]) {
        for (i, p) in params.iter().enumerate() {
            if i < self.model_params.len() && i < self.model_shapes.len() {
                match self.model_params[i].to_cpu() {
                    Ok(arr) => p.set_data(arr),
                    Err(e) => tracing::warn!(
                        "CheckpointGpu::restore_params GPU readback failed for param {i}: {e}"
                    ),
                }
            }
        }
    }

    /// Restore GPU-resident optimizer state to Adam optimizer
    pub fn restore_optimizer(&self, adam: &mut Adam) {
        if let Some(ref opt_state) = self.optimizer_state {
            opt_state.apply_to_adam(adam);
        }
    }
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
            model_params: params
                .iter()
                .map(|p| p.data().iter().copied().collect())
                .collect(),
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

    /// Apply saved weights to model parameters.
    /// Logs error and skips mismatched entries instead of panicking.
    pub fn restore_params(&self, params: &[Tensor]) {
        for (i, p) in params.iter().enumerate() {
            if i < self.model_params.len() && i < self.model_shapes.len() {
                match ArrayD::from_shape_vec(
                    self.model_shapes[i].clone(),
                    self.model_params[i].clone(),
                ) {
                    Ok(arr) => p.set_data(arr),
                    Err(e) => {
                        tracing::error!(
                            "param shape mismatch at index {i}: expected shape {:?} but \
                             data length {} doesn't match product {}: {e}. Skipping.",
                            self.model_shapes[i],
                            self.model_params[i].len(),
                            self.model_shapes[i].iter().product::<usize>()
                        );
                    }
                }
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
        self.losses
            .iter()
            .copied()
            .fold(None, |best, x| Some(best.map_or(x, |b| b.min(x))))
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
        self.losses
            .iter()
            .copied()
            .reduce(|a, b| alpha * b + (1.0 - alpha) * a)
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
    pub checkpoint_callback: Option<Box<dyn Fn(u64) -> super::DLResult<()> + Send>>,
    /// Stored model parameters for direct checkpoint serialization
    pub params: Option<Vec<Tensor>>,
    /// Stored Adam optimizer for direct checkpoint serialization
    pub adam: Option<Adam>,
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
            checkpoint_callback: None,
            params: None,
            adam: None,
        }
    }

    /// Load state from checkpoint.
    /// Restores step/epoch/loss, and if model params are stored on the loop,
    /// applies checkpoint weights and optimizer state to the model.
    pub fn resume(&mut self, ckpt: &Checkpoint) {
        self.step = ckpt.step;
        self.epoch = ckpt.epoch;
        self.best_val_loss = ckpt.best_val_loss;

        // Restore model weights if we have stored params
        if let Some(ref params) = self.params {
            if !ckpt.model_params.is_empty() && !ckpt.model_shapes.is_empty() {
                ckpt.restore_params(params);
                tracing::info!(
                    "Restored {} parameter groups from checkpoint",
                    ckpt.model_params.len()
                );
            }
        }

        // Restore optimizer state
        if let Some(ref mut adam) = self.adam {
            ckpt.restore_optimizer(adam);
            tracing::info!("Restored optimizer state from checkpoint");
        }
    }

    /// Create a shared stop signal
    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stop_flag.clone()
    }

    /// Set a checkpoint callback that the caller provides to serialize model weights.
    /// The callback receives the current step number and should return Ok(()) on success.
    pub fn set_checkpoint_callback<F>(&mut self, callback: F)
    where
        F: Fn(u64) -> super::DLResult<()> + Send + 'static,
    {
        self.checkpoint_callback = Some(Box::new(callback));
    }

    /// Set model parameters and optimizer for direct checkpoint serialization.
    /// When set, `on_step` saves full checkpoint weights in addition to JSON metadata.
    pub fn set_model_state(&mut self, params: Vec<Tensor>, adam: Adam) {
        self.params = Some(params);
        self.adam = Some(adam);
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
    ) -> super::DLResult<bool> {
        self.step += 1;
        self.total_tokens += tokens_in_batch;

        // Throughput
        let elapsed = self
            .start_time
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(1.0);
        let throughput = self.total_tokens as f32 / elapsed;

        self.metrics
            .record(self.step, loss, lr, grad_norm, throughput);

        // Logging
        if self.step % self.config.log_interval == 0 {
            let avg = self
                .metrics
                .avg_loss(self.config.log_interval)
                .unwrap_or(loss);
            tracing::info!(
                "[Step {}] loss={:.4} (avg={:.4}) lr={:.2e} grad_norm={:.4} tok/s={:.0}",
                self.step,
                loss,
                avg,
                lr,
                grad_norm,
                throughput
            );
        }

        // Save checkpoint
        if self.step % self.config.save_interval == 0 {
            if let Some(ref dir) = self.config.checkpoint_dir {
                let path = dir.join(format!("checkpoint-{}.json", self.step));
                tracing::info!("  Saving checkpoint to {:?}", path);

                if let Err(e) = std::fs::create_dir_all(dir) {
                    tracing::warn!("  Failed to create checkpoint dir: {}", e);
                } else if self.params.is_some() && self.adam.is_some() {
                    let params = self.params.as_ref().ok_or_else(|| {
                        super::DeepLearningError::Computation {
                            reason: "params not initialized for checkpoint save".into(),
                        }
                    })?;
                    let adam = self.adam.as_ref().ok_or_else(|| {
                        super::DeepLearningError::Computation {
                            reason: "adam not initialized for checkpoint save".into(),
                        }
                    })?;
                    match Checkpoint::save(
                        &path,
                        params,
                        adam,
                        self.step,
                        self.epoch,
                        self.best_val_loss,
                        None::<&LossScaler>,
                    ) {
                        Ok(_) => tracing::info!("  Full checkpoint saved to {:?}", path),
                        Err(e) => tracing::warn!("  Failed to save full checkpoint: {}", e),
                    }
                } else {
                    // Metadata-only checkpoint when model state is not stored
                    let meta_ckpt = serde_json::json!({
                        "step": self.step,
                        "epoch": self.epoch,
                        "best_val_loss": self.best_val_loss,
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                        "type": "metadata-checkpoint",
                        "note": "full checkpoint requires call to set_model_state()"
                    });
                    if let Ok(json) = serde_json::to_string_pretty(&meta_ckpt) {
                        if let Err(e) = std::fs::write(&path, &json) {
                            tracing::warn!("  Failed to save checkpoint: {}", e);
                        } else {
                            tracing::info!("  Checkpoint metadata saved to {:?}", path);
                        }
                    }
                }

                // Invoke the external checkpoint callback for custom serialization
                if let Some(ref cb) = self.checkpoint_callback {
                    if let Err(e) = cb(self.step as u64) {
                        tracing::warn!("  Checkpoint callback failed: {}", e);
                    }
                }
            }
        }

        // Stop check
        if self.step >= self.config.max_steps || self.should_stop() {
            return Ok(false);
        }

        Ok(true)
    }

    /// Update validation loss, returns true if improved
    pub fn on_eval(&mut self, val_loss: f64) -> bool {
        match self.best_val_loss {
            Some(best) if val_loss < best => {
                self.best_val_loss = Some(val_loss);
                self.patience_counter = 0;
                tracing::info!("  Validation loss improved to {:.4}", val_loss);
                true
            }
            None => {
                self.best_val_loss = Some(val_loss);
                self.patience_counter = 0;
                tracing::info!("  Initial validation loss: {:.4}", val_loss);
                true
            }
            Some(_) => {
                self.patience_counter += 1;
                if self.patience_counter >= self.config.early_stop_patience {
                    tracing::info!(
                        "  Early stopping triggered (patience={})",
                        self.config.early_stop_patience
                    );
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
        let avg = self
            .metrics
            .avg_loss(self.config.log_interval)
            .unwrap_or(0.0);
        tracing::info!(
            "=== Epoch {} complete | avg_loss={:.4} | best_val={:?} ===",
            self.epoch,
            avg,
            self.best_val_loss
        );
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
        self.start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Summary report
    pub fn report(&self) {
        tracing::info!("═══════════════════════════════════════");
        tracing::info!("  Training complete");
        tracing::info!("  Steps: {}", self.step);
        tracing::info!("  Epochs: {}", self.epoch);
        tracing::info!("  Total tokens: {}", self.total_tokens);
        tracing::info!("  Duration: {:.1}s", self.elapsed_secs());
        tracing::info!(
            "  Avg throughput: {:.0} tok/s",
            self.metrics.avg_throughput().unwrap_or(0.0)
        );
        tracing::info!("  Best loss: {:?}", self.best_val_loss);
        tracing::info!(
            "  Final loss: {:?}",
            self.metrics.last_loss().unwrap_or(0.0)
        );
        tracing::info!("═══════════════════════════════════════");
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

/// GPU-native gradient norm: compute L2 norm of GPU-resident gradient tensors
/// using GPU reduce + sqrt — hanya 1 scalar readback (vs N sebelumnya).
#[cfg(feature = "gpu")]
pub fn compute_grad_norm_gpu(grads: &[crate::gpu::GpuTensor], ctx: &crate::gpu::GpuContext) -> f32 {
    if grads.is_empty() {
        return 0.0;
    }

    // 1. Compute per-tensor L2 norm squared on GPU
    let mut norm_sq_tensors = Vec::with_capacity(grads.len());
    for g in grads {
        if let Ok(norm_sq_t) = ctx.l2_norm(g) {
            norm_sq_tensors.push(norm_sq_t);
        }
    }

    if norm_sq_tensors.is_empty() {
        return 0.0;
    }

    // 2. Sum all norms on GPU via elementwise add (no readback)
    let mut total_norm_sq = norm_sq_tensors[0].clone();
    for t in &norm_sq_tensors[1..] {
        if let Ok(result) = ctx.add(&total_norm_sq, t) {
            total_norm_sq = result;
        }
    }

    // 3. Sqrt on GPU, then async readback single scalar via channel
    if let Ok(norm_gpu) = ctx.sqrt(&total_norm_sq) {
        // Non-blocking: mulai readback, GPU masih bisa kerja
        if let Ok(rb) = ctx.readback_f32_async(norm_gpu.buffer(), 4) {
            // Attempt 1: non-blocking poll
            ctx.poll_device();
            if let Some(val) = rb.try_recv() {
                return val[0];
            }
            // Attempt 2: 100μs timeout
            ctx.poll_timeout(100_000);
            if let Some(val) = rb.try_recv() {
                return val[0];
            }
            // Attempt 3: 1ms timeout
            ctx.poll_timeout(1_000_000);
            if let Some(val) = rb.try_recv() {
                return val[0];
            }
            // Non-blocking spin loop with device polling
            ctx.wait_device();
            for _ in 0..100 {
                ctx.poll_timeout(100_000);
                if let Some(val) = rb.try_recv() {
                    return val[0];
                }
            }
            if let Some(val) = rb.try_recv() {
                return val[0];
            }
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, Module, Tensor, TensorOps};

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
            model_params: params
                .iter()
                .map(|p| p.data().iter().copied().collect())
                .collect(),
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
            let cont = loop_.on_step(0.5, 0.001, 0.1, 32).unwrap();
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

        assert!(loop_.on_step(0.5, 0.001, 0.1, 32).unwrap());
        assert!(loop_.on_step(0.4, 0.001, 0.1, 32).unwrap());
        let cont = loop_.on_step(0.3, 0.001, 0.1, 32).unwrap();
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
