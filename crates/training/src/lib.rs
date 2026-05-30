pub mod lora;
pub mod mixed_precision;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ndarray::ArrayD;
use nexora_autograd::compute_grad_norm;
#[cfg(feature = "gpu")]
use nexora_autograd::device::Storage;
#[cfg(feature = "gpu")]
use nexora_autograd::gpu_grad_clip::GpuGradClipResult;
use nexora_autograd::ops::cross_entropy_loss;
#[cfg(feature = "gpu")]
use nexora_autograd::Device;
use nexora_autograd::{clear_tape, Adam, Tensor, TensorOps};
use tracing::{info, warn};

#[cfg(feature = "gpu")]
use nexora_autograd::data_parallel::gpu_allreduce_gradients;
#[cfg(feature = "gpu")]
use nexora_autograd::gpu::{GpuContext, GpuTensor};
#[cfg(feature = "gpu")]
use nexora_autograd::gpu_adam::GpuAdam;
#[cfg(feature = "gpu")]
use nexora_autograd::gpu_async::{AsyncReadback, GpuStagingPool};

use nexora_transformer::{safetensors, CausalLM, TrainableCausalLM, TransformerConfig};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct EvalMetrics {
    pub avg_loss: f64,
    pub perplexity: f64,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct TrainerConfig {
    pub learning_rate: f32,
    pub max_steps: usize,
    pub seq_length: usize,
    pub vocab_size: usize,
    pub save_path: Option<String>,
    pub save_every: usize,
    pub report_every: usize,
    pub batch_size: usize,
    pub weight_decay: f32,
    pub max_grad_norm: Option<f32>,
    pub warmup_steps: usize,
    pub val_every_steps: usize,
    pub early_stop_patience: usize,
    pub use_gpu: bool,
    /// Number of model replicas for data-parallel training (1 = no replication).
    /// When >1, gradients are averaged across replicas via GPU Gradient AllReduce
    /// after each optimizer step. Requires GPU mode.
    pub num_replicas: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            max_steps: 1000,
            seq_length: 128,
            vocab_size: 50257,
            save_path: None,
            save_every: 100,
            report_every: 10,
            batch_size: 1,
            weight_decay: 0.0,
            max_grad_norm: None,
            warmup_steps: 0,
            val_every_steps: 100,
            early_stop_patience: 3,
            use_gpu: true,
            num_replicas: 1,
        }
    }
}

pub struct Trainer {
    pub config: TrainerConfig,
    pub model: CausalLM,
    pub trainable: Option<TrainableCausalLM>,
    pub optimizer: Option<Adam>,
    #[cfg(feature = "gpu")]
    pub gpu_optimizer: Option<GpuAdam>,
    #[cfg(feature = "gpu")]
    pub gpu_input_buf: Option<GpuTensor>,
    #[cfg(feature = "gpu")]
    pub gpu_target_buf: Option<GpuTensor>,
    #[cfg(feature = "gpu")]
    pub gpu_staging: Option<GpuStagingPool>,
    /// GPU gradient tensors per replica for data-parallel allreduce.
    #[cfg(feature = "gpu")]
    pub replica_grads: Vec<Vec<GpuTensor>>,
    pub step: usize,
    pub total_loss: f64,
    pub total_tokens: usize,
    pub accumulation_counter: usize,
    pub best_val_loss: Option<f64>,
    pub patience_counter: usize,
    pub completed_epochs: usize,
    pub stop_flag: Arc<AtomicBool>,
    pub last_grad_norm: Option<f32>,
    pub loss_ema: Option<f32>,
    pub step_times: VecDeque<std::time::Duration>,
    pub token_counts: VecDeque<usize>,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        let model_config = TransformerConfig {
            vocab_size: config.vocab_size,
            max_seq_len: config.seq_length,
            ..Default::default()
        };
        let model = CausalLM::new(model_config);
        Self {
            config,
            model,
            trainable: None,
            optimizer: None,
            #[cfg(feature = "gpu")]
            gpu_optimizer: None,
            #[cfg(feature = "gpu")]
            gpu_input_buf: None,
            #[cfg(feature = "gpu")]
            gpu_target_buf: None,
            #[cfg(feature = "gpu")]
            gpu_staging: None,
            #[cfg(feature = "gpu")]
            replica_grads: Vec::new(),
            step: 0,
            total_loss: 0.0,
            total_tokens: 0,
            accumulation_counter: 0,
            best_val_loss: None,
            patience_counter: 0,
            completed_epochs: 0,
            stop_flag: Arc::new(AtomicBool::new(false)),
            last_grad_norm: None,
            loss_ema: None,
            step_times: VecDeque::with_capacity(100),
            token_counts: VecDeque::with_capacity(100),
        }
    }

    pub fn with_model(model: CausalLM, config: TrainerConfig) -> Self {
        Self {
            config,
            model,
            trainable: None,
            optimizer: None,
            #[cfg(feature = "gpu")]
            gpu_optimizer: None,
            #[cfg(feature = "gpu")]
            gpu_input_buf: None,
            #[cfg(feature = "gpu")]
            gpu_target_buf: None,
            #[cfg(feature = "gpu")]
            gpu_staging: None,
            #[cfg(feature = "gpu")]
            replica_grads: Vec::new(),
            step: 0,
            total_loss: 0.0,
            total_tokens: 0,
            accumulation_counter: 0,
            best_val_loss: None,
            patience_counter: 0,
            completed_epochs: 0,
            stop_flag: Arc::new(AtomicBool::new(false)),
            last_grad_norm: None,
            loss_ema: None,
            step_times: VecDeque::with_capacity(100),
            token_counts: VecDeque::with_capacity(100),
        }
    }

    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stop_flag.clone()
    }

    pub fn request_stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    pub fn should_stop(&self) -> bool {
        self.stop_flag.load(Ordering::SeqCst)
    }

    pub fn reset_stop(&self) {
        self.stop_flag.store(false, Ordering::SeqCst);
    }

    pub fn prepare(&mut self) {
        let trainable = TrainableCausalLM::from_inference(&self.model);
        let params = trainable.parameters();

        #[cfg(feature = "gpu")]
        if self.config.use_gpu {
            let mut gpu_ok = true;
            match GpuContext::global() {
                Ok(ctx) => {
                    for p in &params {
                        match GpuTensor::from_cpu(&p.data()) {
                            Ok(gpu_t) => {
                                p.set_storage(nexora_autograd::Storage::Gpu(gpu_t));
                                p.set_device(Device::Gpu(0));
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to move parameter to GPU: {e}, falling back to CPU"
                                );
                                gpu_ok = false;
                                break;
                            }
                        }
                    }
                    if gpu_ok {
                        let owned_gpu_params: Vec<GpuTensor> = params
                            .iter()
                            .filter_map(|p| match p.storage() {
                                nexora_autograd::Storage::Gpu(g) => Some(g),
                                _ => {
                                    tracing::warn!(
                                        "Non-GPU tensor in GPU training optimizer setup — skipping"
                                    );
                                    None
                                }
                            })
                            .collect();
                        let gpu_params: Vec<&GpuTensor> = owned_gpu_params.iter().collect();
                        gpu_ok = match GpuAdam::new(ctx, &gpu_params, self.config.learning_rate) {
                            Ok(opt) => {
                                let mut gpu_opt = opt;
                                gpu_opt.weight_decay = self.config.weight_decay;
                                gpu_opt.max_grad_norm = self.config.max_grad_norm;
                                self.gpu_optimizer = Some(gpu_opt);
                                true
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to create GpuAdam optimizer: {e}, falling back to CPU"
                                );
                                false
                            }
                        };
                    }
                    if gpu_ok {
                        nexora_autograd::tensor::set_gpu_auto_create(true);
                        let batch_shape = vec![self.config.seq_length];
                        gpu_ok = match GpuTensor::zeros(&batch_shape) {
                            Ok(buf) => {
                                self.gpu_input_buf = Some(buf);
                                true
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to create GPU input buffer: {e}, falling back to CPU"
                                );
                                false
                            }
                        };
                    }
                    if gpu_ok {
                        let batch_shape = vec![self.config.seq_length];
                        gpu_ok = match GpuTensor::zeros(&batch_shape) {
                            Ok(buf) => {
                                self.gpu_target_buf = Some(buf);
                                true
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to create GPU target buffer: {e}, falling back to CPU"
                                );
                                false
                            }
                        };
                    }
                    if gpu_ok {
                        self.gpu_staging = Some(GpuStagingPool::new(
                            ctx,
                            (self.config.seq_length * 4) as u64,
                        ));
                        self.trainable = Some(trainable);
                        return;
                    }
                    nexora_autograd::tensor::set_gpu_auto_create(false);
                    self.gpu_optimizer = None;
                    self.gpu_input_buf = None;
                    self.gpu_target_buf = None;
                }
                Err(e) => {
                    tracing::warn!("GPU init failed ({}), falling back to CPU", e);
                }
            }
        }

        self.trainable = Some(trainable);
        let mut optimizer = Adam::new(params, self.config.learning_rate);
        if self.config.weight_decay > 0.0 {
            optimizer.set_weight_decay(self.config.weight_decay);
        }
        if self.config.max_grad_norm.is_some() {
            optimizer.set_max_grad_norm(self.config.max_grad_norm);
        }
        self.optimizer = Some(optimizer);
    }

    fn lr_at_step(step: usize, base_lr: f32, warmup_steps: usize, max_steps: usize) -> f32 {
        if step < warmup_steps {
            base_lr * (step as f32 / warmup_steps.max(1) as f32)
        } else {
            let progress =
                ((step - warmup_steps) as f32 / (max_steps - warmup_steps).max(1) as f32).min(1.0);
            base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }

    /// TRN-C1: Process multiple chunk pairs in a single call — absorbs chunk loop
    /// into one batch cycle. Each pair runs forward+backward; optimizer step fires
    /// when `accumulation_counter >= batch_size`.
    pub fn train_batch_multi(&mut self, chunks: &[(&[u32], &[u32])]) -> Option<f32> {
        let mut last_loss = None;
        for &(tokens, targets) in chunks {
            last_loss = self.train_batch(tokens, targets);
            if last_loss.is_none() {
                return None;
            }
        }
        last_loss
    }

    pub fn train_batch(&mut self, tokens: &[u32], targets: &[u32]) -> Option<f32> {
        if self.should_stop() {
            return None;
        }

        let trainable = match self.trainable.as_ref() {
            Some(t) => t,
            None => {
                warn!("train_step called without prepare()");
                return None;
            }
        };

        let seq = tokens.len().min(self.config.seq_length);
        if seq == 0 {
            return None;
        }

        // ─── GPU training path ──────────────────────────────────────────────
        #[cfg(feature = "gpu")]
        if self.gpu_optimizer.is_some() {
            let gpu_opt = match self.gpu_optimizer.as_mut() {
                Some(opt) => opt,
                None => return None,
            };
            let gpu_in = match self.gpu_input_buf.as_ref() {
                Some(buf) => buf,
                None => return None,
            };
            let gpu_tgt = match self.gpu_target_buf.as_ref() {
                Some(buf) => buf,
                None => return None,
            };
            let gpu_pool = match self.gpu_staging.as_mut() {
                Some(pool) => pool,
                None => return None,
            };
            return train_batch_gpu(
                &mut self.model,
                trainable,
                gpu_opt,
                tokens,
                targets,
                seq,
                &mut self.total_loss,
                &mut self.total_tokens,
                &mut self.accumulation_counter,
                &mut self.step,
                &mut self.step_times,
                &mut self.token_counts,
                &mut self.loss_ema,
                &mut self.last_grad_norm,
                &self.config,
                &self.stop_flag,
                gpu_in,
                gpu_tgt,
                gpu_pool,
            );
        }

        // ─── CPU training path ──────────────────────────────────────────────
        let optimizer = match self.optimizer.as_mut() {
            Some(o) => o,
            None => {
                warn!("train_step called without prepare()");
                return None;
            }
        };

        let batch_start = std::time::Instant::now();

        let mut input_buf = vec![0.0f32; seq];
        for (i, &t) in tokens[..seq].iter().enumerate() {
            input_buf[i] = t as f32;
        }
        let mut target_buf = vec![0.0f32; seq];
        for (i, &t) in targets[..seq].iter().enumerate() {
            target_buf[i] = t as f32;
        }
        let input_t = Tensor::from_slice(&input_buf, &[seq]);
        let target_t = Tensor::from_slice(&target_buf, &[seq]);

        let logits = trainable.forward(&input_t);
        let loss = cross_entropy_loss(&logits, &target_t).mean();

        loss.backward();

        let loss_val = loss.data()[0];
        if !loss_val.is_finite() {
            warn!("NaN/Inf loss detected ({}) — skipping step", loss_val);
            optimizer.zero_grad();
            return None;
        }
        self.total_loss += loss_val as f64 * seq as f64;
        self.total_tokens += seq;
        self.accumulation_counter += 1;

        if self.accumulation_counter >= self.config.batch_size {
            let new_lr = Self::lr_at_step(
                self.step + 1,
                self.config.learning_rate,
                self.config.warmup_steps,
                self.config.max_steps,
            );
            optimizer.lr = new_lr;

            self.last_grad_norm = Some(compute_grad_norm(&optimizer.parameters));

            optimizer.step();
            optimizer.zero_grad();
            self.step += 1;
            self.accumulation_counter = 0;

            let elapsed = batch_start.elapsed();
            self.step_times.push_back(elapsed);
            self.token_counts.push_back(seq);
            if self.step_times.len() > 100 {
                self.step_times.pop_front();
                self.token_counts.pop_front();
            }

            self.loss_ema = Some(match self.loss_ema {
                Some(ema) => 0.99 * ema + 0.01 * loss_val,
                None => loss_val,
            });

            if let Some(ref path) = self.config.save_path {
                if self.step % self.config.save_every == 0 {
                    if let Err(e) = trainable.sync_to_inference(&mut self.model) {
                        warn!("sync_to_inference failed at step {}: {}", self.step, e);
                    }
                    let save_file = format!("{}.safetensors", path);
                    if let Err(e) = trainable.save_checkpoint(&save_file) {
                        warn!("Failed to save checkpoint at step {}: {}", self.step, e);
                    }
                    if let Some(ref opt) = self.optimizer {
                        let opt_file = format!("{}.opt.safetensors", path);
                        if let Err(e) = save_optimizer_file(&opt_file, opt) {
                            warn!("Failed to save optimizer at step {}: {}", self.step, e);
                        }
                    }
                }
            }

            if self.should_stop() {
                return None;
            }
        }

        Some(loss_val)
    }

    pub fn prepare_batch(&mut self, tokens: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let seq = tokens.len().min(self.config.seq_length + 1);
        if seq < 2 {
            return (vec![], vec![]);
        }
        let input = tokens[..seq - 1].to_vec();
        let target = tokens[1..seq].to_vec();
        (input, target)
    }

    pub fn avg_loss(&self) -> f64 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.total_loss / self.total_tokens as f64
        }
    }

    pub fn save_checkpoint(&self) {
        if let Some(ref path) = self.config.save_path {
            if let Some(ref trainable) = self.trainable {
                let save_file = format!("{}.safetensors", path);
                if let Err(e) = trainable.save_checkpoint(&save_file) {
                    warn!("Failed to save checkpoint: {}", e);
                }
            }
            if let Some(ref opt) = self.optimizer {
                let opt_file = format!("{}.opt.safetensors", path);
                if let Err(e) = save_optimizer_file(&opt_file, opt) {
                    warn!("Failed to save final optimizer: {}", e);
                }
            }
        }
    }

    /// Restore optimizer state from `{save_path}.opt.safetensors` if it exists.
    pub fn try_restore_optimizer(&mut self) {
        let path = match self.config.save_path.as_ref() {
            Some(p) => format!("{}.opt.safetensors", p),
            None => return,
        };
        let opt = match self.optimizer.as_mut() {
            Some(o) => o,
            None => return,
        };
        if !std::path::Path::new(&path).exists() {
            return;
        }
        match load_optimizer_file(&path) {
            Ok(state) => {
                opt.load_optimizer_state(&state);
                self.step = opt.step;
                info!("Restored optimizer from {} (step {})", path, opt.step);
            }
            Err(e) => warn!("Failed to load optimizer state from {}: {}", path, e),
        }
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        if let Some(ref trainable) = self.trainable {
            trainable.save_checkpoint(path)?;
        } else {
            let trainable = TrainableCausalLM::from_inference(&self.model);
            trainable.save_checkpoint(path)?;
        }
        Ok(())
    }

    pub fn load(model: &mut CausalLM, path: &str) -> anyhow::Result<()> {
        TrainableCausalLM::load_checkpoint(model, path)?;
        Ok(())
    }

    pub fn sync_weights(&mut self) -> anyhow::Result<()> {
        if let Some(ref trainable) = self.trainable {
            let mut model_clone = self.model.clone();
            trainable
                .sync_to_inference(&mut model_clone)
                .map_err(|e| anyhow::anyhow!("sync_to_inference failed: {}", e))?;
            self.model = model_clone;
        }
        Ok(())
    }

    pub fn evaluate_loss(&self, sequences: &[Vec<u32>], seq_length: usize) -> EvalMetrics {
        let trainable = match self.trainable.as_ref() {
            Some(t) => t,
            None => {
                warn!("evaluate_loss called without prepare()");
                return EvalMetrics {
                    avg_loss: 0.0,
                    perplexity: 0.0,
                    total_tokens: 0,
                };
            }
        };

        #[cfg(feature = "gpu")]
        if self.gpu_optimizer.is_some() && GpuContext::global().is_ok() {
            if let (Some(gpu_in), Some(gpu_tgt)) =
                (self.gpu_input_buf.as_ref(), self.gpu_target_buf.as_ref())
            {
                return evaluate_loss_gpu(trainable, sequences, seq_length, gpu_in, gpu_tgt);
            }
            tracing::warn!("GPU buffers not initialized, falling back to CPU evaluation");
        }

        let mut total_loss = 0.0f64;
        let mut total_tokens = 0usize;

        for tokens in sequences {
            if tokens.len() < 2 {
                continue;
            }
            for chunk in tokens.chunks(seq_length + 1) {
                if chunk.len() < 2 {
                    continue;
                }
                let input_t = Tensor::from_slice(
                    &chunk[..chunk.len() - 1]
                        .iter()
                        .map(|&x| x as f32)
                        .collect::<Vec<_>>(),
                    &[chunk.len() - 1],
                );
                let target_t = Tensor::from_slice(
                    &chunk[1..].iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[chunk.len() - 1],
                );
                let logits = trainable.forward(&input_t);
                let loss = cross_entropy_loss(&logits, &target_t).mean();
                let step_loss = loss.data()[0] as f64;
                if !step_loss.is_finite() {
                    warn!("NaN/Inf detected during evaluation — skipping chunk");
                    continue;
                }
                total_loss += step_loss * (chunk.len() - 1) as f64;
                total_tokens += chunk.len() - 1;
            }
        }

        // Clear tape entries created during eval (no gradient needed)
        clear_tape();

        EvalMetrics {
            avg_loss: if total_tokens > 0 {
                total_loss / total_tokens as f64
            } else {
                0.0
            },
            perplexity: if total_tokens > 0 {
                (total_loss / total_tokens as f64).exp()
            } else {
                0.0
            },
            total_tokens,
        }
    }

    pub fn should_early_stop(&self) -> bool {
        self.patience_counter >= self.config.early_stop_patience
    }

    pub fn update_val_loss(&mut self, val_loss: f64) -> bool {
        match self.best_val_loss {
            Some(best) if val_loss < best => {
                self.best_val_loss = Some(val_loss);
                self.patience_counter = 0;
                true
            }
            None => {
                self.best_val_loss = Some(val_loss);
                self.patience_counter = 0;
                false
            }
            Some(_) => {
                self.patience_counter += 1;
                false
            }
        }
    }

    pub fn epoch_checkpoint(&self, epoch: usize) {
        if let Some(ref path) = self.config.save_path {
            if let Some(ref trainable) = self.trainable {
                let mut model_clone = self.model.clone();
                if let Err(e) = trainable.sync_to_inference(&mut model_clone) {
                    tracing::error!("sync_to_inference failed in epoch_checkpoint: {}", e);
                    return;
                }
                let save_file = format!("{}.epoch-{}.safetensors", path, epoch);
                if let Err(e) = trainable.save_checkpoint(&save_file) {
                    warn!("Failed to save epoch checkpoint: {}", e);
                }
            }
        }
    }
}

// --- GPU training batch (standalone, avoids borrow conflicts) ---

/// Bounded polling for async GPU readback with 3 escalating timeouts.
#[cfg(feature = "gpu")]
fn poll_async_loss(ctx: &GpuContext, rb: &AsyncReadback<Vec<f32>>) -> f32 {
    // Attempt 1: non-blocking poll
    ctx.poll_device();
    if let Some(v) = rb.try_recv() {
        return v[0];
    }
    // Attempt 2: short timeout (100μs) — GPU likely done with backward
    ctx.poll_timeout(100_000);
    if let Some(v) = rb.try_recv() {
        return v[0];
    }
    // Attempt 3: moderate timeout (1ms) — catch stragglers
    ctx.poll_timeout(1_000_000);
    if let Some(v) = rb.try_recv() {
        return v[0];
    }
    // Non-blocking spin loop
    ctx.wait_device();
    for _ in 0..100 {
        ctx.poll_timeout(100_000);
        if let Some(v) = rb.try_recv() {
            return v[0];
        }
    }
    rb.try_recv().map_or(0.0, |v| v[0])
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn train_batch_gpu(
    model: &mut CausalLM,
    trainable: &nexora_transformer::TrainableCausalLM,
    gpu_opt: &mut GpuAdam,
    tokens: &[u32],
    targets: &[u32],
    seq: usize,
    total_loss: &mut f64,
    total_tokens: &mut usize,
    accumulation_counter: &mut usize,
    step: &mut usize,
    step_times: &mut VecDeque<std::time::Duration>,
    token_counts: &mut VecDeque<usize>,
    loss_ema: &mut Option<f32>,
    last_grad_norm: &mut Option<f32>,
    config: &TrainerConfig,
    stop_flag: &Arc<AtomicBool>,
    input_buffer: &GpuTensor,
    target_buffer: &GpuTensor,
    staging: &mut GpuStagingPool,
) -> Option<f32> {
    use std::sync::atomic::Ordering;
    let batch_start = std::time::Instant::now();
    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(e) => {
            warn!("GpuContext lost: {}", e);
            return None;
        }
    };

    // ── Zero-copy upload into pre-allocated buffers ──
    let input_f32: Vec<f32> = tokens[..seq].iter().map(|&t| t as f32).collect();
    let target_f32: Vec<f32> = targets[..seq].iter().map(|&t| t as f32).collect();
    ctx.queue
        .write_buffer(input_buffer.buffer(), 0, bytemuck::cast_slice(&input_f32));
    ctx.queue
        .write_buffer(target_buffer.buffer(), 0, bytemuck::cast_slice(&target_f32));

    let input_t = Tensor::from_gpu(
        input_buffer.view_as(vec![seq]),
        nexora_autograd::tensor::next_tensor_id(),
        false,
    );
    let target_t = Tensor::from_gpu(
        target_buffer.view_as(vec![seq]),
        nexora_autograd::tensor::next_tensor_id(),
        false,
    );

    // ── Forward / Backward (batched), with pipelined loss readback ──
    ctx.begin_batch_mode();
    let logits = trainable.forward(&input_t);
    let loss = cross_entropy_loss(&logits, &target_t).mean();

    // ── Queue loss copy to staging INSIDE batch (overlaps copy with backward) ──
    let loss_gpu = match loss.storage() {
        nexora_autograd::Storage::Gpu(g) => g,
        _ => return None,
    };
    let loss_staging = ctx.create_readback_staging(loss_gpu.buffer(), 4, "loss");

    loss.backward();
    ctx.end_batch_mode();

    // ── Register map_async after submission ──
    let loss_readback = ctx.complete_readback(&loss_staging, 4);

    // ── Gradients: read from tape for NaN check (CPU work while GPU processes) ──
    let owned_params: Vec<GpuTensor> = trainable
        .parameters()
        .iter()
        .filter_map(|p| match p.storage() {
            nexora_autograd::Storage::Gpu(g) => Some(g),
            _ => {
                tracing::warn!("Non-GPU tensor in GPU backward train — skipping");
                None
            }
        })
        .collect();
    let params: Vec<&GpuTensor> = owned_params.iter().collect();

    // ── Bounded poll + try_recv ──
    let loss_val = poll_async_loss(&ctx, &loss_readback);
    if !loss_val.is_finite() {
        warn!("NaN/Inf loss detected ({}) — skipping GPU step", loss_val);
        let _ = gpu_opt.zero_grad(&ctx, &params);
        return None;
    }

    *total_loss += loss_val as f64 * seq as f64;
    *total_tokens += seq;
    *accumulation_counter += 1;

    if *accumulation_counter >= config.batch_size {
        let owned_params: Vec<GpuTensor> = trainable
            .parameters()
            .iter()
            .filter_map(|p| match p.storage() {
                nexora_autograd::Storage::Gpu(g) => Some(g),
                _ => {
                    tracing::warn!("Non-GPU tensor in GPU gradient accumulation — skipping");
                    None
                }
            })
            .collect();
        let params: Vec<&GpuTensor> = owned_params.iter().collect();
        let mut grad_tensors = Vec::new();
        let mut gpu_ok = true;
        for p in trainable.parameters().iter() {
            match p.grad_storage() {
                Some(Storage::Gpu(g)) => grad_tensors.push(g),
                _ => {
                    let arr = p.grad().unwrap_or_else(|| ArrayD::zeros(p.shape()));
                    match GpuTensor::from_cpu(&arr) {
                        Ok(g) => grad_tensors.push(g),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to upload fallback gradient to GPU: {e}, skipping GPU step"
                            );
                            gpu_ok = false;
                            break;
                        }
                    }
                }
            }
        }
        if !gpu_ok {
            return None;
        }
        let grad_refs: Vec<&GpuTensor> = grad_tensors.iter().collect();

        gpu_opt.lr = Trainer::lr_at_step(
            *step + 1,
            config.learning_rate,
            config.warmup_steps,
            config.max_steps,
        );

        // GPU-native gradient clipping: norm computation + scaling all on GPU
        // Hanya 4 f32 readback — bukan N scalars per gradient tensor
        // Matikan internal clipping di GpuAdam (sekarang dilakukan secara eksternal)
        let internal_max_norm = gpu_opt.max_grad_norm.take();
        let clip_result = if let Some(max_norm) = internal_max_norm.or(config.max_grad_norm) {
            ctx.clip_gradients_gpu(&grad_refs, max_norm)
                .unwrap_or(GpuGradClipResult {
                    was_clipped: false,
                    scale_factor: 1.0,
                    norm: 0.0,
                })
        } else {
            GpuGradClipResult {
                was_clipped: false,
                scale_factor: 1.0,
                norm: 0.0,
            }
        };
        *last_grad_norm = Some(clip_result.norm);

        // Data-parallel gradient allreduce across model replicas
        if config.num_replicas > 1 && !grad_refs.is_empty() {
            let _ = gpu_allreduce_gradients(ctx, &grad_refs);
        }

        let _ = gpu_opt.step(&ctx, &params, &grad_refs);
        let _ = gpu_opt.zero_grad(&ctx, &params);
        for p in trainable.parameters().iter() {
            p.zero_grad();
        }
        *step += 1;
        *accumulation_counter = 0;

        let elapsed = batch_start.elapsed();
        step_times.push_back(elapsed);
        token_counts.push_back(seq);
        if step_times.len() > 100 {
            step_times.pop_front();
            token_counts.pop_front();
        }

        *loss_ema = Some(match *loss_ema {
            Some(ema) => 0.99 * ema + 0.01 * loss_val,
            None => loss_val,
        });

        if *step % config.save_every == 0 {
            if let Some(ref path) = config.save_path {
                if let Err(e) = trainable.sync_to_inference(model) {
                    warn!("sync_to_inference failed: {}", e);
                }
                let save_file = format!("{}.safetensors", path);
                if let Err(e) = trainable.save_checkpoint(&save_file) {
                    warn!("Failed to save checkpoint: {}", e);
                }
            }
        }

        if stop_flag.load(Ordering::SeqCst) {
            return None;
        }
    }

    Some(loss_val)
}

// --- GPU evaluation (forward-only, no backward) ---

#[cfg(feature = "gpu")]
fn evaluate_loss_gpu(
    trainable: &nexora_transformer::TrainableCausalLM,
    sequences: &[Vec<u32>],
    seq_length: usize,
    input_buffer: &GpuTensor,
    target_buffer: &GpuTensor,
) -> EvalMetrics {
    let ctx = match GpuContext::global() {
        Ok(ctx) => ctx,
        Err(e) => {
            tracing::warn!("GpuContext unavailable for evaluate_loss_gpu: {e}, returning zero");
            return EvalMetrics {
                avg_loss: 0.0,
                perplexity: 0.0,
                total_tokens: 0,
            };
        }
    };
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;

    for tokens in sequences {
        if tokens.len() < 2 {
            continue;
        }
        for chunk in tokens.chunks(seq_length + 1) {
            if chunk.len() < 2 {
                continue;
            }
            let seq = chunk.len() - 1;

            let input_f32: Vec<f32> = chunk[..seq].iter().map(|&t| t as f32).collect();
            let target_f32: Vec<f32> = chunk[1..].iter().map(|&t| t as f32).collect();
            ctx.queue
                .write_buffer(input_buffer.buffer(), 0, bytemuck::cast_slice(&input_f32));
            ctx.queue
                .write_buffer(target_buffer.buffer(), 0, bytemuck::cast_slice(&target_f32));

            let input_t = Tensor::from_gpu(
                input_buffer.view_as(vec![seq]),
                nexora_autograd::tensor::next_tensor_id(),
                false,
            );
            let target_t = Tensor::from_gpu(
                target_buffer.view_as(vec![seq]),
                nexora_autograd::tensor::next_tensor_id(),
                false,
            );

            let logits = trainable.forward(&input_t);
            let loss = cross_entropy_loss(&logits, &target_t).mean();

            match loss.storage() {
                nexora_autograd::Storage::Gpu(g) => {
                    let loss_cpu = g.to_cpu().unwrap_or_else(|e| {
                        tracing::warn!("Training eval GPU readback failed: {e}");
                        ndarray::ArrayD::zeros(vec![1])
                    });
                    let step_loss = loss_cpu.as_slice().map(|s| s[0] as f64).unwrap_or(f64::NAN);
                    if !step_loss.is_finite() {
                        warn!("NaN/Inf detected during GPU evaluation — skipping chunk");
                        continue;
                    }
                    total_loss += step_loss * seq as f64;
                    total_tokens += seq;
                }
                _ => {
                    let cpu_val = loss.data()[0] as f64;
                    if cpu_val.is_finite() {
                        total_loss += cpu_val * seq as f64;
                        total_tokens += seq;
                    }
                }
            }
        }
    }

    clear_tape();

    EvalMetrics {
        avg_loss: if total_tokens > 0 {
            total_loss / total_tokens as f64
        } else {
            0.0
        },
        perplexity: if total_tokens > 0 {
            (total_loss / total_tokens as f64).exp()
        } else {
            0.0
        },
        total_tokens,
    }
}

// --- Optimizer save/load helpers ---

fn save_optimizer_file(path: &str, opt: &Adam) -> anyhow::Result<()> {
    let tensors = opt.collect_optimizer_tensors();
    safetensors::save_safetensors(path, &tensors)
        .map_err(|e| anyhow::anyhow!("Optimizer save failed: {}", e))
}

fn load_optimizer_file(
    path: &str,
) -> anyhow::Result<std::collections::HashMap<String, ArrayD<f32>>> {
    safetensors::load_safetensors(path).map_err(|e| anyhow::anyhow!("Optimizer load failed: {}", e))
}
