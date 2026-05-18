pub mod lora;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tracing::warn;
use nexora_deeplearning::autograd::{Tensor, TensorOps, Adam, clear_tape};
use nexora_deeplearning::autograd::ops::cross_entropy_loss;

use crate::models::transformer::{CausalLM, TrainableCausalLM, TransformerConfig};

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
        }
    }
}

pub struct Trainer {
    pub config: TrainerConfig,
    pub model: CausalLM,
    pub trainable: Option<TrainableCausalLM>,
    pub optimizer: Option<Adam>,
    pub step: usize,
    pub total_loss: f64,
    pub total_tokens: usize,
    pub accumulation_counter: usize,
    pub best_val_loss: Option<f64>,
    pub patience_counter: usize,
    pub completed_epochs: usize,
    pub stop_flag: Arc<AtomicBool>,
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
            step: 0,
            total_loss: 0.0,
            total_tokens: 0,
            accumulation_counter: 0,
            best_val_loss: None,
            patience_counter: 0,
            completed_epochs: 0,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_model(model: CausalLM, config: TrainerConfig) -> Self {
        Self {
            config,
            model,
            trainable: None,
            optimizer: None,
            step: 0,
            total_loss: 0.0,
            total_tokens: 0,
            accumulation_counter: 0,
            best_val_loss: None,
            patience_counter: 0,
            completed_epochs: 0,
            stop_flag: Arc::new(AtomicBool::new(false)),
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
        let mut optimizer = Adam::new(params, self.config.learning_rate);
        if self.config.weight_decay > 0.0 {
            optimizer.set_weight_decay(self.config.weight_decay);
        }
        if self.config.max_grad_norm.is_some() {
            optimizer.set_max_grad_norm(self.config.max_grad_norm);
        }
        self.trainable = Some(trainable);
        self.optimizer = Some(optimizer);
    }

    fn lr_at_step(step: usize, base_lr: f32, warmup_steps: usize, max_steps: usize) -> f32 {
        if step < warmup_steps {
            base_lr * (step as f32 / warmup_steps.max(1) as f32)
        } else {
            let progress = ((step - warmup_steps) as f32
                / (max_steps - warmup_steps).max(1) as f32).min(1.0);
            base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }

    pub fn train_batch(&mut self, tokens: &[u32], targets: &[u32]) -> Option<f32> {
        if self.should_stop() {
            return None;
        }

        let trainable = match self.trainable.as_ref() {
            Some(t) => t,
            None => { warn!("train_step called without prepare()"); return None; }
        };
        let optimizer = match self.optimizer.as_mut() {
            Some(o) => o,
            None => { warn!("train_step called without prepare()"); return None; }
        };

        let seq = tokens.len().min(self.config.seq_length);
        if seq == 0 {
            return None;
        }

        let input_t = Tensor::from_slice(
            &tokens[..seq].iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[seq],
        );
        let target_t = Tensor::from_slice(
            &targets[..seq].iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[seq],
        );

        let logits = trainable.forward(&input_t);
        let loss = cross_entropy_loss(&logits, &target_t).mean();

        loss.backward();

        let loss_val = loss.data()[0];
        if !loss_val.is_finite() {
            warn!("NaN/Inf loss detected ({}) — skipping step", loss_val);
            optimizer.zero_grad();
            return None;
        }
        // Accumulate total loss as sum over tokens (loss_val = per-token average * seq length)
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
            optimizer.step();
            optimizer.zero_grad();
            self.step += 1;
            self.accumulation_counter = 0;

            if let Some(ref path) = self.config.save_path {
                if self.step % self.config.save_every == 0 {
                    trainable.sync_to_inference(&mut self.model);
                    let save_file = format!("{}.step-{}.safetensors", path, self.step);
                    if let Err(e) = trainable.save_checkpoint(&save_file) {
                        warn!("Failed to save checkpoint at step {}: {}", self.step, e);
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
        if self.total_tokens == 0 { 0.0 } else { self.total_loss / self.total_tokens as f64 }
    }

    pub fn save_checkpoint(&self) {
        if let Some(ref path) = self.config.save_path {
            if let Some(ref trainable) = self.trainable {
                let save_file = format!("{}.final.safetensors", path);
                if let Err(e) = trainable.save_checkpoint(&save_file) {
                    warn!("Failed to save final checkpoint: {}", e);
                }
            }
        }
    }

    pub fn save(&self, path: &str) -> crate::FoundationResult<()> {
        if let Some(ref trainable) = self.trainable {
            trainable.save_checkpoint(path).map_err(|e| {
                crate::FoundationError::Processing(format!("Failed to save checkpoint: {}", e))
            })
        } else {
            let trainable = TrainableCausalLM::from_inference(&self.model);
            trainable.save_checkpoint(path).map_err(|e| {
                crate::FoundationError::Processing(format!("Failed to save checkpoint: {}", e))
            })
        }
    }

    pub fn load(model: &mut CausalLM, path: &str) -> crate::FoundationResult<()> {
        TrainableCausalLM::load_checkpoint(model, path)
    }

    pub fn sync_weights(&mut self) {
        if let Some(ref trainable) = self.trainable {
            let mut model_clone = self.model.clone();
            trainable.sync_to_inference(&mut model_clone);
            self.model = model_clone;
        }
    }

    pub fn evaluate_loss(&self, sequences: &[Vec<u32>], seq_length: usize) -> EvalMetrics {
        let trainable = match self.trainable.as_ref() {
            Some(t) => t,
            None => {
                warn!("evaluate_loss called without prepare()");
                return EvalMetrics { avg_loss: 0.0, perplexity: 0.0, total_tokens: 0 };
            }
        };

        let mut total_loss = 0.0f64;
        let mut total_tokens = 0usize;

        for tokens in sequences {
            if tokens.len() < 2 { continue; }
            for chunk in tokens.chunks(seq_length + 1) {
                if chunk.len() < 2 { continue; }
                let input_t = Tensor::from_slice(
                    &chunk[..chunk.len()-1].iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[chunk.len()-1],
                );
                let target_t = Tensor::from_slice(
                    &chunk[1..].iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[chunk.len()-1],
                );
                let logits = trainable.forward(&input_t);
                let loss = cross_entropy_loss(&logits, &target_t).mean();
                let step_loss = loss.data()[0] as f64;
                if !step_loss.is_finite() {
                    warn!("NaN/Inf detected during evaluation — skipping chunk");
                    continue;
                }
                total_loss += step_loss * (chunk.len()-1) as f64;
                total_tokens += chunk.len() - 1;
            }
        }

        // Clear tape entries created during eval (no gradient needed)
        clear_tape();

        EvalMetrics {
            avg_loss: if total_tokens > 0 { total_loss / total_tokens as f64 } else { 0.0 },
            perplexity: if total_tokens > 0 { (total_loss / total_tokens as f64).exp() } else { 0.0 },
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
                trainable.sync_to_inference(&mut model_clone);
                let save_file = format!("{}.epoch-{}.safetensors", path, epoch);
                if let Err(e) = trainable.save_checkpoint(&save_file) {
                    warn!("Failed to save epoch checkpoint: {}", e);
                }
            }
        }
    }
}
