use ndarray::ArrayD;
use nexora_autograd::ops::cross_entropy_loss;
use nexora_autograd::{Tensor, TensorOps};

use crate::Trainer;

pub struct LossScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    stable_steps: usize,
}

impl LossScaler {
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            stable_steps: 0,
        }
    }

    pub fn default() -> Self {
        Self::new(65536.0)
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    pub fn unscale_grad(grad: &mut ArrayD<f32>, scale: f32) {
        grad.mapv_inplace(|g| g / scale);
    }

    pub fn has_overflow(grad: &ArrayD<f32>) -> bool {
        grad.iter().any(|&x| x.is_infinite() || x.is_nan())
    }

    pub fn update(&mut self, grad_overflow: bool) {
        if grad_overflow {
            self.scale *= self.backoff_factor;
            self.stable_steps = 0;
        } else {
            self.stable_steps += 1;
            if self.stable_steps >= self.growth_interval {
                self.scale = (self.scale * self.growth_factor).min(2_f32.powi(24));
                self.stable_steps = 0;
            }
        }
    }
}

pub struct MixedPrecisionTrainer {
    pub base: Trainer,
    pub scaler: LossScaler,
    pub use_amp: bool,
}

impl MixedPrecisionTrainer {
    pub fn new(base: Trainer) -> Self {
        Self {
            scaler: LossScaler::default(),
            base,
            use_amp: true,
        }
    }

    pub fn with_amp(base: Trainer, use_amp: bool) -> Self {
        Self {
            scaler: LossScaler::default(),
            base,
            use_amp,
        }
    }

    pub fn set_use_amp(&mut self, use_amp: bool) {
        self.use_amp = use_amp;
    }

    pub fn set_loss_scaler(&mut self, initial_scale: f32, growth_interval: usize) {
        self.scaler = LossScaler::new(initial_scale);
        self.scaler.growth_interval = growth_interval;
    }

    pub fn scaler(&self) -> &LossScaler {
        &self.scaler
    }

    pub fn scaler_mut(&mut self) -> &mut LossScaler {
        &mut self.scaler
    }

    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> Option<f32> {
        if !self.use_amp {
            return self.base.train_batch(tokens, targets);
        }

        let trainable = self.base.trainable.as_ref()?;
        let optimizer = self.base.optimizer.as_mut()?;

        let seq = tokens.len().min(self.base.config.seq_length);
        if seq == 0 {
            return None;
        }

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

        let loss_val = loss.data()[0];
        if !loss_val.is_finite() {
            return None;
        }

        let scale_t = Tensor::from_slice(&[self.scaler.scale], &[1]);
        let scaled_loss = loss.mul(&scale_t);
        scaled_loss.backward();

        let mut overflow = false;
        for p in trainable.parameters() {
            if let Some(g) = p.grad() {
                if LossScaler::has_overflow(&g) {
                    overflow = true;
                    break;
                }
                let mut grad = g;
                LossScaler::unscale_grad(&mut grad, self.scaler.scale);
                p.set_grad(grad);
            }
        }

        if overflow {
            self.scaler.update(true);
            optimizer.zero_grad();
            return None;
        }

        self.base.total_loss += loss_val as f64 * seq as f64;
        self.base.total_tokens += seq;
        self.base.accumulation_counter += 1;

        if self.base.accumulation_counter >= self.base.config.batch_size {
            let new_lr = Trainer::lr_at_step(
                self.base.step + 1,
                self.base.config.learning_rate,
                self.base.config.warmup_steps,
                self.base.config.max_steps,
            );
            optimizer.lr = new_lr;

            optimizer.step();
            optimizer.zero_grad();
            self.base.step += 1;
            self.base.accumulation_counter = 0;

            self.scaler.update(false);
        }

        Some(loss_val)
    }
}
