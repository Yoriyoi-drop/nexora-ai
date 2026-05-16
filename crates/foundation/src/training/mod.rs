use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use nexora_deeplearning::autograd::{Tensor, TensorOps, Adam};
use nexora_deeplearning::autograd::ops::cross_entropy_loss;

use crate::models::transformer::{CausalLM, TrainableCausalLM, TransformerConfig};

pub struct TrainerConfig {
    pub learning_rate: f32,
    pub max_steps: usize,
    pub seq_length: usize,
    pub vocab_size: usize,
    pub save_path: Option<String>,
    pub save_every: usize,
    pub report_every: usize,
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
        let optimizer = Adam::new(params, self.config.learning_rate);
        self.trainable = Some(trainable);
        self.optimizer = Some(optimizer);
    }

    pub fn train_batch(&mut self, tokens: &[u32], targets: &[u32]) -> Option<f32> {
        if self.should_stop() {
            return None;
        }

        let trainable = self.trainable.as_ref().expect("Trainable not prepared");
        let optimizer = self.optimizer.as_mut().expect("Optimizer not prepared");

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
        optimizer.step();
        optimizer.zero_grad();

        let loss_val = loss.data()[0];
        self.step += 1;
        self.total_loss += loss_val as f64;

        if let Some(ref path) = self.config.save_path {
            if self.step % self.config.save_every == 0 {
                trainable.sync_to_inference(&mut self.model);
                let save_file = format!("{}.step-{}.safetensors", path, self.step);
                let _ = trainable.save_checkpoint(&save_file);
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
        if self.step == 0 { 0.0 } else { self.total_loss / self.step as f64 }
    }

    pub fn save_checkpoint(&self) {
        if let Some(ref path) = self.config.save_path {
            if let Some(ref trainable) = self.trainable {
                let save_file = format!("{}.final.safetensors", path);
                let _ = trainable.save_checkpoint(&save_file);
            }
        }
    }

    pub fn save(&self, path: &str) -> crate::FoundationResult<()> {
        if let Some(ref trainable) = self.trainable {
            trainable.save_checkpoint(path)
        } else {
            let trainable = TrainableCausalLM::from_inference(&self.model);
            trainable.save_checkpoint(path)
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
}
