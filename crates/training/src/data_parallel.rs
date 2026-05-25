use std::sync::Mutex;

use ndarray::ArrayD;
use nexora_autograd::ops::cross_entropy_loss;
use nexora_autograd::Tensor;

use crate::Trainer;

pub struct DataParallelTrainer {
    pub num_workers: usize,
    pub workers: Vec<Trainer>,
    pub master: Mutex<Trainer>,
}

impl DataParallelTrainer {
    pub fn new<F>(num_workers: usize, create_trainer_fn: F) -> Self
    where
        F: Fn() -> Trainer,
    {
        let master = Mutex::new(create_trainer_fn());
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let mut worker = create_trainer_fn();
            worker.prepare();
            workers.push(worker);
        }
        Self {
            num_workers,
            workers,
            master,
        }
    }

    pub fn sync_weights_to_workers(&self) {
        let master = self.master.lock().unwrap();
        let master_params = master
            .trainable
            .as_ref()
            .map(|t| {
                t.parameters()
                    .iter()
                    .map(|p| p.data())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for worker in &self.workers {
            if let Some(wt) = worker.trainable.as_ref() {
                let w_params = wt.parameters();
                for (m_data, w) in master_params.iter().zip(w_params.iter()) {
                    w.set_data(m_data.clone());
                }
            }
        }
    }

    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> Option<f32> {
        if self.num_workers == 0 || tokens.is_empty() || targets.is_empty() {
            let mut master = self.master.lock().unwrap();
            return master.train_batch(tokens, targets);
        }

        let total_seq = tokens.len().min({
            let master = self.master.lock().unwrap();
            master.config.seq_length
        });
        if total_seq == 0 {
            return None;
        }

        let chunk_size = total_seq.div_ceil(self.num_workers);

        let worker_losses: Vec<Option<f32>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(self.num_workers);

            for worker_idx in 0..self.num_workers {
                let start = worker_idx * chunk_size;
                let end = (start + chunk_size).min(total_seq);
                if start >= end {
                    continue;
                }
                let w_tokens = tokens[start..end].to_vec();
                let w_targets = targets[start..end].to_vec();
                let worker = &mut self.workers[worker_idx];

                handles.push(s.spawn(move || {
                    let seq = w_tokens.len();
                    let input_t = Tensor::from_slice(
                        &w_tokens.iter().map(|&t| t as f32).collect::<Vec<_>>(),
                        &[seq],
                    );
                    let target_t = Tensor::from_slice(
                        &w_targets.iter().map(|&t| t as f32).collect::<Vec<_>>(),
                        &[seq],
                    );
                    let trainable = match worker.trainable.as_ref() {
                        Some(t) => t,
                        None => return None,
                    };
                    let logits = trainable.forward(&input_t);
                    let loss = cross_entropy_loss(&logits, &target_t).mean();
                    loss.backward();
                    Some(loss.data()[0])
                }));
            }

            handles.into_iter().map(|h| h.join().ok().flatten()).collect()
        });

        let valid_losses: Vec<f32> = worker_losses.iter().filter_map(|&l| l).collect();
        if valid_losses.is_empty() {
            return None;
        }
        let avg_loss = valid_losses.iter().sum::<f32>() / valid_losses.len() as f32;

        self.sync_gradients();
        self.optimizer_step();

        Some(avg_loss)
    }

    pub fn sync_gradients(&self) {
        let master_params = {
            let master = self.master.lock().unwrap();
            master
                .trainable
                .as_ref()
                .map(|t| t.parameters())
                .unwrap_or_default()
        };

        let mut all_grads: Vec<Vec<Option<ArrayD<f32>>>> = Vec::with_capacity(self.num_workers);
        for worker in &self.workers {
            let w_params = worker
                .trainable
                .as_ref()
                .map(|t| t.parameters())
                .unwrap_or_default();
            let grads: Vec<Option<ArrayD<f32>>> = w_params.iter().map(|p| p.grad()).collect();
            all_grads.push(grads);
        }

        let master = self.master.lock().unwrap();
        if let Some(ref t) = master.trainable {
            let master_params = t.parameters();
            for (i, mp) in master_params.iter().enumerate() {
                let mut sum_grad: Option<ArrayD<f32>> = None;
                let mut count = 0usize;
                for w_grads in &all_grads {
                    if let Some(ref g) = w_grads[i] {
                        sum_grad = match sum_grad {
                            Some(ref s) => Some(s + g),
                            None => Some(g.clone()),
                        };
                        count += 1;
                    }
                }
                if let Some(s) = sum_grad {
                    if count > 0 {
                        let mean = s.mapv(|v| v / count as f32);
                        mp.set_grad(mean);
                    }
                }
            }
        }
    }

    pub fn optimizer_step(&self) {
        let master = self.master.lock().unwrap();
        if let Some(ref mut opt) = master.optimizer {
            opt.step();
            opt.zero_grad();
        }
    }

    pub fn prepare_master(&self) {
        let mut master = self.master.lock().unwrap();
        master.prepare();
    }
}
