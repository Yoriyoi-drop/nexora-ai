#![cfg_attr(not(feature = "gpu"), allow(unused))]
use tracing::info;

pub struct GpuEmbedding {
    weight: Option<nexora_deeplearning::autograd::gpu::GpuTensor>,
    vocab_size: usize,
    embed_dim: usize,
}

impl GpuEmbedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let weight = try_init_weight(vocab_size, embed_dim);
        if weight.is_some() {
            info!(
                "GPU embedding initialized: vocab={vocab_size}, dim={embed_dim}"
            );
        }
        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    pub fn is_gpu(&self) -> bool {
        self.weight.is_some()
    }

    pub fn numel(&self) -> usize {
        self.vocab_size * self.embed_dim
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn lookup(&self, token_ids: &[u32]) -> Option<Vec<f32>> {
        let ctx = nexora_deeplearning::autograd::gpu::GpuContext::global().ok()?;
        let weight = self.weight.as_ref()?;

        let ids_f32: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
        let ids_tensor = match nexora_deeplearning::autograd::gpu::GpuTensor::from_slice(
            vec![token_ids.len()],
            &ids_f32,
        ) {
            Ok(t) => t,
            Err(e) => {
                info!("GPU embedding: failed to create ids tensor: {e}");
                return None;
            }
        };

        let result = match ctx.embedding(&ids_tensor, weight) {
            Ok(t) => t,
            Err(e) => {
                info!("GPU embedding lookup failed: {e}");
                return None;
            }
        };

        match result.to_cpu() {
            Ok(arr) => Some(arr.as_slice().unwrap_or_default().to_vec()),
            Err(e) => {
                info!("GPU embedding readback failed: {e}");
                None
            }
        }
    }

    pub fn lookup_batched(&self, batch_ids: &[Vec<u32>]) -> Option<Vec<Vec<f32>>> {
        let flat: Vec<u32> = batch_ids.iter().flat_map(|ids| ids.iter().copied()).collect();
        let result = self.lookup(&flat)?;
        let mut batched = Vec::with_capacity(batch_ids.len());
        let mut offset = 0;
        for ids in batch_ids {
            let len = ids.len() * self.embed_dim;
            batched.push(result[offset..offset + len].to_vec());
            offset += len;
        }
        Some(batched)
    }

    pub fn update_weights(&mut self, cpu_weights: &ndarray::Array2<f32>) {
        let (rows, cols) = cpu_weights.dim();
        if rows != self.vocab_size || cols != self.embed_dim {
            info!(
                "GPU embedding weight shape mismatch: got ({rows},{cols}), expected ({},{})",
                self.vocab_size, self.embed_dim
            );
            return;
        }
        match nexora_deeplearning::autograd::gpu::GpuTensor::from_cpu(&cpu_weights.clone().into_dyn()) {
            Ok(t) => {
                self.weight = Some(t);
                info!("GPU embedding weights updated");
            }
            Err(e) => {
                info!("GPU embedding weight update failed: {e}");
            }
        }
    }

    pub fn reset_weights(&mut self) {
        let arr = ndarray::Array2::<f32>::zeros((self.vocab_size, self.embed_dim));
        if let Ok(t) = nexora_deeplearning::autograd::gpu::GpuTensor::from_cpu(&arr.into_dyn()) {
            self.weight = Some(t);
        }
    }
}

fn try_init_weight(vocab_size: usize, embed_dim: usize) -> Option<nexora_deeplearning::autograd::gpu::GpuTensor> {
    let _ = nexora_deeplearning::autograd::gpu::GpuContext::init().ok()?;
    let ctx = nexora_deeplearning::autograd::gpu::GpuContext::global().ok()?;
    let arr = ndarray::Array2::<f32>::zeros((vocab_size, embed_dim));
    match nexora_deeplearning::autograd::gpu::GpuTensor::from_cpu(&arr.into_dyn()) {
        Ok(t) => {
            ctx.flush();
            Some(t)
        }
        Err(e) => {
            info!("GPU embedding weight init failed: {e}");
            None
        }
    }
}
