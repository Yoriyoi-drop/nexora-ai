use ndarray::{Array1, Array2};
use crate::TransformerResult;

/// Hook for injecting processing between transformer layers.
pub trait LayerInjector: std::fmt::Debug + Send {
    /// Called after a layer's forward pass.
    fn after_layer(
        &mut self,
        layer_idx: usize,
        h: &mut Array2<f32>,
        pos: usize,
    ) -> TransformerResult<()>;

    /// Reset runtime state (ring buffers, caches) for a new sequence.
    /// Default is a no-op — override if the injector maintains per-sequence state.
    fn reset(&mut self) {}

    /// GPU-native injector: receives a mutable `GpuTensor` instead of CPU array.
    /// Default implementation falls back to CPU: reads GPU tensor, calls `after_layer`,
    /// and writes the result back.
    /// Override this for zero-copy GPU-native injection.
    #[cfg(feature = "gpu")]
    fn after_layer_gpu(
        &mut self,
        layer_idx: usize,
        h: &mut nexora_autograd::gpu::GpuTensor,
        pos: usize,
        _ctx: &nexora_autograd::gpu::GpuContext,
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        let h_cpu = h.to_cpu()?;
        let dims = h_cpu.shape().to_vec();
        let mut h_2d = h_cpu.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
            nexora_autograd::gpu::GpuError::Unsupported(format!(
                "injector reshape [{}x{}]: {}",
                dims[0], dims[1], e
            ))
        })?;
        self.after_layer(layer_idx, &mut h_2d, pos).map_err(|e| {
            nexora_autograd::gpu::GpuError::Unsupported(format!("injector after_layer: {}", e))
        })?;
        let h_dyn = h_2d.into_dyn();
        *h = nexora_autograd::gpu::GpuTensor::from_cpu(&h_dyn)?;
        Ok(())
    }
}

pub fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_shape_fn(logits.len(), |i| {
        if sum > 0.0 {
            exps[i] / sum
        } else {
            1.0 / logits.len() as f32
        }
    })
}

#[cfg(feature = "gpu")]
pub(crate) fn sample_token_gpu(
    logits: &Array1<f32>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
) -> u32 {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    let fallback = || sample_token(logits, temperature, top_k);
    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("GPU context unavailable, falling back to CPU: {}", e);
            return fallback();
        }
    };
    let shape = vec![1, logits.len()];
    let cpu = match ndarray::ArrayD::from_shape_vec(shape.clone(), logits.to_vec()) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("Failed to upload tensor to GPU: {}", e);
            return fallback();
        }
    };
    let gpu = match GpuTensor::from_cpu(&cpu) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!("Failed to upload tensor to GPU: {}", e);
            return fallback();
        }
    };
    match ctx.gpu_sample(
        &gpu,
        temperature,
        u32::try_from(top_k).unwrap_or(u32::MAX),
        top_p,
        seed,
    ) {
        Ok(result) => {
            let raw = match result.to_cpu_raw_bytes() {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("Failed to download GPU tensor to CPU: {}", e);
                    return fallback();
                }
            };
            u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]])
        }
        Err(e) => {
            tracing::warn!("GPU sampling failed: {}", e);
            fallback()
        }
    }
}

/// Sample a token from GPU-resident logits — zero CPU round-trip for logits.
/// Takes logits that are ALREADY on GPU, returns sampled token as GPU tensor (still on GPU).
#[cfg(feature = "gpu")]
pub fn sample_token_gpu_keep_gpu(
    logits_gpu: &nexora_autograd::gpu::GpuTensor,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
    use nexora_autograd::gpu::GpuContext;

    let ctx = GpuContext::global()?;
    ctx.gpu_sample(
        logits_gpu,
        temperature,
        u32::try_from(top_k).unwrap_or(u32::MAX),
        top_p,
        seed,
    )
}

pub fn sample_token(logits: &Array1<f32>, temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 {
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return u32::try_from(best).unwrap_or(u32::MAX);
    }
    let scaled: Array1<f32> = logits.mapv(|v| v / temperature);
    let probs = softmax(&scaled);

    if top_k > 0 && top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let k = top_k.min(probs.len());
        let sum_k: f32 = indices[..k].iter().map(|&i| probs[i]).sum();
        if sum_k <= 0.0 {
            return indices[0] as u32;
        }
        let r: f32 = rand::random::<f32>() * sum_k;
        let mut cum = 0.0;
        for &i in &indices[..k] {
            cum += probs[i];
            if cum >= r {
                return i as u32;
            }
        }
        return indices[k - 1] as u32;
    }

    let r: f32 = rand::random();
    let mut cum = 0.0;
    for i in 0..probs.len() {
        cum += probs[i];
        if cum >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
