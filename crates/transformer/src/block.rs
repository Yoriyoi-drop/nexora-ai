use ndarray::{Array1, Array2};

use super::gqa::{KVCacheEntry, PagedCacheReader, GQA};
use super::rms_norm::RMSNorm;
use super::swiglu::SwiGLU;

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
    pub attention: GQA,
    pub ffn: SwiGLU,
}

impl TransformerBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        norm_eps: f32,
    ) -> Self {
        Self {
            attention_norm: RMSNorm::new(hidden_size, norm_eps),
            ffn_norm: RMSNorm::new(hidden_size, norm_eps),
            attention: GQA::new(hidden_size, num_heads, num_kv_heads, head_dim),
            ffn: SwiGLU::new(hidden_size, intermediate_size),
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Array2<f32> {
        #[cfg(feature = "gpu")]
        if let Ok(_ctx) = nexora_autograd::gpu::GpuContext::global() {
            use ndarray::ArrayD;
            use nexora_autograd::gpu::GpuTensor;
            let x_flat: Vec<f32> = x.iter().copied().collect();
            if let Ok(x_cpu) = ArrayD::from_shape_vec(vec![x.shape()[0], x.shape()[1]], x_flat) {
                if let Ok(x_gpu) = GpuTensor::from_cpu(&x_cpu) {
                    if let Ok(result) = self.forward_gpu(&x_gpu, cache, layer_idx, cos, sin) {
                        let cpu = result.to_cpu();
                        if let Ok(arr2) = cpu.into_dimensionality::<ndarray::Ix2>() {
                            return arr2.to_owned();
                        }
                    }
                }
            }
        }

        let normed = self.attention_norm.forward(x);
        let attn_out = self
            .attention
            .forward_with_kv(&normed, cache, layer_idx, cos, sin);
        let after_attn = x + attn_out;

        let normed_ffn = self.ffn_norm.forward(&after_attn);
        let ffn_out = self.ffn.forward(&normed_ffn);
        after_attn + ffn_out
    }

    pub fn forward_no_cache(
        &self,
        x: &Array2<f32>,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Array2<f32> {
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, None, 0, cos, sin);
        let after_attn = x + attn_out;

        let normed_ffn = self.ffn_norm.forward(&after_attn);
        let ffn_out = self.ffn.forward(&normed_ffn);
        after_attn + ffn_out
    }

    pub fn forward_paged(
        &self,
        x: &Array2<f32>,
        cache: &mut dyn PagedCacheReader,
        seq_id: u64,
        layer_idx: usize,
        token_pos: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Array2<f32> {
        let normed = self.attention_norm.forward(x);
        let attn_out = self
            .attention
            .forward_with_paged(&normed, cache, seq_id, layer_idx, token_pos, cos, sin);
        let after_attn = x + attn_out;

        let normed_ffn = self.ffn_norm.forward(&after_attn);
        let ffn_out = self.ffn.forward(&normed_ffn);
        after_attn + ffn_out
    }

    /// GPU forward with pre-uploaded cos/sin GPU tensors.
    /// cos_gpu / sin_gpu shape [1, half] — uploaded once per step, reused across all layers.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_rope_gpu(
        &self,
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos_gpu: &nexora_autograd::gpu::GpuTensor,
        sin_gpu: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuContext;

        let ctx = GpuContext::global()?;

        let normed = self.attention_norm.forward_gpu(x_gpu)?;
        let attn_out = self.attention.forward_gpu_with_rope_gpu(
            &normed, cache, layer_idx, cos_gpu, sin_gpu,
        )?;
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = self.ffn.forward_gpu(&normed_ffn)?;
        ctx.add(&after_attn, &ffn_out)
    }

    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuContext;

        let ctx = GpuContext::global()?;

        // 1. Attention sub-block: RMSNorm -> GQA -> residual add
        let normed = self.attention_norm.forward_gpu(x_gpu)?;
        let attn_out = self.attention.forward_gpu(
            &normed, cache, layer_idx, cos, sin,
        )?;
        // Residual: x + attn_out
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        // 2. FFN sub-block: RMSNorm -> SwiGLU -> residual add
        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = self.ffn.forward_gpu(&normed_ffn)?;
        // Residual: after_attn + ffn_out
        ctx.add(&after_attn, &ffn_out)
    }

    /// GPU forward with GPU-resident KV cache AND pre-uploaded cos/sin GPU tensors.
    /// cos_gpu / sin_gpu shape [1, half] — uploaded once per step, reused across all layers.
    /// No per-layer CPU→GPU upload for RoPE.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache_precomputed_rope(
        &self,
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut [super::gqa::GpuKVCacheEntry],
        layer_idx: usize,
        cos_gpu: &nexora_autograd::gpu::GpuTensor,
        sin_gpu: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuContext;

        let ctx = GpuContext::global()?;

        let normed = self.attention_norm.forward_gpu(x_gpu)?;
        let attn_out = self.attention.forward_gpu_with_cache_precomputed_rope(
            &normed, &mut cache[layer_idx], cos_gpu, sin_gpu,
        )?;
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = self.ffn.forward_gpu(&normed_ffn)?;
        ctx.add(&after_attn, &ffn_out)
    }

    /// GPU forward with GPU-resident KV cache.
    /// No CPU round-trip for K/V — uses GpuKVCacheEntry.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache(
        &self,
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut [super::gqa::GpuKVCacheEntry],
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuContext;

        let ctx = GpuContext::global()?;

        // 1. Attention sub-block: RMSNorm -> GQA -> residual add
        let normed = self.attention_norm.forward_gpu(x_gpu)?;
        let attn_out = self.attention.forward_gpu_with_cache(
            &normed, &mut cache[layer_idx], cos, sin,
        )?;
        // Residual: x + attn_out
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        // 2. FFN sub-block: RMSNorm -> SwiGLU -> residual add
        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = self.ffn.forward_gpu(&normed_ffn)?;
        // Residual: after_attn + ffn_out
        ctx.add(&after_attn, &ffn_out)
    }

    #[cfg(feature = "gpu")]
    pub fn preupload_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        self.attention_norm.preupload_gpu()?;
        self.ffn_norm.preupload_gpu()?;
        self.attention.preupload_gpu()?;
        self.ffn.preupload_gpu()?;
        Ok(())
    }
}
