use ndarray::{Array1, Array2};

use super::gqa::{KVCacheEntry, PagedCacheReader, GQA};
use super::rms_norm::RMSNorm;
use super::sharded::ShardCollective;
use super::swiglu::SwiGLU;
use crate::TransformerResult;

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
    pub attention: GQA,
    pub ffn: SwiGLU,
    pub experts: Option<Vec<SwiGLU>>,
}

impl TransformerBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        norm_eps: f32,
        num_experts: usize,
        expert_intermediate_size: usize,
    ) -> Self {
        let experts = if num_experts > 0 {
            Some(
                (0..num_experts)
                    .map(|_| SwiGLU::new(hidden_size, expert_intermediate_size))
                    .collect(),
            )
        } else {
            None
        };
        Self {
            attention_norm: RMSNorm::new(hidden_size, norm_eps),
            ffn_norm: RMSNorm::new(hidden_size, norm_eps),
            attention: GQA::new(hidden_size, num_heads, num_kv_heads, head_dim),
            ffn: SwiGLU::new(hidden_size, intermediate_size),
            experts,
        }
    }

    pub fn init_random(&mut self, hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize, intermediate_size: usize, expert_intermediate_size: usize) {
        self.attention.init_random(hidden_size, num_heads, num_kv_heads, head_dim);
        self.ffn.init_random(hidden_size, intermediate_size);
        if let Some(experts) = &mut self.experts {
            for e in experts.iter_mut() {
                e.init_random(hidden_size, expert_intermediate_size);
            }
        }
    }

    /// Propagate half-precision flag to attention and FFN sub-layers
    /// and pack f16 weight copies for CPU f16 matmul.
    pub fn set_use_half_precision(&mut self) {
        self.attention.use_half_precision = true;
        self.ffn.use_half_precision = true;
        self.attention.pack_f16_weights();
        self.ffn.pack_f16_weights();
        if let Some(experts) = &mut self.experts {
            for e in experts.iter_mut() {
                e.use_half_precision = true;
                e.pack_f16_weights();
            }
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> TransformerResult<Array2<f32>> {
        self.forward_with_collective(x, cache, layer_idx, cos, sin, None)
    }

    /// Forward pass with optional tensor parallelism all-reduce.
    /// When `collective` is `Some`, attention and FFN partial outputs are
    /// reduced across shards to produce the global result.
    pub fn forward_with_collective(
        &self,
        x: &Array2<f32>,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &[f32],
        sin: &[f32],
        collective: Option<&ShardCollective>,
    ) -> TransformerResult<Array2<f32>> {
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self
            .attention
            .forward_with_kv(&normed, cache, layer_idx, cos, sin)?;

        let after_attn = if let Some(col) = collective {
            x + col.reduce_attn(&attn_out)?
        } else {
            x + attn_out
        };

        let normed_ffn = self.ffn_norm.forward(&after_attn)?;
        let ffn_out = if let Some(experts) = &self.experts {
            let mut sum = Array2::zeros(normed_ffn.dim());
            for expert in experts {
                sum = sum + expert.forward(&normed_ffn)?;
            }
            sum / experts.len() as f32
        } else {
            self.ffn.forward(&normed_ffn)?
        };
        let ffn_global = if let Some(col) = collective {
            col.reduce_ffn(&ffn_out)?
        } else {
            ffn_out
        };
        Ok(after_attn + ffn_global)
    }

    pub fn forward_no_cache(
        &self,
        x: &Array2<f32>,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> TransformerResult<Array2<f32>> {
        self.forward_no_cache_with_collective(x, cos, sin, None)
    }

    pub fn forward_no_cache_with_collective(
        &self,
        x: &Array2<f32>,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
        collective: Option<&ShardCollective>,
    ) -> TransformerResult<Array2<f32>> {
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(
            &normed,
            None,
            0,
            cos.as_slice().unwrap_or(&[]),
            sin.as_slice().unwrap_or(&[]),
        )?;

        let after_attn = if let Some(col) = collective {
            x + col.reduce_attn(&attn_out)?
        } else {
            x + attn_out
        };

        let normed_ffn = self.ffn_norm.forward(&after_attn)?;
        let ffn_out = if let Some(experts) = &self.experts {
            let mut sum = Array2::zeros(normed_ffn.dim());
            for expert in experts {
                sum = sum + expert.forward(&normed_ffn)?;
            }
            sum / experts.len() as f32
        } else {
            self.ffn.forward(&normed_ffn)?
        };
        let ffn_global = if let Some(col) = collective {
            col.reduce_ffn(&ffn_out)?
        } else {
            ffn_out
        };
        Ok(after_attn + ffn_global)
    }

    pub fn forward_paged(
        &self,
        x: &Array2<f32>,
        cache: &mut dyn PagedCacheReader,
        seq_id: u64,
        layer_idx: usize,
        token_pos: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> TransformerResult<Array2<f32>> {
        self.forward_paged_with_collective(x, cache, seq_id, layer_idx, token_pos, cos, sin, None)
    }

    pub fn forward_paged_with_collective(
        &self,
        x: &Array2<f32>,
        cache: &mut dyn PagedCacheReader,
        seq_id: u64,
        layer_idx: usize,
        token_pos: usize,
        cos: &[f32],
        sin: &[f32],
        collective: Option<&ShardCollective>,
    ) -> TransformerResult<Array2<f32>> {
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self
            .attention
            .forward_with_paged(&normed, cache, seq_id, layer_idx, token_pos, cos, sin)?;

        let after_attn = if let Some(col) = collective {
            x + col.reduce_attn(&attn_out)?
        } else {
            x + attn_out
        };

        let normed_ffn = self.ffn_norm.forward(&after_attn)?;
        let ffn_out = if let Some(experts) = &self.experts {
            let mut sum = Array2::zeros(normed_ffn.dim());
            for expert in experts {
                sum = sum + expert.forward(&normed_ffn)?;
            }
            sum / experts.len() as f32
        } else {
            self.ffn.forward(&normed_ffn)?
        };
        let ffn_global = if let Some(col) = collective {
            col.reduce_ffn(&ffn_out)?
        } else {
            ffn_out
        };
        Ok(after_attn + ffn_global)
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
        let attn_out = self
            .attention
            .forward_gpu_with_rope_gpu(&normed, cache, layer_idx, cos_gpu, sin_gpu)?;
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = if let Some(experts) = &self.experts {
            let mut sum = experts[0].forward_gpu(&normed_ffn)?;
            for expert in &experts[1..] {
                let out = expert.forward_gpu(&normed_ffn)?;
                sum = ctx.add(&sum, &out)?;
            }
            ctx.scale_inplace(&sum, 1.0 / experts.len() as f32)?;
            sum
        } else {
            self.ffn.forward_gpu(&normed_ffn)?
        };
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
        let attn_out = self
            .attention
            .forward_gpu(&normed, cache, layer_idx, cos, sin)?;
        // Residual: x + attn_out
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        // 2. FFN sub-block: RMSNorm -> SwiGLU -> residual add
        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = if let Some(experts) = &self.experts {
            let mut sum = experts[0].forward_gpu(&normed_ffn)?;
            for expert in &experts[1..] {
                let out = expert.forward_gpu(&normed_ffn)?;
                sum = ctx.add(&sum, &out)?;
            }
            ctx.scale_inplace(&sum, 1.0 / experts.len() as f32)?;
            sum
        } else {
            self.ffn.forward_gpu(&normed_ffn)?
        };
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
            &normed,
            &mut cache[layer_idx],
            cos_gpu,
            sin_gpu,
        )?;
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = if let Some(experts) = &self.experts {
            let mut sum = experts[0].forward_gpu(&normed_ffn)?;
            for expert in &experts[1..] {
                let out = expert.forward_gpu(&normed_ffn)?;
                sum = ctx.add(&sum, &out)?;
            }
            ctx.scale_inplace(&sum, 1.0 / experts.len() as f32)?;
            sum
        } else {
            self.ffn.forward_gpu(&normed_ffn)?
        };
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
        let attn_out =
            self.attention
                .forward_gpu_with_cache(&normed, &mut cache[layer_idx], cos, sin)?;
        // Residual: x + attn_out
        let after_attn = ctx.add(x_gpu, &attn_out)?;

        // 2. FFN sub-block: RMSNorm -> SwiGLU -> residual add
        let normed_ffn = self.ffn_norm.forward_gpu(&after_attn)?;
        let ffn_out = if let Some(experts) = &self.experts {
            let mut sum = experts[0].forward_gpu(&normed_ffn)?;
            for expert in &experts[1..] {
                let out = expert.forward_gpu(&normed_ffn)?;
                sum = ctx.add(&sum, &out)?;
            }
            ctx.scale_inplace(&sum, 1.0 / experts.len() as f32)?;
            sum
        } else {
            self.ffn.forward_gpu(&normed_ffn)?
        };
        // Residual: after_attn + ffn_out
        ctx.add(&after_attn, &ffn_out)
    }

    #[cfg(feature = "gpu")]
    pub fn preupload_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        self.attention_norm.preupload_gpu()?;
        self.ffn_norm.preupload_gpu()?;
        self.attention.preupload_gpu()?;
        self.ffn.preupload_gpu()?;
        if let Some(experts) = &self.experts {
            for e in experts {
                e.preupload_gpu()?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

fn small_block() -> TransformerBlock {
    let mut block = TransformerBlock::new(8, 4, 2, 4, 16, 1e-6, 0, 0);
    block.init_random(8, 4, 2, 4, 16, 0);
    block
}

    #[test]
    fn test_block_new_shapes() {
        let block = small_block();
        assert_eq!(block.attention_norm.weight.as_ref().unwrap().len(), 8);
        assert_eq!(block.ffn_norm.weight.as_ref().unwrap().len(), 8);
        assert_eq!(block.attention.wq.as_ref().unwrap().dim(), (16, 8));
        assert_eq!(block.ffn.w1.as_ref().unwrap().dim(), (16, 8));
    }

    #[test]
    fn test_block_forward_shape() {
        let block = small_block();
        let x = array![[1.0; 8]];
        let mut cache = vec![KVCacheEntry {
            k: vec![],
            v: vec![],
            kv_dim: 8,
            num_kv_heads: 1,
            head_dim: 8,
            k_compressed: Vec::new(),
            v_compressed: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            compressed_seq_len: 0,
        }];
        let cos = vec![0.0, 1.0];
        let sin = vec![1.0, 0.0];
        let out = block.forward(&x, &mut cache, 0, &cos, &sin).unwrap();
        assert_eq!(out.dim(), (1, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_block_forward_no_cache_shape() {
        let block = small_block();
        let x = array![[1.0; 8]];
        let cos = ndarray::Array1::from(vec![0.0, 1.0]);
        let sin = ndarray::Array1::from(vec![1.0, 0.0]);
        let out = block.forward_no_cache(&x, &cos, &sin).unwrap();
        assert_eq!(out.dim(), (1, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_block_forward_batch2() {
        let block = small_block();
        let x = array![[1.0; 8], [2.0; 8]];
        let mut cache = vec![KVCacheEntry {
            k: vec![],
            v: vec![],
            kv_dim: 8,
            num_kv_heads: 1,
            head_dim: 8,
            k_compressed: Vec::new(),
            v_compressed: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            compressed_seq_len: 0,
        }];
        let cos = vec![0.0, 1.0];
        let sin = vec![1.0, 0.0];
        let out = block.forward(&x, &mut cache, 0, &cos, &sin).unwrap();
        assert_eq!(out.dim(), (2, 8));
    }

    #[test]
    fn test_block_forward_accumulates_cache() {
        let block = small_block();
        let mut cache: Vec<KVCacheEntry> = Vec::new();
        let cos = vec![0.0, 1.0];
        let sin = vec![1.0, 0.0];

        let x1 = array![[1.0; 8]];
        let _ = block.forward(&x1, &mut cache, 0, &cos, &sin).unwrap();
        let seq1 = cache[0].seq_len();

        let x2 = array![[2.0; 8]];
        let _ = block.forward(&x2, &mut cache, 0, &cos, &sin).unwrap();
        let seq2 = cache[0].seq_len();

        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);
    }
}
