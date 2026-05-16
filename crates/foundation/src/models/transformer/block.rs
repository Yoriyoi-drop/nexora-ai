use ndarray::{Array1, Array2};

use super::rms_norm::RMSNorm;
use super::gqa::{GQA, KVCacheEntry};
use super::swiglu::SwiGLU;

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
    pub attention: GQA,
    pub ffn: SwiGLU,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize,
               head_dim: usize, intermediate_size: usize, norm_eps: f32) -> Self {
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
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward_with_kv(&normed, cache, layer_idx, cos, sin);
        let after_attn = x + attn_out;

        let normed_ffn = self.ffn_norm.forward(&after_attn);
        let ffn_out = self.ffn.forward(&normed_ffn);
        after_attn + ffn_out
    }

    pub fn forward_no_cache(&self, x: &Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) -> Array2<f32> {
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, None, 0, cos, sin);
        let after_attn = x + attn_out;

        let normed_ffn = self.ffn_norm.forward(&after_attn);
        let ffn_out = self.ffn.forward(&normed_ffn);
        after_attn + ffn_out
    }
}
