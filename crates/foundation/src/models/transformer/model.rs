use ndarray::{Array1, Array2};
use rand::Rng;

use super::config::TransformerConfig;
use super::block::TransformerBlock;
use super::gqa::KVCacheEntry;
use super::rope::RoPE;

#[derive(Debug, Clone)]
pub struct CausalLM {
    pub config: TransformerConfig,
    pub token_embedding: Array2<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub norm: super::rms_norm::RMSNorm,
    pub lm_head: Array2<f32>,
    pub rope: RoPE,
    pub precomputed_cos: Array1<f32>,
    pub precomputed_sin: Array1<f32>,
}

impl CausalLM {
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (config.hidden_size as f32).sqrt().recip();

        let token_embedding = Array2::from_shape_fn(
            (config.vocab_size, config.hidden_size),
            |_| rng.gen::<f32>() * 2.0 * scale - scale,
        );

        let blocks = (0..config.num_layers)
            .map(|_| TransformerBlock::new(
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim(),
                config.intermediate_size,
                config.norm_eps,
            ))
            .collect();

        let norm = super::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps);

        let lm_head = Array2::from_shape_fn(
            (config.vocab_size, config.hidden_size),
            |_| rng.gen::<f32>() * 2.0 * scale - scale,
        );

        let rope = RoPE::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let (cos_full, sin_full) = rope.precompute_freqs_cis();
        let half = config.head_dim() / 2;
        let precomputed_cos = cos_full.into_shape(config.max_seq_len * half).unwrap_or_else(|_| {
            Array1::zeros(config.max_seq_len * half)
        });
        let precomputed_sin = sin_full.into_shape(config.max_seq_len * half).unwrap_or_else(|_| {
            Array1::zeros(config.max_seq_len * half)
        });

        Self {
            config,
            token_embedding,
            blocks,
            norm,
            lm_head,
            rope,
            precomputed_cos,
            precomputed_sin,
        }
    }

    pub fn forward(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> Array1<f32> {
        let batch_size = 1;
        let seq_len = input_ids.len();

        let mut h = Array2::zeros((batch_size, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = (token_id as usize).min(self.config.vocab_size - 1);
            for j in 0..self.config.hidden_size {
                h[[0, j]] = self.token_embedding[[tid, j]];
            }
        }

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let pos = kv_cache.first().map(|e| e.k.shape()[0]).unwrap_or(0);
            let half = self.config.head_dim() / 2;
            let cos_slice = if pos * half < self.precomputed_cos.len() {
                self.precomputed_cos.slice(ndarray::s![pos * half..(pos + 1) * half]).to_owned()
            } else {
                Array1::zeros(half)
            };
            let sin_slice = if pos * half < self.precomputed_sin.len() {
                self.precomputed_sin.slice(ndarray::s![pos * half..(pos + 1) * half]).to_owned()
            } else {
                Array1::zeros(half)
            };

            h = block.forward(&h, kv_cache, layer_idx, &cos_slice, &sin_slice);
        }

        h = self.norm.forward(&h);

        let h_row = h.row(0);
        let mut logits = Array1::zeros(self.config.vocab_size);
        for i in 0..self.config.vocab_size {
            let mut dot = 0.0;
            for j in 0..self.config.hidden_size {
                dot += h_row[j] * self.lm_head[[i, j]];
            }
            logits[i] = dot;
        }

        logits
    }

    pub fn reset_cache(&self) -> Vec<KVCacheEntry> {
        Vec::with_capacity(self.config.num_layers)
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = self.token_embedding.len();
        count += self.lm_head.len();
        for block in &self.blocks {
            count += block.attention.wq.len();
            count += block.attention.wk.len();
            count += block.attention.wv.len();
            count += block.attention.wo.len();
            count += block.ffn.w1.len();
            count += block.ffn.w2.len();
            count += block.ffn.w3.len();
            count += block.attention_norm.weight.len();
            count += block.ffn_norm.weight.len();
        }
        count += self.norm.weight.len();
        count
    }
}
