use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub use_cache: bool,
    pub norm_eps: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 768,
            num_heads: 12,
            num_kv_heads: 4,
            num_layers: 12,
            max_seq_len: 2048,
            intermediate_size: 3072,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
        }
    }
}

impl TransformerConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn num_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}
