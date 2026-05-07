//! Models configuration

use serde::{Deserialize, Serialize};

/// Models configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f32,
    pub use_rope: bool,
    pub use_gqa: bool,
    pub use_swiglu: bool,
    pub model_path: Option<String>,
    pub enable_caching: bool,
    pub cache_size: usize,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50000,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            d_ff: 3072,
            max_seq_len: 2048,
            dropout: 0.1,
            use_rope: true,
            use_gqa: true,
            use_swiglu: true,
            model_path: None,
            enable_caching: true,
            cache_size: 1000,
        }
    }
}
