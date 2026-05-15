//! Model configuration for Nexora-AI.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelsConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
        }
    }
}
