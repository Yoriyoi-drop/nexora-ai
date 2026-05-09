//! Configuration for HAS-MoE-FFN

use serde::{Serialize, Deserialize};

/// Configuration for layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub use_layer_norm: bool,
    pub activation: String,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            use_layer_norm: true,
            activation: "gelu".to_string(),
        }
    }
}
