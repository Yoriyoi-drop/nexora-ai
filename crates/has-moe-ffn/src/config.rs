//! Configuration for HAS-MoE-FFN

use serde::{Deserialize, Serialize};

/// Configuration for MoE FFN layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeLayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub activation: String,
}

impl Default for MoeLayerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            activation: "gelu".to_string(),
        }
    }
}
