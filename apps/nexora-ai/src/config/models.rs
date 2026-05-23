//! Model configuration for Nexora-AI.

use nexora_foundation::shared::model_identity::NxrModelId;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelsConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    /// Active NXR model to use for inference. Default: Omnis.
    /// Supported: Omnis, Vortex, Aether, Spectra, Nexum, Axiom, Cipher, Swift, Kronos, Genesis
    #[serde(default)]
    pub active_model: Option<NxrModelId>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            active_model: None,
        }
    }
}
