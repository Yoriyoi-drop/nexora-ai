//! Model configuration for Nexora-AI.

use std::collections::HashMap;

use nexora_foundation::shared::model_identity::NxrModelId;

/// Parse a model ID string into NxrModelId (case-insensitive).
fn parse_model_id(s: &str) -> Option<NxrModelId> {
    match s.to_lowercase().as_str() {
        "omnis" => Some(NxrModelId::Omnis),
        "vortex" => Some(NxrModelId::Vortex),
        "aether" => Some(NxrModelId::Aether),
        "spectra" => Some(NxrModelId::Spectra),
        "nexum" => Some(NxrModelId::Nexum),
        "axiom" => Some(NxrModelId::Axiom),
        "cipher" => Some(NxrModelId::Cipher),
        "swift" => Some(NxrModelId::Swift),
        "kronos" => Some(NxrModelId::Kronos),
        "genesis" => Some(NxrModelId::Genesis),
        _ => None,
    }
}

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
    /// Checkpoint paths per model — loads pre-trained weights for standby models at startup.
    /// Example: { "vortex": "./checkpoints/vortex.safetensors", "aether": "./checkpoints/aether.safetensors" }
    #[serde(default)]
    pub model_checkpoints: HashMap<String, String>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            active_model: None,
            model_checkpoints: HashMap::new(),
        }
    }
}

impl ModelsConfig {
    /// Convert string-keyed checkpoint map to NxrModelId-keyed map.
    /// Silently skips invalid model IDs.
    pub fn resolved_checkpoints(&self) -> HashMap<NxrModelId, String> {
        self.model_checkpoints
            .iter()
            .filter_map(|(k, v)| {
                match parse_model_id(k) {
                    Some(id) => Some((id, v.clone())),
                    None => {
                        tracing::warn!("Unknown model ID in checkpoints: {}", k);
                        None
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_id() {
        assert_eq!(parse_model_id("omnis"), Some(NxrModelId::Omnis));
        assert_eq!(parse_model_id("VORTEX"), Some(NxrModelId::Vortex));
        assert_eq!(parse_model_id("Aether"), Some(NxrModelId::Aether));
        assert_eq!(parse_model_id("unknown"), None);
    }

    #[test]
    fn test_resolved_checkpoints() {
        let mut ckpts = HashMap::new();
        ckpts.insert("vortex".into(), "./vtx.safetensors".into());
        ckpts.insert("bogus".into(), "./x.safetensors".into());
        let config = ModelsConfig {
            model_checkpoints: ckpts,
            ..Default::default()
        };
        let resolved = config.resolved_checkpoints();
        assert_eq!(resolved.len(), 1);
        assert!(resolved.contains_key(&NxrModelId::Vortex));
        assert_eq!(resolved[&NxrModelId::Vortex], "./vtx.safetensors");
    }
}
