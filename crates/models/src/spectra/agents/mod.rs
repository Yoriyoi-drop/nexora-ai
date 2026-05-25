//! NXR-SPECTRA Agents Module
//!
//! Individual agent implementations for spectral analysis and processing

pub mod artistic_weaver;
pub mod creative_muse;
pub mod frequency_analyzer;
pub mod innovation_engine;
pub mod spectral_analyzer;
pub mod spectral_mapper;
pub mod spectral_processor;
pub mod spectrum_analyzer;
pub mod style_adapter;

// Re-export all agents
pub use artistic_weaver::*;
pub use creative_muse::*;
pub use frequency_analyzer::*;
pub use innovation_engine::*;
pub use spectral_analyzer::*;
pub use spectral_mapper::*;
pub use spectral_processor::*;
pub use spectrum_analyzer::*;
pub use style_adapter::*;

use nexora_shared::base_model::NxrModelResult;

#[derive(Debug, Clone)]
pub struct SpectraAgents {
    config: super::config::SpectraConfig,
}

impl Default for SpectraAgents {
    fn default() -> Self {
        Self {
            config: super::config::SpectraConfig::default(),
        }
    }
}

impl SpectraAgents {
    pub fn new(config: &super::config::SpectraConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Preprocessing: spectral creative analysis of input text.
    /// Returns a string annotated with spectral processing metadata.
    pub async fn analyze(&self, input: &str) -> NxrModelResult<String> {
        let creativity = self.config.get_creativity_level();
        let modalities = self.config.get_supported_modalities();
        Ok(format!(
            "[Spectra creative analysis] Input ({} chars) processed — creativity: {:?}, modalities: {:?}",
            input.len(),
            creativity,
            modalities,
        ))
    }
}
