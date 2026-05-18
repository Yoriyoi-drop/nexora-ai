// Multimodal AI framework
//
// Re-exports from `nexora-multimodal` crate, plus CaffeineSpectraIntegration
// that depends on NXR-SPECTRA (model series still in foundation).

pub use nexora_multimodal::*;

use nexora_multimodal::caffeine::MultiModalInputs as CaffeineInputs;
use nexora_multimodal::caffeine::TextInput;
use nexora_shared::base_model::{NxrModel, NxrInput, InputData, NxrOutput, OutputData};
use crate::models::spectra::NxrSpectraModel;

/// Enhanced CAFFEINE with NXR-SPECTRA integration
pub struct CaffeineSpectraIntegration {
    pub caffeine_processor: nexora_multimodal::caffeine::CaffeineProcessor,
    pub spectra_model: NxrSpectraModel,
    pub integration_config: CaffeineSpectraConfig,
}

#[derive(Debug, Clone)]
pub struct CaffeineSpectraConfig {
    pub enable_creative_synthesis: bool,
    pub creativity_level: f32,
    pub style_adaptation: bool,
}

impl Default for CaffeineSpectraConfig {
    fn default() -> Self {
        Self {
            enable_creative_synthesis: true,
            creativity_level: 0.9,
            style_adaptation: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalInputs {
    pub text: Option<String>,
    pub image: Option<String>,
    pub audio: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EnhancedMultimodalResult {
    pub caffeine_processing: Option<nexora_multimodal::caffeine::MultimodalResult>,
    pub spectra_synthesis: Option<NxrOutput>,
    pub combined_insights: Vec<String>,
    pub creative_outputs: Vec<String>,
}

impl EnhancedMultimodalResult {
    pub fn new() -> Self {
        Self {
            caffeine_processing: None,
            spectra_synthesis: None,
            combined_insights: Vec::new(),
            creative_outputs: Vec::new(),
        }
    }

    pub fn combine_results(&mut self) {
        if let Some(ref _caf) = self.caffeine_processing {
            self.combined_insights.push("CAFFEINE processing completed".to_string());
        }
        if self.spectra_synthesis.is_some() {
            self.combined_insights.push("SPECTRA synthesis completed".to_string());
        }
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![];
        if self.caffeine_processing.is_some() {
            parts.push("caffeine");
        }
        if self.spectra_synthesis.is_some() {
            parts.push("spectra");
        }
        format!("EnhancedMultimodalResult({})", parts.join("+"))
    }
}

impl CaffeineSpectraIntegration {
    pub fn new() -> Self {
        Self {
            caffeine_processor: nexora_multimodal::caffeine::CaffeineProcessor::new(),
            spectra_model: NxrSpectraModel::new(),
            integration_config: CaffeineSpectraConfig::default(),
        }
    }

    pub async fn enhanced_multimodal_processing(&self, inputs: &MultimodalInputs) -> std::result::Result<EnhancedMultimodalResult, Box<dyn std::error::Error>> {
        let mut result = EnhancedMultimodalResult::new();

        let caffeine_inputs = self.to_caffeine_inputs(inputs);
        if let Ok(caffeine_result) = self.caffeine_processor.process_multimodal(&caffeine_inputs).await {
            result.caffeine_processing = Some(caffeine_result);
        }
        if self.integration_config.enable_creative_synthesis {
            let combined_input = self.combine_multimodal_inputs(inputs);
            let spectra_input = NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: InputData::Text(combined_input),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };
            if let Ok(spectra_result) = self.spectra_model.infer(&spectra_input).await {
                result.spectra_synthesis = Some(spectra_result);
            }
        }
        result.combine_results();
        Ok(result)
    }

    fn to_caffeine_inputs(&self, inputs: &MultimodalInputs) -> CaffeineInputs {
        CaffeineInputs {
            text: inputs.text.as_ref().map(|t| TextInput {
                text: t.clone(),
                tokens: None,
                language: "en".to_string(),
            }),
            image: None,
            audio: None,
            video: None,
            context: None,
        }
    }

    fn combine_multimodal_inputs(&self, inputs: &MultimodalInputs) -> String {
        let mut combined = String::new();
        if let Some(text) = &inputs.text {
            combined.push_str(&format!("Text: {}\n", text));
        }
        if let Some(image) = &inputs.image {
            combined.push_str(&format!("Image: {}\n", image));
        }
        if let Some(audio) = &inputs.audio {
            combined.push_str(&format!("Audio: {}\n", audio));
        }
        combined
    }
}

impl Default for CaffeineSpectraIntegration {
    fn default() -> Self { Self::new() }
}
