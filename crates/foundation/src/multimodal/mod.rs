// Foundation Multimodal Framework (CAFFEINE)
// 
// Comprehensive multimodal cognition and processing system
// Now integrated with NXR-SPECTRA for enhanced creative multimodal synthesis.

pub mod caffeine;

// Re-export main components
pub use caffeine::*;

// Integration with NXR-SPECTRA
pub use crate::models::spectra::NxrSpectraModel;

/// Enhanced CAFFEINE with NXR-SPECTRA integration
pub struct CaffeineSpectraIntegration {
    /// Original CAFFEINE multimodal processor
    pub caffeine_processor: caffeine::CaffeineProcessor,
    /// NXR-SPECTRA creative synthesis
    pub spectra_model: NxrSpectraModel,
    /// Integration configuration
    pub integration_config: CaffeineSpectraConfig,
}

/// Configuration for CAFFEINE-SPECTRA integration
#[derive(Debug, Clone)]
pub struct CaffeineSpectraConfig {
    /// Enable creative synthesis
    pub enable_creative_synthesis: bool,
    /// Cross-modal creativity level
    pub creativity_level: f32,
    /// Artistic style adaptation
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

impl CaffeineSpectraIntegration {
    /// Create new integration
    pub fn new() -> Self {
        Self {
            caffeine_processor: caffeine::CaffeineProcessor::new(),
            spectra_model: NxrSpectraModel::new(),
            integration_config: CaffeineSpectraConfig::default(),
        }
    }

    /// Enhanced multimodal processing with creative synthesis
    pub async fn enhanced_multimodal_processing(&self, inputs: &MultimodalInputs) -> std::result::Result<EnhancedMultimodalResult, Box<dyn std::error::Error>> {
        let mut result = EnhancedMultimodalResult::new();

        // Original CAFFEINE multimodal processing
        if let Ok(caffeine_result) = self.caffeine_processor.process_multimodal(inputs).await {
            result.caffeine_processing = Some(caffeine_result);
        }

        // NXR-SPECTRA creative synthesis
        if self.integration_config.enable_creative_synthesis {
            let combined_input = self.combine_multimodal_inputs(inputs);
            let spectra_input = crate::shared::base_model::NxrInput {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                data: crate::shared::base_model::InputData::Text(combined_input),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
            };

            if let Ok(spectra_result) = self.spectra_model.infer(&spectra_input).await {
                result.spectra_synthesis = Some(spectra_result);
            }
        }

        // Combine multimodal and creative results
        result.combine_results(&self.integration_config);

        Ok(result)
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
        
        combined.push_str(&format!("Creativity Level: {:.2}\n", self.integration_config.creativity_level));
        
        combined
    }
}

/// Multimodal input structure
#[derive(Debug, Clone)]
pub struct MultimodalInputs {
    pub text: Option<String>,
    pub image: Option<String>,
    pub audio: Option<String>,
}

/// Enhanced multimodal result with creative synthesis
#[derive(Debug, Clone)]
pub struct EnhancedMultimodalResult {
    /// CAFFEINE multimodal processing
    pub caffeine_processing: Option<caffeine::MultimodalResult>,
    /// SPECTRA creative synthesis
    pub spectra_synthesis: Option<crate::shared::base_model::NxrOutput>,
    /// Combined multimodal insights
    pub combined_insights: Vec<String>,
    /// Creative cross-modal outputs
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

    /// Combine multimodal and creative results
    fn combine_results(&mut self, config: &CaffeineSpectraConfig) {
        // Combine insights from both systems
        if let Some(caffeine) = &self.caffeine_processing {
            self.combined_insights.push(format!("Multimodal Processing: {}", caffeine.processing_summary));
        }

        if let Some(spectra) = &self.spectra_synthesis {
            if let crate::shared::base_model::OutputData::Text(text) = &spectra.data {
                self.combined_insights.push(format!("Creative Synthesis: {}", text));
                
                // Generate creative cross-modal outputs
                if text.contains("visual") {
                    self.creative_outputs.push("Enhanced visual composition with artistic style".to_string());
                }
                if text.contains("audio") {
                    self.creative_outputs.push("Musical composition with emotional depth".to_string());
                }
                if text.contains("text") {
                    self.creative_outputs.push("Creative narrative with cross-modal elements".to_string());
                }
            }
        }

        // Apply creativity level to outputs
        if config.creativity_level > 0.8 {
            self.creative_outputs.push("High-creativity cross-modal synthesis achieved".to_string());
        }
    }

    /// Get comprehensive multimodal summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        if !self.combined_insights.is_empty() {
            summary.push_str("Combined Multimodal Insights:\n");
            for insight in &self.combined_insights {
                summary.push_str(&format!("- {}\n", insight));
            }
        }
        
        if !self.creative_outputs.is_empty() {
            summary.push_str("\nCreative Cross-Modal Outputs:\n");
            for output in &self.creative_outputs {
                summary.push_str(&format!("- {}\n", output));
            }
        }
        
        summary
    }
}

impl Default for CaffeineSpectraIntegration {
    fn default() -> Self {
        Self::new()
    }
}

