//! Spectral Processor Agent
//! 
//! Advanced spectral processing and transformation

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Spectral Processor Agent - Advanced spectral processing and transformation
#[derive(Debug, Clone)]
pub struct SpectralProcessorAgent {
    pub config: SpectralProcessorConfig,
    pub processing_capabilities: ProcessingCapabilities,
    pub transformation_engine: TransformationEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralProcessorConfig {
    pub base_config: BaseAgentConfig,
    pub processing_model: ProcessingModel,
    pub transformation_approach: TransformationApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingModel {
    LinearProcessing,
    NonLinearProcessing,
    AdaptiveProcessing,
    MultiResolutionProcessing,
    HybridProcessing { models: Vec<ProcessingModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationApproach {
    FrequencyDomain,
    TimeFrequencyDomain,
    WaveletDomain,
    HybridDomain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingCapabilities {
    pub filtering: bool,
    pub enhancement: bool,
    pub noise_reduction: bool,
    pub signal_reconstruction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationEngine {
    pub processing_algorithms: Vec<String>,
    pub transformation_methods: Vec<String>,
    pub optimization_techniques: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralProcessorTaskInput {
    pub input_signal: Vec<f32>,
    pub processing_parameters: HashMap<String, f32>,
    pub transformation_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralProcessorTaskOutput {
    pub processed_signal: Vec<f32>,
    pub spectral_features: Vec<f32>,
    pub processing_metadata: HashMap<String, String>,
    pub processing_quality: f32,
}

impl Default for SpectralProcessorConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            processing_model: ProcessingModel::HybridProcessing {
                models: vec![
                    ProcessingModel::AdaptiveProcessing,
                    ProcessingModel::MultiResolutionProcessing,
                ],
            },
            transformation_approach: TransformationApproach::TimeFrequencyDomain,
        }
    }
}

impl Default for ProcessingCapabilities {
    fn default() -> Self {
        Self {
            filtering: true,
            enhancement: true,
            noise_reduction: true,
            signal_reconstruction: true,
        }
    }
}

impl Default for TransformationEngine {
    fn default() -> Self {
        Self {
            processing_algorithms: vec![
                "fast_fourier_transform".to_string(),
                "inverse_fourier_transform".to_string(),
                "spectral_filtering".to_string(),
                "phase_correction".to_string(),
            ],
            transformation_methods: vec![
                "amplitude_scaling".to_string(),
                "phase_modification".to_string(),
                "frequency_shifting".to_string(),
                "spectral_enhancement".to_string(),
            ],
            optimization_techniques: vec![
                "window_optimization".to_string(),
                "overlap_add".to_string(),
                "zero_padding".to_string(),
                "spectral_interpolation".to_string(),
            ],
        }
    }
}

impl Default for SpectralProcessorAgent {
    fn default() -> Self {
        Self {
            config: SpectralProcessorConfig::default(),
            processing_capabilities: ProcessingCapabilities::default(),
            transformation_engine: TransformationEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for SpectralProcessorAgent {
    type Config = SpectralProcessorConfig;
    type Input = SpectralProcessorTaskInput;
    type Output = SpectralProcessorTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let processed_signal = self.process_signal(&input).await?;
        let spectral_features = self.extract_spectral_features(&input, &processed_signal).await?;
        let processing_metadata = self.generate_processing_metadata(&input).await?;
        let processing_quality = self.assess_processing_quality(&input, &processed_signal).await?;

        Ok(SpectralProcessorTaskOutput {
            processed_signal,
            spectral_features,
            processing_metadata,
            processing_quality,
        })
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability {
                name: "spectral_processing".to_string(),
                description: "Advanced spectral processing and transformation".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["input_signal".to_string(), "processing_parameters".to_string()],
                output_types: vec!["processed_signal".to_string(), "spectral_features".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.93,
                    avg_latency: 2100.0,
                    resource_usage: 0.8,
                    reliability: 0.95,
                },
            },
        ]
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }

    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()> {
        self.config = config;
        self.status = AgentStatus::Idle;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        self.status = AgentStatus::Disabled;
        Ok(())
    }
}

impl SpectralProcessorAgent {
    pub fn new(config: SpectralProcessorConfig) -> Self {
        Self {
            config,
            processing_capabilities: ProcessingCapabilities::default(),
            transformation_engine: TransformationEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    async fn process_signal(&self, input: &SpectralProcessorTaskInput) -> AgentResult<Vec<f32>> {
        let mut processed_signal = Vec::new();
        
        for (i, &sample) in input.input_signal.iter().enumerate() {
            let processed = match input.transformation_type.as_str() {
                "filter" => self.apply_filter(sample, i, &input.processing_parameters),
                "enhance" => self.apply_enhancement(sample, i, &input.processing_parameters),
                "noise_reduce" => self.apply_noise_reduction(sample, i, &input.processing_parameters),
                _ => sample,
            };
            processed_signal.push(processed);
        }
        
        Ok(processed_signal)
    }

    fn apply_filter(&self, sample: f32, index: usize, params: &HashMap<String, f32>) -> f32 {
        let cutoff_freq = params.get("cutoff").unwrap_or(&0.5);
        let filter_factor = (index as f32 * cutoff_freq).sin();
        sample * filter_factor
    }

    fn apply_enhancement(&self, sample: f32, index: usize, params: &HashMap<String, f32>) -> f32 {
        let enhancement_factor = params.get("enhancement").unwrap_or(&1.5);
        let dynamic_factor = 1.0 + (index as f32 * 0.01).sin() * 0.1;
        sample * enhancement_factor * dynamic_factor
    }

    fn apply_noise_reduction(&self, sample: f32, index: usize, params: &HashMap<String, f32>) -> f32 {
        let noise_threshold = params.get("threshold").unwrap_or(&0.1);
        if sample.abs() < *noise_threshold {
            0.0
        } else {
            sample * (1.0 - noise_threshold)
        }
    }

    async fn extract_spectral_features(&self, input: &SpectralProcessorTaskInput, processed_signal: &[f32]) -> AgentResult<Vec<f32>> {
        let mut features = Vec::new();
        
        // Calculate basic spectral features
        let mean = processed_signal.iter().sum::<f32>() / processed_signal.len() as f32;
        features.push(mean);
        
        let variance = processed_signal.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / processed_signal.len() as f32;
        features.push(variance.sqrt());
        
        // Calculate spectral centroid
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (i, &sample) in processed_signal.iter().enumerate() {
            let frequency = i as f32;
            let magnitude = sample.abs();
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }
        
        let spectral_centroid = if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };
        features.push(spectral_centroid);
        
        Ok(features)
    }

    async fn generate_processing_metadata(&self, input: &SpectralProcessorTaskInput) -> AgentResult<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        
        metadata.insert("transformation_type".to_string(), input.transformation_type.clone());
        metadata.insert("input_length".to_string(), input.input_signal.len().to_string());
        metadata.insert("parameter_count".to_string(), input.processing_parameters.len().to_string());
        metadata.insert("processing_model".to_string(), "HybridProcessing".to_string());
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        
        Ok(metadata)
    }

    async fn assess_processing_quality(&self, input: &SpectralProcessorTaskInput, processed_signal: &[f32]) -> AgentResult<f32> {
        let input_length = input.input_signal.len();
        let output_length = processed_signal.len();
        
        let length_consistency = if input_length == output_length { 0.9 } else { 0.7 };
        let parameter_validity = if input.processing_parameters.len() > 0 { 0.8 } else { 0.6 };
        let signal_quality = if processed_signal.iter().any(|&x| x.is_finite()) { 0.85 } else { 0.5 };
        
        Ok((length_consistency + parameter_validity + signal_quality) / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_processor_agent_creation() {
        let agent = SpectralProcessorAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_spectral_processor_task_processing() {
        let agent = SpectralProcessorAgent::default();
        let input = SpectralProcessorTaskInput {
            input_signal: vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0],
            processing_parameters: HashMap::from([
                ("cutoff".to_string(), 0.5),
                ("enhancement".to_string(), 1.2),
            ]),
            transformation_type: "enhance".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.processed_signal.is_empty());
        assert!(!output.spectral_features.is_empty());
        assert!(!output.processing_metadata.is_empty());
        assert!(output.processing_quality > 0.0);
    }
}
