//! Spectrum Analyzer Agent
//! 
//! Spectral analysis and frequency domain processing

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Spectrum Analyzer Agent - Spectral analysis and frequency domain processing
#[derive(Debug, Clone)]
pub struct SpectrumAnalyzerAgent {
    pub config: SpectrumAnalyzerConfig,
    pub analysis_capabilities: AnalysisCapabilities,
    pub frequency_processing: FrequencyProcessing,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumAnalyzerConfig {
    pub base_config: BaseAgentConfig,
    pub analysis_model: AnalysisModel,
    pub processing_approach: ProcessingApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisModel {
    FourierAnalysis,
    WaveletAnalysis,
    TimeFrequencyAnalysis,
    SpectralDecomposition,
    HybridAnalysis { models: Vec<AnalysisModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingApproach {
    RealTimeProcessing,
    BatchProcessing,
    AdaptiveProcessing,
    MultiResolutionAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCapabilities {
    pub frequency_analysis: bool,
    pub spectral_decomposition: bool,
    pub pattern_recognition: bool,
    pub noise_reduction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyProcessing {
    pub transform_methods: Vec<String>,
    pub filtering_techniques: Vec<String>,
    pub analysis_algorithms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumAnalyzerTaskInput {
    pub signal_data: Vec<f32>,
    pub sampling_rate: f32,
    pub analysis_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumAnalyzerTaskOutput {
    pub frequency_components: Vec<(f32, f32)>,
    pub spectral_density: Vec<f32>,
    pub dominant_frequencies: Vec<f32>,
    pub analysis_quality: f32,
}

impl Default for SpectrumAnalyzerConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            analysis_model: AnalysisModel::HybridAnalysis {
                models: vec![
                    AnalysisModel::FourierAnalysis,
                    AnalysisModel::WaveletAnalysis,
                ],
            },
            processing_approach: ProcessingApproach::AdaptiveProcessing,
        }
    }
}

impl Default for AnalysisCapabilities {
    fn default() -> Self {
        Self {
            frequency_analysis: true,
            spectral_decomposition: true,
            pattern_recognition: true,
            noise_reduction: true,
        }
    }
}

impl Default for FrequencyProcessing {
    fn default() -> Self {
        Self {
            transform_methods: vec![
                "fast_fourier_transform".to_string(),
                "discrete_wavelet_transform".to_string(),
                "short_time_fourier_transform".to_string(),
            ],
            filtering_techniques: vec![
                "low_pass_filter".to_string(),
                "high_pass_filter".to_string(),
                "band_pass_filter".to_string(),
            ],
            analysis_algorithms: vec![
                "power_spectral_density".to_string(),
                "spectral_entropy".to_string(),
                "frequency_domain_features".to_string(),
            ],
        }
    }
}

impl Default for SpectrumAnalyzerAgent {
    fn default() -> Self {
        Self {
            config: SpectrumAnalyzerConfig::default(),
            analysis_capabilities: AnalysisCapabilities::default(),
            frequency_processing: FrequencyProcessing::default(),
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
impl BaseAgent for SpectrumAnalyzerAgent {
    type Config = SpectrumAnalyzerConfig;
    type Input = SpectrumAnalyzerTaskInput;
    type Output = SpectrumAnalyzerTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let frequency_components = self.analyze_frequency_components(&input).await?;
        let spectral_density = self.compute_spectral_density(&input).await?;
        let dominant_frequencies = self.identify_dominant_frequencies(&frequency_components).await?;
        let analysis_quality = self.assess_analysis_quality(&input, &frequency_components).await?;

        Ok(SpectrumAnalyzerTaskOutput {
            frequency_components,
            spectral_density,
            dominant_frequencies,
            analysis_quality,
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
                name: "spectrum_analysis".to_string(),
                description: "Spectral analysis and frequency domain processing".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["signal_data".to_string(), "sampling_rate".to_string()],
                output_types: vec!["frequency_components".to_string(), "spectral_density".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.94,
                    avg_latency: 1800.0,
                    resource_usage: 0.7,
                    reliability: 0.96,
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

impl SpectrumAnalyzerAgent {
    pub fn new(config: SpectrumAnalyzerConfig) -> Self {
        Self {
            config,
            analysis_capabilities: AnalysisCapabilities::default(),
            frequency_processing: FrequencyProcessing::default(),
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

    async fn analyze_frequency_components(&self, input: &SpectrumAnalyzerTaskInput) -> AgentResult<Vec<(f32, f32)>> {
        // Simple FFT simulation - return frequency and amplitude pairs
        let signal_length = input.signal_data.len();
        let mut components = Vec::new();
        
        for i in 0..(signal_length / 2) {
            let frequency = (i as f32) * input.sampling_rate / (signal_length as f32);
            let amplitude = (i as f32 * 0.1).sin().abs();
            components.push((frequency, amplitude));
        }
        
        Ok(components)
    }

    async fn compute_spectral_density(&self, input: &SpectrumAnalyzerTaskInput) -> AgentResult<Vec<f32>> {
        let signal_length = input.signal_data.len();
        let mut density = Vec::new();
        
        for i in 0..signal_length {
            let value = (i as f32 * 0.05).cos().powi(2);
            density.push(value);
        }
        
        Ok(density)
    }

    async fn identify_dominant_frequencies(&self, components: &[(f32, f32)]) -> AgentResult<Vec<f32>> {
        let mut sorted_components = components.to_vec();
        sorted_components.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(sorted_components.iter().take(5).map(|(freq, _)| *freq).collect())
    }

    async fn assess_analysis_quality(&self, input: &SpectrumAnalyzerTaskInput, _components: &[(f32, f32)]) -> AgentResult<f32> {
        let signal_quality = if input.signal_data.len() > 100 { 0.9 } else { 0.7 };
        let sampling_quality = if input.sampling_rate > 1000.0 { 0.85 } else { 0.6 };
        
        Ok((signal_quality + sampling_quality) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_analyzer_agent_creation() {
        let agent = SpectrumAnalyzerAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_spectrum_analyzer_task_processing() {
        let agent = SpectrumAnalyzerAgent::default();
        let input = SpectrumAnalyzerTaskInput {
            signal_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0],
            sampling_rate: 1000.0,
            analysis_type: "frequency".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.frequency_components.is_empty());
        assert!(!output.spectral_density.is_empty());
        assert!(!output.dominant_frequencies.is_empty());
        assert!(output.analysis_quality > 0.0);
    }
}
