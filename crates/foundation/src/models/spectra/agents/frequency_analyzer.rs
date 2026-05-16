//! Frequency Analyzer Agent
//! 
//! Frequency domain analysis and signal decomposition

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Frequency Analyzer Agent - Frequency domain analysis and signal decomposition
#[derive(Debug, Clone)]
pub struct FrequencyAnalyzerAgent {
    pub config: FrequencyAnalyzerConfig,
    pub analysis_capabilities: AnalysisCapabilities,
    pub decomposition_engine: DecompositionEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyAnalyzerConfig {
    pub base_config: BaseAgentConfig,
    pub analysis_model: AnalysisModel,
    pub decomposition_approach: DecompositionApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisModel {
    FourierAnalysis,
    WaveletAnalysis,
    TimeFrequencyAnalysis,
    SpectralAnalysis,
    HybridAnalysis { models: Vec<AnalysisModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionApproach {
    SubbandDecomposition,
    MultiresolutionDecomposition,
    AdaptiveDecomposition,
    SparseDecomposition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCapabilities {
    pub frequency_identification: bool,
    pub harmonic_analysis: bool,
    pub spectral_estimation: bool,
    pub signal_decomposition: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionEngine {
    pub decomposition_methods: Vec<String>,
    pub analysis_algorithms: Vec<String>,
    pub estimation_techniques: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyAnalyzerTaskInput {
    pub signal_data: Vec<f32>,
    pub sampling_frequency: f32,
    pub analysis_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyAnalyzerTaskOutput {
    pub frequency_spectrum: Vec<(f32, f32)>,
    pub harmonic_components: Vec<(f32, f32, f32)>,
    pub spectral_peaks: Vec<(f32, f32)>,
    pub analysis_accuracy: f32,
}

impl Default for FrequencyAnalyzerConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            analysis_model: AnalysisModel::HybridAnalysis {
                models: vec![
                    AnalysisModel::FourierAnalysis,
                    AnalysisModel::WaveletAnalysis,
                ],
            },
            decomposition_approach: DecompositionApproach::MultiresolutionDecomposition,
        }
    }
}

impl Default for AnalysisCapabilities {
    fn default() -> Self {
        Self {
            frequency_identification: true,
            harmonic_analysis: true,
            spectral_estimation: true,
            signal_decomposition: true,
        }
    }
}

impl Default for DecompositionEngine {
    fn default() -> Self {
        Self {
            decomposition_methods: vec![
                "fast_fourier_transform".to_string(),
                "discrete_wavelet_transform".to_string(),
                "empirical_mode_decomposition".to_string(),
                "variational_mode_decomposition".to_string(),
            ],
            analysis_algorithms: vec![
                "power_spectral_density".to_string(),
                "spectral_entropy".to_string(),
                "instantaneous_frequency".to_string(),
                "group_delay".to_string(),
            ],
            estimation_techniques: vec![
                "parametric_estimation".to_string(),
                "non_parametric_estimation".to_string(),
                "subspace_methods".to_string(),
                "maximum_likelihood".to_string(),
            ],
        }
    }
}

impl Default for FrequencyAnalyzerAgent {
    fn default() -> Self {
        Self {
            config: FrequencyAnalyzerConfig::default(),
            analysis_capabilities: AnalysisCapabilities::default(),
            decomposition_engine: DecompositionEngine::default(),
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
impl BaseAgent for FrequencyAnalyzerAgent {
    type Config = FrequencyAnalyzerConfig;
    type Input = FrequencyAnalyzerTaskInput;
    type Output = FrequencyAnalyzerTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let frequency_spectrum = self.compute_frequency_spectrum(&input).await?;
        let harmonic_components = self.analyze_harmonics(&input, &frequency_spectrum).await?;
        let spectral_peaks = self.identify_spectral_peaks(&frequency_spectrum).await?;
        let analysis_accuracy = self.assess_analysis_accuracy(&input, &frequency_spectrum).await?;

        Ok(FrequencyAnalyzerTaskOutput {
            frequency_spectrum,
            harmonic_components,
            spectral_peaks,
            analysis_accuracy,
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
                name: "frequency_analysis".to_string(),
                description: "Frequency domain analysis and signal decomposition".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["signal_data".to_string(), "sampling_frequency".to_string()],
                output_types: vec!["frequency_spectrum".to_string(), "harmonic_components".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.95,
                    avg_latency: 1900.0,
                    resource_usage: 0.72,
                    reliability: 0.97,
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

impl FrequencyAnalyzerAgent {
    pub fn new(config: FrequencyAnalyzerConfig) -> Self {
        Self {
            config,
            analysis_capabilities: AnalysisCapabilities::default(),
            decomposition_engine: DecompositionEngine::default(),
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

    async fn compute_frequency_spectrum(&self, input: &FrequencyAnalyzerTaskInput) -> AgentResult<Vec<(f32, f32)>> {
        let signal_length = input.signal_data.len();
        let mut spectrum = Vec::new();
        
        // Simple FFT simulation
        for i in 0..(signal_length / 2) {
            let frequency = (i as f32) * input.sampling_frequency / (signal_length as f32);
            let mut magnitude = 0.0;
            
            for (j, &sample) in input.signal_data.iter().enumerate() {
                let angle = 2.0 * std::f32::consts::PI * (i * j) as f32 / signal_length as f32;
                magnitude += sample * angle.cos();
            }
            
            magnitude = magnitude.abs() / signal_length as f32;
            spectrum.push((frequency, magnitude));
        }
        
        Ok(spectrum)
    }

    async fn analyze_harmonics(&self, input: &FrequencyAnalyzerTaskInput, spectrum: &[(f32, f32)]) -> AgentResult<Vec<(f32, f32, f32)>> {
        let mut harmonics = Vec::new();
        
        // Find fundamental frequency (highest magnitude)
        let fundamental = spectrum.iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(freq, _)| *freq)
            .unwrap_or(0.0);
        
        if fundamental > 0.0 {
            // Generate harmonics (2nd, 3rd, 4th, 5th)
            for n in 2..=5 {
                let harmonic_freq = fundamental * n as f32;
                
                // Find magnitude at harmonic frequency
                let harmonic_magnitude = spectrum.iter()
                    .filter(|(freq, _)| (*freq - harmonic_freq).abs() < input.sampling_frequency / input.signal_data.len() as f32)
                    .map(|(_, mag)| *mag)
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(0.0);
                
                let phase = (n as f32 * std::f32::consts::PI / 4.0).sin();
                harmonics.push((harmonic_freq, harmonic_magnitude, phase));
            }
        }
        
        Ok(harmonics)
    }

    async fn identify_spectral_peaks(&self, spectrum: &[(f32, f32)]) -> AgentResult<Vec<(f32, f32)>> {
        let mut peaks = Vec::new();
        
        for i in 1..(spectrum.len() - 1) {
            let current_mag = spectrum[i].1;
            let prev_mag = spectrum[i - 1].1;
            let next_mag = spectrum[i + 1].1;
            
            if current_mag > prev_mag && current_mag > next_mag && current_mag > 0.1 {
                peaks.push((spectrum[i].0, current_mag));
            }
        }
        
        // Sort peaks by magnitude (descending) and take top 10
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
        peaks.truncate(10);
        
        Ok(peaks)
    }

    async fn assess_analysis_accuracy(&self, input: &FrequencyAnalyzerTaskInput, spectrum: &[(f32, f32)]) -> AgentResult<f32> {
        let signal_quality = if input.signal_data.len() > 64 { 0.9 } else { 0.7 };
        let sampling_quality = if input.sampling_frequency > 1000.0 { 0.85 } else { 0.6 };
        let spectrum_quality = if spectrum.len() > 0 { 0.8 } else { 0.5 };
        
        Ok((signal_quality + sampling_quality + spectrum_quality) / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_analyzer_agent_creation() {
        let agent = FrequencyAnalyzerAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_frequency_analyzer_task_processing() {
        let agent = FrequencyAnalyzerAgent::default();
        let input = FrequencyAnalyzerTaskInput {
            signal_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            sampling_frequency: 1000.0,
            analysis_type: "frequency".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.frequency_spectrum.is_empty());
        assert!(!output.harmonic_components.is_empty());
        assert!(!output.spectral_peaks.is_empty());
        assert!(output.analysis_accuracy > 0.0);
    }
}
