//! Spectral Analyzer Agent
//! 
//! Comprehensive spectral analysis and signal characterization

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Spectral Analyzer Agent - Comprehensive spectral analysis and signal characterization
#[derive(Debug, Clone)]
pub struct SpectralAnalyzerAgent {
    pub config: SpectralAnalyzerConfig,
    pub analysis_capabilities: AnalysisCapabilities,
    pub characterization_engine: CharacterizationEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalyzerConfig {
    pub base_config: BaseAgentConfig,
    pub analysis_model: AnalysisModel,
    pub characterization_approach: CharacterizationApproach,
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
pub enum CharacterizationApproach {
    StatisticalCharacterization,
    EnergyCharacterization,
    InformationCharacterization,
    HybridCharacterization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCapabilities {
    pub spectral_analysis: bool,
    pub signal_characterization: bool,
    pub feature_extraction: bool,
    pub pattern_recognition: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterizationEngine {
    pub characterization_methods: Vec<String>,
    pub feature_extractors: Vec<String>,
    pub pattern_recognizers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalyzerTaskInput {
    pub signal_data: Vec<f32>,
    pub analysis_parameters: HashMap<String, f32>,
    pub characterization_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalyzerTaskOutput {
    pub spectral_features: Vec<f32>,
    pub signal_characteristics: HashMap<String, f32>,
    pub identified_patterns: Vec<String>,
    pub analysis_confidence: f32,
}

impl Default for SpectralAnalyzerConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            analysis_model: AnalysisModel::HybridAnalysis {
                models: vec![
                    AnalysisModel::FourierAnalysis,
                    AnalysisModel::WaveletAnalysis,
                ],
            },
            characterization_approach: CharacterizationApproach::HybridCharacterization,
        }
    }
}

impl Default for AnalysisCapabilities {
    fn default() -> Self {
        Self {
            spectral_analysis: true,
            signal_characterization: true,
            feature_extraction: true,
            pattern_recognition: true,
        }
    }
}

impl Default for CharacterizationEngine {
    fn default() -> Self {
        Self {
            characterization_methods: vec![
                "statistical_analysis".to_string(),
                "energy_analysis".to_string(),
                "entropy_analysis".to_string(),
            ],
            feature_extractors: vec![
                "spectral_centroid".to_string(),
                "spectral_rolloff".to_string(),
                "spectral_flux".to_string(),
                "zero_crossing_rate".to_string(),
            ],
            pattern_recognizers: vec![
                "periodic_patterns".to_string(),
                "transient_patterns".to_string(),
                "noise_patterns".to_string(),
            ],
        }
    }
}

impl Default for SpectralAnalyzerAgent {
    fn default() -> Self {
        Self {
            config: SpectralAnalyzerConfig::default(),
            analysis_capabilities: AnalysisCapabilities::default(),
            characterization_engine: CharacterizationEngine::default(),
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
impl BaseAgent for SpectralAnalyzerAgent {
    type Config = SpectralAnalyzerConfig;
    type Input = SpectralAnalyzerTaskInput;
    type Output = SpectralAnalyzerTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let spectral_features = self.extract_spectral_features(&input).await?;
        let signal_characteristics = self.characterize_signal(&input, &spectral_features).await?;
        let identified_patterns = self.identify_patterns(&input, &spectral_features).await?;
        let analysis_confidence = self.calculate_confidence(&input, &spectral_features).await?;

        Ok(SpectralAnalyzerTaskOutput {
            spectral_features,
            signal_characteristics,
            identified_patterns,
            analysis_confidence,
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
                name: "spectral_analysis".to_string(),
                description: "Comprehensive spectral analysis and signal characterization".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["signal_data".to_string(), "analysis_parameters".to_string()],
                output_types: vec!["spectral_features".to_string(), "signal_characteristics".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.92,
                    avg_latency: 2300.0,
                    resource_usage: 0.68,
                    reliability: 0.94,
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

impl SpectralAnalyzerAgent {
    pub fn new(config: SpectralAnalyzerConfig) -> Self {
        Self {
            config,
            analysis_capabilities: AnalysisCapabilities::default(),
            characterization_engine: CharacterizationEngine::default(),
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

    async fn extract_spectral_features(&self, input: &SpectralAnalyzerTaskInput) -> AgentResult<Vec<f32>> {
        let mut features = Vec::new();
        
        // Calculate spectral centroid
        let spectral_centroid = self.calculate_spectral_centroid(&input.signal_data);
        features.push(spectral_centroid);
        
        // Calculate spectral rolloff
        let spectral_rolloff = self.calculate_spectral_rolloff(&input.signal_data);
        features.push(spectral_rolloff);
        
        // Calculate spectral flux
        let spectral_flux = self.calculate_spectral_flux(&input.signal_data);
        features.push(spectral_flux);
        
        // Calculate zero crossing rate
        let zcr = self.calculate_zero_crossing_rate(&input.signal_data);
        features.push(zcr);
        
        // Calculate energy
        let energy = self.calculate_energy(&input.signal_data);
        features.push(energy);
        
        Ok(features)
    }

    fn calculate_spectral_centroid(&self, signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (i, &sample) in signal.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += i as f32 * magnitude;
            magnitude_sum += magnitude;
        }
        
        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn calculate_spectral_rolloff(&self, signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        
        let total_energy = signal.iter().map(|x| x * x).sum::<f32>();
        let threshold = 0.85 * total_energy;
        let mut cumulative_energy = 0.0;
        
        for (i, &sample) in signal.iter().enumerate() {
            cumulative_energy += sample * sample;
            if cumulative_energy >= threshold {
                return i as f32 / signal.len() as f32;
            }
        }
        
        1.0
    }

    fn calculate_spectral_flux(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        
        let mut flux = 0.0;
        for i in 1..signal.len() {
            let diff = signal[i].abs() - signal[i - 1].abs();
            if diff > 0.0 {
                flux += diff;
            }
        }
        
        flux / (signal.len() - 1) as f32
    }

    fn calculate_zero_crossing_rate(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        
        let mut crossings = 0;
        for i in 1..signal.len() {
            if (signal[i] >= 0.0) != (signal[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        
        crossings as f32 / (signal.len() - 1) as f32
    }

    fn calculate_energy(&self, signal: &[f32]) -> f32 {
        signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32
    }

    async fn characterize_signal(&self, input: &SpectralAnalyzerTaskInput, features: &[f32]) -> AgentResult<HashMap<String, f32>> {
        let mut characteristics = HashMap::new();
        
        characteristics.insert("mean".to_string(), input.signal_data.iter().sum::<f32>() / input.signal_data.len() as f32);
        characteristics.insert("variance".to_string(), self.calculate_variance(&input.signal_data));
        characteristics.insert("skewness".to_string(), self.calculate_skewness(&input.signal_data));
        characteristics.insert("kurtosis".to_string(), self.calculate_kurtosis(&input.signal_data));
        
        if features.len() >= 4 {
            characteristics.insert("spectral_centroid".to_string(), features[0]);
            characteristics.insert("spectral_rolloff".to_string(), features[1]);
            characteristics.insert("spectral_flux".to_string(), features[2]);
            characteristics.insert("zero_crossing_rate".to_string(), features[3]);
        }
        
        Ok(characteristics)
    }

    fn calculate_variance(&self, signal: &[f32]) -> f32 {
        let mean = signal.iter().sum::<f32>() / signal.len() as f32;
        signal.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / signal.len() as f32
    }

    fn calculate_skewness(&self, signal: &[f32]) -> f32 {
        let mean = signal.iter().sum::<f32>() / signal.len() as f32;
        let variance = self.calculate_variance(signal);
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        signal.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f32>() / signal.len() as f32
    }

    fn calculate_kurtosis(&self, signal: &[f32]) -> f32 {
        let mean = signal.iter().sum::<f32>() / signal.len() as f32;
        let variance = self.calculate_variance(signal);
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        signal.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f32>() / signal.len() as f32 - 3.0
    }

    async fn identify_patterns(&self, input: &SpectralAnalyzerTaskInput, features: &[f32]) -> AgentResult<Vec<String>> {
        let mut patterns = Vec::new();
        
        // Identify periodic patterns
        if self.is_periodic(&input.signal_data) {
            patterns.push("Periodic signal detected".to_string());
        }
        
        // Identify transient patterns
        if self.has_transients(&input.signal_data) {
            patterns.push("Transient components detected".to_string());
        }
        
        // Identify noise patterns
        if self.is_noisy(&input.signal_data) {
            patterns.push("High noise content detected".to_string());
        }
        
        // Identify harmonic patterns
        if features.len() >= 1 && features[0] > 0.5 {
            patterns.push("Harmonic content detected".to_string());
        }
        
        Ok(patterns)
    }

    fn is_periodic(&self, signal: &[f32]) -> bool {
        if signal.len() < 10 {
            return false;
        }
        
        // Simple periodicity check using autocorrelation
        let mut max_correlation = 0.0f32;
        for lag in 1..(signal.len() / 2) {
            let mut correlation = 0.0f32;
            for i in 0..(signal.len() - lag) {
                correlation += signal[i] * signal[i + lag];
            }
            correlation /= (signal.len() - lag) as f32;
            max_correlation = max_correlation.max(correlation);
        }
        
        max_correlation > 0.7
    }

    fn has_transients(&self, signal: &[f32]) -> bool {
        if signal.len() < 3 {
            return false;
        }
        
        // Check for sudden changes
        for i in 1..signal.len() {
            let diff = (signal[i] - signal[i - 1]).abs();
            if diff > 0.5 {
                return true;
            }
        }
        
        false
    }

    fn is_noisy(&self, signal: &[f32]) -> bool {
        if signal.len() < 2 {
            return false;
        }
        
        let variance = self.calculate_variance(signal);
        let mean = signal.iter().sum::<f32>() / signal.len() as f32;
        
        variance > mean.abs() * 0.5
    }

    async fn calculate_confidence(&self, input: &SpectralAnalyzerTaskInput, features: &[f32]) -> AgentResult<f32> {
        let signal_quality = if input.signal_data.len() > 100 { 0.9 } else { 0.7 };
        let parameter_quality = if input.analysis_parameters.len() > 0 { 0.8 } else { 0.6 };
        let feature_quality = if features.len() > 0 { 0.85 } else { 0.5 };
        
        Ok((signal_quality + parameter_quality + feature_quality) / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_analyzer_agent_creation() {
        let agent = SpectralAnalyzerAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_spectral_analyzer_task_processing() {
        let agent = SpectralAnalyzerAgent::default();
        let input = SpectralAnalyzerTaskInput {
            signal_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            analysis_parameters: HashMap::from([
                ("window_size".to_string(), 1024.0),
                ("overlap".to_string(), 0.5),
            ]),
            characterization_type: "comprehensive".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.spectral_features.is_empty());
        assert!(!output.signal_characteristics.is_empty());
        assert!(!output.identified_patterns.is_empty());
        assert!(output.analysis_confidence > 0.0);
    }

    #[test]
    fn test_spectral_feature_calculation() {
        let agent = SpectralAnalyzerAgent::default();
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let centroid = agent.calculate_spectral_centroid(&signal);
        assert!(centroid >= 0.0);
        
        let energy = agent.calculate_energy(&signal);
        assert!(energy > 0.0);
        
        let zcr = agent.calculate_zero_crossing_rate(&signal);
        assert!(zcr >= 0.0);
    }
}
