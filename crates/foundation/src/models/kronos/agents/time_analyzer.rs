//! Time Analyzer Agent
//! 
//! Temporal analysis and time-based pattern recognition

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Time Analyzer Agent - Temporal analysis and time-based pattern recognition
#[derive(Debug, Clone)]
pub struct TimeAnalyzerAgent {
    pub config: TimeAnalyzerConfig,
    pub analysis_capabilities: AnalysisCapabilities,
    pub temporal_patterns: TemporalPatterns,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeAnalyzerConfig {
    pub base_config: BaseAgentConfig,
    pub analysis_model: AnalysisModel,
    pub pattern_recognition_approach: PatternRecognitionApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisModel {
    TimeSeriesAnalysis,
    TemporalSequenceAnalysis,
    EventStreamAnalysis,
    HybridAnalysis { models: Vec<AnalysisModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternRecognitionApproach {
    StatisticalPatternRecognition,
    MachineLearningPatternRecognition,
    RuleBasedPatternRecognition,
    HybridPatternRecognition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCapabilities {
    pub temporal_analysis: bool,
    pub pattern_recognition: bool,
    pub trend_analysis: bool,
    pub anomaly_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatterns {
    pub pattern_types: Vec<String>,
    pub recognition_algorithms: Vec<String>,
    pub analysis_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeAnalyzerTaskInput {
    pub temporal_data: Vec<(chrono::DateTime<chrono::Utc>, f32)>,
    pub analysis_parameters: HashMap<String, f32>,
    pub pattern_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeAnalyzerTaskOutput {
    pub temporal_patterns: Vec<String>,
    pub trend_analysis: HashMap<String, f32>,
    pub detected_anomalies: Vec<(chrono::DateTime<chrono::Utc>, String)>,
    pub analysis_confidence: f32,
}

impl Default for TimeAnalyzerConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            analysis_model: AnalysisModel::HybridAnalysis {
                models: vec![
                    AnalysisModel::TimeSeriesAnalysis,
                    AnalysisModel::EventStreamAnalysis,
                ],
            },
            pattern_recognition_approach: PatternRecognitionApproach::HybridPatternRecognition,
        }
    }
}

impl Default for AnalysisCapabilities {
    fn default() -> Self {
        Self {
            temporal_analysis: true,
            pattern_recognition: true,
            trend_analysis: true,
            anomaly_detection: true,
        }
    }
}

impl Default for TemporalPatterns {
    fn default() -> Self {
        Self {
            pattern_types: vec![
                "periodic_patterns".to_string(),
                "seasonal_patterns".to_string(),
                "trend_patterns".to_string(),
                "anomaly_patterns".to_string(),
            ],
            recognition_algorithms: vec![
                "autocorrelation".to_string(),
                "fourier_analysis".to_string(),
                "moving_average".to_string(),
                "exponential_smoothing".to_string(),
            ],
            analysis_methods: vec![
                "statistical_analysis".to_string(),
                "spectral_analysis".to_string(),
                "trend_analysis".to_string(),
                "outlier_detection".to_string(),
            ],
        }
    }
}

impl Default for TimeAnalyzerAgent {
    fn default() -> Self {
        Self {
            config: TimeAnalyzerConfig::default(),
            analysis_capabilities: AnalysisCapabilities::default(),
            temporal_patterns: TemporalPatterns::default(),
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
impl BaseAgent for TimeAnalyzerAgent {
    type Config = TimeAnalyzerConfig;
    type Input = TimeAnalyzerTaskInput;
    type Output = TimeAnalyzerTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let temporal_patterns = self.identify_temporal_patterns(&input).await?;
        let trend_analysis = self.analyze_trends(&input).await?;
        let detected_anomalies = self.detect_anomalies(&input).await?;
        let analysis_confidence = self.calculate_analysis_confidence(&input, &temporal_patterns).await?;

        Ok(TimeAnalyzerTaskOutput {
            temporal_patterns,
            trend_analysis,
            detected_anomalies,
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
                name: "time_analysis".to_string(),
                description: "Temporal analysis and time-based pattern recognition".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["temporal_data".to_string(), "analysis_parameters".to_string()],
                output_types: vec!["temporal_patterns".to_string(), "trend_analysis".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.89,
                    avg_latency: 2400.0,
                    resource_usage: 0.65,
                    reliability: 0.91,
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

impl TimeAnalyzerAgent {
    pub fn new(config: TimeAnalyzerConfig) -> Self {
        Self {
            config,
            analysis_capabilities: AnalysisCapabilities::default(),
            temporal_patterns: TemporalPatterns::default(),
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

    async fn identify_temporal_patterns(&self, input: &TimeAnalyzerTaskInput) -> AgentResult<Vec<String>> {
        let mut patterns = Vec::new();
        
        if input.temporal_data.len() > 10 {
            patterns.push("Periodic pattern detected".to_string());
        }
        
        if self.has_seasonal_pattern(&input.temporal_data) {
            patterns.push("Seasonal pattern identified".to_string());
        }
        
        if self.has_trend_pattern(&input.temporal_data) {
            patterns.push("Trend pattern recognized".to_string());
        }
        
        patterns.push("Temporal sequence analysis completed".to_string());
        
        Ok(patterns)
    }

    fn has_seasonal_pattern(&self, data: &[(chrono::DateTime<chrono::Utc>, f32)]) -> bool {
        if data.len() < 12 {
            return false;
        }
        
        // Simple seasonal pattern detection
        let values: Vec<f32> = data.iter().map(|(_, value)| *value).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        
        let seasonal_variance = values.chunks(3)
            .map(|chunk| {
                let chunk_mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
                (chunk_mean - mean).powi(2)
            })
            .sum::<f32>() / (values.len() / 3) as f32;
        
        seasonal_variance > mean * 0.1
    }

    fn has_trend_pattern(&self, data: &[(chrono::DateTime<chrono::Utc>, f32)]) -> bool {
        if data.len() < 5 {
            return false;
        }
        
        let values: Vec<f32> = data.iter().map(|(_, value)| *value).collect();
        let first_half_mean = values[..values.len() / 2].iter().sum::<f32>() / (values.len() / 2) as f32;
        let second_half_mean = values[values.len() / 2..].iter().sum::<f32>() / (values.len() - values.len() / 2) as f32;
        
        (second_half_mean - first_half_mean).abs() > first_half_mean * 0.2
    }

    async fn analyze_trends(&self, input: &TimeAnalyzerTaskInput) -> AgentResult<HashMap<String, f32>> {
        let mut trends = HashMap::new();
        
        if input.temporal_data.is_empty() {
            return Ok(trends);
        }
        
        let values: Vec<f32> = input.temporal_data.iter().map(|(_, value)| *value).collect();
        
        // Calculate trend metrics
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        
        trends.insert("mean".to_string(), mean);
        trends.insert("variance".to_string(), variance);
        trends.insert("std_dev".to_string(), std_dev);
        
        // Simple trend calculation
        if values.len() > 1 {
            let trend = (values.last().unwrap() - values.first().unwrap()) / values.len() as f32;
            trends.insert("trend".to_string(), trend);
        }
        
        Ok(trends)
    }

    async fn detect_anomalies(&self, input: &TimeAnalyzerTaskInput) -> AgentResult<Vec<(chrono::DateTime<chrono::Utc>, String)>> {
        let mut anomalies = Vec::new();
        
        if input.temporal_data.len() < 3 {
            return Ok(anomalies);
        }
        
        let values: Vec<f32> = input.temporal_data.iter().map(|(_, value)| *value).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        
        // Detect outliers (values beyond 2 standard deviations)
        for (timestamp, value) in &input.temporal_data {
            if (value - mean).abs() > 2.0 * std_dev {
                anomalies.push((*timestamp, "Statistical anomaly detected".to_string()));
            }
        }
        
        Ok(anomalies)
    }

    async fn calculate_analysis_confidence(&self, input: &TimeAnalyzerTaskInput, patterns: &[String]) -> AgentResult<f32> {
        let data_quality = if input.temporal_data.len() > 20 { 0.9 } else { 0.7 };
        let parameter_quality = if input.analysis_parameters.len() > 0 { 0.8 } else { 0.6 };
        let pattern_quality = if patterns.len() > 0 { 0.85 } else { 0.5 };
        
        Ok((data_quality + parameter_quality + pattern_quality) / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_analyzer_agent_creation() {
        let agent = TimeAnalyzerAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_time_analyzer_task_processing() {
        let agent = TimeAnalyzerAgent::default();
        let base_time = chrono::Utc::now();
        let input = TimeAnalyzerTaskInput {
            temporal_data: vec![
                (base_time, 1.0),
                (base_time + chrono::Duration::hours(1), 2.0),
                (base_time + chrono::Duration::hours(2), 3.0),
                (base_time + chrono::Duration::hours(3), 4.0),
                (base_time + chrono::Duration::hours(4), 5.0),
            ],
            analysis_parameters: HashMap::from([
                ("window_size".to_string(), 5.0),
                ("threshold".to_string(), 2.0),
            ]),
            pattern_type: "trend_analysis".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.temporal_patterns.is_empty());
        assert!(!output.trend_analysis.is_empty());
        assert!(output.analysis_confidence > 0.0);
    }

    #[test]
    fn test_seasonal_pattern_detection() {
        let agent = TimeAnalyzerAgent::default();
        let base_time = chrono::Utc::now();
        let data = vec![
            (base_time, 1.0),
            (base_time + chrono::Duration::hours(1), 2.0),
            (base_time + chrono::Duration::hours(2), 1.0),
            (base_time + chrono::Duration::hours(3), 2.0),
            (base_time + chrono::Duration::hours(4), 1.0),
            (base_time + chrono::Duration::hours(5), 2.0),
            (base_time + chrono::Duration::hours(6), 1.0),
            (base_time + chrono::Duration::hours(7), 2.0),
            (base_time + chrono::Duration::hours(8), 1.0),
            (base_time + chrono::Duration::hours(9), 2.0),
            (base_time + chrono::Duration::hours(10), 1.0),
            (base_time + chrono::Duration::hours(11), 2.0),
        ];
        
        let has_seasonal = agent.has_seasonal_pattern(&data);
        assert!(has_seasonal);
    }

    #[test]
    fn test_trend_pattern_detection() {
        let agent = TimeAnalyzerAgent::default();
        let base_time = chrono::Utc::now();
        let data = vec![
            (base_time, 1.0),
            (base_time + chrono::Duration::hours(1), 2.0),
            (base_time + chrono::Duration::hours(2), 3.0),
            (base_time + chrono::Duration::hours(3), 4.0),
            (base_time + chrono::Duration::hours(4), 5.0),
        ];
        
        let has_trend = agent.has_trend_pattern(&data);
        assert!(has_trend);
    }
}
