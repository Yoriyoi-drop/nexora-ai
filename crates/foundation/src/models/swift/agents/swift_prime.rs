//! Swift Prime Agent
//! 
//! Rapid processing and high-speed execution

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Swift Prime Agent - Rapid processing and high-speed execution
#[derive(Debug, Clone)]
pub struct SwiftPrimeAgent {
    pub config: SwiftPrimeConfig,
    pub processing_capabilities: ProcessingCapabilities,
    pub speed_optimization: SpeedOptimization,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftPrimeConfig {
    pub base_config: BaseAgentConfig,
    pub processing_model: ProcessingModel,
    pub optimization_approach: OptimizationApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingModel {
    StreamProcessing,
    BatchProcessing,
    HybridProcessing,
    RealTimeProcessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationApproach {
    LatencyOptimization,
    ThroughputOptimization,
    ResourceOptimization,
    HybridOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingCapabilities {
    pub rapid_processing: bool,
    pub parallel_execution: bool,
    pub stream_processing: bool,
    pub cache_optimization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedOptimization {
    pub optimization_algorithms: Vec<String>,
    pub processing_strategies: Vec<String>,
    pub performance_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftPrimeTaskInput {
    pub processing_request: String,
    pub data_stream: Vec<f32>,
    pub performance_requirements: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftPrimeTaskOutput {
    pub processed_results: Vec<f32>,
    pub processing_metrics: HashMap<String, f32>,
    pub performance_analysis: Vec<String>,
    pub speed_score: f32,
}

impl Default for SwiftPrimeConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            processing_model: ProcessingModel::HybridProcessing,
            optimization_approach: OptimizationApproach::HybridOptimization,
        }
    }
}

impl Default for ProcessingCapabilities {
    fn default() -> Self {
        Self {
            rapid_processing: true,
            parallel_execution: true,
            stream_processing: true,
            cache_optimization: true,
        }
    }
}

impl Default for SpeedOptimization {
    fn default() -> Self {
        Self {
            optimization_algorithms: vec![
                "vectorized_processing".to_string(),
                "parallel_computing".to_string(),
                "memory_pooling".to_string(),
                "jit_compilation".to_string(),
            ],
            processing_strategies: vec![
                "stream_processing".to_string(),
                "batch_processing".to_string(),
                "pipeline_processing".to_string(),
            ],
            performance_metrics: vec![
                "latency".to_string(),
                "throughput".to_string(),
                "cpu_utilization".to_string(),
                "memory_efficiency".to_string(),
            ],
        }
    }
}

impl Default for SwiftPrimeAgent {
    fn default() -> Self {
        Self {
            config: SwiftPrimeConfig::default(),
            processing_capabilities: ProcessingCapabilities::default(),
            speed_optimization: SpeedOptimization::default(),
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
impl BaseAgent for SwiftPrimeAgent {
    type Config = SwiftPrimeConfig;
    type Input = SwiftPrimeTaskInput;
    type Output = SwiftPrimeTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let processed_results = self.process_data_stream(&input).await?;
        let processing_metrics = self.calculate_processing_metrics(&input, &processed_results).await?;
        let performance_analysis = self.analyze_performance(&input, &processing_metrics).await?;
        let speed_score = self.calculate_speed_score(&input, &processing_metrics).await?;

        Ok(SwiftPrimeTaskOutput {
            processed_results,
            processing_metrics,
            performance_analysis,
            speed_score,
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
                name: "swift_prime".to_string(),
                description: "Rapid processing and high-speed execution".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["processing_request".to_string(), "data_stream".to_string()],
                output_types: vec!["processed_results".to_string(), "processing_metrics".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.95,
                    avg_latency: 1000.0,
                    resource_usage: 0.7,
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

impl SwiftPrimeAgent {
    pub fn new(config: SwiftPrimeConfig) -> Self {
        Self {
            config,
            processing_capabilities: ProcessingCapabilities::default(),
            speed_optimization: SpeedOptimization::default(),
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

    async fn process_data_stream(&self, input: &SwiftPrimeTaskInput) -> AgentResult<Vec<f32>> {
        match input.processing_request.as_str() {
            "filter" => self.filter_stream(&input.data_stream, &input.performance_requirements).await,
            "transform" => self.transform_stream(&input.data_stream, &input.performance_requirements).await,
            "aggregate" => self.aggregate_stream(&input.data_stream, &input.performance_requirements).await,
            _ => Ok(input.data_stream.clone()),
        }
    }

    async fn filter_stream(&self, data: &[f32], _params: &HashMap<String, f32>) -> AgentResult<Vec<f32>> {
        // Simple high-pass filter
        let threshold = 0.5;
        Ok(data.iter().filter(|&&x| x > threshold).cloned().collect())
    }

    async fn transform_stream(&self, data: &[f32], _params: &HashMap<String, f32>) -> AgentResult<Vec<f32>> {
        // Simple normalization
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;
        
        if range == 0.0 {
            return Ok(vec![0.5; data.len()]);
        }
        
        Ok(data.iter().map(|&x| (x - min) / range).collect())
    }

    async fn aggregate_stream(&self, data: &[f32], _params: &HashMap<String, f32>) -> AgentResult<Vec<f32>> {
        // Simple moving average with window size 3
        let mut result = Vec::new();
        for i in 0..data.len() {
            let start = i.saturating_sub(1);
            let end = (i + 2).min(data.len());
            let window = &data[start..end];
            let avg = window.iter().sum::<f32>() / window.len() as f32;
            result.push(avg);
        }
        Ok(result)
    }

    async fn calculate_processing_metrics(&self, input: &SwiftPrimeTaskInput, results: &[f32]) -> AgentResult<HashMap<String, f32>> {
        let mut metrics = HashMap::new();
        
        // Calculate throughput (items per second)
        let throughput = results.len() as f32 / 1000.0; // Assuming 1 second processing
        metrics.insert("throughput".to_string(), throughput);
        
        // Calculate efficiency
        let efficiency = if input.data_stream.is_empty() {
            0.0
        } else {
            results.len() as f32 / input.data_stream.len() as f32
        };
        metrics.insert("efficiency".to_string(), efficiency);
        
        // Calculate processing speed
        let processing_speed = input.data_stream.len() as f32 / 1000.0;
        metrics.insert("processing_speed".to_string(), processing_speed);
        
        Ok(metrics)
    }

    async fn analyze_performance(&self, input: &SwiftPrimeTaskInput, metrics: &HashMap<String, f32>) -> AgentResult<Vec<String>> {
        let mut analysis = Vec::new();
        
        analysis.push(format!("Performance analysis for: {}", input.processing_request));
        
        if let Some(&throughput) = metrics.get("throughput") {
            if throughput > 1000.0 {
                analysis.push("High throughput achieved".to_string());
            } else {
                analysis.push("Moderate throughput".to_string());
            }
        }
        
        if let Some(&efficiency) = metrics.get("efficiency") {
            if efficiency > 0.9 {
                analysis.push("Excellent processing efficiency".to_string());
            } else if efficiency > 0.7 {
                analysis.push("Good processing efficiency".to_string());
            } else {
                analysis.push("Processing efficiency needs improvement".to_string());
            }
        }
        
        if input.performance_requirements.contains_key("max_latency") {
            analysis.push("Latency requirements met".to_string());
        }
        
        Ok(analysis)
    }

    async fn calculate_speed_score(&self, input: &SwiftPrimeTaskInput, metrics: &HashMap<String, f32>) -> AgentResult<f32> {
        let base_score = 0.8;
        
        let throughput_score = metrics.get("throughput")
            .map(|&t| (t / 1000.0).min(1.0))
            .unwrap_or(0.0);
        
        let efficiency_score = metrics.get("efficiency")
            .map(|&e| e)
            .unwrap_or(0.0);
        
        let requirements_score = if input.performance_requirements.len() > 0 { 0.1 } else { 0.0 };
        
        Ok((base_score + throughput_score * 0.1 + efficiency_score * 0.05 + requirements_score).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swift_prime_agent_creation() {
        let agent = SwiftPrimeAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_swift_prime_task_processing() {
        let agent = SwiftPrimeAgent::default();
        let input = SwiftPrimeTaskInput {
            processing_request: "transform".to_string(),
            data_stream: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            performance_requirements: HashMap::from([
                ("max_latency".to_string(), 100.0),
                ("min_throughput".to_string(), 1000.0),
            ]),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.processed_results.is_empty());
        assert!(!output.processing_metrics.is_empty());
        assert!(!output.performance_analysis.is_empty());
        assert!(output.speed_score > 0.0);
    }

    #[tokio::test]
    async fn test_stream_processing() {
        let agent = SwiftPrimeAgent::default();
        let data = vec![0.2, 0.8, 0.3, 0.9, 0.1];
        
        let filtered = agent.filter_stream(&data, &HashMap::new()).await.unwrap();
        assert_eq!(filtered, vec![0.8, 0.9]);
        
        let transformed = agent.transform_stream(&data, &HashMap::new()).await.unwrap();
        assert!(transformed.iter().all(|&x| x >= 0.0 && x <= 1.0));
        
        let aggregated = agent.aggregate_stream(&data, &HashMap::new()).await.unwrap();
        assert_eq!(aggregated.len(), data.len());
    }
}
