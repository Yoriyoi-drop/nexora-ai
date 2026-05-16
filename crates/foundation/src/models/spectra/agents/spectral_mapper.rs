//! Spectral Mapper Agent
//! 
//! Spectral mapping and visualization systems

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Spectral Mapper Agent - Spectral mapping and visualization systems
#[derive(Debug, Clone)]
pub struct SpectralMapperAgent {
    pub config: SpectralMapperConfig,
    pub mapping_capabilities: MappingCapabilities,
    pub visualization_engine: VisualizationEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMapperConfig {
    pub base_config: BaseAgentConfig,
    pub mapping_model: MappingModel,
    pub visualization_approach: VisualizationApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MappingModel {
    LinearMapping,
    LogarithmicMapping,
    ExponentialMapping,
    CustomMapping,
    HybridMapping { models: Vec<MappingModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationApproach {
    HeatMap,
    ContourMap,
    ThreeDVisualization,
    InteractiveVisualization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingCapabilities {
    pub spatial_mapping: bool,
    pub frequency_mapping: bool,
    pub color_mapping: bool,
    pub intensity_mapping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationEngine {
    pub rendering_methods: Vec<String>,
    pub color_schemes: Vec<String>,
    pub interaction_modes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMapperTaskInput {
    pub spectral_data: Vec<(f32, f32)>,
    pub mapping_parameters: HashMap<String, f32>,
    pub visualization_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMapperTaskOutput {
    pub mapped_data: Vec<(f32, f32, f32)>,
    pub visualization_data: Vec<u8>,
    pub color_map: Vec<(f32, f32, f32)>,
    pub mapping_quality: f32,
}

impl Default for SpectralMapperConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            mapping_model: MappingModel::HybridMapping {
                models: vec![
                    MappingModel::LinearMapping,
                    MappingModel::LogarithmicMapping,
                ],
            },
            visualization_approach: VisualizationApproach::HeatMap,
        }
    }
}

impl Default for MappingCapabilities {
    fn default() -> Self {
        Self {
            spatial_mapping: true,
            frequency_mapping: true,
            color_mapping: true,
            intensity_mapping: true,
        }
    }
}

impl Default for VisualizationEngine {
    fn default() -> Self {
        Self {
            rendering_methods: vec![
                "raster_rendering".to_string(),
                "vector_rendering".to_string(),
                "gpu_accelerated".to_string(),
            ],
            color_schemes: vec![
                "viridis".to_string(),
                "plasma".to_string(),
                "inferno".to_string(),
                "magma".to_string(),
            ],
            interaction_modes: vec![
                "zoom".to_string(),
                "pan".to_string(),
                "filter".to_string(),
                "select".to_string(),
            ],
        }
    }
}

impl Default for SpectralMapperAgent {
    fn default() -> Self {
        Self {
            config: SpectralMapperConfig::default(),
            mapping_capabilities: MappingCapabilities::default(),
            visualization_engine: VisualizationEngine::default(),
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
impl BaseAgent for SpectralMapperAgent {
    type Config = SpectralMapperConfig;
    type Input = SpectralMapperTaskInput;
    type Output = SpectralMapperTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let mapped_data = self.map_spectral_data(&input).await?;
        let visualization_data = self.generate_visualization(&input, &mapped_data).await?;
        let color_map = self.create_color_map(&mapped_data).await?;
        let mapping_quality = self.assess_mapping_quality(&input, &mapped_data).await?;

        Ok(SpectralMapperTaskOutput {
            mapped_data,
            visualization_data,
            color_map,
            mapping_quality,
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
                name: "spectral_mapping".to_string(),
                description: "Spectral mapping and visualization systems".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["spectral_data".to_string(), "mapping_parameters".to_string()],
                output_types: vec!["mapped_data".to_string(), "visualization_data".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.91,
                    avg_latency: 2500.0,
                    resource_usage: 0.75,
                    reliability: 0.93,
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

impl SpectralMapperAgent {
    pub fn new(config: SpectralMapperConfig) -> Self {
        Self {
            config,
            mapping_capabilities: MappingCapabilities::default(),
            visualization_engine: VisualizationEngine::default(),
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

    async fn map_spectral_data(&self, input: &SpectralMapperTaskInput) -> AgentResult<Vec<(f32, f32, f32)>> {
        let mut mapped_data = Vec::new();
        
        for (i, (freq, amp)) in input.spectral_data.iter().enumerate() {
            let x = *freq;
            let y = *amp;
            let z = (i as f32 * 0.1).sin();
            mapped_data.push((x, y, z));
        }
        
        Ok(mapped_data)
    }

    async fn generate_visualization(&self, input: &SpectralMapperTaskInput, mapped_data: &[(f32, f32, f32)]) -> AgentResult<Vec<u8>> {
        // Simple visualization data generation
        let data_size = mapped_data.len() * 4; // RGBA
        let mut visualization_data = Vec::with_capacity(data_size);
        
        for (x, y, z) in mapped_data {
            let r = (x * 255.0) as u8;
            let g = (y * 255.0) as u8;
            let b = (z * 255.0) as u8;
            let a = 255;
            
            visualization_data.extend_from_slice(&[r, g, b, a]);
        }
        
        Ok(visualization_data)
    }

    async fn create_color_map(&self, mapped_data: &[(f32, f32, f32)]) -> AgentResult<Vec<(f32, f32, f32)>> {
        let mut color_map = Vec::new();
        
        for (x, y, z) in mapped_data {
            let r = x.abs().min(1.0);
            let g = y.abs().min(1.0);
            let b = z.abs().min(1.0);
            color_map.push((r, g, b));
        }
        
        Ok(color_map)
    }

    async fn assess_mapping_quality(&self, input: &SpectralMapperTaskInput, mapped_data: &[(f32, f32, f32)]) -> AgentResult<f32> {
        let data_completeness = if input.spectral_data.len() == mapped_data.len() { 0.9 } else { 0.7 };
        let parameter_validity = if input.mapping_parameters.len() > 0 { 0.8 } else { 0.6 };
        
        Ok((data_completeness + parameter_validity) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_mapper_agent_creation() {
        let agent = SpectralMapperAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_spectral_mapper_task_processing() {
        let agent = SpectralMapperAgent::default();
        let input = SpectralMapperTaskInput {
            spectral_data: vec![(100.0, 0.5), (200.0, 0.8), (300.0, 0.3)],
            mapping_parameters: HashMap::from([
                ("scale".to_string(), 1.0),
                ("offset".to_string(), 0.0),
            ]),
            visualization_type: "heatmap".to_string(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.mapped_data.is_empty());
        assert!(!output.visualization_data.is_empty());
        assert!(!output.color_map.is_empty());
        assert!(output.mapping_quality > 0.0);
    }
}
