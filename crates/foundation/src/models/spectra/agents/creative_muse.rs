//! Creative Muse Agent
//! 
//! Core creative synthesis agent for NXR-SPECTRA

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig, AgentLifecycle},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
    base_model::NxrModelResult,
};

/// Creative Muse Agent - Core creative synthesis
#[derive(Debug, Clone)]
pub struct CreativeMuseAgent {
    /// Agent configuration
    pub config: CreativeMuseConfig,
    /// Creative capabilities
    pub creative_capabilities: CreativeCapabilities,
    /// Content generation
    pub content_generation: ContentGeneration,
    /// Creative inspiration
    pub creative_inspiration: CreativeInspiration,
    /// Agent status
    status: AgentStatus,
    /// Agent metrics
    metrics: AgentMetrics,
}

/// Creative Muse Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeMuseConfig {
    /// Base agent configuration
    pub base_config: BaseAgentConfig,
    /// Creativity level
    pub creativity_level: CreativityLevel,
    /// Originality weight
    pub originality_weight: f32,
    /// Innovation threshold
    pub innovation_threshold: f32,
    /// Creative diversity
    pub creative_diversity: f32,
    /// Inspiration sources
    pub inspiration_sources: Vec<String>,
    /// Creative domains
    pub creative_domains: Vec<String>,
}

/// Creativity Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativityLevel {
    /// Conservative creativity
    Conservative,
    /// Moderate creativity
    Moderate,
    /// High creativity
    High,
    /// Maximum creativity
    Maximum,
    /// Transcendent creativity
    Transcendent,
}

impl CreativityLevel {
    /// Get numeric value for creativity level
    pub fn value(&self) -> f32 {
        match self {
            CreativityLevel::Conservative => 0.2,
            CreativityLevel::Moderate => 0.4,
            CreativityLevel::High => 0.6,
            CreativityLevel::Maximum => 0.8,
            CreativityLevel::Transcendent => 1.0,
        }
    }
}

/// Creative Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeCapabilities {
    /// Visual creativity
    pub visual_creativity: bool,
    /// Audio creativity
    pub audio_creativity: bool,
    /// Text creativity
    pub text_creativity: bool,
    /// Multimedia creativity
    pub multimedia_creativity: bool,
    /// Interactive creativity
    pub interactive_creativity: bool,
    /// Performance creativity
    pub performance_creativity: bool,
}

/// Content Generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentGeneration {
    /// Generation strategies
    pub strategies: Vec<GenerationStrategy>,
    /// Output formats
    pub output_formats: Vec<String>,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Generation parameters
    pub parameters: HashMap<String, f32>,
}

/// Generation Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationStrategy {
    /// Transformative generation
    Transformative,
    /// Generative generation
    Generative,
    /// Hybrid generation
    Hybrid,
    /// Collaborative generation
    Collaborative,
    /// Adaptive generation
    Adaptive,
}

/// Quality Thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum originality score
    pub min_originality: f32,
    /// Minimum creativity score
    pub min_creativity: f32,
    /// Minimum coherence score
    pub min_coherence: f32,
    /// Minimum innovation score
    pub min_innovation: f32,
}

/// Creative Inspiration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeInspiration {
    /// Inspiration sources
    pub sources: Vec<InspirationSource>,
    /// Inspiration cache
    pub inspiration_cache: HashMap<String, String>,
    /// Inspiration weights
    pub inspiration_weights: HashMap<String, f32>,
}

/// Inspiration Source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspirationSource {
    /// Source ID
    pub id: String,
    /// Source type
    pub source_type: InspirationSourceType,
    /// Source weight
    pub weight: f32,
    /// Source metadata
    pub metadata: HashMap<String, String>,
}

/// Inspiration Source Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InspirationSourceType {
    /// External database
    ExternalDatabase,
    /// Internal knowledge
    InternalKnowledge,
    /// User input
    UserInput,
    /// Environmental context
    EnvironmentalContext,
    /// Cross-modal synthesis
    CrossModalSynthesis,
}

/// Creative Task Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeTaskInput {
    /// Task description
    pub description: String,
    /// Creative domain
    pub domain: String,
    /// Style requirements
    pub style_requirements: Option<String>,
    /// Constraints
    pub constraints: Vec<String>,
    /// Inspiration hints
    pub inspiration_hints: Vec<String>,
    /// Quality requirements
    pub quality_requirements: Option<QualityThresholds>,
}

/// Creative Task Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeTaskOutput {
    /// Generated content
    pub content: String,
    /// Creativity score
    pub creativity_score: f32,
    /// Originality score
    pub originality_score: f32,
    /// Innovation score
    pub innovation_score: f32,
    /// Coherence score
    pub coherence_score: f32,
    /// Generation metadata
    pub metadata: HashMap<String, String>,
    /// Inspiration sources used
    pub inspiration_sources: Vec<String>,
}

impl Default for CreativeMuseConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            creativity_level: CreativityLevel::Moderate,
            originality_weight: 0.5,
            innovation_threshold: 0.6,
            creative_diversity: 0.7,
            inspiration_sources: vec![
                "internal_knowledge".to_string(),
                "cross_modal".to_string(),
            ],
            creative_domains: vec![
                "visual".to_string(),
                "text".to_string(),
                "multimedia".to_string(),
            ],
        }
    }
}

impl Default for CreativeCapabilities {
    fn default() -> Self {
        Self {
            visual_creativity: true,
            audio_creativity: true,
            text_creativity: true,
            multimedia_creativity: true,
            interactive_creativity: false,
            performance_creativity: false,
        }
    }
}

impl Default for ContentGeneration {
    fn default() -> Self {
        Self {
            strategies: vec![
                GenerationStrategy::Hybrid,
                GenerationStrategy::Adaptive,
            ],
            output_formats: vec![
                "text".to_string(),
                "json".to_string(),
            ],
            quality_thresholds: QualityThresholds {
                min_originality: 0.6,
                min_creativity: 0.5,
                min_coherence: 0.7,
                min_innovation: 0.4,
            },
            parameters: HashMap::new(),
        }
    }
}

impl Default for CreativeInspiration {
    fn default() -> Self {
        Self {
            sources: vec![],
            inspiration_cache: HashMap::new(),
            inspiration_weights: HashMap::new(),
        }
    }
}

impl Default for CreativeMuseAgent {
    fn default() -> Self {
        Self {
            config: CreativeMuseConfig::default(),
            creative_capabilities: CreativeCapabilities::default(),
            content_generation: ContentGeneration::default(),
            creative_inspiration: CreativeInspiration::default(),
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
impl BaseAgent for CreativeMuseAgent {
    type Config = CreativeMuseConfig;
    type Input = CreativeTaskInput;
    type Output = CreativeTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        self.validate_input(&input)?;
        
        // Generate creative content
        let content = self.generate_creative_content(&input).await?;
        
        // Calculate quality scores
        let creativity_score = self.calculate_creativity_score(&input, &content);
        let originality_score = self.calculate_originality_score(&content);
        let innovation_score = self.calculate_innovation_score(&input, &content);
        let coherence_score = self.calculate_coherence_score(&content);
        
        // Build output
        let output = CreativeTaskOutput {
            content,
            creativity_score,
            originality_score,
            innovation_score,
            coherence_score,
            metadata: HashMap::new(),
            inspiration_sources: self.get_inspiration_sources_used(&input),
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(output)
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
                name: "creative_synthesis".to_string(),
                description: "Core creative synthesis across multiple domains".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["creative_task".to_string()],
                output_types: vec!["creative_content".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.85,
                    avg_latency: 500.0,
                    resource_usage: 0.6,
                    reliability: 0.9,
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

impl CreativeMuseAgent {
    /// Create a new Creative Muse Agent
    pub fn new(config: CreativeMuseConfig) -> Self {
        Self {
            config,
            creative_capabilities: CreativeCapabilities::default(),
            content_generation: ContentGeneration::default(),
            creative_inspiration: CreativeInspiration::default(),
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

    /// Validate creative task input
    fn validate_input(&self, input: &CreativeTaskInput) -> AgentResult<()> {
        if input.description.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Task description cannot be empty".to_string()
            ));
        }
        
        if input.domain.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Creative domain cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }

    /// Generate creative content
    async fn generate_creative_content(&self, input: &CreativeTaskInput) -> AgentResult<String> {
        // This is a simplified implementation
        // In a real system, this would involve complex creative generation algorithms
        let creativity_level = self.config.creativity_level.value();
        let originality_weight = self.config.originality_weight;
        
        // Simulate creative content generation
        let base_content = format!(
            "Creative content for: {} in domain: {} with creativity level: {:.2}",
            input.description, input.domain, creativity_level
        );
        
        let enhanced_content = if originality_weight > 0.7 {
            format!("{} [High Originality]", base_content)
        } else {
            base_content
        };
        
        Ok(enhanced_content)
    }

    /// Calculate creativity score
    fn calculate_creativity_score(&self, input: &CreativeTaskInput, content: &str) -> f32 {
        let base_score = self.config.creativity_level.value();
        let domain_bonus = if self.config.creative_domains.contains(&input.domain) {
            0.2
        } else {
            0.0
        };
        
        (base_score + domain_bonus).min(1.0)
    }

    /// Calculate originality score
    fn calculate_originality_score(&self, content: &str) -> f32 {
        // Simplified originality calculation
        let unique_words = content.split_whitespace().collect::<std::collections::HashSet<_>>().len();
        let total_words = content.split_whitespace().count();
        
        if total_words == 0 {
            return 0.0;
        }
        
        let uniqueness_ratio = unique_words as f32 / total_words as f32;
        (uniqueness_ratio * self.config.originality_weight).min(1.0)
    }

    /// Calculate innovation score
    fn calculate_innovation_score(&self, input: &CreativeTaskInput, content: &str) -> f32 {
        let base_score = content.len() as f32 / 1000.0; // Simplified
        let threshold_adjustment = self.config.innovation_threshold;
        
        (base_score * threshold_adjustment).min(1.0)
    }

    /// Calculate coherence score
    fn calculate_coherence_score(&self, content: &str) -> f32 {
        // Simplified coherence calculation
        let sentences = content.split('.').count();
        if sentences == 0 {
            return 0.0;
        }
        
        // Base coherence on sentence structure
        let avg_sentence_length = content.len() as f32 / sentences as f32;
        let optimal_length = 50.0;
        let length_score = 1.0 - (avg_sentence_length - optimal_length).abs() / optimal_length;
        
        length_score.max(0.0).min(1.0)
    }

    /// Get inspiration sources used for this task
    fn get_inspiration_sources_used(&self, input: &CreativeTaskInput) -> Vec<String> {
        input.inspiration_hints.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creative_muse_agent_creation() {
        let agent = CreativeMuseAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[test]
    fn test_creativity_levels() {
        assert_eq!(CreativityLevel::Conservative.value(), 0.2);
        assert_eq!(CreativityLevel::Transcendent.value(), 1.0);
    }

    #[tokio::test]
    async fn test_creative_task_processing() {
        let agent = CreativeMuseAgent::default();
        let input = CreativeTaskInput {
            description: "Create a beautiful landscape".to_string(),
            domain: "visual".to_string(),
            style_requirements: None,
            constraints: vec![],
            inspiration_hints: vec![],
            quality_requirements: None,
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.content.is_empty());
        assert!(output.creativity_score > 0.0);
        assert!(output.originality_score >= 0.0);
        assert!(output.innovation_score >= 0.0);
        assert!(output.coherence_score >= 0.0);
    }
}
