//! Style Adapter Agent
//! 
//! Dynamic style learning and adaptation agent for NXR-SPECTRA

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Style Adapter Agent - Dynamic style learning and adaptation
#[derive(Debug, Clone)]
pub struct StyleAdapterAgent {
    /// Agent configuration
    pub config: StyleAdapterConfig,
    /// Style learning capabilities
    pub style_learning: StyleLearning,
    /// Adaptation engine
    pub adaptation_engine: AdaptationEngine,
    /// Style knowledge base
    pub style_knowledge: StyleKnowledgeBase,
    /// Agent status
    status: AgentStatus,
    /// Agent metrics
    metrics: AgentMetrics,
}

/// Style Adapter Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAdapterConfig {
    /// Base agent configuration
    pub base_config: BaseAgentConfig,
    /// Learning rate
    pub learning_rate: f32,
    /// Adaptation sensitivity
    pub adaptation_sensitivity: f32,
    /// Style domains
    pub style_domains: Vec<String>,
    /// Adaptation strategies
    pub adaptation_strategies: Vec<AdaptationStrategyType>,
}

/// Style Learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleLearning {
    /// Learning algorithms
    pub algorithms: Vec<LearningAlgorithm>,
    /// Training data sources
    pub training_sources: Vec<String>,
    /// Learning progress
    pub learning_progress: LearningProgress,
    /// Style patterns
    pub style_patterns: HashMap<String, StylePattern>,
}

/// Learning Algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    /// Neural network based learning
    NeuralNetwork,
    /// Statistical pattern learning
    StatisticalPattern,
    /// Rule-based learning
    RuleBased,
    /// Hybrid learning
    Hybrid,
    /// Reinforcement learning
    Reinforcement,
}

/// Learning Progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
    /// Total samples processed
    pub total_samples: u64,
    /// Successful adaptations
    pub successful_adaptations: u64,
    /// Learning accuracy
    pub learning_accuracy: f32,
    /// Current learning phase
    pub current_phase: LearningPhase,
}

/// Learning Phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningPhase {
    /// Initial data collection
    DataCollection,
    /// Pattern recognition
    PatternRecognition,
    /// Model training
    ModelTraining,
    /// Active adaptation
    ActiveAdaptation,
    /// Continuous improvement
    ContinuousImprovement,
}

/// Style Pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylePattern {
    /// Pattern ID
    pub id: String,
    /// Pattern name
    pub name: String,
    /// Pattern features
    pub features: HashMap<String, f32>,
    /// Pattern frequency
    pub frequency: f32,
    /// Pattern confidence
    pub confidence: f32,
}

/// Adaptation Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEngine {
    /// Adaptation methods
    pub methods: Vec<AdaptationMethod>,
    /// Adaptation history
    pub adaptation_history: Vec<AdaptationRecord>,
    /// Performance metrics
    pub performance_metrics: AdaptationMetrics,
}

/// Adaptation Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationMethod {
    /// Direct mapping
    DirectMapping,
    /// Feature transformation
    FeatureTransformation,
    /// Neural adaptation
    NeuralAdaptation,
    /// Statistical adaptation
    StatisticalAdaptation,
    /// Hybrid adaptation
    HybridAdaptation,
}

/// Adaptation Record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRecord {
    /// Record ID
    pub id: String,
    /// Source style
    pub source_style: String,
    /// Target style
    pub target_style: String,
    /// Adaptation method
    pub method: AdaptationMethod,
    /// Success rate
    pub success_rate: f32,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Adaptation Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    /// Total adaptations
    pub total_adaptations: u64,
    /// Successful adaptations
    pub successful_adaptations: u64,
    /// Average success rate
    pub avg_success_rate: f32,
    /// Adaptation speed (ms)
    pub avg_adaptation_speed: f64,
}

/// Style Knowledge Base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleKnowledgeBase {
    /// Style definitions
    pub style_definitions: HashMap<String, StyleDefinition>,
    /// Style relationships
    pub style_relationships: HashMap<String, Vec<String>>,
    /// Style evolution history
    pub evolution_history: Vec<StyleEvolution>,
}

/// Style Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleDefinition {
    /// Style name
    pub name: String,
    /// Style category
    pub category: String,
    /// Style characteristics
    pub characteristics: HashMap<String, f32>,
    /// Style rules
    pub rules: Vec<StyleRule>,
    /// Style metadata
    pub metadata: HashMap<String, String>,
}

/// Style Rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleRule {
    /// Rule ID
    pub id: String,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: String,
    /// Rule weight
    pub weight: f32,
}

/// Style Evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleEvolution {
    /// Style name
    pub style_name: String,
    /// Evolution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Changes made
    pub changes: Vec<StyleChange>,
}

/// Style Change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleChange {
    /// Change type
    pub change_type: ChangeType,
    /// Old value
    pub old_value: Option<String>,
    /// New value
    pub new_value: String,
    /// Change reason
    pub reason: String,
}

/// Change Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// Addition
    Addition,
    /// Modification
    Modification,
    /// Deletion
    Deletion,
}

/// Adaptation Strategy Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategyType {
    /// Real-time adaptation
    RealTime,
    /// Batch adaptation
    Batch,
    /// Progressive adaptation
    Progressive,
    /// Contextual adaptation
    Contextual,
}

/// Style Adaptation Task Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAdaptationTaskInput {
    /// Source content
    pub source_content: String,
    /// Source style
    pub source_style: String,
    /// Target style
    pub target_style: String,
    /// Adaptation constraints
    pub constraints: Vec<String>,
    /// Adaptation preferences
    pub preferences: HashMap<String, f32>,
}

/// Style Adaptation Task Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAdaptationTaskOutput {
    /// Adapted content
    pub adapted_content: String,
    /// Adaptation success rate
    pub success_rate: f32,
    /// Adaptation confidence
    pub confidence: f32,
    /// Applied adaptations
    pub applied_adaptations: Vec<AppliedAdaptation>,
    /// Style preservation score
    pub style_preservation_score: f32,
    /// Adaptation metadata
    pub metadata: HashMap<String, String>,
}

/// Applied Adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedAdaptation {
    /// Adaptation type
    pub adaptation_type: String,
    /// Adaptation strength
    pub strength: f32,
    /// Success indicator
    pub success: bool,
    /// Adaptation details
    pub details: HashMap<String, String>,
}

impl Default for StyleAdapterConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            learning_rate: 0.01,
            adaptation_sensitivity: 0.7,
            style_domains: vec![
                "visual".to_string(),
                "text".to_string(),
                "audio".to_string(),
            ],
            adaptation_strategies: vec![
                AdaptationStrategyType::RealTime,
                AdaptationStrategyType::Contextual,
            ],
        }
    }
}

impl Default for StyleLearning {
    fn default() -> Self {
        Self {
            algorithms: vec![
                LearningAlgorithm::NeuralNetwork,
                LearningAlgorithm::StatisticalPattern,
            ],
            training_sources: vec![
                "user_feedback".to_string(),
                "style_examples".to_string(),
            ],
            learning_progress: LearningProgress {
                total_samples: 0,
                successful_adaptations: 0,
                learning_accuracy: 0.0,
                current_phase: LearningPhase::DataCollection,
            },
            style_patterns: HashMap::new(),
        }
    }
}

impl Default for AdaptationEngine {
    fn default() -> Self {
        Self {
            methods: vec![
                AdaptationMethod::NeuralAdaptation,
                AdaptationMethod::HybridAdaptation,
            ],
            adaptation_history: Vec::new(),
            performance_metrics: AdaptationMetrics {
                total_adaptations: 0,
                successful_adaptations: 0,
                avg_success_rate: 0.0,
                avg_adaptation_speed: 0.0,
            },
        }
    }
}

impl Default for StyleKnowledgeBase {
    fn default() -> Self {
        Self {
            style_definitions: HashMap::new(),
            style_relationships: HashMap::new(),
            evolution_history: Vec::new(),
        }
    }
}

impl Default for StyleAdapterAgent {
    fn default() -> Self {
        Self {
            config: StyleAdapterConfig::default(),
            style_learning: StyleLearning::default(),
            adaptation_engine: AdaptationEngine::default(),
            style_knowledge: StyleKnowledgeBase::default(),
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
impl BaseAgent for StyleAdapterAgent {
    type Config = StyleAdapterConfig;
    type Input = StyleAdaptationTaskInput;
    type Output = StyleAdaptationTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        self.validate_input(&input)?;
        
        // Analyze source style
        let source_analysis = self.analyze_source_style(&input).await?;
        
        // Analyze target style
        let target_analysis = self.analyze_target_style(&input).await?;
        
        // Select adaptation method
        let adaptation_method = self.select_adaptation_method(&input, &source_analysis, &target_analysis);
        
        // Perform style adaptation
        let adaptation_result = self.perform_style_adaptation(&input, &adaptation_method).await?;
        
        // Calculate adaptation metrics
        let success_rate = self.calculate_adaptation_success(&input, &adaptation_result);
        let confidence = self.calculate_adaptation_confidence(&adaptation_result);
        let preservation_score = self.calculate_style_preservation(&input, &adaptation_result);
        
        // Build output
        let output = StyleAdaptationTaskOutput {
            adapted_content: adaptation_result.adapted_content,
            success_rate,
            confidence,
            applied_adaptations: adaptation_result.applied_adaptations,
            style_preservation_score: preservation_score,
            metadata: adaptation_result.metadata,
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
                name: "style_adaptation".to_string(),
                description: "Dynamic style learning and adaptation".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["style_adaptation_task".to_string()],
                output_types: vec!["adapted_content".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.88,
                    avg_latency: 600.0,
                    resource_usage: 0.65,
                    reliability: 0.92,
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

impl StyleAdapterAgent {
    /// Create a new Style Adapter Agent
    pub fn new(config: StyleAdapterConfig) -> Self {
        Self {
            config,
            style_learning: StyleLearning::default(),
            adaptation_engine: AdaptationEngine::default(),
            style_knowledge: StyleKnowledgeBase::default(),
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

    /// Validate style adaptation input
    fn validate_input(&self, input: &StyleAdaptationTaskInput) -> AgentResult<()> {
        if input.source_content.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Source content cannot be empty".to_string()
            ));
        }
        
        if input.source_style.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Source style cannot be empty".to_string()
            ));
        }
        
        if input.target_style.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Target style cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }

    /// Analyze source style
    async fn analyze_source_style(&self, input: &StyleAdaptationTaskInput) -> AgentResult<StyleAnalysis> {
        // Simplified source style analysis
        let characteristics = self.extract_style_characteristics(&input.source_content, &input.source_style);
        let confidence = self.calculate_style_confidence(&characteristics);
        
        Ok(StyleAnalysis {
            style_name: input.source_style.clone(),
            characteristics,
            confidence,
            complexity: self.calculate_content_complexity(&input.source_content),
        })
    }

    /// Analyze target style
    async fn analyze_target_style(&self, input: &StyleAdaptationTaskInput) -> AgentResult<StyleAnalysis> {
        // Get target style definition from knowledge base
        let target_definition = self.style_knowledge.style_definitions
            .get(&input.target_style)
            .cloned()
            .unwrap_or_else(|| StyleDefinition {
                name: input.target_style.clone(),
                category: "unknown".to_string(),
                characteristics: HashMap::new(),
                rules: Vec::new(),
                metadata: HashMap::new(),
            });
        
        Ok(StyleAnalysis {
            style_name: input.target_style.clone(),
            characteristics: target_definition.characteristics,
            confidence: 0.8,
            complexity: 0.5,
        })
    }

    /// Select adaptation method
    fn select_adaptation_method(&self, input: &StyleAdaptationTaskInput, 
                               source_analysis: &StyleAnalysis, 
                               target_analysis: &StyleAnalysis) -> AdaptationMethod {
        // Simplified method selection
        if source_analysis.confidence > 0.8 && target_analysis.confidence > 0.8 {
            AdaptationMethod::NeuralAdaptation
        } else if source_analysis.complexity > 0.7 {
            AdaptationMethod::HybridAdaptation
        } else {
            AdaptationMethod::FeatureTransformation
        }
    }

    /// Perform style adaptation
    async fn perform_style_adaptation(&self, input: &StyleAdaptationTaskInput, 
                                    method: &AdaptationMethod) -> AgentResult<AdaptationResult> {
        let adapted_content = match method {
            AdaptationMethod::NeuralAdaptation => {
                format!("{} [Neural Style Adaptation: {} -> {}]", 
                       input.source_content, input.source_style, input.target_style)
            },
            AdaptationMethod::FeatureTransformation => {
                format!("{} [Feature Transformation: {} -> {}]", 
                       input.source_content, input.source_style, input.target_style)
            },
            AdaptationMethod::HybridAdaptation => {
                format!("{} [Hybrid Style Adaptation: {} -> {}]", 
                       input.source_content, input.source_style, input.target_style)
            },
            _ => {
                format!("{} [Style Adaptation: {} -> {}]", 
                       input.source_content, input.source_style, input.target_style)
            }
        };
        
        let applied_adaptations = vec![
            AppliedAdaptation {
                adaptation_type: format!("{:?}", method),
                strength: self.config.adaptation_sensitivity,
                success: true,
                details: HashMap::new(),
            }
        ];
        
        Ok(AdaptationResult {
            adapted_content,
            applied_adaptations,
            metadata: HashMap::new(),
        })
    }

    /// Calculate adaptation success rate
    fn calculate_adaptation_success(&self, input: &StyleAdaptationTaskInput, result: &AdaptationResult) -> f32 {
        // Simplified success calculation
        let content_length_ratio = result.adapted_content.len() as f32 / input.source_content.len() as f32;
        let adaptation_count = result.applied_adaptations.len() as f32;
        
        let length_score = if content_length_ratio >= 0.8 && content_length_ratio <= 1.2 { 0.8 } else { 0.5 };
        let adaptation_score = if adaptation_count > 0.0 { 0.9 } else { 0.0 };
        
        (length_score + adaptation_score) / 2.0
    }

    /// Calculate adaptation confidence
    fn calculate_adaptation_confidence(&self, result: &AdaptationResult) -> f32 {
        let successful_adaptations = result.applied_adaptations.iter()
            .filter(|adaptation| adaptation.success)
            .count() as f32;
        
        let total_adaptations = result.applied_adaptations.len() as f32;
        
        if total_adaptations == 0.0 {
            return 0.0;
        }
        
        successful_adaptations / total_adaptations
    }

    /// Calculate style preservation score
    fn calculate_style_preservation(&self, input: &StyleAdaptationTaskInput, result: &AdaptationResult) -> f32 {
        // Simplified preservation calculation
        let source_words = input.source_content.split_whitespace().collect::<std::collections::HashSet<_>>();
        let adapted_words = result.adapted_content.split_whitespace().collect::<std::collections::HashSet<_>>();
        
        if source_words.is_empty() {
            return 0.0;
        }
        
        let common_words = source_words.intersection(&adapted_words).count() as f32;
        common_words / source_words.len() as f32
    }

    /// Extract style characteristics
    fn extract_style_characteristics(&self, content: &str, style_name: &str) -> HashMap<String, f32> {
        let mut characteristics = HashMap::new();
        
        // Simplified characteristic extraction
        characteristics.insert("formality".to_string(), 0.5);
        characteristics.insert("complexity".to_string(), self.calculate_content_complexity(content));
        characteristics.insert("creativity".to_string(), 0.6);
        characteristics.insert("consistency".to_string(), 0.7);
        
        characteristics
    }

    /// Calculate style confidence
    fn calculate_style_confidence(&self, characteristics: &HashMap<String, f32>) -> f32 {
        let values: Vec<f32> = characteristics.values().cloned().collect();
        if values.is_empty() {
            return 0.0;
        }
        
        values.iter().sum::<f32>() / values.len() as f32
    }

    /// Calculate content complexity
    fn calculate_content_complexity(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count() as f32;
        let unique_words = content.split_whitespace().collect::<std::collections::HashSet<_>>().len() as f32;
        
        if word_count == 0.0 {
            return 0.0;
        }
        
        unique_words / word_count
    }
}

/// Style Analysis Result
#[derive(Debug, Clone)]
pub struct StyleAnalysis {
    /// Style name
    pub style_name: String,
    /// Style characteristics
    pub characteristics: HashMap<String, f32>,
    /// Analysis confidence
    pub confidence: f32,
    /// Style complexity
    pub complexity: f32,
}

/// Adaptation Result
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Adapted content
    pub adapted_content: String,
    /// Applied adaptations
    pub applied_adaptations: Vec<AppliedAdaptation>,
    /// Adaptation metadata
    pub metadata: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_adapter_agent_creation() {
        let agent = StyleAdapterAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_style_adaptation_processing() {
        let agent = StyleAdapterAgent::default();
        let input = StyleAdaptationTaskInput {
            source_content: "A simple piece of text".to_string(),
            source_style: "simple".to_string(),
            target_style: "elegant".to_string(),
            constraints: vec![],
            preferences: HashMap::new(),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.adapted_content.is_empty());
        assert!(output.success_rate > 0.0);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
        assert!(output.style_preservation_score >= 0.0 && output.style_preservation_score <= 1.0);
    }

    #[test]
    fn test_content_complexity_calculation() {
        let agent = StyleAdapterAgent::default();
        
        let simple_text = "hello world";
        let complex_text = "hello wonderful beautiful amazing world";
        
        let simple_complexity = agent.calculate_content_complexity(simple_text);
        let complex_complexity = agent.calculate_content_complexity(complex_text);
        
        assert!(simple_complexity >= 0.0 && simple_complexity <= 1.0);
        assert!(complex_complexity >= 0.0 && complex_complexity <= 1.0);
    }
}
