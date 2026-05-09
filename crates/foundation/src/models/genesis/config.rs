//! NXR-GENESIS Configuration
//! 
//! Model-specific configuration for NXR-GENESIS

use serde::{Deserialize, Serialize};
use crate::shared::model_config::NxrModelConfig;

/// NXR-GENESIS Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Generative configuration
    pub generative: GenerativeConfig,
    /// Evolutionary configuration
    pub evolutionary: EvolutionaryConfig,
    /// Creative configuration
    pub creative: CreativeConfig,
    /// Agent configuration
    pub agents: AgentConfig,
}

/// Generative Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeConfig {
    /// Generation mode
    pub generation_mode: GenerationMode,
    /// Diversity control
    pub diversity_control: DiversityControl,
    /// Quality threshold
    pub quality_threshold: QualityThreshold,
}

/// GenerationMode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationMode {
    /// Random generation
    Random,
    /// Guided generation
    Guided,
    /// Hybrid generation
    Hybrid,
    /// Custom generation
    Custom { parameters: std::collections::HashMap<String, f32> },
}

/// DiversityControl
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityControl {
    /// Temperature
    pub temperature: f32,
    /// Top-k sampling
    pub top_k: u32,
    /// Top-p sampling
    pub top_p: f32,
}

/// QualityThreshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThreshold {
    /// Minimum quality score
    pub minimum_quality: f32,
    /// Coherence threshold
    pub coherence_threshold: f32,
    /// Novelty threshold
    pub novelty_threshold: f32,
}

/// Evolutionary Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryConfig {
    /// Evolution strategy
    pub evolution_strategy: EvolutionStrategy,
    /// Mutation rate
    pub mutation_rate: f32,
    /// Selection pressure
    pub selection_pressure: SelectionPressure,
}

/// EvolutionStrategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionStrategy {
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Evolutionary strategy
    EvolutionaryStrategy,
    /// Differential evolution
    DifferentialEvolution,
    /// Custom strategy
    Custom { strategy: String },
}

/// SelectionPressure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionPressure {
    /// Low pressure
    Low,
    /// Medium pressure
    Medium,
    /// High pressure
    High,
    /// Adaptive pressure
    Adaptive,
}

/// Creative Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeConfig {
    /// Creativity level
    pub creativity_level: CreativityLevel,
    /// Domain crossing
    pub domain_crossing: DomainCrossing,
    /// Innovation focus
    pub innovation_focus: InnovationFocus,
}

/// CreativityLevel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativityLevel {
    /// Conservative creativity
    Conservative,
    /// Moderate creativity
    Moderate,
    /// High creativity
    High,
    /// Radical creativity
    Radical,
}

/// DomainCrossing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCrossing {
    /// Cross-domain enabled
    pub enabled: bool,
    /// Domain weights
    pub domain_weights: std::collections::HashMap<String, f32>,
}

/// InnovationFocus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InnovationFocus {
    /// Incremental innovation
    Incremental,
    /// Radical innovation
    Radical,
    /// Disruptive innovation
    Disruptive,
}

/// Agent Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// CREATIVE-ENGINE configuration
    pub creative_engine: CreativeEngineConfig,
    /// INNOVATOR configuration
    pub innovator: InnovatorConfig,
    /// SYNTHESIZER configuration
    pub synthesizer: SynthesizerConfig,
    /// EVALUATOR configuration
    pub evaluator: EvaluatorConfig,
}

/// CreativeEngineConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeEngineConfig {
    /// Generation style
    pub generation_style: GenerationStyle,
    /// Exploration rate
    pub exploration_rate: f32,
    /// Iteration limit
    pub iteration_limit: u32,
}

/// GenerationStyle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationStyle {
    /// Divergent style
    Divergent,
    /// Convergent style
    Convergent,
    /// Hybrid style
    Hybrid,
}

/// InnovatorConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovatorConfig {
    /// Innovation strategy
    pub innovation_strategy: InnovationStrategy,
    /// Novelty detection
    pub novelty_detection: NoveltyDetection,
}

/// InnovationStrategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InnovationStrategy {
    /// Combinatorial innovation
    Combinatorial,
    /// Analogical innovation
    Analogical,
    /// First principles innovation
    FirstPrinciples,
}

/// NoveltyDetection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyDetection {
    /// Detection method
    pub detection_method: DetectionMethod,
    /// Threshold
    pub threshold: f32,
}

/// DetectionMethod
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Statistical detection
    Statistical,
    /// Semantic detection
    Semantic,
    /// Hybrid detection
    Hybrid,
}

/// SynthesizerConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizerConfig {
    /// Synthesis method
    pub synthesis_method: SynthesisMethod,
    /// Integration depth
    pub integration_depth: IntegrationDepth,
}

/// SynthesisMethod
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynthesisMethod {
    /// Linear synthesis
    Linear,
    /// Hierarchical synthesis
    Hierarchical,
    /// Network synthesis
    Network,
}

/// IntegrationDepth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationDepth {
    /// Shallow integration
    Shallow,
    /// Medium integration
    Medium,
    /// Deep integration
    Deep,
}

/// EvaluatorConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorConfig {
    /// Evaluation criteria
    pub evaluation_criteria: Vec<EvaluationCriterion>,
    /// Scoring method
    pub scoring_method: ScoringMethod,
}

/// EvaluationCriterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationCriterion {
    /// Criterion name
    pub name: String,
    /// Criterion weight
    pub weight: f32,
}

/// ScoringMethod
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringMethod {
    /// Weighted sum
    WeightedSum,
    /// Multi-criteria
    MultiCriteria,
    /// Machine learning scoring
    MachineLearning,
}

impl Default for GenesisConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Genesis),
            generative: GenerativeConfig::default(),
            evolutionary: EvolutionaryConfig::default(),
            creative: CreativeConfig::default(),
            agents: AgentConfig::default(),
        }
    }
}

impl Default for GenerativeConfig {
    fn default() -> Self {
        Self {
            generation_mode: GenerationMode::Hybrid,
            diversity_control: DiversityControl {
                temperature: 0.8,
                top_k: 50,
                top_p: 0.95,
            },
            quality_threshold: QualityThreshold {
                minimum_quality: 0.7,
                coherence_threshold: 0.75,
                novelty_threshold: 0.6,
            },
        }
    }
}

impl Default for EvolutionaryConfig {
    fn default() -> Self {
        Self {
            evolution_strategy: EvolutionStrategy::GeneticAlgorithm,
            mutation_rate: 0.1,
            selection_pressure: SelectionPressure::Medium,
        }
    }
}

impl Default for CreativeConfig {
    fn default() -> Self {
        Self {
            creativity_level: CreativityLevel::High,
            domain_crossing: DomainCrossing {
                enabled: true,
                domain_weights: std::collections::HashMap::new(),
            },
            innovation_focus: InnovationFocus::Radical,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            creative_engine: CreativeEngineConfig::default(),
            innovator: InnovatorConfig::default(),
            synthesizer: SynthesizerConfig::default(),
            evaluator: EvaluatorConfig::default(),
        }
    }
}

impl Default for CreativeEngineConfig {
    fn default() -> Self {
        Self {
            generation_style: GenerationStyle::Hybrid,
            exploration_rate: 0.3,
            iteration_limit: 100,
        }
    }
}

impl Default for InnovatorConfig {
    fn default() -> Self {
        Self {
            innovation_strategy: InnovationStrategy::Combinatorial,
            novelty_detection: NoveltyDetection {
                detection_method: DetectionMethod::Hybrid,
                threshold: 0.7,
            },
        }
    }
}

impl Default for SynthesizerConfig {
    fn default() -> Self {
        Self {
            synthesis_method: SynthesisMethod::Network,
            integration_depth: IntegrationDepth::Deep,
        }
    }
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            evaluation_criteria: vec![
                EvaluationCriterion {
                    name: "creativity".to_string(),
                    weight: 0.4,
                },
                EvaluationCriterion {
                    name: "novelty".to_string(),
                    weight: 0.3,
                },
                EvaluationCriterion {
                    name: "quality".to_string(),
                    weight: 0.3,
                },
            ],
            scoring_method: ScoringMethod::WeightedSum,
        }
    }
}

impl GenesisConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate diversity control
        if self.generative.diversity_control.temperature < 0.0 || self.generative.diversity_control.temperature > 2.0 {
            return Err("Temperature must be between 0.0 and 2.0".to_string());
        }

        // Validate quality threshold
        if self.generative.quality_threshold.minimum_quality < 0.0 || self.generative.quality_threshold.minimum_quality > 1.0 {
            return Err("Minimum quality must be between 0.0 and 1.0".to_string());
        }

        // Validate evolutionary config
        if self.evolutionary.mutation_rate < 0.0 || self.evolutionary.mutation_rate > 1.0 {
            return Err("Mutation rate must be between 0.0 and 1.0".to_string());
        }

        // Validate evaluation criteria
        let total_weight: f32 = self.agents.evaluator.evaluation_criteria.iter().map(|c| c.weight).sum();
        if (total_weight - 1.0).abs() > 0.01 {
            return Err("Evaluation criterion weights must sum to 1.0".to_string());
        }

        Ok(())
    }

    /// Get configuration for specific agent
    pub fn get_agent_config(&self, agent_name: &str) -> Option<serde_json::Value> {
        match agent_name {
            "creative_engine" => Some(serde_json::to_value(&self.agents.creative_engine).unwrap_or_default()),
            "innovator" => Some(serde_json::to_value(&self.agents.innovator).unwrap_or_default()),
            "synthesizer" => Some(serde_json::to_value(&self.agents.synthesizer).unwrap_or_default()),
            "evaluator" => Some(serde_json::to_value(&self.agents.evaluator).unwrap_or_default()),
            _ => None,
        }
    }
}
