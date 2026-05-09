//! NXR-OMNIS Configuration
//! 
//! Model-specific configuration for NXR-OMNIS

use serde::{Deserialize, Serialize};
use crate::shared::{model_config::NxrModelConfig, deeplearning_integration::DeepLearningConfig};

/// NXR-OMNIS Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmnisConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Reasoning configuration
    pub reasoning: ReasoningConfig,
    /// World model configuration
    pub world_model: WorldModelConfig,
    /// Agent configuration
    pub agents: AgentConfig,
    /// Meta-reasoning configuration
    pub meta_reasoning: MetaReasoningConfig,
    /// Truth arbitration configuration
    pub truth_arbitration: TruthArbitrationConfig,
    /// Deep learning configuration
    pub deep_learning: DeepLearningConfig,
}

/// Reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Maximum reasoning depth
    pub max_reasoning_depth: u8,
    /// Reasoning timeout in seconds
    pub reasoning_timeout_seconds: u64,
    /// Enable chain-of-thought
    pub enable_chain_of_thought: bool,
    /// Enable tree-of-thought
    pub enable_tree_of_thought: bool,
    /// Enable graph-of-thought
    pub enable_graph_of_thought: bool,
    /// Decomposition strategy
    pub decomposition_strategy: DecompositionStrategy,
    /// Synthesis strategy
    pub synthesis_strategy: SynthesisStrategy,
}

/// Decomposition strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Hierarchical decomposition
    Hierarchical,
    /// Goal-oriented decomposition
    GoalOriented,
    /// Constraint-based decomposition
    ConstraintBased,
    /// Hybrid decomposition
    Hybrid { weights: DecompositionWeights },
}

/// Decomposition weights for hybrid strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionWeights {
    /// Hierarchical weight
    pub hierarchical: f32,
    /// Goal-oriented weight
    pub goal_oriented: f32,
    /// Constraint-based weight
    pub constraint_based: f32,
}

/// Synthesis strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynthesisStrategy {
    /// Weighted averaging
    WeightedAverage,
    /// Hierarchical synthesis
    Hierarchical,
    /// Constraint satisfaction
    ConstraintSatisfaction,
    /// Neural synthesis
    Neural,
    /// Hybrid synthesis
    Hybrid { weights: SynthesisWeights },
}

/// Synthesis weights for hybrid strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisWeights {
    /// Weighted average weight
    pub weighted_average: f32,
    /// Hierarchical weight
    pub hierarchical: f32,
    /// Constraint satisfaction weight
    pub constraint_satisfaction: f32,
    /// Neural synthesis weight
    pub neural: f32,
}

/// World model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConfig {
    /// World model size
    pub world_model_size: WorldModelSize,
    /// Update frequency
    pub update_frequency_ms: u64,
    /// Context window management
    pub context_management: ContextManagement,
    /// Knowledge graph integration
    pub knowledge_graph: KnowledgeGraphConfig,
    /// Simulation parameters
    pub simulation: SimulationConfig,
}

/// World model size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldModelSize {
    /// Small (limited scope)
    Small,
    /// Medium (domain-specific)
    Medium,
    /// Large (multi-domain)
    Large,
    /// Unlimited (full world)
    Unlimited,
}

/// Context management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextManagement {
    /// Maximum context windows
    pub max_context_windows: usize,
    /// Context expiration time in minutes
    pub context_expiration_minutes: u64,
    /// Context compression enabled
    pub enable_compression: bool,
    /// Context prioritization
    pub prioritization: ContextPrioritization,
}

/// Context prioritization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextPrioritization {
    /// FIFO (First In, First Out)
    FIFO,
    /// LRU (Least Recently Used)
    LRU,
    /// Relevance-based
    Relevance,
    /// Importance-based
    Importance,
    /// Hybrid prioritization
    Hybrid { weights: PrioritizationWeights },
}

/// Prioritization weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizationWeights {
    /// Recency weight
    pub recency: f32,
    /// Relevance weight
    pub relevance: f32,
    /// Importance weight
    pub importance: f32,
}

/// Knowledge graph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphConfig {
    /// Enable knowledge graph
    pub enabled: bool,
    /// Graph size
    pub graph_size: GraphSize,
    /// Update strategy
    pub update_strategy: GraphUpdateStrategy,
    /// Reasoning over graph
    pub enable_reasoning: bool,
}

/// Graph size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphSize {
    /// Small (10K entities)
    Small,
    /// Medium (1M entities)
    Medium,
    /// Large (100M entities)
    Large,
    /// Massive (1B+ entities)
    Massive,
}

/// Graph update strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphUpdateStrategy {
    /// Incremental updates
    Incremental,
    /// Periodic rebuilding
    Periodic { interval_hours: u32 },
    /// Event-driven
    EventDriven,
    /// Hybrid strategy
    Hybrid,
}

/// Simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Enable simulation
    pub enabled: bool,
    /// Simulation fidelity
    pub fidelity: SimulationFidelity,
    /// Maximum simulation steps
    pub max_steps: usize,
    /// Simulation timeout in seconds
    pub timeout_seconds: u64,
}

/// Simulation fidelity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationFidelity {
    /// Low fidelity (fast)
    Low,
    /// Medium fidelity
    Medium,
    /// High fidelity (slow)
    High,
    /// Ultra fidelity (very slow)
    Ultra,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// ORACLE-7 configuration
    pub oracle_7: Oracle7Config,
    /// SYNTH-PRIME configuration
    pub synth_prime: SynthPrimeConfig,
    /// META-REASONER configuration
    pub meta_reasoner: MetaReasonerAgentConfig,
    /// WORLD-MODEL-X configuration
    pub world_model_x: WorldModelXConfig,
    /// TRUTH-ARBITER configuration
    pub truth_arbiter: TruthArbiterConfig,
    /// CHAIN-EXECUTOR configuration
    pub chain_executor: ChainExecutorConfig,
}

/// ORACLE-7 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Oracle7Config {
    /// Knowledge base size
    pub knowledge_base_size: KnowledgeBaseSize,
    /// Reasoning depth
    pub reasoning_depth: u8,
    /// Verification enabled
    pub enable_verification: bool,
    /// Confidence threshold
    pub confidence_threshold: f32,
}

/// Knowledge base size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeBaseSize {
    /// Small (1GB)
    Small,
    /// Medium (10GB)
    Medium,
    /// Large (100GB)
    Large,
    /// Massive (1TB+)
    Massive,
}

/// SYNTH-PRIME configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthPrimeConfig {
    /// Synthesis strategy
    pub strategy: SynthesisStrategy,
    /// Quality threshold
    pub quality_threshold: f32,
    /// Creativity level
    pub creativity_level: CreativityLevel,
    /// Enable cross-modal synthesis
    pub enable_cross_modal: bool,
}

/// Creativity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativityLevel {
    /// Conservative
    Conservative,
    /// Balanced
    Balanced,
    /// Creative
    Creative,
    /// Highly creative
    HighlyCreative,
    /// Transcendent
    Transcendent,
}

/// META-REASONER configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaReasonerAgentConfig {
    /// Meta-reasoning depth
    pub meta_reasoning_depth: u8,
    /// Self-reflection enabled
    pub enable_self_reflection: bool,
    /// Strategy selection
    pub strategy_selection: StrategySelection,
    /// Performance monitoring
    pub enable_performance_monitoring: bool,
}

/// Strategy selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategySelection {
    /// Fixed strategy
    Fixed { strategy: String },
    /// Adaptive selection
    Adaptive,
    /// Learning-based selection
    Learning,
    /// Hybrid selection
    Hybrid,
}

/// WORLD-MODEL-X configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelXConfig {
    /// World model type
    pub model_type: WorldModelType,
    /// Update frequency
    pub update_frequency_ms: u64,
    /// Context window size
    pub context_window_size: usize,
    /// Enable prediction
    pub enable_prediction: bool,
}

/// World model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldModelType {
    /// Physics-based simulation
    Physics,
    /// Neural world model
    Neural,
    /// Hybrid simulation
    Hybrid,
    /// Symbolic reasoning
    Symbolic,
}

/// TRUTH-ARBITER configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthArbiterConfig {
    /// Arbitration strategy
    pub arbitration_strategy: ArbitrationStrategy,
    /// Evidence threshold
    pub evidence_threshold: f32,
    /// Contradiction detection
    pub enable_contradiction_detection: bool,
    /// Resolution confidence threshold
    pub resolution_confidence_threshold: f32,
}

/// Arbitration strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArbitrationStrategy {
    /// Majority vote
    MajorityVote,
    /// Weighted evidence
    WeightedEvidence,
    /// Bayesian arbitration
    Bayesian,
    /// Neural arbitration
    Neural,
    /// Hybrid arbitration
    Hybrid,
}

/// CHAIN-EXECUTOR configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainExecutorConfig {
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Parallel execution enabled
    pub enable_parallel: bool,
    /// Maximum parallel tasks
    pub max_parallel_tasks: usize,
    /// Dependency resolution
    pub dependency_resolution: DependencyResolution,
}

/// Execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Adaptive execution
    Adaptive,
    /// Priority-based execution
    PriorityBased,
}

/// Dependency resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyResolution {
    /// Topological sort
    TopologicalSort,
    /// Priority-based resolution
    PriorityBased,
    /// Dynamic resolution
    Dynamic,
}

/// Meta-reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaReasoningConfig {
    /// Meta-reasoning enabled
    pub enabled: bool,
    /// Meta-reasoning depth
    pub depth: u8,
    /// Self-awareness level
    pub self_awareness_level: SelfAwarenessLevel,
    /// Reflection frequency
    pub reflection_frequency_ms: u64,
    /// Strategy learning enabled
    pub enable_strategy_learning: bool,
}

/// Self-awareness level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelfAwarenessLevel {
    /// No self-awareness
    None,
    /// Basic self-awareness
    Basic,
    /// Advanced self-awareness
    Advanced,
    /// Full self-awareness
    Full,
    /// Transcendent self-awareness
    Transcendent,
}

/// Truth arbitration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthArbitrationConfig {
    /// Arbitration method
    pub method: ArbitrationMethod,
    /// Evidence weighting
    pub evidence_weighting: EvidenceWeighting,
    /// Contradiction handling
    pub contradiction_handling: ContradictionHandling,
    /// Resolution confidence threshold
    pub confidence_threshold: f32,
}

/// Arbitration method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArbitrationMethod {
    /// Consensus-based
    Consensus,
    /// Evidence-based
    EvidenceBased,
    /// Bayesian
    Bayesian,
    /// Neural network
    Neural,
    /// Hybrid method
    Hybrid,
}

/// Evidence weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceWeighting {
    /// Equal weighting
    Equal,
    /// Source reliability weighting
    SourceReliability,
    /// Recency weighting
    Recency,
    /// Confidence weighting
    Confidence,
    /// Hybrid weighting
    Hybrid,
}

/// Contradiction handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContradictionHandling {
    /// Ignore contradictions
    Ignore,
    /// Flag contradictions
    Flag,
    /// Resolve contradictions
    Resolve,
    /// Synthesize contradictions
    Synthesize,
    /// Escalate contradictions
    Escalate,
}

impl Default for OmnisConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Omnis),
            reasoning: ReasoningConfig::default(),
            world_model: WorldModelConfig::default(),
            agents: AgentConfig::default(),
            meta_reasoning: MetaReasoningConfig::default(),
            truth_arbitration: TruthArbitrationConfig::default(),
            deep_learning: DeepLearningConfig::star_x(),
        }
    }
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            max_reasoning_depth: 10,
            reasoning_timeout_seconds: 300,
            enable_chain_of_thought: true,
            enable_tree_of_thought: true,
            enable_graph_of_thought: true,
            decomposition_strategy: DecompositionStrategy::Hybrid {
                weights: DecompositionWeights {
                    hierarchical: 0.4,
                    goal_oriented: 0.4,
                    constraint_based: 0.2,
                },
            },
            synthesis_strategy: SynthesisStrategy::Hybrid {
                weights: SynthesisWeights {
                    weighted_average: 0.2,
                    hierarchical: 0.3,
                    constraint_satisfaction: 0.2,
                    neural: 0.3,
                },
            },
        }
    }
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            world_model_size: WorldModelSize::Unlimited,
            update_frequency_ms: 1000,
            context_management: ContextManagement::default(),
            knowledge_graph: KnowledgeGraphConfig::default(),
            simulation: SimulationConfig::default(),
        }
    }
}

impl Default for ContextManagement {
    fn default() -> Self {
        Self {
            max_context_windows: 1000,
            context_expiration_minutes: 60,
            enable_compression: true,
            prioritization: ContextPrioritization::Hybrid {
                weights: PrioritizationWeights {
                    recency: 0.3,
                    relevance: 0.4,
                    importance: 0.3,
                },
            },
        }
    }
}

impl Default for KnowledgeGraphConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            graph_size: GraphSize::Massive,
            update_strategy: GraphUpdateStrategy::Hybrid,
            enable_reasoning: true,
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fidelity: SimulationFidelity::High,
            max_steps: 1000,
            timeout_seconds: 60,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            oracle_7: Oracle7Config::default(),
            synth_prime: SynthPrimeConfig::default(),
            meta_reasoner: MetaReasonerAgentConfig::default(),
            world_model_x: WorldModelXConfig::default(),
            truth_arbiter: TruthArbiterConfig::default(),
            chain_executor: ChainExecutorConfig::default(),
        }
    }
}

impl Default for Oracle7Config {
    fn default() -> Self {
        Self {
            knowledge_base_size: KnowledgeBaseSize::Massive,
            reasoning_depth: 8,
            enable_verification: true,
            confidence_threshold: 0.9,
        }
    }
}

impl Default for SynthPrimeConfig {
    fn default() -> Self {
        Self {
            strategy: SynthesisStrategy::Neural,
            quality_threshold: 0.95,
            creativity_level: CreativityLevel::Transcendent,
            enable_cross_modal: true,
        }
    }
}

impl Default for MetaReasonerAgentConfig {
    fn default() -> Self {
        Self {
            meta_reasoning_depth: 5,
            enable_self_reflection: true,
            strategy_selection: StrategySelection::Adaptive,
            enable_performance_monitoring: true,
        }
    }
}

impl Default for WorldModelXConfig {
    fn default() -> Self {
        Self {
            model_type: WorldModelType::Hybrid,
            update_frequency_ms: 500,
            context_window_size: 10_000_000,
            enable_prediction: true,
        }
    }
}

impl Default for TruthArbiterConfig {
    fn default() -> Self {
        Self {
            arbitration_strategy: ArbitrationStrategy::Bayesian,
            evidence_threshold: 0.8,
            enable_contradiction_detection: true,
            resolution_confidence_threshold: 0.9,
        }
    }
}

impl Default for ChainExecutorConfig {
    fn default() -> Self {
        Self {
            execution_strategy: ExecutionStrategy::Adaptive,
            enable_parallel: true,
            max_parallel_tasks: 10,
            dependency_resolution: DependencyResolution::Dynamic,
        }
    }
}

impl Default for MetaReasoningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            depth: 5,
            self_awareness_level: SelfAwarenessLevel::Transcendent,
            reflection_frequency_ms: 5000,
            enable_strategy_learning: true,
        }
    }
}

impl Default for TruthArbitrationConfig {
    fn default() -> Self {
        Self {
            method: ArbitrationMethod::Hybrid,
            evidence_weighting: EvidenceWeighting::Hybrid,
            contradiction_handling: ContradictionHandling::Synthesize,
            confidence_threshold: 0.9,
        }
    }
}

impl OmnisConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate reasoning configuration
        if self.reasoning.max_reasoning_depth == 0 {
            return Err("max_reasoning_depth must be > 0".to_string());
        }

        if self.reasoning.reasoning_timeout_seconds == 0 {
            return Err("reasoning_timeout_seconds must be > 0".to_string());
        }

        // Validate world model configuration
        if self.world_model.update_frequency_ms == 0 {
            return Err("update_frequency_ms must be > 0".to_string());
        }

        // Validate agent configurations
        if self.agents.oracle_7.confidence_threshold < 0.0 || self.agents.oracle_7.confidence_threshold > 1.0 {
            return Err("oracle_7 confidence_threshold must be between 0.0 and 1.0".to_string());
        }

        if self.agents.synth_prime.quality_threshold < 0.0 || self.agents.synth_prime.quality_threshold > 1.0 {
            return Err("synth_prime quality_threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate meta-reasoning configuration
        if self.meta_reasoning.depth == 0 {
            return Err("meta_reasoning depth must be > 0".to_string());
        }

        if self.meta_reasoning.reflection_frequency_ms == 0 {
            return Err("reflection_frequency_ms must be > 0".to_string());
        }

        // Validate truth arbitration configuration
        if self.truth_arbitration.confidence_threshold < 0.0 || self.truth_arbitration.confidence_threshold > 1.0 {
            return Err("truth_arbitration confidence_threshold must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }

    /// Get configuration for specific agent
    pub fn get_agent_config(&self, agent_name: &str) -> Option<serde_json::Value> {
        match agent_name {
            "oracle_7" => Some(serde_json::to_value(&self.agents.oracle_7).unwrap_or_default()),
            "synth_prime" => Some(serde_json::to_value(&self.agents.synth_prime).unwrap_or_default()),
            "meta_reasoner" => Some(serde_json::to_value(&self.agents.meta_reasoner).unwrap_or_default()),
            "world_model_x" => Some(serde_json::to_value(&self.agents.world_model_x).unwrap_or_default()),
            "truth_arbiter" => Some(serde_json::to_value(&self.agents.truth_arbiter).unwrap_or_default()),
            "chain_executor" => Some(serde_json::to_value(&self.agents.chain_executor).unwrap_or_default()),
            _ => None,
        }
    }
}
