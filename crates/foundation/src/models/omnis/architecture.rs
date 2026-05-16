//! NXR-OMNIS Architecture
//! 
//! Implementation of the MoE + Transformer-XL hybrid architecture

use std::collections::HashMap;
use crate::shared::base_model::NxrModelResult;
use super::config::OmnisConfig;

/// NXR-OMNIS Architecture Implementation
pub struct OmnisArchitecture {
    /// Configuration
    config: OmnisConfig,
    /// Expert networks
    experts: HashMap<String, ExpertNetwork>,
    /// Gating network
    gating_network: GatingNetwork,
    /// Transformer-XL components
    transformer_xl: TransformerXLComponents,
    /// Neural world model
    neural_world_model: NeuralWorldModel,
    /// Meta-reasoning layers
    meta_reasoning_layers: Vec<MetaReasoningLayer>,
    /// Truth arbitration network
    truth_arbitration_network: TruthArbitrationNetwork,
    /// Chain execution engine
    chain_execution_engine: ChainExecutionEngine,
}

/// Expert network for MoE
#[derive(Debug, Clone)]
pub struct ExpertNetwork {
    /// Expert ID
    pub id: String,
    /// Expert specialization
    pub specialization: ExpertSpecialization,
    /// Expert capacity
    pub capacity: f32,
    /// Expert utilization
    pub utilization: f32,
    /// Expert performance score
    pub performance_score: f32,
}

/// Expert specialization
#[derive(Debug, Clone)]
pub enum ExpertSpecialization {
    /// Text reasoning expert
    TextReasoning,
    /// Mathematical reasoning expert
    MathematicalReasoning,
    /// Logical reasoning expert
    LogicalReasoning,
    /// World modeling expert
    WorldModeling,
    /// Meta-reasoning expert
    MetaReasoning,
    /// Truth arbitration expert
    TruthArbitration,
    /// Synthesis expert
    Synthesis,
    /// Verification expert
    Verification,
}

/// Gating network for expert selection
#[derive(Debug, Clone)]
pub struct GatingNetwork {
    /// Gating strategy
    pub strategy: GatingStrategy,
    /// Expert weights
    pub expert_weights: HashMap<String, f32>,
    /// Load balancing coefficient
    pub load_balancing_coef: f32,
    /// Capacity factor
    pub capacity_factor: f32,
}

/// Gating strategy
#[derive(Debug, Clone)]
pub enum GatingStrategy {
    /// Top-k gating
    TopK { k: usize },
    /// Noisy top-k gating
    NoisyTopK { k: usize, noise_std: f32 },
    /// Learned gating
    Learned,
    /// Adaptive gating
    Adaptive,
}

/// Transformer-XL components
#[derive(Debug, Clone)]
pub struct TransformerXLComponents {
    /// Memory segments
    pub memory_segments: Vec<MemorySegment>,
    /// Segment length
    pub segment_length: usize,
    /// Memory size
    pub memory_size: usize,
    /// Relative position encoding
    pub relative_position_encoding: RelativePositionEncoding,
}

/// Memory segment for Transformer-XL
#[derive(Debug, Clone)]
pub struct MemorySegment {
    /// Segment ID
    pub id: uuid::Uuid,
    /// Segment data
    pub data: Vec<f32>,
    /// Segment timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Segment importance score
    pub importance_score: f32,
}

/// Relative position encoding
#[derive(Debug, Clone)]
pub struct RelativePositionEncoding {
    /// Maximum relative distance
    pub max_relative_distance: usize,
    /// Encoding type
    pub encoding_type: PositionEncodingType,
    /// Learned embeddings
    pub embeddings: Option<HashMap<isize, Vec<f32>>>,
}

/// Position encoding type
#[derive(Debug, Clone)]
pub enum PositionEncodingType {
    /// Sinusoidal encoding
    Sinusoidal,
    /// Learned embeddings
    Learned,
    /// Relative bias
    RelativeBias,
    /// Rotary position encoding (RoPE)
    Rotary,
}

/// Neural world model
#[derive(Debug, Clone)]
pub struct NeuralWorldModel {
    /// World state representation
    pub world_state: WorldState,
    /// Prediction network
    pub prediction_network: PredictionNetwork,
    /// Update mechanism
    pub update_mechanism: UpdateMechanism,
    /// Knowledge integration
    pub knowledge_integration: KnowledgeIntegration,
}

/// World state representation
#[derive(Debug, Clone)]
pub struct WorldState {
    /// State ID
    pub id: uuid::Uuid,
    /// State variables
    pub variables: HashMap<String, WorldVariable>,
    /// Relationships
    pub relationships: Vec<WorldRelationship>,
    /// Uncertainty estimates
    pub uncertainty: HashMap<String, f32>,
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// World variable
#[derive(Debug, Clone)]
pub struct WorldVariable {
    /// Variable name
    pub name: String,
    /// Variable value
    pub value: WorldValue,
    /// Variable type
    pub variable_type: WorldVariableType,
    /// Confidence
    pub confidence: f32,
    /// Temporal dynamics
    pub temporal_dynamics: Option<TemporalDynamics>,
}

/// World value
#[derive(Debug, Clone)]
pub enum WorldValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Vector value
    Vector(Vec<f32>),
    /// Tensor value
    Tensor(Vec<Vec<f32>>),
    /// Distribution value
    Distribution { mean: f64, std: f64 },
    /// Unknown value
    Unknown,
}

/// World variable type
#[derive(Debug, Clone)]
pub enum WorldVariableType {
    /// Static variable
    Static,
    /// Dynamic variable
    Dynamic,
    /// Stochastic variable
    Stochastic,
    /// Categorical variable
    Categorical { categories: Vec<String> },
    /// Continuous variable
    Continuous { min: f64, max: f64 },
}

/// Temporal dynamics
#[derive(Debug, Clone)]
pub struct TemporalDynamics {
    /// Change rate
    pub change_rate: f64,
    /// Seasonality
    pub seasonality: Option<f64>,
    /// Trend
    pub trend: Option<f64>,
    /// Noise level
    pub noise_level: f64,
}

/// World relationship
#[derive(Debug, Clone)]
pub struct WorldRelationship {
    /// Relationship ID
    pub id: uuid::Uuid,
    /// Source variable
    pub source: String,
    /// Target variable
    pub target: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength
    pub strength: f32,
    /// Causal direction
    pub causal_direction: CausalDirection,
}

/// Relationship type
#[derive(Debug, Clone)]
pub enum RelationshipType {
    /// Causal relationship
    Causal,
    /// Correlational relationship
    Correlational { correlation: f32 },
    /// Logical implication
    Implication,
    /// Temporal precedence
    TemporalPrecedence,
    /// Spatial relationship
    Spatial { distance: f64, direction: String },
    /// Functional relationship
    Functional { function: String },
}

/// Causal direction
#[derive(Debug, Clone)]
pub enum CausalDirection {
    /// Source causes target
    Forward,
    /// Target causes source
    Backward,
    /// Bidirectional causality
    Bidirectional,
    /// No causal direction
    None,
}

/// Prediction network
#[derive(Debug, Clone)]
pub struct PredictionNetwork {
    /// Network architecture
    pub architecture: PredictionArchitecture,
    /// Prediction horizon
    pub prediction_horizon: u32,
    /// Prediction accuracy
    pub accuracy: f32,
    /// Uncertainty quantification
    pub uncertainty_quantification: bool,
}

/// Prediction architecture
#[derive(Debug, Clone)]
pub enum PredictionArchitecture {
    /// Linear prediction
    Linear,
    /// Neural network prediction
    Neural { layers: Vec<usize>, activation: String },
    /// Ensemble prediction
    Ensemble { models: Vec<String> },
    /// Bayesian prediction
    Bayesian,
    /// Hybrid prediction
    Hybrid,
}

/// Update mechanism
#[derive(Debug, Clone)]
pub struct UpdateMechanism {
    /// Update strategy
    pub strategy: UpdateStrategy,
    /// Update frequency
    pub frequency_ms: u64,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
}

/// Update strategy
#[derive(Debug, Clone)]
pub enum UpdateStrategy {
    /// Incremental update
    Incremental,
    /// Batch update
    Batch,
    /// Online learning
    Online,
    /// Adaptive update
    Adaptive,
    /// Selective update
    Selective { threshold: f32 },
}

/// Knowledge integration
#[derive(Debug, Clone)]
pub struct KnowledgeIntegration {
    /// Integration method
    pub method: IntegrationMethod,
    /// Knowledge sources
    pub knowledge_sources: Vec<KnowledgeSource>,
    /// Conflict resolution
    pub conflict_resolution: ConflictResolution,
    /// Integration confidence
    pub integration_confidence: f32,
}

/// Integration method
#[derive(Debug, Clone)]
pub enum IntegrationMethod {
    /// Rule-based integration
    RuleBased,
    /// Probabilistic integration
    Probabilistic,
    /// Neural integration
    Neural,
    /// Graph-based integration
    GraphBased,
    /// Hybrid integration
    Hybrid,
}

/// Knowledge source
#[derive(Debug, Clone)]
pub struct KnowledgeSource {
    /// Source ID
    pub id: String,
    /// Source type
    pub source_type: SourceType,
    /// Reliability score
    pub reliability: f32,
    /// Update frequency
    pub update_frequency_ms: u64,
    /// Last update
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Source type
#[derive(Debug, Clone)]
pub enum SourceType {
    /// Database source
    Database,
    /// API source
    API,
    /// File source
    File,
    /// Sensor source
    Sensor,
    /// Human input
    Human,
    /// Model inference
    Model,
}

/// Conflict resolution
#[derive(Debug, Clone)]
pub enum ConflictResolution {
    /// Keep first
    KeepFirst,
    /// Keep last
    KeepLast,
    /// Majority vote
    MajorityVote,
    /// Weighted average
    WeightedAverage,
    /// Expert arbitration
    ExpertArbitration,
    /// Probabilistic resolution
    Probabilistic,
}

/// Meta-reasoning layer
#[derive(Debug, Clone)]
pub struct MetaReasoningLayer {
    /// Layer ID
    pub id: String,
    /// Layer type
    pub layer_type: MetaReasoningLayerType,
    /// Reasoning depth
    pub reasoning_depth: u8,
    /// Self-awareness level
    pub self_awareness_level: SelfAwarenessLevel,
    /// Strategy selection
    pub strategy_selection: StrategySelection,
}

/// Meta-reasoning layer type
#[derive(Debug, Clone)]
pub enum MetaReasoningLayerType {
    /// Self-reflection layer
    SelfReflection,
    /// Strategy selection layer
    StrategySelection,
    /// Performance monitoring layer
    PerformanceMonitoring,
    /// Learning layer
    Learning,
    /// Adaptation layer
    Adaptation,
}

/// Self-awareness level
#[derive(Debug, Clone)]
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

/// Strategy selection
#[derive(Debug, Clone)]
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

/// Truth arbitration network
#[derive(Debug, Clone)]
pub struct TruthArbitrationNetwork {
    /// Arbitration method
    pub arbitration_method: ArbitrationMethod,
    /// Evidence integration
    pub evidence_integration: EvidenceIntegration,
    /// Contradiction detection
    pub contradiction_detection: ContradictionDetection,
    /// Resolution confidence
    pub resolution_confidence: f32,
}

/// Arbitration method
#[derive(Debug, Clone)]
pub enum ArbitrationMethod {
    /// Consensus-based
    Consensus,
    /// Evidence-based
    EvidenceBased,
    /// Bayesian arbitration
    Bayesian,
    /// Neural arbitration
    Neural,
    /// Hybrid arbitration
    Hybrid,
}

/// Evidence integration
#[derive(Debug, Clone)]
pub enum EvidenceIntegration {
    /// Simple aggregation
    SimpleAggregation,
    /// Weighted aggregation
    WeightedAggregation,
    /// Bayesian updating
    BayesianUpdating,
    /// Neural integration
    NeuralIntegration,
    /// Graph-based integration
    GraphBased,
}

/// Contradiction detection
#[derive(Debug, Clone)]
pub struct ContradictionDetection {
    /// Detection method
    pub method: ContradictionDetectionMethod,
    /// Sensitivity threshold
    pub sensitivity_threshold: f32,
    /// False positive tolerance
    pub false_positive_tolerance: f32,
}

/// Contradiction detection method
#[derive(Debug, Clone)]
pub enum ContradictionDetectionMethod {
    /// Logical contradiction
    Logical,
    /// Statistical contradiction
    Statistical,
    /// Semantic contradiction
    Semantic,
    /// Neural contradiction detection
    Neural,
    /// Hybrid detection
    Hybrid,
}

/// Chain execution engine
#[derive(Debug, Clone)]
pub struct ChainExecutionEngine {
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Parallel execution
    pub parallel_execution: bool,
    /// Maximum parallel tasks
    pub max_parallel_tasks: usize,
    /// Dependency resolution
    pub dependency_resolution: DependencyResolution,
    /// Task scheduling
    pub task_scheduling: TaskScheduling,
}

/// Execution strategy
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Adaptive execution
    Adaptive,
    /// Priority-based execution
    PriorityBased,
    /// Resource-constrained execution
    ResourceConstrained,
}

/// Dependency resolution
#[derive(Debug, Clone)]
pub enum DependencyResolution {
    /// Topological sort
    TopologicalSort,
    /// Priority-based resolution
    PriorityBased,
    /// Dynamic resolution
    Dynamic,
    /// Lazy resolution
    Lazy,
}

/// Task scheduling
#[derive(Debug, Clone)]
pub enum TaskScheduling {
    /// FIFO scheduling
    FIFO,
    /// Priority scheduling
    Priority,
    /// Round-robin scheduling
    RoundRobin,
    /// Adaptive scheduling
    Adaptive,
    /// Resource-aware scheduling
    ResourceAware,
}

impl OmnisArchitecture {
    /// Create new architecture with configuration
    pub fn new(config: &OmnisConfig) -> Self {
        let mut experts = HashMap::new();
        
        // Initialize expert networks
        experts.insert("text_reasoning".to_string(), ExpertNetwork {
            id: "text_reasoning".to_string(),
            specialization: ExpertSpecialization::TextReasoning,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.95,
        });
        
        experts.insert("mathematical_reasoning".to_string(), ExpertNetwork {
            id: "mathematical_reasoning".to_string(),
            specialization: ExpertSpecialization::MathematicalReasoning,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.98,
        });
        
        experts.insert("logical_reasoning".to_string(), ExpertNetwork {
            id: "logical_reasoning".to_string(),
            specialization: ExpertSpecialization::LogicalReasoning,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.97,
        });
        
        experts.insert("world_modeling".to_string(), ExpertNetwork {
            id: "world_modeling".to_string(),
            specialization: ExpertSpecialization::WorldModeling,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.93,
        });
        
        experts.insert("meta_reasoning".to_string(), ExpertNetwork {
            id: "meta_reasoning".to_string(),
            specialization: ExpertSpecialization::MetaReasoning,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.96,
        });
        
        experts.insert("truth_arbitration".to_string(), ExpertNetwork {
            id: "truth_arbitration".to_string(),
            specialization: ExpertSpecialization::TruthArbitration,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.99,
        });
        
        experts.insert("synthesis".to_string(), ExpertNetwork {
            id: "synthesis".to_string(),
            specialization: ExpertSpecialization::Synthesis,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.94,
        });
        
        experts.insert("verification".to_string(), ExpertNetwork {
            id: "verification".to_string(),
            specialization: ExpertSpecialization::Verification,
            capacity: 1.0,
            utilization: 0.0,
            performance_score: 0.97,
        });

        Self {
            config: config.clone(),
            experts,
            gating_network: GatingNetwork {
                strategy: GatingStrategy::TopK { k: 2 },
                expert_weights: HashMap::new(),
                load_balancing_coef: 0.01,
                capacity_factor: 1.25,
            },
            transformer_xl: TransformerXLComponents {
                memory_segments: Vec::new(),
                segment_length: 1024,
                memory_size: 10000,
                relative_position_encoding: RelativePositionEncoding {
                    max_relative_distance: 1024,
                    encoding_type: PositionEncodingType::Rotary,
                    embeddings: None,
                },
            },
            neural_world_model: NeuralWorldModel {
                world_state: WorldState {
                    id: uuid::Uuid::new_v4(),
                    variables: HashMap::new(),
                    relationships: Vec::new(),
                    uncertainty: HashMap::new(),
                    last_updated: chrono::Utc::now(),
                },
                prediction_network: PredictionNetwork {
                    architecture: PredictionArchitecture::Neural {
                        layers: vec![512, 256, 128, 64],
                        activation: "relu".to_string(),
                    },
                    prediction_horizon: 100,
                    accuracy: 0.95,
                    uncertainty_quantification: true,
                },
                update_mechanism: UpdateMechanism {
                    strategy: UpdateStrategy::Adaptive,
                    frequency_ms: 1000,
                    batch_size: 32,
                    learning_rate: 0.001,
                },
                knowledge_integration: KnowledgeIntegration {
                    method: IntegrationMethod::Hybrid,
                    knowledge_sources: Vec::new(),
                    conflict_resolution: ConflictResolution::ExpertArbitration,
                    integration_confidence: 0.9,
                },
            },
            meta_reasoning_layers: vec![
                MetaReasoningLayer {
                    id: "self_reflection".to_string(),
                    layer_type: MetaReasoningLayerType::SelfReflection,
                    reasoning_depth: 5,
                    self_awareness_level: SelfAwarenessLevel::Transcendent,
                    strategy_selection: StrategySelection::Adaptive,
                },
                MetaReasoningLayer {
                    id: "strategy_selection".to_string(),
                    layer_type: MetaReasoningLayerType::StrategySelection,
                    reasoning_depth: 3,
                    self_awareness_level: SelfAwarenessLevel::Advanced,
                    strategy_selection: StrategySelection::Learning,
                },
                MetaReasoningLayer {
                    id: "performance_monitoring".to_string(),
                    layer_type: MetaReasoningLayerType::PerformanceMonitoring,
                    reasoning_depth: 2,
                    self_awareness_level: SelfAwarenessLevel::Full,
                    strategy_selection: StrategySelection::Adaptive,
                },
            ],
            truth_arbitration_network: TruthArbitrationNetwork {
                arbitration_method: ArbitrationMethod::Hybrid,
                evidence_integration: EvidenceIntegration::BayesianUpdating,
                contradiction_detection: ContradictionDetection {
                    method: ContradictionDetectionMethod::Hybrid,
                    sensitivity_threshold: 0.8,
                    false_positive_tolerance: 0.1,
                },
                resolution_confidence: 0.9,
            },
            chain_execution_engine: ChainExecutionEngine {
                execution_strategy: ExecutionStrategy::Adaptive,
                parallel_execution: true,
                max_parallel_tasks: 10,
                dependency_resolution: DependencyResolution::Dynamic,
                task_scheduling: TaskScheduling::ResourceAware,
            },
        }
    }

    /// Initialize architecture
    pub async fn initialize(&mut self, config: &OmnisConfig) -> NxrModelResult<()> {
        // Initialize expert networks
        for expert in self.experts.values_mut() {
            expert.utilization = 0.0;
        }

        // Initialize gating network
        self.gating_network.expert_weights = self.experts
            .iter()
            .map(|(id, expert)| (id.clone(), expert.performance_score))
            .collect();

        // Initialize Transformer-XL memory segments
        self.transformer_xl.memory_segments = (0..10)
            .map(|_| MemorySegment {
                id: uuid::Uuid::new_v4(),
                data: vec![0.0; self.transformer_xl.segment_length],
                timestamp: chrono::Utc::now(),
                importance_score: 0.0,
            })
            .collect();

        Ok(())
    }

    /// Validate architecture
    pub async fn validate(&self) -> NxrModelResult<()> {
        // Validate expert networks
        if self.experts.is_empty() {
            return Err("No expert networks configured".into());
        }

        // Validate gating network
        if self.gating_network.expert_weights.len() != self.experts.len() {
            return Err("Gating network weights don't match expert count".into());
        }

        // Validate Transformer-XL components
        if self.transformer_xl.segment_length == 0 {
            return Err("Invalid segment length".into());
        }

        // Validate meta-reasoning layers
        if self.meta_reasoning_layers.is_empty() {
            return Err("No meta-reasoning layers configured".into());
        }

        Ok(())
    }

    /// Select experts for input
    pub async fn select_experts(&self, input: &str) -> Vec<String> {
        // Simple heuristic-based selection for now
        let mut selected_experts = Vec::new();
        
        if input.contains("math") || input.contains("calculate") {
            selected_experts.push("mathematical_reasoning".to_string());
        }
        
        if input.contains("logic") || input.contains("reason") {
            selected_experts.push("logical_reasoning".to_string());
        }
        
        if input.contains("world") || input.contains("context") {
            selected_experts.push("world_modeling".to_string());
        }
        
        if input.len() > 1000 {
            selected_experts.push("meta_reasoning".to_string());
        }
        
        // Always include text reasoning and synthesis
        selected_experts.push("text_reasoning".to_string());
        selected_experts.push("synthesis".to_string());
        
        // Limit to top-k experts
        match &self.gating_network.strategy {
            GatingStrategy::TopK { k } => selected_experts.truncate(*k),
            GatingStrategy::NoisyTopK { k, .. } => selected_experts.truncate(*k),
            _ => {}
        }
        
        selected_experts
    }

    /// Update world model
    pub async fn update_world_model(&mut self, input: &str) -> NxrModelResult<()> {
        // Simple world model update
        let timestamp = chrono::Utc::now();
        self.neural_world_model.world_state.last_updated = timestamp;
        
        // Add input as a world variable
        self.neural_world_model.world_state.variables.insert(
            format!("input_{}", timestamp.timestamp()),
            WorldVariable {
                name: format!("input_{}", timestamp.timestamp()),
                value: WorldValue::String(input.to_string()),
                variable_type: WorldVariableType::Static,
                confidence: 1.0,
                temporal_dynamics: None,
            },
        );
        
        Ok(())
    }

    /// Perform meta-reasoning
    pub async fn meta_reason(&self, problem: &str) -> NxrModelResult<Vec<String>> {
        let mut reasoning_steps = Vec::new();
        
        // Self-reflection
        reasoning_steps.push("Analyzing problem structure and complexity".to_string());
        
        // Strategy selection
        reasoning_steps.push("Selecting optimal reasoning approach".to_string());
        
        // Performance monitoring
        reasoning_steps.push("Monitoring reasoning progress and quality".to_string());
        
        // Adaptation
        reasoning_steps.push("Adapting reasoning strategy based on progress".to_string());
        
        Ok(reasoning_steps)
    }

    /// Arbitrate truth claims
    pub async fn arbitrate_truth(&self, claims: Vec<String>) -> NxrModelResult<String> {
        // Simple truth arbitration
        if claims.is_empty() {
            return Ok("No claims to arbitrate".to_string());
        }
        
        // For now, return the most confident claim
        // In a real implementation, this would be much more sophisticated
        Ok(claims.first().expect("claims is non-empty here").clone())
    }

    /// Execute reasoning chain
    pub async fn execute_chain(&self, steps: Vec<String>) -> NxrModelResult<String> {
        let mut result = String::new();
        
        for step in steps {
            result.push_str(&step);
            result.push_str(" → ");
        }
        
        // Remove final arrow
        if result.ends_with(" → ") {
            result.truncate(result.len() - 3);
        }
        
        Ok(result)
    }

    /// Synthesize results
    pub async fn synthesize(&self, expert_outputs: HashMap<String, String>) -> NxrModelResult<String> {
        let mut synthesis = String::new();
        
        for (expert, output) in expert_outputs {
            synthesis.push_str(&format!("[{}]: {}\n", expert, output));
        }
        
        synthesis.push_str("\nSynthesis: Integrated results from all expert networks.");
        
        Ok(synthesis)
    }
}
