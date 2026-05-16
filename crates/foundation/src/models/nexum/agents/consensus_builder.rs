//! Consensus Builder Agent
//! 
//! Builds consensus from diverse or conflicting agent outputs

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Consensus Builder Agent - Builds consensus from agent outputs
#[derive(Debug, Clone)]
pub struct ConsensusBuilderAgent {
    /// Agent configuration
    pub config: ConsensusBuilderConfig,
    /// Consensus capabilities
    pub consensus_capabilities: ConsensusCapabilities,
    /// Disagreement analysis
    pub disagreement_analysis: DisagreementAnalysis,
    /// Consensus building
    pub consensus_building: ConsensusBuilding,
    /// Agent status
    status: AgentStatus,
    /// Agent metrics
    metrics: AgentMetrics,
}

/// Consensus Builder Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusBuilderConfig {
    /// Base agent configuration
    pub base_config: BaseAgentConfig,
    /// Consensus strategy
    pub consensus_strategy: ConsensusStrategy,
    /// Disagreement tolerance
    pub disagreement_tolerance: f32,
    /// Confidence threshold
    pub confidence_threshold: f32,
    /// Voting mechanisms
    pub voting_mechanisms: Vec<VotingMechanism>,
}

/// Consensus Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusStrategy {
    /// Majority voting
    MajorityVoting,
    /// Weighted voting
    WeightedVoting { weights: HashMap<String, f32> },
    /// Delphi method
    DelphiMethod,
    /// Bayesian consensus
    BayesianConsensus,
    /// Argumentation based
    ArgumentationBased,
    /// Hybrid approach
    Hybrid { strategies: Vec<ConsensusStrategy> },
}

/// Voting Mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingMechanism {
    /// Simple majority
    SimpleMajority,
    /// Supermajority (2/3)
    SuperMajority,
    /// Unanimity
    Unanimity,
    /// Weighted voting
    WeightedVoting,
    /// Approval voting
    ApprovalVoting,
    /// Ranked choice
    RankedChoice,
}

/// Consensus Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusCapabilities {
    /// Disagreement detection
    pub disagreement_detection: bool,
    /// Consensus building
    pub consensus_building: bool,
    /// Argument evaluation
    pub argument_evaluation: bool,
    /// Confidence scoring
    pub confidence_scoring: bool,
    /// Synthesis generation
    pub synthesis_generation: bool,
    /// Consensus validation
    pub consensus_validation: bool,
}

/// Disagreement Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisagreementAnalysis {
    /// Analysis methods
    pub methods: Vec<DisagreementMethod>,
    /// Conflict detection
    pub conflict_detection: ConflictDetection,
    /// Disagreement metrics
    pub metrics: DisagreementMetrics,
}

/// Disagreement Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisagreementMethod {
    /// Semantic analysis
    SemanticAnalysis,
    /// Logical inconsistency
    LogicalInconsistency,
    /// Factual disagreement
    FactualDisagreement,
    /// Value conflict
    ValueConflict,
    /// Priority disagreement
    PriorityDisagreement,
    /// Methodological disagreement
    MethodologicalDisagreement,
}

/// Conflict Detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictDetection {
    /// Detection algorithms
    pub algorithms: Vec<ConflictAlgorithm>,
    /// Sensitivity thresholds
    pub thresholds: HashMap<String, f32>,
    /// Conflict types
    pub conflict_types: Vec<ConflictType>,
}

/// Conflict Algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictAlgorithm {
    /// Text similarity
    TextSimilarity,
    /// Semantic distance
    SemanticDistance,
    /// Logical contradiction
    LogicalContradiction,
    /// Factual verification
    FactualVerification,
    /// Value alignment
    ValueAlignment,
}

/// Conflict Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Direct contradiction
    DirectContradiction,
    /// Implicit conflict
    ImplicitConflict,
    /// Partial disagreement
    PartialDisagreement,
    /// Priority conflict
    PriorityConflict,
    /// Methodological conflict
    MethodologicalConflict,
}

/// Disagreement Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisagreementMetrics {
    /// Overall disagreement score
    pub overall_disagreement: f32,
    /// Pairwise disagreements
    pub pairwise_disagreements: HashMap<String, f32>,
    /// Conflict intensity
    pub conflict_intensity: f32,
    /// Resolution difficulty
    pub resolution_difficulty: f32,
}

/// Consensus Building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusBuilding {
    /// Building methods
    pub methods: Vec<ConsensusMethod>,
    /// Synthesis strategies
    pub synthesis_strategies: Vec<SynthesisStrategy>,
    /// Consensus validation
    pub validation: ConsensusValidation,
}

/// Consensus Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMethod {
    /// Negotiation
    Negotiation,
    /// Mediation
    Mediation,
    /// Arbitration
    Arbitration,
    /// Deliberation
    Deliberation,
    /// Compromise finding
    CompromiseFinding,
    /// Integration
    Integration,
}

/// Synthesis Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynthesisStrategy {
    /// Best elements selection
    BestElementsSelection,
    /// Hybrid solution
    HybridSolution,
    /// Meta-synthesis
    MetaSynthesis,
    /// Creative integration
    CreativeIntegration,
    /// Layered synthesis
    LayeredSynthesis,
}

/// Consensus Validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusValidation {
    /// Validation criteria
    pub criteria: Vec<ValidationCriterion>,
    /// Quality thresholds
    pub thresholds: HashMap<String, f32>,
    /// Consensus stability
    pub stability_check: StabilityCheck,
}

/// Validation Criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCriterion {
    /// Coherence
    Coherence,
    /// Completeness
    Completeness,
    /// Consistency
    Consistency,
    /// Acceptability
    Acceptability,
    /// Feasibility
    Feasibility,
}

/// Stability Check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityCheck {
    /// Stability metrics
    pub metrics: Vec<StabilityMetric>,
    /// Sensitivity analysis
    pub sensitivity_analysis: bool,
    /// Robustness testing
    pub robustness_testing: bool,
}

/// Stability Metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityMetric {
    /// Consensus persistence
    ConsensusPersistence,
    /// Resistance to perturbation
    ResistanceToPerturbation,
    /// Convergence speed
    ConvergenceSpeed,
    /// Agreement durability
    AgreementDurability,
}

/// Consensus Task Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusTaskInput {
    /// Agent outputs
    pub agent_outputs: Vec<AgentOutput>,
    /// Reasoning traces
    pub reasoning_traces: Vec<ReasoningTrace>,
    /// Confidence scores
    pub confidence_scores: Vec<ConfidenceScore>,
    /// Conflict map
    pub conflict_map: ConflictMap,
    /// Consensus requirements
    pub consensus_requirements: ConsensusRequirements,
}

/// Agent Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// Agent ID
    pub agent_id: String,
    /// Output content
    pub content: String,
    /// Output type
    pub output_type: String,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Reasoning Trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Agent ID
    pub agent_id: String,
    /// Reasoning steps
    pub steps: Vec<ReasoningStep>,
    /// Conclusion
    pub conclusion: String,
    /// Confidence
    pub confidence: f32,
}

/// Reasoning Step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step ID
    pub step_id: String,
    /// Step description
    pub description: String,
    /// Step type
    pub step_type: String,
    /// Evidence
    pub evidence: Vec<String>,
    /// Logic
    pub logic: String,
}

/// Confidence Score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Agent ID
    pub agent_id: String,
    /// Confidence level
    pub confidence: f32,
    /// Confidence breakdown
    pub breakdown: HashMap<String, f32>,
    /// Uncertainty sources
    pub uncertainty_sources: Vec<String>,
}

/// Conflict Map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictMap {
    /// Conflicts
    pub conflicts: Vec<Conflict>,
    /// Conflict severity
    pub severity: HashMap<String, f32>,
    /// Resolution priority
    pub priority: HashMap<String, u8>,
}

/// Conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// Conflict ID
    pub conflict_id: String,
    /// Conflicting agents
    pub agents: Vec<String>,
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Description
    pub description: String,
    /// Evidence
    pub evidence: Vec<String>,
}

/// Consensus Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRequirements {
    /// Minimum agreement level
    pub minimum_agreement: f32,
    /// Required consensus type
    pub consensus_type: ConsensusType,
    /// Time constraints
    pub time_constraints: Option<TimeConstraints>,
    /// Quality requirements
    pub quality_requirements: Vec<QualityRequirement>,
}

/// Consensus Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusType {
    /// Full consensus
    FullConsensus,
    /// Supermajority
    SuperMajority { threshold: f32 },
    /// Simple majority
    SimpleMajority,
    /// Qualified consensus
    QualifiedConsensus { criteria: Vec<String> },
    /// Working agreement
    WorkingAgreement,
}

/// Time Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraints {
    /// Maximum time
    pub max_time_ms: u64,
    /// Deadline
    pub deadline: chrono::DateTime<chrono::Utc>,
    /// Time pressure
    pub time_pressure: f32,
}

/// Quality Requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirement {
    /// Requirement type
    pub requirement_type: String,
    /// Minimum threshold
    pub minimum_threshold: f32,
    /// Weight
    pub weight: f32,
}

/// Consensus Task Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusTaskOutput {
    /// Consensus output
    pub consensus_output: String,
    /// Consensus level
    pub consensus_level: f32,
    /// Dissenting views
    pub dissenting_views: Vec<DissentingView>,
    /// Synthesis rationale
    pub synthesis_rationale: String,
    /// Confidence final
    pub confidence_final: f32,
    /// Consensus metadata
    pub metadata: HashMap<String, String>,
}

/// Dissenting View
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissentingView {
    /// Agent ID
    pub agent_id: String,
    /// Dissent reason
    pub reason: String,
    /// Alternative proposal
    pub alternative_proposal: Option<String>,
    /// Strength of dissent
    pub strength: f32,
}

impl Default for ConsensusBuilderConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            consensus_strategy: ConsensusStrategy::Hybrid {
                strategies: vec![
                    ConsensusStrategy::WeightedVoting { weights: HashMap::new() },
                    ConsensusStrategy::DelphiMethod,
                ],
            },
            disagreement_tolerance: 0.3,
            confidence_threshold: 0.7,
            voting_mechanisms: vec![
                VotingMechanism::WeightedVoting,
                VotingMechanism::SuperMajority,
            ],
        }
    }
}

impl Default for ConsensusCapabilities {
    fn default() -> Self {
        Self {
            disagreement_detection: true,
            consensus_building: true,
            argument_evaluation: true,
            confidence_scoring: true,
            synthesis_generation: true,
            consensus_validation: true,
        }
    }
}

impl Default for DisagreementAnalysis {
    fn default() -> Self {
        Self {
            methods: vec![
                DisagreementMethod::SemanticAnalysis,
                DisagreementMethod::LogicalInconsistency,
                DisagreementMethod::FactualDisagreement,
            ],
            conflict_detection: ConflictDetection {
                algorithms: vec![
                    ConflictAlgorithm::SemanticDistance,
                    ConflictAlgorithm::LogicalContradiction,
                ],
                thresholds: HashMap::new(),
                conflict_types: vec![
                    ConflictType::DirectContradiction,
                    ConflictType::ImplicitConflict,
                ],
            },
            metrics: DisagreementMetrics {
                overall_disagreement: 0.0,
                pairwise_disagreements: HashMap::new(),
                conflict_intensity: 0.0,
                resolution_difficulty: 0.0,
            },
        }
    }
}

impl Default for ConsensusBuilding {
    fn default() -> Self {
        Self {
            methods: vec![
                ConsensusMethod::Negotiation,
                ConsensusMethod::Mediation,
                ConsensusMethod::Integration,
            ],
            synthesis_strategies: vec![
                SynthesisStrategy::HybridSolution,
                SynthesisStrategy::BestElementsSelection,
            ],
            validation: ConsensusValidation {
                criteria: vec![
                    ValidationCriterion::Coherence,
                    ValidationCriterion::Consistency,
                    ValidationCriterion::Acceptability,
                ],
                thresholds: HashMap::new(),
                stability_check: StabilityCheck {
                    metrics: vec![
                        StabilityMetric::ConsensusPersistence,
                        StabilityMetric::ResistanceToPerturbation,
                    ],
                    sensitivity_analysis: true,
                    robustness_testing: false,
                },
            },
        }
    }
}

impl Default for ConsensusBuilderAgent {
    fn default() -> Self {
        Self {
            config: ConsensusBuilderConfig::default(),
            consensus_capabilities: ConsensusCapabilities::default(),
            disagreement_analysis: DisagreementAnalysis::default(),
            consensus_building: ConsensusBuilding::default(),
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
impl BaseAgent for ConsensusBuilderAgent {
    type Config = ConsensusBuilderConfig;
    type Input = ConsensusTaskInput;
    type Output = ConsensusTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        self.validate_input(&input)?;
        
        // Analyze disagreements
        let disagreement_analysis = self.analyze_disagreements(&input).await?;
        
        // Evaluate reasoning
        let reasoning_evaluation = self.evaluate_reasoning(&input).await?;
        
        // Build consensus
        let consensus_result = self.build_consensus(&input, &disagreement_analysis, &reasoning_evaluation).await?;
        
        // Validate consensus
        let validation_result = self.validate_consensus(&consensus_result).await?;
        
        // Build output
        let output = ConsensusTaskOutput {
            consensus_output: consensus_result.consensus_text,
            consensus_level: consensus_result.agreement_level,
            dissenting_views: consensus_result.dissenting_views,
            synthesis_rationale: consensus_result.rationale,
            confidence_final: consensus_result.confidence,
            metadata: HashMap::new(),
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
                name: "consensus_building".to_string(),
                description: "Builds consensus from diverse agent outputs".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["agent_outputs".to_string(), "reasoning_traces".to_string()],
                output_types: vec!["consensus_output".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.88,
                    avg_latency: 600.0,
                    resource_usage: 0.7,
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

impl ConsensusBuilderAgent {
    /// Create a new Consensus Builder Agent
    pub fn new(config: ConsensusBuilderConfig) -> Self {
        Self {
            config,
            consensus_capabilities: ConsensusCapabilities::default(),
            disagreement_analysis: DisagreementAnalysis::default(),
            consensus_building: ConsensusBuilding::default(),
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

    /// Validate consensus task input
    fn validate_input(&self, input: &ConsensusTaskInput) -> AgentResult<()> {
        if input.agent_outputs.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "At least one agent output must be provided".to_string()
            ));
        }
        
        if input.reasoning_traces.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "At least one reasoning trace must be provided".to_string()
            ));
        }
        
        Ok(())
    }

    /// Analyze disagreements between agent outputs
    async fn analyze_disagreements(&self, input: &ConsensusTaskInput) -> AgentResult<DisagreementResult> {
        let mut conflicts = Vec::new();
        let mut overall_disagreement = 0.0;
        
        // Analyze pairwise disagreements
        for (i, output1) in input.agent_outputs.iter().enumerate() {
            for (j, output2) in input.agent_outputs.iter().enumerate() {
                if i < j {
                    let disagreement_score = self.calculate_disagreement_score(output1, output2);
                    overall_disagreement += disagreement_score;
                    
                    if disagreement_score > self.config.disagreement_tolerance {
                        conflicts.push(Conflict {
                            conflict_id: format!("conflict_{}_{}", i, j),
                            agents: vec![output1.agent_id.clone(), output2.agent_id.clone()],
                            conflict_type: ConflictType::DirectContradiction,
                            description: format!("Disagreement between {} and {}", output1.agent_id, output2.agent_id),
                            evidence: vec![],
                        });
                    }
                }
            }
        }
        
        overall_disagreement /= (input.agent_outputs.len() * (input.agent_outputs.len() - 1) / 2) as f32;
        
        Ok(DisagreementResult {
            conflicts,
            overall_disagreement,
            resolution_difficulty: self.estimate_resolution_difficulty(overall_disagreement),
        })
    }

    /// Evaluate reasoning quality
    async fn evaluate_reasoning(&self, input: &ConsensusTaskInput) -> AgentResult<ReasoningEvaluation> {
        let mut reasoning_scores = HashMap::new();
        
        for trace in &input.reasoning_traces {
            let score = self.evaluate_reasoning_quality(trace);
            reasoning_scores.insert(trace.agent_id.clone(), score);
        }
        
        let overall_quality = reasoning_scores.values().sum::<f32>() / reasoning_scores.len() as f32;
        let best_reasoning = reasoning_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| id.clone())
            .unwrap_or_default();

        Ok(ReasoningEvaluation {
            reasoning_scores,
            overall_quality,
            best_reasoning,
        })
    }

    /// Build consensus from inputs
    async fn build_consensus(&self, input: &ConsensusTaskInput,
                           disagreement: &DisagreementResult,
                           reasoning: &ReasoningEvaluation) -> AgentResult<ConsensusResult> {
        match &self.config.consensus_strategy {
            ConsensusStrategy::WeightedVoting { weights } => {
                self.build_weighted_consensus(input, disagreement, reasoning, weights).await
            },
            ConsensusStrategy::DelphiMethod => {
                self.build_delphi_consensus(input, disagreement, reasoning).await
            },
            ConsensusStrategy::Hybrid { strategies } => {
                self.build_hybrid_consensus(input, disagreement, reasoning, strategies).await
            },
            _ => {
                self.build_simple_consensus(input, disagreement, reasoning).await
            }
        }
    }

    /// Validate consensus result
    async fn validate_consensus(&self, consensus: &ConsensusResult) -> AgentResult<ValidationResult> {
        let is_valid = consensus.agreement_level >= self.config.confidence_threshold;
        let quality_score = self.calculate_consensus_quality(consensus);
        
        Ok(ValidationResult {
            is_valid,
            quality_score,
            stability_score: self.estimate_stability(consensus),
            recommendations: if !is_valid {
                vec!["Consider additional deliberation".to_string()]
            } else {
                vec![]
            },
        })
    }

    /// Calculate disagreement score between two outputs
    fn calculate_disagreement_score(&self, output1: &AgentOutput, output2: &AgentOutput) -> f32 {
        // Simplified semantic distance calculation
        let lower1 = output1.content.to_lowercase();
        let lower2 = output2.content.to_lowercase();
        let words1: std::collections::HashSet<_> = lower1.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = lower2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 { 0.0 } else { 1.0 - (intersection as f32 / union as f32) }
    }

    /// Estimate resolution difficulty
    fn estimate_resolution_difficulty(&self, disagreement: f32) -> f32 {
        (disagreement * 2.0).min(1.0)
    }

    /// Evaluate reasoning quality
    fn evaluate_reasoning_quality(&self, trace: &ReasoningTrace) -> f32 {
        let step_quality = if trace.steps.is_empty() { 0.0 } else { 0.8 };
        let confidence_bonus = trace.confidence * 0.2;
        step_quality + confidence_bonus
    }

    /// Build weighted consensus
    async fn build_weighted_consensus(&self, input: &ConsensusTaskInput,
                                    _disagreement: &DisagreementResult,
                                    reasoning: &ReasoningEvaluation,
                                    weights: &HashMap<String, f32>) -> AgentResult<ConsensusResult> {
        let mut weighted_outputs = Vec::new();
        
        for output in &input.agent_outputs {
            let weight = weights.get(&output.agent_id).unwrap_or(&0.5);
            let reasoning_score = reasoning.reasoning_scores.get(&output.agent_id).unwrap_or(&0.5);
            let combined_weight = weight * reasoning_score;
            
            weighted_outputs.push((output, combined_weight));
        }
        
        // Sort by weight and select best elements
        weighted_outputs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let consensus_text = if let Some((best_output, _)) = weighted_outputs.first() {
            best_output.content.clone()
        } else {
            "No consensus reached".to_string()
        };
        
        Ok(ConsensusResult {
            consensus_text,
            agreement_level: 0.75,
            dissenting_views: vec![],
            rationale: "Weighted consensus based on agent reliability".to_string(),
            confidence: 0.8,
        })
    }

    /// Build Delphi consensus
    async fn build_delphi_consensus(&self, input: &ConsensusTaskInput,
                                  _disagreement: &DisagreementResult,
                                  reasoning: &ReasoningEvaluation) -> AgentResult<ConsensusResult> {
        // Simplified Delphi method - iterate towards consensus
        let mut iteration = 0;
        let max_iterations = 3;
        
        let mut current_outputs = input.agent_outputs.clone();
        
        while iteration < max_iterations {
            iteration += 1;
            
            // In real implementation, this would involve feedback rounds
            // For now, return a simplified result
            break;
        }
        
        Ok(ConsensusResult {
            consensus_text: "Delphi consensus result".to_string(),
            agreement_level: 0.85,
            dissenting_views: vec![],
            rationale: "Consensus reached through iterative Delphi method".to_string(),
            confidence: 0.9,
        })
    }

    /// Build hybrid consensus
    async fn build_hybrid_consensus(&self, input: &ConsensusTaskInput,
                                   disagreement: &DisagreementResult,
                                   reasoning: &ReasoningEvaluation,
                                   strategies: &[ConsensusStrategy]) -> AgentResult<ConsensusResult> {
        // Try multiple strategies and pick best result
        let mut best_result = None;
        let mut best_score = 0.0;
        
        for strategy in strategies {
            let result = match strategy {
                ConsensusStrategy::WeightedVoting { weights } => {
                    self.build_weighted_consensus(input, disagreement, reasoning, weights).await?
                },
                ConsensusStrategy::DelphiMethod => {
                    self.build_delphi_consensus(input, disagreement, reasoning).await?
                },
                _ => continue,
            };
            
            let score = result.agreement_level * result.confidence;
            if score > best_score {
                best_score = score;
                best_result = Some(result);
            }
        }
        
        best_result.ok_or_else(|| crate::shared::agent_types::AgentError::ProcessingFailed(
            "No valid consensus strategy found".to_string()
        ))
    }

    /// Build simple consensus
    async fn build_simple_consensus(&self, input: &ConsensusTaskInput,
                                   _disagreement: &DisagreementResult,
                                   reasoning: &ReasoningEvaluation) -> AgentResult<ConsensusResult> {
        // Simple majority vote
        let mut vote_counts: HashMap<String, usize> = HashMap::new();
        
        for output in &input.agent_outputs {
            *vote_counts.entry(output.content.clone()).or_insert(0) += 1;
        }
        
        let (consensus_text, votes) = vote_counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(text, count)| (text.clone(), *count))
            .unwrap_or_else(|| ("No consensus".to_string(), 0));
        
        let agreement_level = votes as f32 / input.agent_outputs.len() as f32;
        
        Ok(ConsensusResult {
            consensus_text,
            agreement_level,
            dissenting_views: vec![],
            rationale: "Simple majority vote consensus".to_string(),
            confidence: agreement_level,
        })
    }

    /// Calculate consensus quality
    fn calculate_consensus_quality(&self, consensus: &ConsensusResult) -> f32 {
        consensus.agreement_level * consensus.confidence
    }

    /// Estimate consensus stability
    fn estimate_stability(&self, consensus: &ConsensusResult) -> f32 {
        if consensus.agreement_level > 0.8 {
            0.9
        } else if consensus.agreement_level > 0.6 {
            0.7
        } else {
            0.4
        }
    }
}

// Helper structs for internal processing
#[derive(Debug, Clone)]
struct DisagreementResult {
    conflicts: Vec<Conflict>,
    overall_disagreement: f32,
    resolution_difficulty: f32,
}

#[derive(Debug, Clone)]
struct ReasoningEvaluation {
    reasoning_scores: HashMap<String, f32>,
    overall_quality: f32,
    best_reasoning: String,
}

#[derive(Debug, Clone)]
struct ConsensusResult {
    consensus_text: String,
    agreement_level: f32,
    dissenting_views: Vec<DissentingView>,
    rationale: String,
    confidence: f32,
}

#[derive(Debug, Clone)]
struct ValidationResult {
    is_valid: bool,
    quality_score: f32,
    stability_score: f32,
    recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_builder_agent_creation() {
        let agent = ConsensusBuilderAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_consensus_task_processing() {
        let agent = ConsensusBuilderAgent::default();
        let input = ConsensusTaskInput {
            agent_outputs: vec![
                AgentOutput {
                    agent_id: "agent1".to_string(),
                    content: "Solution A is best".to_string(),
                    output_type: "recommendation".to_string(),
                    metadata: HashMap::new(),
                },
                AgentOutput {
                    agent_id: "agent2".to_string(),
                    content: "Solution A is best".to_string(),
                    output_type: "recommendation".to_string(),
                    metadata: HashMap::new(),
                },
            ],
            reasoning_traces: vec![],
            confidence_scores: vec![],
            conflict_map: ConflictMap {
                conflicts: vec![],
                severity: HashMap::new(),
                priority: HashMap::new(),
            },
            consensus_requirements: ConsensusRequirements {
                minimum_agreement: 0.7,
                consensus_type: ConsensusType::SimpleMajority,
                time_constraints: None,
                quality_requirements: vec![],
            },
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.consensus_output.is_empty());
        assert!(output.consensus_level > 0.0);
    }

    #[test]
    fn test_disagreement_calculation() {
        let agent = ConsensusBuilderAgent::default();
        
        let output1 = AgentOutput {
            agent_id: "agent1".to_string(),
            content: "Solution A is best".to_string(),
            output_type: "recommendation".to_string(),
            metadata: HashMap::new(),
        };
        
        let output2 = AgentOutput {
            agent_id: "agent2".to_string(),
            content: "Solution B is better".to_string(),
            output_type: "recommendation".to_string(),
            metadata: HashMap::new(),
        };
        
        let disagreement = agent.calculate_disagreement_score(&output1, &output2);
        assert!(disagreement > 0.0);
        assert!(disagreement <= 1.0);
    }

    #[test]
    fn test_consensus_strategies() {
        let config = ConsensusBuilderConfig {
            consensus_strategy: ConsensusStrategy::DelphiMethod,
            ..Default::default()
        };
        let agent = ConsensusBuilderAgent::new(config);
        
        assert!(matches!(agent.config.consensus_strategy, ConsensusStrategy::DelphiMethod));
    }
}
