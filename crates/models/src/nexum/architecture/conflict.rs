//! Conflict resolution system and detection mechanisms

use std::collections::HashMap;

/// Conflict Resolution System
#[derive(Debug, Clone)]
pub struct ConflictResolutionSystem {
    /// Resolution strategy
    pub strategy: ConflictResolutionStrategy,
    /// Conflict detector
    pub conflict_detector: ConflictDetector,
    /// Resolution engine
    pub resolution_engine: ResolutionEngine,
    /// Resolution metrics
    pub metrics: ConflictResolutionMetrics,
}

/// Conflict Resolution Strategy
#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    /// Negotiation
    Negotiation,
    /// Arbitration
    Arbitration,
    /// Mediation
    Mediation,
    /// Consensus building
    ConsensusBuilding,
    /// Compromise
    Compromise,
    /// Escalation
    Escalation,
}

/// Conflict Detector
#[derive(Debug, Clone)]
pub struct ConflictDetector {
    /// Detection algorithm
    pub algorithm: ConflictDetectionAlgorithm,
    /// Sensitivity level
    pub sensitivity: f32,
    /// Detection history
    pub detection_history: Vec<ConflictDetectionRecord>,
    /// Detector metrics
    pub metrics: ConflictDetectorMetrics,
}

/// Conflict Detection Algorithm
#[derive(Debug, Clone)]
pub enum ConflictDetectionAlgorithm {
    /// Pattern-based detection
    PatternBased,
    /// Rule-based detection
    RuleBased,
    /// Machine learning detection
    MachineLearning,
    /// Statistical detection
    Statistical,
    /// Hybrid detection
    Hybrid,
}

/// Conflict Detection Record
#[derive(Debug, Clone)]
pub struct ConflictDetectionRecord {
    /// Detection ID
    pub id: String,
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Conflict parties
    pub parties: Vec<String>,
    /// Conflict severity
    pub severity: ConflictSeverity,
    /// Detection time
    pub detected_at: chrono::DateTime<chrono::Utc>,
    /// Detection confidence
    pub confidence: f32,
}

/// Conflict Type
#[derive(Debug, Clone)]
pub enum ConflictType {
    /// Resource conflict
    Resource,
    /// Priority conflict
    Priority,
    /// Goal conflict
    Goal,
    /// Communication conflict
    Communication,
    /// Coordination conflict
    Coordination,
    /// Value conflict
    Value,
}

/// Conflict Severity
#[derive(Debug, Clone)]
pub enum ConflictSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Conflict Detector Metrics
#[derive(Debug, Clone)]
pub struct ConflictDetectorMetrics {
    /// Detection accuracy
    pub detection_accuracy: f32,
    /// False positive rate
    pub false_positive_rate: f32,
    /// Detection speed
    pub detection_speed_ms: f64,
    /// Detection coverage
    pub detection_coverage: f32,
}

/// Resolution Engine
#[derive(Debug, Clone)]
pub struct ResolutionEngine {
    /// Resolution methods
    pub methods: Vec<ResolutionMethod>,
    /// Resolution strategies
    pub strategies: Vec<ResolutionStrategyType>,
    /// Resolution history
    pub resolution_history: Vec<ConflictResolutionRecord>,
    /// Engine metrics
    pub metrics: ResolutionEngineMetrics,
}

/// Resolution Method
#[derive(Debug, Clone)]
pub enum ResolutionMethod {
    /// Negotiation method
    Negotiation,
    /// Arbitration method
    Arbitration,
    /// Mediation method
    Mediation,
    /// Consensus method
    Consensus,
    /// Compromise method
    Compromise,
    /// Voting method
    Voting,
}

/// Conflict Resolution Record
#[derive(Debug, Clone)]
pub struct ConflictResolutionRecord {
    /// Resolution ID
    pub id: String,
    /// Conflict ID
    pub conflict_id: String,
    /// Resolution method
    pub method: ResolutionMethod,
    /// Resolution outcome
    pub outcome: ResolutionStatus,
    /// Resolution time
    pub resolved_at: chrono::DateTime<chrono::Utc>,
    /// Resolution duration
    pub resolution_duration_ms: u64,
}

/// Resolution Status
#[derive(Debug, Clone)]
pub enum ResolutionStatus {
    /// Resolved
    Resolved,
    /// Partially resolved
    PartiallyResolved,
    /// Escalated
    Escalated,
    /// Unresolved
    Unresolved,
}

/// Resolution Engine Metrics
#[derive(Debug, Clone)]
pub struct ResolutionEngineMetrics {
    /// Resolution success rate
    pub success_rate: f32,
    /// Average resolution time
    pub avg_resolution_time_ms: f64,
    /// Resolution quality
    pub resolution_quality: f32,
    /// Escalation rate
    pub escalation_rate: f32,
}

/// Conflict Resolution Metrics
#[derive(Debug, Clone)]
pub struct ConflictResolutionMetrics {
    /// Resolution success rate
    pub resolution_success_rate: f32,
    /// Average resolution time
    pub avg_resolution_time_ms: f64,
    /// Conflict prevention rate
    pub prevention_rate: f32,
    /// Resolution satisfaction
    pub resolution_satisfaction: f32,
}

/// Conflict
#[derive(Debug, Clone)]
pub struct Conflict {
    pub id: String,
    pub description: String,
    pub parties: Vec<String>,
    pub conflict_time: chrono::DateTime<chrono::Utc>,
}

/// Conflict Detection
#[derive(Debug, Clone)]
pub struct ConflictDetection {
    pub conflict_type: ConflictType,
    pub severity: ConflictSeverity,
    pub confidence: f32,
}

/// Conflict Resolution
#[derive(Debug, Clone)]
pub struct ConflictResolution {
    pub conflict_id: String,
    pub conflict_type: ConflictType,
    pub conflict_severity: ConflictSeverity,
    pub resolution_method: ResolutionMethod,
    pub resolution_outcome: ResolutionOutcome,
    pub resolution_time_ms: u64,
    pub resolution_quality: f32,
}

impl ConflictResolution {
    pub fn new() -> Self {
        Self {
            conflict_id: String::new(),
            conflict_type: ConflictType::Resource,
            conflict_severity: ConflictSeverity::Medium,
            resolution_method: ResolutionMethod::Negotiation,
            resolution_outcome: ResolutionOutcome {
                method: ResolutionMethod::Negotiation,
                outcome: ResolutionOutcomeType::Unresolved,
                resolution_details: String::new(),
                participant_satisfaction: HashMap::new(),
            },
            resolution_time_ms: 0,
            resolution_quality: 0.0,
        }
    }
}

/// Resolution Outcome
#[derive(Debug, Clone)]
pub struct ResolutionOutcome {
    pub method: ResolutionMethod,
    pub outcome: ResolutionOutcomeType,
    pub resolution_details: String,
    pub participant_satisfaction: HashMap<String, f32>,
}

/// Resolution Outcome Type
#[derive(Debug, Clone)]
pub enum ResolutionOutcomeType {
    Resolved,
    PartiallyResolved,
    Escalated,
    Unresolved,
}

/// Resolution Strategy Type
#[derive(Debug, Clone)]
pub enum ResolutionStrategyType {
    Negotiation,
    Arbitration,
    Mediation,
    Consensus,
    Compromise,
    Escalation,
    Voting,
}

use super::super::config as nexum_config;

impl From<nexum_config::ConflictResolutionStrategy> for ConflictResolutionStrategy {
    fn from(c: nexum_config::ConflictResolutionStrategy) -> Self {
        match c {
            nexum_config::ConflictResolutionStrategy::Negotiation => Self::Negotiation,
            nexum_config::ConflictResolutionStrategy::Arbitration => Self::Arbitration,
            nexum_config::ConflictResolutionStrategy::Mediation => Self::Mediation,
            nexum_config::ConflictResolutionStrategy::ConsensusBuilding => Self::ConsensusBuilding,
            nexum_config::ConflictResolutionStrategy::Compromise => Self::Compromise,
            nexum_config::ConflictResolutionStrategy::Escalation => Self::Escalation,
        }
    }
}

impl From<nexum_config::ResolutionMethod> for ResolutionMethod {
    fn from(c: nexum_config::ResolutionMethod) -> Self {
        match c {
            nexum_config::ResolutionMethod::Negotiation => Self::Negotiation,
            nexum_config::ResolutionMethod::Arbitration => Self::Arbitration,
            nexum_config::ResolutionMethod::Mediation => Self::Mediation,
            nexum_config::ResolutionMethod::Consensus => Self::Consensus,
            nexum_config::ResolutionMethod::Compromise => Self::Compromise,
            nexum_config::ResolutionMethod::Voting => Self::Voting,
        }
    }
}

impl From<nexum_config::ConflictResolutionStrategy> for ResolutionStrategyType {
    fn from(strategy: nexum_config::ConflictResolutionStrategy) -> Self {
        match strategy {
            nexum_config::ConflictResolutionStrategy::Negotiation => {
                ResolutionStrategyType::Negotiation
            }
            nexum_config::ConflictResolutionStrategy::Arbitration => {
                ResolutionStrategyType::Arbitration
            }
            nexum_config::ConflictResolutionStrategy::Mediation => {
                ResolutionStrategyType::Mediation
            }
            nexum_config::ConflictResolutionStrategy::ConsensusBuilding => {
                ResolutionStrategyType::Consensus
            }
            nexum_config::ConflictResolutionStrategy::Compromise => {
                ResolutionStrategyType::Compromise
            }
            nexum_config::ConflictResolutionStrategy::Escalation => {
                ResolutionStrategyType::Escalation
            }
        }
    }
}
