//! Consensus building and voting mechanisms

use std::collections::HashMap;

/// Consensus System
#[derive(Debug, Clone)]
pub struct ConsensusSystem {
    /// Consensus algorithm
    pub algorithm: ConsensusAlgorithm,
    /// Voting mechanism
    pub voting_mechanism: VotingMechanism,
    /// Consensus builder
    pub consensus_builder: ConsensusBuilder,
    /// Consensus metrics
    pub metrics: ConsensusMetrics,
}

/// Consensus Algorithm
#[derive(Debug, Clone)]
pub enum ConsensusAlgorithm {
    /// Majority voting
    MajorityVoting,
    /// Weighted voting
    WeightedVoting,
    /// Consensus ranking
    ConsensusRanking,
    /// Delphi method
    DelphiMethod,
    /// Byzantine fault tolerance
    ByzantineFaultTolerance,
    /// Practical Byzantine fault tolerance
    PracticalByzantineFaultTolerance,
    /// Raft consensus
    Raft,
    /// Hybrid consensus
    Hybrid { algorithms: Vec<ConsensusAlgorithm> },
}

/// Voting Mechanism
#[derive(Debug, Clone)]
pub enum VotingMechanism {
    /// Simple voting
    Simple,
    /// Weighted voting
    Weighted,
    /// Ranked voting
    Ranked,
    /// Approval voting
    Approval,
    /// Delegated voting
    Delegated,
    /// Quadratic voting
    Quadratic,
}

/// Consensus Builder
#[derive(Debug, Clone)]
pub struct ConsensusBuilder {
    /// Builder type
    pub builder_type: ConsensusBuilderType,
    /// Consensus threshold
    pub threshold: f32,
    /// Timeout
    pub timeout_ms: u64,
    /// Builder metrics
    pub metrics: ConsensusBuilderMetrics,
}

/// Consensus Builder Type
#[derive(Debug, Clone)]
pub enum ConsensusBuilderType {
    /// Incremental builder
    Incremental,
    /// Batch builder
    Batch,
    /// Real-time builder
    RealTime,
    /// Optimized builder
    Optimized,
}

/// Consensus Builder Metrics
#[derive(Debug, Clone)]
pub struct ConsensusBuilderMetrics {
    /// Consensus success rate
    pub success_rate: f32,
    /// Average consensus time
    pub avg_consensus_time_ms: f64,
    /// Participation rate
    pub participation_rate: f32,
    /// Consensus quality
    pub consensus_quality: f32,
}

/// Consensus Metrics
#[derive(Debug, Clone)]
pub struct ConsensusMetrics {
    /// Consensus accuracy
    pub consensus_accuracy: f32,
    /// Consensus speed
    pub consensus_speed: f32,
    /// Consensus quality
    pub consensus_quality: f32,
    /// Consensus efficiency
    pub consensus_efficiency: f32,
}

/// Consensus Result
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub votes: HashMap<String, Vote>,
    pub consensus_outcome: ConsensusOutcome,
    pub consensus_time_ms: u64,
    pub consensus_quality: f32,
}

impl ConsensusResult {
    pub fn new() -> Self {
        Self {
            votes: HashMap::new(),
            consensus_outcome: ConsensusOutcome {
                consensus_reached: false,
                winning_option: String::new(),
                vote_counts: HashMap::new(),
                consensus_strength: 0.0,
            },
            consensus_time_ms: 0,
            consensus_quality: 0.0,
        }
    }
}

/// Vote
#[derive(Debug, Clone)]
pub struct Vote {
    pub voter: String,
    pub selected_option: String,
    pub vote_weight: f32,
    pub vote_time: chrono::DateTime<chrono::Utc>,
}

/// Consensus Outcome
#[derive(Debug, Clone)]
pub struct ConsensusOutcome {
    pub consensus_reached: bool,
    pub winning_option: String,
    pub vote_counts: HashMap<String, usize>,
    pub consensus_strength: f32,
}

use super::super::config as nexum_config;

impl From<nexum_config::ConsensusAlgorithm> for ConsensusAlgorithm {
    fn from(c: nexum_config::ConsensusAlgorithm) -> Self {
        match c {
            nexum_config::ConsensusAlgorithm::MajorityVoting => Self::MajorityVoting,
            nexum_config::ConsensusAlgorithm::WeightedVoting => Self::WeightedVoting,
            nexum_config::ConsensusAlgorithm::ConsensusRanking => Self::ConsensusRanking,
            nexum_config::ConsensusAlgorithm::DelphiMethod => Self::DelphiMethod,
            nexum_config::ConsensusAlgorithm::ByzantineFaultTolerance => {
                Self::ByzantineFaultTolerance
            }
            nexum_config::ConsensusAlgorithm::PracticalByzantineFaultTolerance => {
                Self::PracticalByzantineFaultTolerance
            }
            nexum_config::ConsensusAlgorithm::Raft => Self::Raft,
            nexum_config::ConsensusAlgorithm::Hybrid { algorithms } => Self::Hybrid {
                algorithms: algorithms.into_iter().map(Into::into).collect(),
            },
        }
    }
}

impl From<nexum_config::VotingMechanism> for VotingMechanism {
    fn from(c: nexum_config::VotingMechanism) -> Self {
        match c {
            nexum_config::VotingMechanism::Simple => Self::Simple,
            nexum_config::VotingMechanism::Weighted => Self::Weighted,
            nexum_config::VotingMechanism::Ranked => Self::Ranked,
            nexum_config::VotingMechanism::Approval => Self::Approval,
            nexum_config::VotingMechanism::Delegated => Self::Delegated,
            nexum_config::VotingMechanism::Quadratic => Self::Quadratic,
        }
    }
}
