//! Type definitions for HAS-MoE-FFN

use crate::domains::ExpertDomain;
use serde::{Deserialize, Serialize};

/// Expert output type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertOutput {
    pub expert_id: usize,
    pub output: Vec<f32>,
    pub confidence: f32,
    pub domain: Option<String>,
    pub tier: Option<String>,
}

/// Routing weights with domain info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingWeights {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
    pub domains: Vec<Option<String>>,
}

/// Routing decision per expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub expert_id: usize,
    pub confidence: f32,
    pub domain: Option<ExpertDomain>,
}

/// MoE output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEOutput {
    pub combined_output: Vec<f32>,
    pub expert_contributions: Vec<ExpertOutput>,
    pub routing_info: RoutingWeights,
}

/// Domain-aware routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRoutingConfig {
    /// Bias weight for shared experts (0.0–1.0).
    pub shared_bias: f64,
    /// Minimum number of shared experts in top-k.
    pub min_shared_experts: usize,
    /// How many domain-specific experts to force per token.
    pub tier_quota: usize,
    /// Whether to apply domain bias during routing.
    pub use_domain_bias: bool,
}

impl Default for DomainRoutingConfig {
    fn default() -> Self {
        Self {
            shared_bias: 0.3,
            min_shared_experts: 1,
            tier_quota: 4,
            use_domain_bias: true,
        }
    }
}
