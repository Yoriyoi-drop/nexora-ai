//! Type definitions for HAS-MoE-FFN

use serde::{Serialize, Deserialize};

/// Expert output type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertOutput {
    pub expert_id: usize,
    pub output: Vec<f32>,
    pub confidence: f32,
}

/// Routing weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingWeights {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub expert_id: usize,
    pub confidence: f32,
}

/// MoE output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEOutput {
    pub combined_output: Vec<f32>,
    pub expert_contributions: Vec<ExpertOutput>,
    pub routing_info: RoutingWeights,
}
