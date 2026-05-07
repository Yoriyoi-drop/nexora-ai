//! Core types for HAS-MoE-FFN implementation

use ndarray::{ArrayD, Array2, Array1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Expert specialization types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExpertSpecialization {
    General,
    Reasoning,
    Coding,
    Mathematics,
    Language,
    Knowledge,
    Creative,
    Analytical,
}

/// Routing decision for expert selection
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub expert_id: usize,
    pub confidence: f32,
    pub specialization: ExpertSpecialization,
    pub input_tokens: Vec<usize>,
}

/// Expert output with metadata
#[derive(Debug, Clone)]
pub struct ExpertOutput {
    pub expert_id: usize,
    pub output: ArrayD<f32>,
    pub confidence: f32,
    pub computation_cost: f32,
    pub specialization: ExpertSpecialization,
}

/// Load balancing metrics
#[derive(Debug, Clone)]
pub struct LoadMetrics {
    pub expert_id: usize,
    pub current_load: f32,
    pub average_response_time: f32,
    pub queue_length: usize,
    pub success_rate: f32,
}

/// Structured matrix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredMatrixConfig {
    pub rank: usize,
    pub block_size: usize,
    pub sparsity_ratio: f32,
    pub use_low_rank: bool,
    pub use_block_diagonal: bool,
}

impl Default for StructuredMatrixConfig {
    fn default() -> Self {
        Self {
            rank: 64,
            block_size: 128,
            sparsity_ratio: 0.1,
            use_low_rank: true,
            use_block_diagonal: true,
        }
    }
}

/// SwiGLU expert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiGLUExpertConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub matrix_config: StructuredMatrixConfig,
    pub specialization: ExpertSpecialization,
    pub activation_dropout: f32,
}

impl Default for SwiGLUExpertConfig {
    fn default() -> Self {
        Self {
            input_dim: 4096,
            hidden_dim: 11008,
            output_dim: 4096,
            matrix_config: StructuredMatrixConfig::default(),
            specialization: ExpertSpecialization::General,
            activation_dropout: 0.0,
        }
    }
}

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub temperature: f32,
    pub use_context_analysis: bool,
    pub load_balance_factor: f32,
    pub capacity_factor: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            temperature: 1.0,
            use_context_analysis: true,
            load_balance_factor: 0.1,
            capacity_factor: 1.25,
        }
    }
}

/// Aggregation layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub method: AggregationMethod,
    pub learnable_weights: bool,
    pub attention_mechanism: bool,
    pub normalization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    WeightedSum,
    Attention,
    Gating,
    LearnedMixing,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            method: AggregationMethod::Attention,
            learnable_weights: true,
            attention_mechanism: true,
            normalization: true,
        }
    }
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    pub strategy: LoadBalancingStrategy,
    pub max_queue_length: usize,
    pub timeout_ms: u64,
    pub rebalance_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin,
    Adaptive,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::Adaptive,
            max_queue_length: 100,
            timeout_ms: 5000,
            rebalance_interval_ms: 1000,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_tokens_processed: usize,
    pub average_expert_utilization: f32,
    pub load_balance_score: f32,
    pub average_response_time_ms: f32,
    pub throughput_tokens_per_second: f32,
    pub memory_usage_mb: f32,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_tokens_processed: 0,
            average_expert_utilization: 0.0,
            load_balance_score: 0.0,
            average_response_time_ms: 0.0,
            throughput_tokens_per_second: 0.0,
            memory_usage_mb: 0.0,
        }
    }
}

/// Context analysis result
#[derive(Debug, Clone)]
pub struct ContextAnalysis {
    pub content_type: ExpertSpecialization,
    pub complexity_score: f32,
    pub required_experts: Vec<usize>,
    pub confidence_scores: Vec<f32>,
}

/// Training state for experts
#[derive(Debug, Clone)]
pub struct ExpertTrainingState {
    pub expert_id: usize,
    pub training_steps: usize,
    pub loss: f32,
    pub gradient_norm: f32,
    pub learning_rate: f32,
    pub last_update: std::time::Instant,
}
