//! Error types for HAS-MoE-FFN

use thiserror::Error;

/// Main error type for HAS-MoE-FFN
#[derive(Error, Debug)]
pub enum HasMoeFfnError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Expert error: {0}")]
    Expert(String),
    
    #[error("Router error: {0}")]
    Router(String),
    
    #[error("Aggregation error: {0}")]
    Aggregation(String),
    
    #[error("Load balancer error: {0}")]
    LoadBalancer(String),
    
    #[error("Tensor operation error: {0}")]
    Tensor(String),
    
    #[error("Memory error: {0}")]
    Memory(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Index out of bounds: expert {index}, total {total}")]
    IndexOutOfBounds { index: usize, total: usize },
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Expert not found: {expert_id}")]
    ExpertNotFound { expert_id: usize },
    
    #[error("All experts are busy")]
    AllExpertsBusy,
    
    #[error("Load balancing failed: {reason}")]
    LoadBalancingFailed { reason: String },
    
    #[error("Routing failed: {reason}")]
    RoutingFailed { reason: String },
    
    #[error("Aggregation failed: {reason}")]
    AggregationFailed { reason: String },
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("NDArray error: {0}")]
    NdArray(String),
}

impl HasMoeFfnError {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    
    pub fn expert(msg: impl Into<String>) -> Self {
        Self::Expert(msg.into())
    }
    
    pub fn router(msg: impl Into<String>) -> Self {
        Self::Router(msg.into())
    }
    
    pub fn aggregation(msg: impl Into<String>) -> Self {
        Self::Aggregation(msg.into())
    }
    
    pub fn load_balancer(msg: impl Into<String>) -> Self {
        Self::LoadBalancer(msg.into())
    }
    
    pub fn tensor(msg: impl Into<String>) -> Self {
        Self::Tensor(msg.into())
    }
    
    pub fn memory(msg: impl Into<String>) -> Self {
        Self::Memory(msg.into())
    }
    
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
    
    pub fn index_out_of_bounds(index: usize, total: usize) -> Self {
        Self::IndexOutOfBounds { index, total }
    }
    
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }
    
    pub fn expert_not_found(expert_id: usize) -> Self {
        Self::ExpertNotFound { expert_id }
    }
    
    pub fn all_experts_busy() -> Self {
        Self::AllExpertsBusy
    }
    
    pub fn load_balancing_failed(reason: impl Into<String>) -> Self {
        Self::LoadBalancingFailed { reason: reason.into() }
    }
    
    pub fn routing_failed(reason: impl Into<String>) -> Self {
        Self::RoutingFailed { reason: reason.into() }
    }
    
    pub fn aggregation_failed(reason: impl Into<String>) -> Self {
        Self::AggregationFailed { reason: reason.into() }
    }
    
    pub fn training(msg: impl Into<String>) -> Self {
        Self::Training(msg.into())
    }
    
    pub fn ndarray(msg: impl Into<String>) -> Self {
        Self::NdArray(msg.into())
    }
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, HasMoeFfnError>;

impl From<crate::atqs::error::ATQSError> for HasMoeFfnError {
    fn from(error: crate::atqs::error::ATQSError) -> Self {
        Self::Aggregation(format!("ATQS error: {:?}", error))
    }
}
