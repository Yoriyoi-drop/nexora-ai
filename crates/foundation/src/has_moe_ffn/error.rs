//! Error types for HAS-MoE-FFN

use thiserror::Error;

/// HAS-MoE-FFN specific errors
#[derive(Debug, Error)]
pub enum HasMoeFfnError {
    #[error("Routing error: {0}")]
    RoutingError(String),
    
    #[error("Expert error: {0}")]
    ExpertError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl HasMoeFfnError {
    /// Create a new routing error
    pub fn routing(msg: impl Into<String>) -> Self {
        Self::RoutingError(msg.into())
    }
    
    /// Create a new expert error
    pub fn expert(msg: impl Into<String>) -> Self {
        Self::ExpertError(msg.into())
    }
    
    /// Create a new configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }
    
    /// Create a new computation error
    pub fn computation(msg: impl Into<String>) -> Self {
        Self::ComputationError(msg.into())
    }
    
    /// Create a new invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}
