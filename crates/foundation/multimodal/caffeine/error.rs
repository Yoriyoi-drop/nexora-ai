//! Error types for CAFFEINE

use thiserror::Error;

/// Main CAFFEINE error type
#[derive(Error, Debug)]
pub enum CaffeineError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Encoder error
    #[error("Encoder error: {0}")]
    Encoder(String),
    
    /// Q-Former error
    #[error("Q-Former error: {0}")]
    QFormer(String),
    
    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    
    /// Action head error
    #[error("Action head error: {0}")]
    ActionHead(String),
    
    /// ATQS compression error
    #[error("ATQS compression error: {0}")]
    AtqsCompression(String),
    
    /// HAS-MoE-FFN routing error
    #[error("HAS-MoE-FFN routing error: {0}")]
    HasMoeRouting(String),
    
    /// Input validation error
    #[error("Input validation error: {0}")]
    InputValidation(String),
    
    /// Output generation error
    #[error("Output generation error: {0}")]
    OutputGeneration(String),
    
    /// Memory allocation error
    #[error("Memory allocation error: {0}")]
    MemoryAllocation(String),
    
    /// Tensor operation error
    #[error("Tensor operation error: {0}")]
    TensorOperation(String),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// NDArray error
    #[error("NDArray error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    
    /// Generic error
    #[error("Generic error: {0}")]
    Generic(String),
}

/// Result type for CAFFEINE operations
pub type Result<T> = std::result::Result<T, CaffeineError>;

impl CaffeineError {
    /// Create a new configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }
    
    /// Create a new encoder error
    pub fn encoder<S: Into<String>>(msg: S) -> Self {
        Self::Encoder(msg.into())
    }
    
    /// Create a new Q-Former error
    pub fn qformer<S: Into<String>>(msg: S) -> Self {
        Self::QFormer(msg.into())
    }
    
    /// Create a new tokenizer error
    pub fn tokenizer<S: Into<String>>(msg: S) -> Self {
        Self::Tokenizer(msg.into())
    }
    
    /// Create a new action head error
    pub fn action_head<S: Into<String>>(msg: S) -> Self {
        Self::ActionHead(msg.into())
    }
    
    /// Create a new input validation error
    pub fn input_validation<S: Into<String>>(msg: S) -> Self {
        Self::InputValidation(msg.into())
    }
    
    /// Create a new output generation error
    pub fn output_generation<S: Into<String>>(msg: S) -> Self {
        Self::OutputGeneration(msg.into())
    }
    
    /// Create a new memory allocation error
    pub fn memory_allocation<S: Into<String>>(msg: S) -> Self {
        Self::MemoryAllocation(msg.into())
    }
    
    /// Create a new tensor operation error
    pub fn tensor_operation<S: Into<String>>(msg: S) -> Self {
        Self::TensorOperation(msg.into())
    }
    
    /// Create a new generic error
    pub fn generic<S: Into<String>>(msg: S) -> Self {
        Self::Generic(msg.into())
    }
}

/// Convert ATQS errors to CAFFEINE errors
impl From<crate::atqs::error::ATQSError> for CaffeineError {
    fn from(err: crate::atqs::error::ATQSError) -> Self {
        Self::AtqsCompression(format!("ATQS error: {}", err))
    }
}

/// Convert HAS-MoE-FFN errors to CAFFEINE errors
impl From<crate::has_moe_ffn::error::HasMoeFfnError> for CaffeineError {
    fn from(err: crate::has_moe_ffn::error::HasMoeFfnError) -> Self {
        Self::HasMoeRouting(format!("HAS-MoE-FFN error: {}", err))
    }
}
