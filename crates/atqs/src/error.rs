//! Error types for ATQS (Attention Tensor Quantum System)

use thiserror::Error;

/// ATQS-specific error type
#[derive(Error, Debug)]
pub enum ATQSError {
    #[error("Tensor computation error: {0}")]
    TensorError(String),
    
    #[error("Calibration error: {0}")]
    CalibrationError(String),
    
    #[error("Compression error: {0}")]
    CompressionError(String),
    
    #[error("Profiling error: {0}")]
    ProfilingError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
    
    #[error("Generic error: {0}")]
    GenericError(#[from] Box<dyn std::error::Error>),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type ATQSResult<T> = Result<T, ATQSError>;
