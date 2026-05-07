//! Error handling untuk Nexora Core

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum CoreError {
    #[error("Input validation failed: {0}")]
    InputValidation(String),
    
    #[error("Intent detection failed: {0}")]
    IntentDetection(String),
    
    #[error("Routing failed: {0}")]
    Routing(String),
    
    #[error("Memory access failed: {0}")]
    MemoryAccess(String),
    
    #[error("Task execution failed: {0}")]
    TaskExecution(String),
    
    #[error("General error: {0}")]
    General(String),
    
    #[error("Fusion failed: {0}")]
    Fusion(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Model not available: {model_id}")]
    ModelNotAvailable { model_id: u8 },
    
    #[error("Task limit exceeded: {current}/{max}")]
    TaskLimitExceeded { current: usize, max: usize },
    
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

impl From<anyhow::Error> for CoreError {
    fn from(err: anyhow::Error) -> Self {
        CoreError::General(err.to_string())
    }
}

pub type CoreResult<T> = Result<T, CoreError>;
