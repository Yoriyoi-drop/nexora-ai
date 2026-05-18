//! Validation utilities for Nexora AI
//! 
//! Common validation functions and types

use std::fmt;

/// Validation error type
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Validation error: {}", self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Validation result type
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validate tensor shape
pub fn validate_tensor_shape(shape: &[usize]) -> ValidationResult<()> {
    if shape.is_empty() {
        return Err(ValidationError {
            message: "Tensor shape cannot be empty".to_string(),
        });
    }
    
    if shape.iter().any(|&dim| dim == 0) {
        return Err(ValidationError {
            message: "Tensor dimensions cannot be zero".to_string(),
        });
    }
    
    Ok(())
}

/// Validate tensor data size matches shape
pub fn validate_tensor_data_size(data_len: usize, shape: &[usize]) -> ValidationResult<()> {
    let expected_size: usize = shape.iter().product();
    if data_len != expected_size {
        return Err(ValidationError {
            message: format!(
                "Data size {} does not match expected size {} for shape {:?}",
                data_len, expected_size, shape
            ),
        });
    }
    Ok(())
}
