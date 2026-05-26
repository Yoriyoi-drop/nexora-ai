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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError {
            message: "test error".to_string(),
        };
        assert_eq!(format!("{}", err), "Validation error: test error");
    }

    #[test]
    fn test_validation_error_impl_error() {
        let err = ValidationError {
            message: "test".to_string(),
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_validate_tensor_shape_empty() {
        let result = validate_tensor_shape(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("empty"));
    }

    #[test]
    fn test_validate_tensor_shape_zero_dim() {
        let result = validate_tensor_shape(&[2, 0, 3]);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("zero"));
    }

    #[test]
    fn test_validate_tensor_shape_valid() {
        assert!(validate_tensor_shape(&[1]).is_ok());
        assert!(validate_tensor_shape(&[2, 3, 4]).is_ok());
        assert!(validate_tensor_shape(&[256, 256, 3]).is_ok());
    }

    #[test]
    fn test_validate_tensor_data_size_match() {
        assert!(validate_tensor_data_size(12, &[2, 3, 2]).is_ok());
        assert!(validate_tensor_data_size(1, &[1]).is_ok());
        assert!(validate_tensor_data_size(0, &[0]).is_err());
    }

    #[test]
    fn test_validate_tensor_data_size_mismatch() {
        let result = validate_tensor_data_size(10, &[2, 3]);
        assert!(result.is_err());
        let msg = result.unwrap_err().message;
        assert!(msg.contains("10"));
        assert!(msg.contains("6"));
    }

    #[test]
    fn test_validate_tensor_data_size_large() {
        assert!(validate_tensor_data_size(1000, &[10, 10, 10]).is_ok());
    }
}
