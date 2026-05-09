//! Utility functions for validation framework

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Validation utility functions
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate tensor shape
    pub fn validate_tensor_shape(shape: &[usize], expected: &[usize]) -> bool {
        shape == expected
    }

    /// Check if value is within valid range
    pub fn is_in_range<T: PartialOrd>(value: T, min: T, max: T) -> bool {
        value >= min && value <= max
    }

    /// Validate string format
    pub fn validate_string_format(value: &str, pattern: &str) -> bool {
        // Simple pattern matching - in real implementation, use regex
        value.contains(pattern)
    }

    /// Calculate validation statistics
    pub fn calculate_stats(results: &[bool]) -> HashMap<String, f32> {
        let total = results.len() as f32;
        let passed = results.iter().filter(|&&x| x).count() as f32;
        
        let mut stats = HashMap::new();
        stats.insert("pass_rate".to_string(), passed / total);
        stats.insert("fail_rate".to_string(), (total - passed) / total);
        stats.insert("total_checks".to_string(), total);
        
        stats
    }
}

/// Common validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    /// Value must be within range
    Range { min: f32, max: f32 },
    /// Value must not be empty
    NotEmpty,
    /// Value must match pattern
    Pattern(String),
    /// Value must be positive
    Positive,
    /// Custom validation function
    Custom(String),
}

impl ValidationRule {
    /// Validate a value against this rule
    pub fn validate(&self, value: &str) -> bool {
        match self {
            ValidationRule::Range { min, max } => {
                if let Ok(val) = value.parse::<f32>() {
                    ValidationUtils::is_in_range(val, *min, *max)
                } else {
                    false
                }
            }
            ValidationRule::NotEmpty => !value.is_empty(),
            ValidationRule::Pattern(pattern) => {
                ValidationUtils::validate_string_format(value, pattern)
            }
            ValidationRule::Positive => {
                if let Ok(val) = value.parse::<f32>() {
                    val > 0.0
                } else {
                    false
                }
            }
            ValidationRule::Custom(_) => {
                // In real implementation, call custom validation function
                true
            }
        }
    }
}
