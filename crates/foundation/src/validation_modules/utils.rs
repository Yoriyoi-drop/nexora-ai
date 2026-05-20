//! Utility functions for validation framework

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    Range { min: f32, max: f32 },
    NotEmpty,
    Pattern(String),
    Positive,
    Custom(String),
}

impl ValidationRule {
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
            ValidationRule::Custom(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tensor_shape_exact_match() {
        assert!(ValidationUtils::validate_tensor_shape(&[3, 4], &[3, 4]));
    }

    #[test]
    fn test_validate_tensor_shape_mismatch() {
        assert!(!ValidationUtils::validate_tensor_shape(&[3, 4], &[4, 3]));
    }

    #[test]
    fn test_is_in_range_within() {
        assert!(ValidationUtils::is_in_range(5.0, 0.0, 10.0));
    }

    #[test]
    fn test_is_in_range_outside() {
        assert!(!ValidationUtils::is_in_range(15.0, 0.0, 10.0));
    }

    #[test]
    fn test_validate_string_format_contains() {
        assert!(ValidationUtils::validate_string_format(
            "hello world",
            "world"
        ));
        assert!(!ValidationUtils::validate_string_format(
            "hello world",
            "xyz"
        ));
    }

    #[test]
    fn test_calculate_stats_all_pass() {
        let stats = ValidationUtils::calculate_stats(&[true, true, true]);
        assert!((stats["pass_rate"] - 1.0).abs() < 1e-6);
        assert!((stats["fail_rate"] - 0.0).abs() < 1e-6);
        assert!((stats["total_checks"] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_stats_mixed() {
        let stats = ValidationUtils::calculate_stats(&[true, false, true]);
        assert!((stats["pass_rate"] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_validation_rule_range_valid() {
        let rule = ValidationRule::Range {
            min: 0.0,
            max: 100.0,
        };
        assert!(rule.validate("50"));
        assert!(!rule.validate("200"));
    }

    #[test]
    fn test_validation_rule_not_empty() {
        let rule = ValidationRule::NotEmpty;
        assert!(rule.validate("hello"));
        assert!(!rule.validate(""));
    }

    #[test]
    fn test_validation_rule_positive() {
        let rule = ValidationRule::Positive;
        assert!(rule.validate("5"));
        assert!(!rule.validate("-1"));
        assert!(!rule.validate("0"));
    }
}
