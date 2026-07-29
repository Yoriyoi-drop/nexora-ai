//! Foundation model error types.
//!
//! Single responsibility: define and implement error types for the foundation model layer.

use std::fmt;

/// Error type for foundation model operations (checkpoint loading, initialization, etc.).
#[derive(Debug)]
pub struct FoundationError(pub String);

impl fmt::Display for FoundationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FoundationError {}

impl From<nexora_transformer::TransformerError> for FoundationError {
    fn from(e: nexora_transformer::TransformerError) -> Self {
        FoundationError(e.to_string())
    }
}
