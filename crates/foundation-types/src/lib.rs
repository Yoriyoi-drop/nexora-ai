//! Type definitions for Nexora Foundation — minimal dependency crate.
//!
//! Contains `FoundationError` and `FoundationResult` used across the workspace.
//! This crate has ZERO internal dependencies so it compiles instantly.

pub type FoundationResult<T> = std::result::Result<T, FoundationError>;

#[derive(Debug, thiserror::Error)]
pub enum FoundationError {
    #[error("Implementation error: {0}")]
    Implementation(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Resource error: {0}")]
    Resource(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Timeout error")]
    Timeout,
}
