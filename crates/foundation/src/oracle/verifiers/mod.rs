//! Code Verifiers untuk ORACLE
//! 
//! Verifiers khusus untuk kode yang memeriksa security,
//! efficiency, correctness, dan best practices
//! berbagai bahasa pemrograman.
//! 
//! This module has been refactored from a single large file into modular components:
//! - manager.rs: Main verifier manager
//! - security.rs: Security verification functionality
//! - performance.rs: Performance verification functionality
//! - correctness.rs: Correctness verification functionality
//! - style.rs: Style verification functionality

pub mod manager;
pub mod security;
pub mod performance;
pub mod correctness;
pub mod style;

// Re-export main components for backward compatibility
pub use manager::CodeVerifierManager;
pub use security::SecurityVerifier;
pub use performance::PerformanceVerifier;
pub use correctness::CorrectnessVerifier;
pub use style::StyleVerifier;

// Re-export types
pub use manager::{CodeVerifier, VerifierType, VerificationResult, CodeIssue, IssueSeverity};
