//! Pattern Detectors untuk ORACLE
//!
//! Pattern detectors berbasis regex untuk kode yang memeriksa security,
//! performance, correctness, dan best practices berbagai bahasa pemrograman.
//!
//! NOTE: Detectors ini menggunakan **pattern matching (regex + string containment)**,
//! bukan semantic/static analysis. False positive/negative mungkin terjadi.
//! Untuk analisis mendalam, diperlukan AST-based analysis di masa depan.
//!
//! Module structure:
//! - manager.rs: Main linter manager
//! - security.rs: Security pattern detection
//! - performance.rs: Performance pattern detection
//! - correctness.rs: Correctness pattern detection
//! - style.rs: Style pattern detection

pub mod correctness;
pub mod manager;
pub mod performance;
pub mod security;
pub mod style;

// Re-export main components
pub use correctness::CorrectnessLinter;
pub use manager::CodeLinterManager;
pub use performance::PerformanceLinter;
pub use security::SecurityLinter;
pub use style::StyleLinter;

// Re-export types
pub use manager::{CodeIssue, CodeLinter, IssueSeverity, LintResult, LinterType, LintSummary};

// Backward-compat deprecated aliases
#[allow(deprecated)]
pub type VerifierType = LinterType;
#[allow(deprecated)]
pub type VerificationResult = LintResult;
#[allow(deprecated)]
pub type VerificationSummary = LintSummary;
#[allow(deprecated)]
pub type SecurityVerifier = SecurityLinter;
#[allow(deprecated)]
pub type PerformanceVerifier = PerformanceLinter;
#[allow(deprecated)]
pub type CorrectnessVerifier = CorrectnessLinter;
#[allow(deprecated)]
pub type StyleVerifier = StyleLinter;
