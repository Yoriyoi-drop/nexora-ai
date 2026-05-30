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

pub mod ast_analyzer;
pub mod correctness;
pub mod dep_scanner;
pub mod go_analyzer;
pub mod java_analyzer;
pub mod javascript_analyzer;
pub mod manager;
pub mod performance;
pub mod python_analyzer;
pub mod security;
pub mod style;

// Re-export main components
pub use correctness::CorrectnessLinter;
pub use manager::CodeLinterManager;
pub use performance::PerformanceLinter;
pub use security::{detect_language, SecurityLinter};
pub use style::StyleLinter;

// Re-export types
pub use manager::{CodeIssue, CodeLinter, IssueSeverity, LintResult, LintSummary, LinterType};

// Backward-compat aliases — prefer Linter-prefixed names
#[deprecated(note = "use LinterType instead")]
pub type VerifierType = LinterType;
#[deprecated(note = "use LintResult instead")]
pub type VerificationResult = LintResult;
#[deprecated(note = "use LintSummary instead")]
pub type VerificationSummary = LintSummary;
#[deprecated(note = "use SecurityLinter instead")]
pub type SecurityVerifier = SecurityLinter;
#[deprecated(note = "use PerformanceLinter instead")]
pub type PerformanceVerifier = PerformanceLinter;
#[deprecated(note = "use CorrectnessLinter instead")]
pub type CorrectnessVerifier = CorrectnessLinter;
#[deprecated(note = "use StyleLinter instead")]
pub type StyleVerifier = StyleLinter;
