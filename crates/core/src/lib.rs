//! Nexora Core - Core controller dan tipe data dasar
//!
//! Module ini menyediakan tipe data fundamental dan core controller
//! untuk sistem Nexora AI yang dimigrasi dari C ke Rust.

pub mod async_executor;
pub mod context;
pub mod controller;
pub mod coordination;
pub mod error;
pub mod error_recovery;
pub use error_recovery::{
    global_error_recovery, with_circuit_breaker, CircuitBreaker, CircuitBreakerConfig,
    CircuitBreakerStats, CircuitState, ErrorCategory, ErrorInfo, ErrorRecoveryManager,
    ErrorSeverity, ErrorStats, RecoveryAction, RecoveryStrategy, RetryHandler, RetryPolicy,
};
pub mod execution;
pub mod fusion;
pub mod input;
pub mod intent;
pub mod ml_intent;
pub mod task;
pub mod types;
pub mod utils;

#[cfg(test)]
mod tests;

// Re-export execution layer
pub use execution::*;

// Re-export tipe data penting
pub use controller::CoreController;
pub use error::{CoreError, CoreResult};
pub use types::*;
