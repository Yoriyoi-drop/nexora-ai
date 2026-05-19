//! SACA Prelude - Common imports and utilities

// Re-export main types (excluding ExecutionResult to avoid collision)
pub use super::types::{SACAExecutionResult as ExecutionResult, *};
pub use super::config::*;
pub use super::error::*;

// Re-export main components
pub use super::SACA;

// Re-export phase engines
pub use super::cot::CoTEngine;
pub use super::decompose::DecomposeEngine;
pub use super::context::ContextEngine;
pub use super::sampling::SamplingEngine;
pub use super::execute::ExecuteEngine;
pub use super::rerank::RerankEngine;
pub use super::feedback::FeedbackSystem;

// Re-export pipeline and integration
pub use super::pipeline::SACAPipeline;
pub use super::integration::{SACAIntegration, EnhancedSACASolution, SACAFactory};

// Common result type
pub type Result<T> = SACAResult<T>;

// Utility macros
#[macro_export]
macro_rules! saca_info {
    ($($arg:tt)*) => {
        tracing::info!($($arg)*);
    };
}

#[macro_export]
macro_rules! saca_debug {
    ($($arg:tt)*) => {
        tracing::debug!($($arg)*);
    };
}

#[macro_export]
macro_rules! saca_warn {
    ($($arg:tt)*) => {
        tracing::warn!($($arg)*);
    };
}

#[macro_export]
macro_rules! saca_error {
    ($($arg:tt)*) => {
        tracing::error!($($arg)*);
    };
}
