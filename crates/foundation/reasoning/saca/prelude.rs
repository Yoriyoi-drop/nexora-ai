//! SACA Prelude - Common imports and utilities

// Re-export main types (excluding ExecutionResult to avoid collision)
pub use crate::saca::types::{SACAExecutionResult as ExecutionResult, *};
pub use crate::saca::config::*;
pub use crate::saca::error::*;

// Re-export main components
pub use crate::saca::SACA;

// Re-export phase engines
pub use crate::saca::cot::CoTEngine;
pub use crate::saca::decompose::DecomposeEngine;
pub use crate::saca::context::ContextEngine;
pub use crate::saca::sampling::SamplingEngine;
pub use crate::saca::execute::ExecuteEngine;
pub use crate::saca::rerank::RerankEngine;
pub use crate::saca::feedback::FeedbackSystem;

// Re-export pipeline and integration
pub use crate::saca::pipeline::SACAPipeline;
pub use crate::saca::integration::{SACAIntegration, EnhancedSACASolution, SACAFactory};

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
