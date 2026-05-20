//! Execution Strategies
//!
//! This module contains different execution strategies for running candidates.

pub mod adaptive;
pub mod parallel;
pub mod sequential;

// Re-export all strategies
pub use adaptive::AdaptiveExecutionStrategy;
pub use parallel::ParallelExecutionStrategy;
pub use sequential::SequentialExecutionStrategy;
