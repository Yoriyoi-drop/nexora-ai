//! Execution Strategies
//! 
//! This module contains different execution strategies for running candidates.

pub mod sequential;
pub mod parallel;
pub mod adaptive;

// Re-export all strategies
pub use sequential::SequentialExecutionStrategy;
pub use parallel::ParallelExecutionStrategy;
pub use adaptive::AdaptiveExecutionStrategy;
