//! Execute-Fail-Fix Loop Engine
//!
//! Phase 5 of SACA: Execute candidates, capture errors, and fix iteratively
//! Implements self-debugging with real error log analysis
//!
//! This module has been refactored from a single large file into modular components:
//! - engine.rs: Core execution engine implementation
//! - strategies/: Different execution strategies (sequential, parallel, adaptive)
//! - testing/: Test generation and execution functionality
//! - fixing/: Error analysis and fix generation functionality

pub mod engine;
pub mod fixing;
pub mod strategies;
pub mod testing;

// Re-export main components for backward compatibility
pub use engine::{
    CodeExecutor, ExecuteEngine, ExecutionEnvironment, ExecutionOutput, PerformanceMetrics,
    PerformanceMonitor, SandboxCodeExecutor,
};
pub use fixing::*;
pub use strategies::*;
pub use testing::*;
