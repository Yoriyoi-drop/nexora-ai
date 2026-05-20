//! Large-Scale Sampling Engine
//!
//! Phase 4 of SACA: Generate N ≥ 5 diverse implementations for each module
//! Implements diversity sampling to maximize optimal solution discovery
//!
//! This module has been refactored from a single large file into modular components:
//! - engine.rs: Core sampling engine implementation
//! - strategies/: Different sampling strategies (random, diverse, quality-focused, performance-focused)
//! - generators/: Algorithm generators for different implementation approaches

pub mod engine;
pub mod generators;
pub mod strategies;

// Re-export main components for backward compatibility
pub use engine::{DiversityCalculator, SamplingEngine};
pub use generators::*;
pub use strategies::*;
