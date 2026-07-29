//! Sampling Strategies
//!
//! This module contains different sampling strategies for generating candidates.

pub mod diverse;
pub mod performance_focused;
pub mod quality_focused;
pub mod random;

// Re-export all strategies
pub use diverse::DiverseSamplingStrategy;
pub use performance_focused::PerformanceFocusedSamplingStrategy;
pub use quality_focused::QualityFocusedSamplingStrategy;
pub use random::RandomSamplingStrategy;
