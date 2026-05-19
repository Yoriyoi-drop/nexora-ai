//! Sampling Strategies
//! 
//! This module contains different sampling strategies for generating candidates.

pub mod random;
pub mod diverse;
pub mod quality_focused;
pub mod performance_focused;

// Re-export all strategies
pub use random::RandomSamplingStrategy;
pub use diverse::DiverseSamplingStrategy;
pub use quality_focused::QualityFocusedSamplingStrategy;
pub use performance_focused::PerformanceFocusedSamplingStrategy;
