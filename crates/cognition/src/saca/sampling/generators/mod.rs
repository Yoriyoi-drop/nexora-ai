//! Algorithm Generators for Sampling
//!
//! This module contains various algorithm generators that create
//! different implementation approaches for sampled candidates.

use super::super::error::*;
use super::super::types::*;

/// Trait for algorithm generators
pub trait AlgorithmGenerator: Send + Sync {
    fn generate(
        &self,
        module: &Module,
        context: &RepositoryContext,
        cot_result: &CoTResult,
        algorithm_type: AlgorithmType,
    ) -> SACAResult<SamplingCandidate>;
}

/// Algorithm types for sampling
#[derive(Debug, Clone)]
pub enum AlgorithmType {
    Standard,
    Optimized,
    Alternative,
    Experimental,
    Hybrid,
    Random,
}

pub mod alternative;
pub mod experimental;
pub mod hybrid;
pub mod optimized;
pub mod standard;

// Re-export all generators
pub use alternative::AlternativeAlgorithmGenerator;
pub use experimental::ExperimentalAlgorithmGenerator;
pub use hybrid::HybridAlgorithmGenerator;
pub use optimized::OptimizedAlgorithmGenerator;
pub use standard::StandardAlgorithmGenerator;
