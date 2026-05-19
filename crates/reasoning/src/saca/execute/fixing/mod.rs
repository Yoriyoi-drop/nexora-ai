//! Fixing Module
//! 
//! This module contains error analysis and fix generation functionality.

pub mod analyzer;
pub mod generator;

// Re-export fixing components
pub use analyzer::ErrorAnalyzer;
pub use generator::FixGenerator;
