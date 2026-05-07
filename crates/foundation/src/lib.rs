//! Foundation AI Components for Nexora
//! 
//! Shared tensor operations, validation, and core utilities
//! for all AI frameworks in the Nexora ecosystem.

// Include framework modules as separate crates
pub mod tensor;
pub mod validation;

// Re-export main components for easier access
pub use tensor::*;
pub use validation::*;
