//! Foundation Runtime Abstraction Layer
//! 
//! Shared runtime utilities and abstractions for all AI frameworks

pub mod executor;
pub mod scheduler;
pub mod resource;
pub mod monitoring;

// Re-export main components
pub use executor::*;
pub use scheduler::*;
pub use resource::*;
pub use monitoring::*;

/// Foundation runtime abstraction layer
/// 
/// Provides shared runtime utilities for:
/// - Async execution patterns
/// - Task scheduling
/// - Resource management
/// - Performance monitoring
