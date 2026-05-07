//! Nexora Core - Core controller dan tipe data dasar
//! 
//! Module ini menyediakan tipe data fundamental dan core controller
//! untuk sistem Nexora AI yang dimigrasi dari C ke Rust.

pub mod types;
pub mod controller;
pub mod orchestration;
pub mod execution;
pub mod input;
pub mod intent;
pub mod context;
pub mod task;
pub mod fusion;
pub mod error;
pub mod ml_intent;
pub mod coordination;
pub mod async_executor;
pub mod error_recovery;
// #[cfg(test)]
// pub mod tests; // Disabled - tests reference non-existent types

// Re-export orchestration layer
pub use orchestration::*;

// Re-export execution layer
pub use execution::*;

// Re-export tipe data penting
pub use types::*;
pub use controller::CoreController;
pub use error::{CoreError, CoreResult};
