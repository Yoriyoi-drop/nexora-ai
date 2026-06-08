//! Core module for Nexora-AI functionality

pub mod debate;
pub mod processing;
pub mod system;
pub mod tier_router;
pub mod types;

// Re-export core types for backward compatibility
pub use processing::RequestProcessor;
pub use system::SystemMonitor;
pub use types::*;
