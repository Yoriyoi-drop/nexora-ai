//! Core module for Nexora-AI functionality

pub mod chat;
pub mod debate;
pub mod generation;
pub mod processing;
pub mod system;
pub mod tier_router;
pub mod types;

// Re-export core types for backward compatibility
pub use processing::RequestProcessor;
pub use system::SystemMonitor;
pub use types::*;
