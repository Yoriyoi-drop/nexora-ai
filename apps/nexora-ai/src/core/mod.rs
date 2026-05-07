//! Core module for Nexora-AI functionality

pub mod types;
pub mod system;
pub mod generation;
pub mod chat;
pub mod processing;

// Re-export core types for backward compatibility
pub use types::*;
pub use system::SystemMonitor;
pub use generation::TextGenerator;
pub use chat::ChatEngine;
pub use processing::RequestProcessor;
