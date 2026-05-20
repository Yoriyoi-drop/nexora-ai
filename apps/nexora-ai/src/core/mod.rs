//! Core module for Nexora-AI functionality

pub mod chat;
pub mod generation;
pub mod processing;
pub mod system;
pub mod types;

// Re-export core types for backward compatibility
pub use chat::ChatEngine;
pub use generation::TextGenerator;
pub use processing::RequestProcessor;
pub use system::SystemMonitor;
pub use types::*;
