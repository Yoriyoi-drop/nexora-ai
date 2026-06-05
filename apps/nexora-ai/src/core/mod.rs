//! Core module for Nexora-AI functionality

#[cfg(feature = "legacy-fastpath")]
pub mod chat;
pub mod debate;
#[cfg(feature = "legacy-fastpath")]
pub mod generation;
pub mod processing;
pub mod system;
pub mod tier_router;
pub mod types;

// Re-export core types for backward compatibility
#[cfg(feature = "legacy-fastpath")]
pub use chat::ChatEngine;
#[cfg(feature = "legacy-fastpath")]
pub use generation::TextGenerator;
pub use processing::RequestProcessor;
pub use system::SystemMonitor;
pub use types::*;
