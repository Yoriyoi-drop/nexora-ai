//! CLI module for Nexora-AI

pub mod commands;
pub mod handlers;
pub mod chat;
pub mod training;

// Re-export main CLI types
pub use commands::{Cli, Commands, ConfigAction, TokenizerAction, MemoryAction};
