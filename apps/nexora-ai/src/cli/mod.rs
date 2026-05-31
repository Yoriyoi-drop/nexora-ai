//! CLI module for Nexora-AI

pub mod benchmark;
pub mod chat;
pub mod commands;
pub mod handlers;
pub mod training;

// Re-export main CLI types
pub use commands::{Cli, Commands, ConfigAction, MemoryAction, TokenizerAction};
