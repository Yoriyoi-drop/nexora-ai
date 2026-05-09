//! Configuration management module for Nexora-AI

pub mod core;
pub mod tokenizer;
pub mod memory;
pub mod server;
pub mod api;
pub mod logging;
pub mod utils;
pub mod loader;

// Re-export main configuration types
pub use loader::NexoraConfig;
pub use core::CoreConfig;
pub use tokenizer::{TokenizerConfig, SpecialTokens};
pub use memory::MemoryConfig;
pub use server::ServerConfig;
pub use api::ApiConfig;
pub use logging::LoggingConfig;
pub use utils::UtilsConfig;
