//! Logging Initialization Utilities
//! 
//! Helper functions untuk initialize logging system.

use anyhow::Result;
use super::{LoggingSystem, LoggingConfig};

/// Initialize logging with default config
pub fn init_default() -> Result<()> {
    let config = LoggingConfig::default();
    let mut logging = LoggingSystem::new(config);
    logging.init()
}

/// Initialize logging with minimal config
pub fn init_minimal() -> Result<()> {
    let config = LoggingConfig::minimal();
    let mut logging = LoggingSystem::new(config);
    logging.init()
}

/// Initialize logging with development config
pub fn init_development() -> Result<()> {
    let config = LoggingConfig::development();
    let mut logging = LoggingSystem::new(config);
    logging.init()
}

/// Initialize logging with production config
pub fn init_production() -> Result<()> {
    let config = LoggingConfig::production();
    let mut logging = LoggingSystem::new(config);
    logging.init()
}

/// Initialize logging with test config
pub fn init_test() -> Result<()> {
    let config = LoggingConfig::test();
    let mut logging = LoggingSystem::new(config);
    logging.init()
}

/// Initialize logging with custom config
pub fn init_with_config(config: LoggingConfig) -> Result<()> {
    // Validate config first
    config.validate()
        .map_err(|e| anyhow::anyhow!("Invalid logging config: {}", e))?;
    
    let mut logging = LoggingSystem::new(config);
    logging.init()
}

/// Initialize logging from environment variables
pub fn init_from_env() -> Result<()> {
    use std::env;
    
    let level = env::var("NEXORA_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    let console = env::var("NEXORA_LOG_CONSOLE").unwrap_or_else(|_| "true".to_string()) == "true";
    let file = env::var("NEXORA_LOG_FILE").unwrap_or_else(|_| "false".to_string()) == "true";
    let json_format = env::var("NEXORA_LOG_JSON").unwrap_or_else(|_| "false".to_string()) == "true";
    
    let file_dir = env::var("NEXORA_LOG_DIR").ok()
        .map(|dir| std::path::PathBuf::from(dir));
    
    let config = LoggingConfig {
        level,
        console,
        file,
        file_dir,
        json_format,
        ..LoggingConfig::default()
    };
    
    init_with_config(config)
}

/// Get logging system instance
pub fn get_logging_system(config: LoggingConfig) -> LoggingSystem {
    LoggingSystem::new(config)
}
